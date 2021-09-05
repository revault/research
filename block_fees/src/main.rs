use bitcoin::{blockdata::constants::WITNESS_SCALE_FACTOR, consensus::encode::VarInt};
use blocks_iterator::Config;
use chrono::NaiveDateTime;
use structopt::StructOpt;

use std::{fs, io::Write, process, sync::mpsc::sync_channel};

fn metadata(block: &blocks_iterator::BlockExtra) -> (u64, Vec<u64>, Vec<u64>) {
    let mut total_weight = 0;
    let mut fees = Vec::with_capacity(block.block.txdata.len());
    let mut feerates = Vec::with_capacity(block.block.txdata.len());

    for tx in block.block.txdata[1..].iter() {
        // NOTE: there should not (tm) be any None here as we restrict the use of skip_prevout.
        if let Some(fee) = block.tx_fee(&tx) {
            let weight = tx.get_weight() as u64;

            total_weight += weight;

            fees.push(fee);
            feerates.push(fee / weight);
        }
    }

    (total_weight, fees, feerates)
}

fn mean(collection: &[u64]) -> u64 {
    if collection.len() == 0 {
        0
    } else {
        collection.iter().sum::<u64>() / collection.len() as u64
    }
}

fn block_weight(txs_weight: u64, n_txs: u64) -> u64 {
    let base_weight = WITNESS_SCALE_FACTOR * (80 + VarInt(n_txs).len());
    base_weight as u64 + txs_weight
}

/// Assumes a sorted slice
fn median(collection: &[u64]) -> u64 {
    if collection.len() == 0 {
        0
    } else if collection.len() % 2 == 0 {
        collection[collection.len() / 2]
    } else {
        mean(&collection[(collection.len() - 1) / 2..collection.len() / 2 + 1])
    }
}

fn main() {
    // We only care about the blocks directory, make sure to not skip the prevouts to be able to
    // compute the fees!
    let mut config = Config::from_args();
    config.skip_prevout = false;
    let network = config.network;

    let (send, recv) = sync_channel(1000);
    let handle = blocks_iterator::iterate(config, send);

    let mut out = fs::File::create("historical_fees.csv").unwrap_or_else(|e| {
        eprintln!("Error creating output file: {}", e);
        process::exit(1);
    });

    while let Some(block) = recv.recv().unwrap() {
        let n_txs = block.block.txdata.len() as u64;
        let (total_weight, mut fees, mut feerates) = metadata(&block);
        fees.sort();
        feerates.sort();

        if fees.len() == 0 {
            write!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                block.height,
                block.block_hash,
                block.block.header.difficulty(network),
                block_weight(total_weight, n_txs),
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                "NA",
                NaiveDateTime::from_timestamp(block.block.header.time as i64, 0),
            )
            .unwrap_or_else(|e| {
                eprintln!("Error writing to file: {}", e);
                process::exit(1);
            });
        } else {
            write!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                block.height,
                block.block_hash,
                block.block.header.difficulty(network),
                block_weight(total_weight, n_txs),
                mean(&fees),
                median(&fees),
                fees[0],
                fees[fees.len() - 1],
                mean(&feerates),
                median(&feerates),
                feerates[0],
                feerates[feerates.len() - 1],
                NaiveDateTime::from_timestamp(block.block.header.time as i64, 0)
            )
            .unwrap_or_else(|e| {
                eprintln!("Error writing to file: {}", e);
                process::exit(1);
            });
        }
    }

    handle.join().unwrap();
}
