use blocks_iterator::Config;
use structopt::StructOpt;

use std::{fs, io::Write, process, sync::mpsc::sync_channel};

/// NOTE: there should not (tm) be any None here as we restrict the use of skip_prevout.
fn block_fees(block: blocks_iterator::BlockExtra) -> Vec<u64> {
    block
        .block
        .txdata
        .iter()
        .filter_map(|tx| block.tx_fee(&tx))
        .collect()
}

fn mean(fees: &[u64]) -> u64 {
    fees.iter().sum::<u64>() / fees.len() as u64
}

/// Assumes a sorted slice
fn median(fees: &[u64]) -> u64 {
    if fees.len() % 2 == 0 {
        fees[fees.len() / 2]
    } else {
        mean(&fees[(fees.len() - 1) / 2..fees.len() / 2 + 1])
    }
}

fn main() {
    // We only care about the blocks directory, make sure to not skip the prevouts to be able to
    // compute the fees!
    let mut config = Config::from_args();
    config.skip_prevout = false;

    let (send, recv) = sync_channel(1000);
    let handle = blocks_iterator::iterate(config, send);

    let mut out = fs::File::create("historical_fees.csv").unwrap_or_else(|e| {
        eprintln!("Error creating output file: {}", e);
        process::exit(1);
    });

    while let Some(block) = recv.recv().unwrap() {
        let height = block.height;
        let mut fees = block_fees(block);
        fees.sort();

        write!(
            out,
            "{},{},{},{},{}\n",
            height,
            mean(&fees),
            median(&fees),
            fees[0],
            fees[fees.len() - 1],
        )
        .unwrap_or_else(|e| {
            eprintln!("Error writing to file: {}", e);
            process::exit(1);
        });
    }

    handle.join().unwrap();
}
