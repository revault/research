# Size of the non-input and non-output parts of a Segwit tx, in virtual bytes
TX_OVERHEAD_SIZE = 10.5
# Size of a P2WPKH input in virtual bytes
P2WPKH_INPUT_SIZE = 68
# Size of a P2WPKH output in virtual bytes
P2WPKH_OUTPUT_SIZE = 31
# Maximum standard transaction size, in virtual bytes
MAX_TX_SIZE = 100_000

BLOCKS_PER_DAY = 144

# A mapping from the number of stakeholders and managers to the Cancel transaction
# weight. Generated in advance for performance reasons.
CANCEL_TX_WEIGHT = {
    2: {
        1: 607,
        2: 715,
        3: 822,
        4: 931,
        5: 1038,
        6: 1145,
        7: 1252,
        8: 1359,
        9: 1466,
        10: 1573,
    },
    3: {
        1: 771,
        2: 852,
        3: 959,
        4: 1066,
        5: 1173,
        6: 1280,
        7: 1387,
        8: 1494,
        9: 1601,
        10: 1708,
    },
    4: {
        1: 942,
        2: 987,
        3: 1094,
        4: 1201,
        5: 1308,
        6: 1415,
        7: 1522,
        8: 1629,
        9: 1736,
        10: 1843,
    },
    5: {
        1: 1111,
        2: 1147,
        3: 1229,
        4: 1336,
        5: 1443,
        6: 1550,
        7: 1657,
        8: 1764,
        9: 1871,
        10: 1978,
    },
    6: {
        1: 1280,
        2: 1316,
        3: 1364,
        4: 1471,
        5: 1578,
        6: 1685,
        7: 1792,
        8: 1899,
        9: 2006,
        10: 2113,
    },
    7: {
        1: 1449,
        2: 1485,
        3: 1520,
        4: 1606,
        5: 1713,
        6: 1820,
        7: 1927,
        8: 2034,
        9: 2141,
        10: 2248,
    },
    8: {
        1: 1618,
        2: 1654,
        3: 1689,
        4: 1741,
        5: 1848,
        6: 1955,
        7: 2062,
        8: 2169,
        9: 2276,
        10: 2384,
    },
    9: {
        1: 1787,
        2: 1823,
        3: 1858,
        4: 1893,
        5: 1983,
        6: 2090,
        7: 2197,
        8: 2304,
        9: 2411,
        10: 2518,
    },
    10: {
        1: 1956,
        2: 1992,
        3: 2027,
        4: 2062,
        5: 2118,
        6: 2225,
        7: 2332,
        8: 2439,
        9: 2546,
        10: 2653,
    },
}


# The feerate at which the Cancel transaction is presigned in sat/vb
CANCEL_TX_PRESIGNED_FEERATE = 88


def cf_tx_size(n_inputs, n_outputs):
    """Size of the consolidate-fanout transaction, in vbytes"""
    return (
        TX_OVERHEAD_SIZE + n_inputs * P2WPKH_INPUT_SIZE + n_outputs * P2WPKH_OUTPUT_SIZE
    )
