from utils import TX_OVERHEAD_SIZE, P2WPKH_INPUT_SIZE, P2WPKH_OUTPUT_SIZE


class Transaction:
    """A Transaction in the mempool relevant for the WT wallet."""

    def __init__(self, broadcast_height, txins, txouts):
        self.broadcast_height = broadcast_height
        self.txins = txins
        self.txouts = txouts

    def get_fee(self):
        input_total = sum([c.amount for c in self.ins])
        output_total = sum([c.amount for c in self.outs])
        fee = input_total - output_total
        return fee

    def get_size(self):
        return (
            TX_OVERHEAD_SIZE
            + len(self.txins) * P2WPKH_INPUT_SIZE
            + len(self.txouts) * P2WPKH_OUTPUT_SIZE
        )


class ConsolidateFanoutTx(Transaction):
    pass


class CancelTx(Transaction):
    pass


class SpendTx(Transaction):
    pass
