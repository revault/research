from utils import (
    TX_OVERHEAD_SIZE,
    P2WPKH_INPUT_SIZE,
    P2WPKH_OUTPUT_SIZE,
    CANCEL_TX_WEIGHT,
    CANCEL_TX_PRESIGNED_FEERATE,
)


class Transaction:
    """A Transaction in the mempool relevant for the WT wallet."""

    def __init__(self, broadcast_height):
        self.broadcast_height = broadcast_height

    def feerate(self):
        return self.fee / self.size


class ConsolidateFanoutTx(Transaction):
    def __init__(self, broadcast_height, txins, txouts):
        super().__init__(broadcast_height)
        self.txins = txins
        self.txouts = txouts

        input_total = sum([c.amount for c in self.txins])
        output_total = sum([c.amount for c in self.txouts])
        self.fee = input_total - output_total

        self.size = (
            TX_OVERHEAD_SIZE
            + len(self.txins) * P2WPKH_INPUT_SIZE
            + len(self.txouts) * P2WPKH_OUTPUT_SIZE
        )


class CancelTx(Transaction):
    def __init__(self, broadcast_height, vault_id, size_vb, fbcoins):
        super().__init__(broadcast_height)
        self.vault_id = vault_id
        self.size = size_vb
        # Note: we assume the presigned feerate to be 0, while it's 88 by
        # practical-revault specs
        self.fee = sum(c.amount for c in fbcoins)
        self.fbcoins = fbcoins
