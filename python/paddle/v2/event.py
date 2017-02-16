"""
All training events.
"""
__all__ = ['EndIteration']


class EndIteration(object):
    """
    Event On One Batch Training Complete.
    """

    def __init__(self, pass_id, batch_id, cost):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost
