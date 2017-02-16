"""
All training events.

There are:

* BeginTraining
* EndTraining
* BeginIteration
* EndIteration
* BeginPass
* EndPass

TODO(yuyang18): Complete it!
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
