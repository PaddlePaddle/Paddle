"""
All training events.

There are:

* BeginIteration
* EndIteration
* BeginPass
* EndPass

TODO(yuyang18): Complete it!
"""
__all__ = ['EndIteration', 'BeginIteration', 'BeginPass', 'EndPass']


class BeginPass(object):
    """
    Event On One Pass Training Start.
    """

    def __init__(self, pass_id, tester):
        self.pass_id = pass_id
        self.tester = tester


class EndPass(object):
    """
    Event On One Pass Training Complete.
    """

    def __init__(self, pass_id, tester):
        self.pass_id = pass_id
        self.tester = tester


class BeginIteration(object):
    """
    Event On One Batch Training Start.
    """

    def __init__(self, pass_id, batch_id, tester):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.tester = tester


class EndIteration(object):
    """
    Event On One Batch Training Complete.
    """

    def __init__(self, pass_id, batch_id, cost, tester):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost
        self.tester = tester
