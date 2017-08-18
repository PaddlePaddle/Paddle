from session import g_session


class SGD(object):
    def __init__(self, cost, parameters, update_equatio, session=g_session):
        self.session = session
        self.cost = cost

    def init_parameters(self):
        '''
        # borrowed from tf, no need to expose parameters.create outside
        '''
        pass

    def train(self, reader, event_hander, num_passes, targets=None):
        if not targets:
            targets = [cost]
        self.session.run(targets, need_backward=True)

    def run(self, reader, targets):
        '''
        run a sub-graph
        '''
        self.session.run(targets)
