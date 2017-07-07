import logging


class Variable(object):
    def __init__(self, name='', ref=None):
        '''
        ref: Variable
            reference
        '''
        self.name = name
        self.ref = ref


class Tensor:
    pass


class Scope:
    def __init__(self, father=None):
        self.father = father
        self.dic = {}

    def lookup(self, key):
        return self.dic[key]

    def lookup_grad(self, key):
        return self.lookup(key + '__grad__')

    def update(self, key, var):
        self.dic[key] = var

    def update_grad(self, key, var):
        self.dic[key + '__grad__'] += var


global_scope = Scope()


class Net:
    def __init__(self, proto, name):
        self.name = name

    def forward(self):
        logging.warning('Net %s forward' % self.name)

    def backward(self):
        logging.warning('Net %s backward' % self.name)


def merge_var(li):
    '''
    merge multiple records to a Variable
    '''
    return Variable


class StepNet:
    def __init__(self,
                 net_desc,
                 scope,
                 pre_scope,
                 in_links,
                 out_links,
                 states,
                 name=''):
        '''
        net_desc: proto
        scope: Scope
        pre_scope: Scope
            previous stepnet's scope
        in_links: list of str
        out_links: list of str
        states: list of str
        '''
        self.scope = scope
        self.in_links = in_links
        self.out_links = out_links
        self.states = states
        self.net = Net(net_desc, name)

    def forward(self, inputs):
        self._copy_state_as_inlink()
        self.net.forward()

    def backward(self):
        self.net.backward()


class RNNOP(object):
    def __init__(self, scope, inputs, outputs, states, num_rnns):
        '''
        scope: Scope
        inputs: list of str
        outputs: list of str
        states: list of str
        '''
        self.scope = scope
        self.inputs = inputs
        self.outputs = outputs
        self.states = states
        self.num_rnns = num_rnns

        self._build_rnn_net()

    def forward(self, input_vars):
        self._apply_inlinks(input_vars)
        for step in self.stepnets:
            step.forward()
        self._apply_outlinks()

    def backward(self):
        for step in self.stepnets:
            step.backward()

    def _apply_inlinks(self, input_vars):
        '''
        input_vars: list of Variables
            NOTE each var is a tensor which store multiple records
        '''
        assert len(input_vars) == len(self.inputs)
        for input_id, input in enumerate(self.inputs):
            assert len(input_vars[input_id]) == self.num_rnns
            for id, scope in enumerate(self.stepnets):
                # set a stepnet's input
                scope.update(input, input_vars[input_id][id])

    def _apply_outlinks(self):
        for output_id, output in enumerate(self.outputs):
            var = []
            for id, scope in enumerate(self.stepnets):
                var.append(scope.lookup(output))
            var = merge_var(var)
            global_scope.update(output, var)

    def _apply_state_references(self):
        '''
        TODO step net config pre-state
        '''
        for state in self.states:
            pre_state_name = 'pre_' + state
            self.scope.update(
                pre_state_name, Variable(ref=self.scope.father.lookup(state)))

    def _build_rnn_net(self, net_desc):
        self.scopes = [Scope() for i in range()]
        self.stepnets = []
        for i in range(self.num_rnns):
            step = StepNet(
                net_desc,
                scope=self.scopes[i],
                pre_scope=self.scopes[i - 1],
                in_links=self.inputs,
                out_links=self.outputs,
                states=self.states,
                name='rnn_%d' % i)
            self.stepnets.append(step)
