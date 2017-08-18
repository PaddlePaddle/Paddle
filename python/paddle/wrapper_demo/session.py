'''
session stores all the vars, ops for topology alalysis.
'''
from common import logger, g_scope, g_device
from topology import Topology
import paddle.v2.framework.core as core


class Session(object):
    '''
    NOTE this maybe renamed to Context, but it acts like tf.Session
    only one Session is needed, so it maybe hiddened from user.
    '''

    def __init__(self, scope=g_scope, device=g_device):
        self.ops = {}
        self.vars = {}

        self.scope = scope
        self.device = device
        self.topology = Topology(self)

    def add_var(self, var):
        '''
        register a var in the context
        '''
        logger.info('register a var %s' % repr(var))
        assert repr(var) not in self.vars
        self.vars[repr(var)] = var

    def add_op(self, op):
        '''
        register a op in the context
        '''
        logger.info('register an op %s' % repr(op))
        assert repr(op) not in self.ops
        self.ops[repr(op)] = op

    def run(self, targets=[], need_backward=False):
        '''
        just run a sub-graph, compatible with paddle.infer
        '''
        self.topology.build()
        visited_nodes = self.topology.DFS_to_targets(targets)
        ops = filter(lambda _: _.startswith('<Op'), visited_nodes)

        # create a core.Net
        net = core.Net.create()
        for op in ops:
            core_op = op.op
            net.add(core_op)
        net.complete_add_op(True)

        ctx = core.DeviceContext.create(self.device)
        net.infer_shape(self.scope)
        net.run(self.scope, ctx)

    def deserialize(self, str):
        '''
        to load 
        '''
        pass

    def serialize(self):
        pass

    def __str__(self):
        pass


g_session = Session()
