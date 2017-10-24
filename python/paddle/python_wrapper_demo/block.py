import paddle.v2.framework.core as core

__all__ = [
    'Block',
    'g_block',
    'block',
    'eval',
]


class Block(object):
    '''
    Block is the concept of code block, which has a sequence of local Variables
    and Operators.
    '''

    def __init__(self):
        '''
        namespace: str
        '''
        self.cmds = []

    def append(self, cmd):
        '''
        cmd: Block or Op or Layer
        '''
        self.cmds.append(cmd)

    def execute(self):
        '''
        Execute this block, this will run all the operators and update the coresponding
        output variables.
        '''
        self._build_nn()
        self.net.run()

    def _build_nn(self):
        self.net = core.Net.create()
        ops = self.__extract_op_from_block(self.cmds)
        for op in ops:
            self.net.append_op(op)
        self.net.complete_add_op(True)

    def __extract_op_from_block(self, cmds):
        ops = []
        for cmd in cmds:
            if type(cmd) is Block:
                child_ops = self.__extract_op_from_block([cmd])
                ops += child_ops
            else:
                ops.append(cmd)
        return ops


g_block = Block()


#TODO this need to be renamed
class block_guard(object):
    '''
    a wrapper for Block, which automatically change g_block, Namespace.

    usage:

        import paddle as pd

        with pd.block() as block:
            a = pd.Variable()
            b = pd.Variable()
            c = pd.add_two(a, b)
            block.execute()
    '''
    cur_block = g_block
    last_block = None
    counter = 0

    def __init__(self, namespace='', block=None, execute_immediately=True):
        '''
        namespace: str
            current block's namespace, if leave default, father's namespace will be used.
        block: Block
            current block, if set None, a new Block will be created.
        execute_immediately: bool
            if execute_immediately is True, then all the operators of this block will be
            inserted into father's block immediately, if false, this block is independent
            from father's block.
        '''
        self.namespace = namespace if namespace else block_guard.inc_counter()
        self.block = block if block else Block()
        self.execute_immediately = execute_immediately

    def __enter__(self):
        Namespace.begin(self.namespace)
        block_guard.last_block = block_guard.cur_block
        block_guard.cur_block = self.block

        if self.execute_immediately:
            block_guard.last_block.append(block_guard.cur_block)

    def __exit__(self):
        Namespace.end()
        block_guard.cur_block = block_guard.last_block

    def inc_counter():
        c = block_guard.counter
        block_guard.counter += 1
        return c


def eval(fetches, block=g_block):
    '''
    fetches: list of Variable
    block: Block

    evaluate all the variables in `fetches`. In details, it will trace the sub-graph which
    the variables of `fetches` depends and compile a new corespondding block, and execute it.

    usage:

        # use g_block as default
        var_value = pd.eval([var])

        # use a specific block
        var_value = pd.eval([var], block=a_block)
    '''
    assert all(isinstance(_, Variable) for _ in fetches), "fetches should be Variables"
    graph = DependenceGraph(block.cmds)
    op_or_layers = graph.DFS_with_targets(fetches)

    with block_guard() as B:
        for cmd in op_or_layers:
            B.block.append(cmd)
        B.execute()
    # return python values
    return [v.val() for v in fetches]
