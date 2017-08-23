__all__ = [
    'block',
    'global_block',
]

from namespace import Namespace

class Block(object):
    '''
    Block is an implementation concept like a programming code's scope between two curly braces.

    It stores variables and ops in a core Scope.
    '''
    def __init__(self, namespace=''):
        '''
        A namespace is used to make variables and ops from every block to store in the same
        `Scope`.
        '''
        self.namespace = namespace

        self.ops = []
        self.local_vars = []
        # a local core_scope, and all the local_vars are stored in this scope.
        self._core_scope = core.Scope()

    def append(self, op):
        self.ops.append(op)

    def scope(self):
        return self._core_scope

    def execute(self):
        '''
        Execute a block.
        '''
        for op in self.ops:
            op.run()



# A global block as default, all the variables or operators defined are stored in g_block.
#
# for example:
#
#   a = 1
#
#   int main() {
#     b = 2
#   }
#
# `a` will stored in global block.
#
g_block = Block()
last_block = g_block


class BlockMap(object):
    __map__ = {}

    cur_block = None
    last_block = None

    @staticmethod
    def retrieve(namespace):
        if namespace == '':
            if not BlockMap.cur_block:
                BlockMap.cur_block = Block('')
        if namespace not in BlockMap.__map__:
            BlockMap.__map__[namespace] = Block(namespace)
        return BlockMap.__map__[namespace]

    @staticmethod
    def global_block():
        return BlockMap.retrieve('')


def global_block():
    return BlockMap.retrieve('')


class block(object):
    '''
    a guard like std::lock_guard in c++11

    usage:

      with pd.block(namespace) as d:
        c = pd.fc(a)
        ...
    '''
    def __init__(self, namespace=''):
        self.namespace = namespace

    def __enter__(self):
        Namespace.begin(self.namespace)
        global g_block, last_block
        last_block = g_block
        g_block = BlockMap.retrieve(self.namespace)

    def __exit__(self):
        Namespace.end()
        global g_block, last_block
        g_block = last_block

    def execute(self):
        g_block.execute()
