import os
from common import g_scope


class Namespace(object):
    '''
    Namespace is similar to tf.variable_scope, which helps to store local
    variables  of different `Block`s into one Scope, and make it possible
    to reference Variable across different Blocks.
    '''

    stack = []

    def __init__(self, name):
        '''
        parents: list of str
        '''
        self.name = name
        parents = [n.name for n in Namespace.stack]
        self.prefix = os.path.join(*parents) if parents else ''

    def sub_namespace(self, name):
        prefix = os.path.join(self.name, name)
        return Namespace(prefix)

    def parent_namespace(self):
        if not stack:
            return None
        return stack[-1]

    def getname(name):
        return os.path.join(self.prefix, self.name, name)

    @staticmethod
    def cur():
        if Namespace.stack:
            return Namespace.stack[-1]
        return Namespace('')

    def enter(self):
        Namespace.stack.append(self)

    def exit(self):
        Namespace.stack.pop()

    def __repr__(self):
        return "<Namespace '%s'>" % os.path.join(self.prefix, self.name)

    def __enter__(self):
        self.enter()

    def __exit__(self, type, value, traceback):
        self.exit()


if __name__ == '__main__':
    with Namespace('apple'):
        print Namespace.cur()

        with Namespace('beer'):
            print Namespace.cur()
        print 'should be apple', Namespace.cur()
    print 'should be empty', Namespace.cur()
