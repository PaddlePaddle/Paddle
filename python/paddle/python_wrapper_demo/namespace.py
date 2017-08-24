from common import g_scope


class Namespace:
    '''
    Namespace is similar to tf.variable_scope, which helps to store local
    variables  of different `Block`s into one Scope, and make it possible
    to reference Variable across different Blocks.
    '''
    scope = g_scope
    cur_namespace = None
    last_namespace = None

    @staticmethod
    def gen_name(name):
        '''
        generate a name within this namespace.
        '''
        assert '/' not in name
        return Namespace.cur_namespace + '/' + name

    @staticmethod
    def begin(namespace):
        '''
        namespace: string
        '''
        if namespace:
            Namespace.last_namespace = Namespace.cur_namespace
            Namespace.cur_namespace = namespace

    @staticmethod
    def end():
        Namespace.cur_namespace = Namespace.last_namespace
