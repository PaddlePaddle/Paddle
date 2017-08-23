class Namespace(object):
    '''
    Similar to tf.scope, help to store multiple `Block`s in a `core.Scope`,
    and make reference across different blocks possible.
    '''
    namespace = ''
    last_namespace = ''

    @staticmethod
    def begin(namespace):
        if namespace:
            Namespace.last_namespace = Namespace.namespace
            Namespace.namespace += '/' + namespace

    @staticmethod
    def end():
        Namespace.namespace = Namespace.last_namespace
