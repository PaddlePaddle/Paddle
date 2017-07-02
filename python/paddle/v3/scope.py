from lib_loader import _C_LIB


class Scope(object):
    """
    Scope is used to manage variables, in python end, all the value of
    Variable should be type of numpy array.
    """
    def __init__(self, name):
        self._scope = _C_LIB.CreateScope(name)

    def vars(self):
        return self._scope.Vars

    def reset(self):
        """
        remove all variables hold in this scope
        :return:
        """
        self._scope.reset()

    def create_var(self, name):
        """
        create a variable in the current scope without value in it, you
        must call feed_var to initialize the value of this variable, that
        should be a numpy array.
        :param name:
        :return:
        """
        assert isinstance(name, basestring)
        return

    def feed_var(self, name, value):
        """
        initialize the value in variable, value should be a numpy array object.
        :param name:
        :return:
        """
        pass

    def get_var(self, name):
        """
        get the value of variable, if not initialized, return
        :param name:
        :return:
        """
        assert isinstance(name, basestring)
        return self._scope.GetVariable(name)

    def has_var(self, name):
        return self._scope.HasVar(name)
