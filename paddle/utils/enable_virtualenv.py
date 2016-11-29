import os


def __activate_virtual_env__():
    __path__ = os.getenv('VIRTUAL_ENV')
    if __path__ is None:
        return
    __script__ = os.path.join(__path__, 'bin', 'activate_this.py')
    execfile(__script__, {'__file__': __script__})


__activate_virtual_env__()
