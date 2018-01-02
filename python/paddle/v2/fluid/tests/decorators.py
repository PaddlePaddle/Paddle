import paddle.v2.fluid as fluid

__all__ = ['many_times', 'prog_scope']


def many_times(times):
    def __impl__(fn):
        def __fn__(*args, **kwargs):
            for _ in range(times):
                fn(*args, **kwargs)

        return __fn__

    return __impl__


def prog_scope():
    def __impl__(fn):
        def __fn__(*args, **kwargs):
            prog = fluid.Program()
            startup_prog = fluid.Program()
            scope = fluid.core.Scope()
            with fluid.scope_guard(scope):
                with fluid.program_guard(prog, startup_prog):
                    fn(*args, **kwargs)

        return __fn__

    return __impl__
