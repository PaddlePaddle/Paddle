import paddle.v2.framework.core as core
from paddle.v2.framework.framework import Block, Program, g_main_program

g_scope = core.Scope()


class Executor(object):
    def __init__(self, places):
        if not isinstance(places, list) and not isinstance(places, tuple):
            places = [places]

        act_places = []
        for each in places:
            p = core.Place()
            p.set_place(each)
            act_places.append(p)

        self.executor = core.Executor(act_places)

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None):
        if feed is None:
            feed = {}
        if fetch_list is None:
            fetch_list = []

        if program is None:
            program = g_main_program

        if not isinstance(program, Program):
            raise TypeError()

        if scope is None:
            scope = g_scope

        program = program.clone()
        global_block = program.global_block()
        feed_var = global_block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)

        for i, name in enumerate(feed):
            out = global_block.var(name)
            global_block.prepend_op(
                'feed',
                inputs={'X': [feed_var]},
                outputs={'Out': [out]},
                attrs={'col': i})
            core.set_feed_variable(scope, feed[name], feed_var.name, i)

        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)
        for i, var in enumerate(fetch_list):
            global_block.append_op(
                type='fetch',
                inputs={'X': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})

        self.executor.run(program.desc, scope, 0, True)
        return [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in xrange(len(fetch_list))
        ]
