import paddle.v2.framework.core as core
import paddle.v2.framework.utility as utility
from paddle.v2.framework.framework import Block, Program

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
            program,
            feed,
            fetch_list,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None):
        if not isinstance(program, Program):
            raise TypeError()

        if scope is None:
            scope = g_scope

        program = program.clone()
        global_block = program.global_block()
        feed_indexes = utility.add_feed_components(global_block,
                                                   feed.keys(), feed_var_name)
        for feeded_name, idx in feed_indexes.iteritems():
            core.set_feed_variable(scope, feed[feeded_name], feed_var_name, idx)

        fetch_indexes = utility.add_fetch_components(global_block, fetch_list,
                                                     fetch_var_name)

        self.executor.run(program.desc, scope, 0)
        return [
            core.get_fetch_variable(scope,
                                    fetch_var_name, fetch_indexes[var.name])
            for var in fetch_list
        ]
