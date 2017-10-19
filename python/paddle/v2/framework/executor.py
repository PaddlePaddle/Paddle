import paddle.v2.framework.core as core
from paddle.v2.framework.framework import Block, Program


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
            fetch_var_name='fetch'):
        if not isinstance(program, Program):
            raise TypeError()

        program = program.clone()
        global_block = program.global_block()
        assert isinstance(global_block, Block)
        feed_var = global_block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)

        for i, name in enumerate(feed):
            out = global_block.var(name)
            global_block.prepend_op(
                'feed',
                inputs={'Input': [feed_var]},
                outputs={'Out': [out]},
                attrs={'col': i})
            # FIXME
            core.set_feed_variable_float(feed[name], feed_var.name, i)

        fetch_var = global_block.create_var(
            name=fetch_var_name,
            type=core.VarDesc.VarType.FETCH_LIST,
            persistable=True)
        for i, var in enumerate(fetch_list):
            global_block.append_op(
                type='fetch',
                inputs={'Input': [var]},
                outputs={'Out': [fetch_var]},
                attrs={'col': 1})

        assert isinstance(global_block, Block)
        self.executor.run(program.desc, 0)
        for i, _ in enumerate(fetch_list):
            yield core.get_fetch_variable(fetch_var_name, i)
