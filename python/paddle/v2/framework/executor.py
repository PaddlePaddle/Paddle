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
        # global_block.create_var(
        #     name=feed_var_name,
        #     type=
        # )

        for i, name in enumerate(feed):
            global_block.prepend_op(
                'feed',
                inputs={'Input': [feed_var_name]},
                outputs={'Out': [name]},
                attrs={'col': i})
            # FIXME
            core.set_feed_variable_float(feed[name], feed_var_name, i)

        for fetch_name in fetch_list:
            var = block.var(fetch_name)
            print var.op

        assert isinstance(global_block, Block)
        self.executor.run(block.program.desc, block.idx)
