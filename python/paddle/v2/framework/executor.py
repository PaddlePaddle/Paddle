import paddle.v2.framework.core as core
from paddle.v2.framework.framework import Block, Program, g_main_program
import numpy as np

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
        self.places = act_places

    def aslodtensor(self, data):
        def accumulate(data):
            if not isinstance(data, list):
                return 1
            return sum([accumulate(sub) for sub in data])

        def parselod(data):
            seq_lens = [accumulate(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            return lod

        assert len(self.places) != 0
        if not isinstance(data, list):
            # pure tensor case
            tensor = core.LoDTensor()
            numpy_data = np.array(data).reshape([len(data), 1])
            tensor.set(numpy_data, self.places[0])
            return tensor
        else:
            # lodtensor case
            lod = []
            flattened_data = None
            if not isinstance(data[0], list):
                lod.append(parselod(data))
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            else:
                while isinstance(data[0], list):
                    lod.append(parselod(seq))
                    flattened_data = [item for seq in data for item in seq]
                    data = flattened_data
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            flattened_data = flattened_data.reshape([len(flattened_data), 1])
            tensor = core.LoDTensor()
            tensor.set(flattened_data, self.places[0])
            tensor.set_lod(lod)
            return tensor

    def asnumpy(self, tensor):
        if isinstance(tensor, list):
            return [self.asnumpy(t) for t in tensor]
        assert isinstance(tensor, core.LoDTensor)
        lod = tensor.lod()
        ans = None
        tensor_data = np.array(tensor)
        if len(lod) == 0:
            ans = tensor_data
        elif len(lod) == 1:
            ans = []
            idx = 0
            while idx < len(lod) - 1:
                ans.append(tensor_data[lod[idx]:lod[idx + 1]])
                idx += 1
        else:
            for l in reversed(lod):
                ans = []
                idx = 0
                while idx < len(l) - 1:
                    ans.append(tensor_data[l[idx]:l[idx + 1]])
                    idx += 1
                tensor_data = ans
            ans = tensor_data
        return ans

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
            # core.set_feed_variable(scope, self.aslodtensor(feed[name]), feed_var.name, i)
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
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in xrange(len(fetch_list))
        ]
        return [self.asnumpy(out) for out in outs]
