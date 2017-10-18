import unittest
import numpy as np
from op_test import OpTest


def repeat(list, starts, times, is_first):
    newlist = [list[0]]
    if is_first:
        for i, time in enumerate(times):
            size = list[i + 1] - list[i]
            newlist.append(newlist[-1] + size * time)
    else:
        for i, time in enumerate(times):
            start = list.index(starts[i])
            end = list.index(starts[i + 1]) + 1
            for t in range(time):
                for index in range(start, end - 1):
                    newlist.append(newlist[-1] + list[index + 1] - list[index])
    return newlist


def repeat_array(array, starts, times):
    newlist = []
    for i, time in enumerate(times):
        for t in range(time):
            newlist.extend(array[starts[i]:starts[i + 1]])
    return newlist


class TestSeqExpand(OpTest):
    def set_data(self):
        self.op_type = 'seq_expand'
        x = np.random.uniform(0.1, 1, [3, 2, 2]).astype('float32')
        y = np.zeros((6, 2, 2)).astype('float32')
        y_lod = [[0, 2, 3, 6]]
        self.inputs = {'X': (x, None), 'Y': (y, y_lod)}
        self.repeat = 2

    def compute(self):
        x_data, x_lod = self.inputs['X']
        print "x_data: %s" % x_data
        print "x_lod: %s" % x_lod
        if not x_lod:
            x_lod = [[i for i in range(1 + x_data.shape[0])]]
        else:
            x_lod = [x_lod[0]] + x_lod
        if self.repeat:
            self.attrs = {'repeat': self.repeat}
            repeats = (len(x_lod[0]) - 1) * [self.repeat]
            # get out shape
            # out_shape = np.copy(x_data.shape)
            # out_shape[0] = out_shape[0] * self.repeat
        else:
            y_data, y_lod = self.inputs['Y']
            print "y_lod: %s" % y_lod
            #print "y_lod: %s" % y_lod
            # get repeats
            repeats = [((y_lod[0][i + 1] - y_lod[0][i]) /
                        (x_lod[0][i + 1] - x_lod[0][i]))
                       for i in range(len(y_lod[0]) - 1)]
            # get out shape
            # out_shape = y_data.shape
        # get out lod

        out_lod = [repeat(x_lod[0], x_lod[0], repeats, True)] + [
            repeat(lod, x_lod[0], repeats, False) for lod in x_lod[1:]
        ]
        # copy data
        out = repeat_array(x_data.tolist(), x_lod[0], repeats)
        self.outputs = {'Out': (out, out_lod)}
        print "outputs: %s" % self.outputs

    def setUp(self):
        self.op_type = 'seq_expand'
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()


#    def test_check_grad(self):
#        self.check_grad(["X"], "Out")


class TestSeqExpandCase1(TestSeqExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [7, 1]).astype('float32')
        x_lod = [[0, 5, 7], [0, 2, 5, 7]]
        self.inputs = {'X': (x_data, x_lod)}
        self.repeat = 2


class TestSeqExpandCase2(TestSeqExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [4, 1]).astype('float32')
        self.inputs = {'X': (x_data, None)}
        self.repeat = 2


class TestSeqExpandCase3(TestSeqExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [3, 1]).astype('float32')
        y_data = np.random.uniform(0.1, 1, [8, 1]).astype('float32')
        y_lod = [[0, 1, 4, 8]]
        self.inputs = {'X': (x_data, None), 'Y': (y_data, y_lod)}
        self.repeat = None


class TestSeqExpandCase4(TestSeqExpand):
    def set_data(self):
        x_data = np.random.uniform(0.1, 1, [5, 1]).astype('float32')
        x_lod = [[0, 2, 5]]
        y_data = np.random.uniform(0.1, 1, [13, 1]).astype('float32')
        y_lod = [[0, 4, 13], [0, 2, 4, 7, 10, 13]]
        self.inputs = {'X': (x_data, x_lod), 'Y': (y_data, y_lod)}
        self.repeat = None


if __name__ == '__main__':
    unittest.main()
