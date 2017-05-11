import paddle.trainer_config_helpers.layers as layers
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config


class Tape(object):
    def __init__(self):
        self.__calls__ = []
        self.__is_calling__ = False

    def append(self, call):
        if not self.__is_calling__:
            self.__calls__.append(call)
            return len(self.__calls__) - 1
        else:
            return -1

    def __call__(self, end=-1):
        self.__is_calling__ = True
        tape_items = []
        for i, each in enumerate(self.__calls__):
            tape_items.append(each(tape=tape_items))
            if i == end:
                break


class TapeItem(object):
    def __init__(self, idx, tape):
        self.__idx__ = idx
        self.__tape__ = tape

    def idx(self):
        return self.__idx__

    def to_proto(self):
        return self.__tape__(self.__idx__)


gTape = Tape()


def __convert_v2__(name, module):
    def __convert_tape_item__(tape, a):
        if isinstance(a, TapeItem):
            return tape[a.idx()]
        elif isinstance(a, list) or isinstance(a, tuple):
            return map(lambda x: __convert_tape_item__(tape=tape, a=x), a)
        else:
            return a

    def __impl__(*args, **kwargs):
        func = getattr(module, name)
        tape = kwargs.get('tape', None)
        if tape is None:
            return func(*args, **kwargs)
        else:
            # Convert arguments
            args = list(args)
            for i in xrange(len(args)):
                args[i] = __convert_tape_item__(tape, args[i])

            for key in kwargs:
                kwargs[key] = __convert_tape_item__(tape, kwargs[key])

            del kwargs['tape']

            return func(*args, **kwargs)

    def __v2_impl__(*args, **kwargs):
        idx = gTape.append(lambda tape: __impl__(tape=tape, *args, **kwargs))
        if idx != -1:
            return TapeItem(idx, gTape)
        else:
            return __impl__(*args, **kwargs)

    return __v2_impl__


for nm in layers.__all__:
    if nm.endswith("_layer"):
        globals()[nm] = __convert_v2__(nm, layers)
    if nm == "recurrent_group":
        globals()[nm] = __convert_v2__(nm, layers)

if __name__ == '__main__':
    dat = data_layer(name="a", size=10)
    hidden = fc_layer(input=dat, size=100)

    def step(input):
        return fc_layer(input=input, size=200)

    hidden = recurrent_group(step=step, input=[hidden])

    output = fc_layer(input=hidden, size=10)

    print parse_network_config(output.to_proto)
