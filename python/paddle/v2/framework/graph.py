import paddle.v2.framework.core as core


class Block(object):
    def __init__(self, program, idx):
        self.proto = program.proto.block(idx)
        self.vars = dict()  # var_name --> var
        self.ops = list()  # operator list
        self.program = program

    @property
    def parent_idx(self):
        return self.proto.parent

    @property
    def idx(self):
        return self.proto.id


class Program(object):
    @classmethod
    def instance(cls):
        # From https://stackoverflow.com/questions/8212053
        # Making Program as a Singleton class.
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(self.__class__,
                           '_instance'), 'Do not call constructor directly!'
        self.proto = core.ProgramDesc.instance()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0

    def global_block(self):
        return self.blocks[0]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def create_block(self):
        new_block_idx = len(self.blocks)
        self.proto.append_block(self.current_block().proto)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def rollback(self):
        self.current_block_idx = self.current_block().parent_idx


# program is a global instance.
g_program = Program.instance()
