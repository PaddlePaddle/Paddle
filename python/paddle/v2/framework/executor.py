import paddle.v2.framework.core as core
from paddle.v2.framework.framework import Block, Program


class Executor(object):
    def __init__(self, places):
        self.executor = core.Executor(places)

    def run(self, block):
        if isinstance(block, Program):
            block = block.global_block()
        if not isinstance(block, Block):
            raise TypeError("Block should be a block or program")

        self.executor.run(block.program.desc, block.idx)
