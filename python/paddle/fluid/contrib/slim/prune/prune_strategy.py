from ..core.strategy import Strategy
from ....framework import Program, program_guard
from .... import layers
import numpy as np

__all__ = ['SensetivePruneStrategy', 'PruneStrategy']
class SensetivePruneStrategy(Strategy):
    def __init__(self,
                 pruner=None, 
                 start_epoch=0,
                 end_epoch=10,
                 delta_rate=0.20,
                 acc_loss_threshold=0.2,
                 sensitivities=None):
        super(SensetivePruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.delta_rate = delta_rate
        self.acc_loss_threshold = acc_loss_threshold
        self.sensitivities = sensitivities
        

class PruneStrategy(Strategy):
    """
    The strategy that pruning weights by threshold or ratio iteratively.
    """
    def __init__(self, pruner, mini_batch_pruning_frequency=1, start_epoch=0, end_epoch=10):
        super(PruneStrategy, self).__init__(start_epoch, end_epoch)
        self.pruner = pruner
        self.mini_batch_pruning_frequency = mini_batch_pruning_frequency

    def _triger(self, context):
        return (context.batch_id % self.mini_batch_pruning_frequency == 0 and 
           context.epoch_id >= self.start_epoch and context.epoch_id < self.end_epoch)

    def on_batch_end(self, context):
        if self._triger(context):
            prune_program = Program()
            with program_guard(prune_program):
                for param in context.graph.all_parameters():
                    prune_program.global_block().clone_variable(param)
                    p = prune_program.global_block().var(param.name)
                    zeros_mask = self.pruner.prune(p)
                    pruned_param = p * zeros_mask
                    layers.assign(input=pruned_param, output=param)
            context.program_exe.run(prune_program, scope=context.scope)
