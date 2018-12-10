import paddle.fluid as fluid
import paddle
from strategy import Strategy
from parl.policy_gradient import PolicyGradientAgent
import numpy as np

class KnowledgeStrategy(Strategy):

    def __init__(self, start_epoch=0, end_epoch=1, student_weight=0.9, teacher_weight=0.1, teacher_program=None, teacher_logits=None):
        super(PruneStrategy, self).__init__(start_epoch, end_epoch)
        self.teacher_program = None
        self.teacher_logits = None
        self.teacher_weight = teacher_weight
        self.student_weight = student_weight

    def _triger(self, context):
        return (context.epoch_id >= self.start_epoch and context.epoch_id < self.end_epoch)

    def on_compress_begin(self, context):
        """
        Modify the train_program in context for distillation trainning.
        1. Merge studnet program and teacher program.
        2. Calculate distillation_loss according to the logits of studnet and teacher.
        3. Append backward operators by optimizer of student
        4. Replace the train_program in context with new program.
        """
        program = _merge_programs(context.train_program, self.teacher_program)
        self._remove_backward(context.train_program, context.loss)
        self._remove_backward(self.teacher_program, self.teacher_logits)
        program = self._merge_program(context.train_program, self.teacher_program)
        with fluid.ProgramGuard(program):
            distillation_loss = _cal_distillation_loss(self.teacher_logits, context.logits)
            context.loss = self.loss_weight * context.loss + self.dist_weight * distillation_loss
            context.optimizer.minimize(context.loss)
        self.train_program = program
            
        
    def _cal_distillation_loss(self, teacher_logits, student_logits):
        """Calculate distillation_loss according to the logits of studnet and teacher."""
        return

    def _merge_programs(self, program_1, program_2):
        """Copy the operators and variables of program_1 and program_2 into new program."""
        program = fluid.Program()
        return program

    def _remove_backward(self, program, last_op):
        """Remove the backward op and variables from program."""
        pass
