import numpy as np
import time
import sys
import logging
import paddle
import paddle.fluid as fluid
import six
import paddleslim.dist as dist
import math
import copy
from ....log_helper import get_logger
from .utils import load_variable_data, set_variable_data, stable_sigmoid, quant_tensor, dequant_tensor, _channelwise_quant_axis1_ops, calculate_quant_cos_error, bias_correction_w, isolate_blocks, _get_op_output_var_names, _get_op_input_var_names, insert_drop_quant_deqaunt, insert_soft_rounding

_logger = get_logger(__name__,
                     logging.INFO,
                     fmt='%(asctime)s-%(levelname)s: %(message)s')
GAMMA = -0.1
ZETA = 1.1

def _sigmoid(x):
    return 1/(1+paddle.exp(-x))

def compute_soft_rounding(alpha_v, model_name):

    return paddle.clip(paddle.nn.functional.sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, 0, 1)

def compute_soft_rounding_np(alpha_v):
    return np.clip(stable_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA,
                   a_min=0,
                   a_max=1)


class LossFunction:
    def __init__(self,
                 program,
                 model_name,
                 weight_block_names: list = None,
                 round_loss: str = 'relaxation',
                 weight: float = 0.1,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (20, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 beta_mode: str = 'const'):

        self.program = program
        self.model_name = model_name
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.weight_block_names = weight_block_names
        self.beta_mode = beta_mode

    def get_loss(self, s_v, t_v, scheduler):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param s_v: output from quantized model
        :param t_v: output from FP model
        :param scheduler: beta
        :return: total loss function
        """

        if self.rec_loss == 'mse':
            rec_loss = paddle.nn.functional.mse_loss(s_v, t_v)
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        if self.beta_mode == 'const':
            self.beta = 3
        else:
            self.beta = scheduler.get_lr()
        
        if self.round_loss == 'relaxation':
            round_loss = 0.0
            for name in self.weight_block_names:
                alpha_v = self.program.global_block().var(name+'.alpha')
                h_v = compute_soft_rounding(alpha_v, self.model_name)
                round_loss += self.weight * paddle.sum(-paddle.pow(paddle.abs(2 * h_v-1), self.beta) + 1)
        else:
            raise NotImplementedError

        total_loss = rec_loss+round_loss
        return total_loss, rec_loss, round_loss



def run_adaround(data_loader,
                 fp32_program,
                 feed_list,
                 fetch_list,
                 exe,
                 scope,
                 place,
                 quantized_op_pairs,
                 input_weight_pairs,
                 weight_op_pairs,
                 scale_dict,
                 blocks,
                 block_weights_names,
                 num_iterations=1000,
                 lr=0.1,
                 bias_correction=False,
                 epochs=20,
                 weight_quantize_type='channel_wise_abs_max',
                 model_name=None,
                 qdrop=False,
                 ):
    

    assert blocks is not None, "The blocks cannot be None."
    assert block_weights_names is not None, "The block_weights_names cannot be None."

    def _floor(weight_var_names, scale_dict, scope, place):
        for name in weight_var_names:
            weight_np = load_variable_data(scope, name)
            scale = scale_dict[name]
            weight_np_floor = np.floor(quant_tensor(weight_np, scale))
            set_variable_data(scope, place, name, weight_np_floor)

    student_program = fp32_program.clone()
    for param in student_program.global_block().all_parameters():
        if param.trainable:
            param.trainable=False

    weight_var_names = list(quantized_op_pairs.keys())
    
    data_name_map = {}
    for name in feed_list:
        data_name_map[name] = name

    dist.merge(
        fp32_program,
        student_program,
        data_name_map,
        place,
        teacher_scope=None,
        name_prefix="teacher_",
        merge_feed=True)

    _floor(weight_var_names=weight_var_names, scope=scope, place=place, scale_dict=scale_dict)
    
    if qdrop:
        #insert quant/dequant func on the mul/conv/depthwise_conv input
        insert_drop_quant_deqaunt(student_program, scale_dict)
    
    #insert soft rounding on the weights
    insert_soft_rounding(program=student_program, weight_names=weight_var_names, scales=copy.deepcopy(scale_dict), scope=scope, weight_quantize_type=weight_quantize_type)

    #Divided into blocks
    isolate_blocks(student_program, blocks)

    #build and run adaround/brecq/qdrop program
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1
    startup_program = fluid.Program()
    for k in range(len(blocks)):
        block_ = blocks[k]
        names = block_weights_names[k]
        tmp_program = student_program.clone()
        quant_op_out_name = block_[1]
        with paddle.static.program_guard(tmp_program, startup_program):
            adaroundloss = LossFunction(tmp_program, model_name, names)
            quant_op_out_name = block_[1]
            student_var = tmp_program.global_block().var(quant_op_out_name)
            teacher_var = tmp_program.global_block().var("teacher_"+quant_op_out_name)
            scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=20, eta_min=2, T_max=2000, verbose=True)
            total_loss, recon_loss, round_loss = adaroundloss.get_loss(student_var, teacher_var, scheduler)
            train_fetches_loss = {"total_loss":total_loss, "recon_loss":recon_loss, "round_loss":round_loss}
            optimizer = paddle.optimizer.Adam(learning_rate=lr)
            optimizer.minimize(total_loss)

        exe.run(startup_program)
        start_time = time.time()
        prev_start_time = start_time
        for epoch in range(epochs):
            for i, data in enumerate(data_loader()):
                prev_start_time = start_time
                start_time = time.time()
                out = exe.run(
                    tmp_program,
                    feed=data,
                    fetch_list=[v.name for v in train_fetches_loss.values()],
                    return_numpy=True)
                _logger.info(
                    "Iter {:d}, lr {}, total_loss {:.5f}, recon_loss {:.5f}, round_loss {:.5f}, time {:.5f}s"
                    .format(epoch, lr, np.mean(out[0]), np.mean(out[1]), np.mean(out[2]), start_time - prev_start_time))
                sys.stdout.flush()
                if i == num_iterations:
                    break

    # update adarounded calibrated weights
    for weight_var_name in weight_var_names:
        alpha_tensor = load_variable_data(scope, weight_var_name+'.alpha')
        h_alpha_tensor = compute_soft_rounding_np(alpha_tensor)
        weight_quant_tensor = load_variable_data(scope, weight_var_name)
        set_variable_data(scope, place, weight_var_name, np.round(weight_quant_tensor+h_alpha_tensor))

    if bias_correction:
        for weight_var_name in weight_var_names:
            weight_var_tensor = load_variable_data(scope, "teacher_"+weight_var_name)
            weight_quant_tensor = load_variable_data(scope, weight_var_name)
            scale = scale_dict[weight_var_name]
            final_weight_tensor_quant_dict[weight_var_name] = bias_correction_w(
                weight_var_tensor,
                weight_quant_tensor,
                scale,
                quant_axis=0,
                weight_bits=8)    
    
    return fp32_program







