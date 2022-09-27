#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    if model_name in ['YOLOv5', 'YOLOv7']:
        return paddle.clip(_sigmoid(alpha_v) * (ZETA - GAMMA) + GAMMA, 0, 1)
    else:
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
                 num_iterations=1000,
                 lr=0.1,
                 bias_correction=False,
                 epochs=20,
                 weight_quantize_type='channel_wise_abs_max',
                 model_name=None,
                 qdrop=False):

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
    
    # blocks
    if model_name=="MobileNetV1":
        data_name_map = {'inputs':'inputs'}
        blocks = [['inputs']]
        block_weights_names = []
        def get_blocks(block_size=5):
            index = 0
            for i in range(0, len(weight_var_names), block_size):
                weight_var_name = weight_var_names[i]
                input_var_name = input_weight_pairs[weight_var_name][0]
                blocks[index].append(input_var_name)
                block_ = []
                block_.append(input_var_name)
                blocks.append(block_)
                index+=1

                names = []
                for j in range(i,min(i+block_size,len(weight_var_names)-1)):
                    #print(j)
                    names.append(weight_var_names[j])
                block_weights_names.append(names)
        get_blocks()
        blocks.pop(0)
        blocks[-1].append('batch_norm_26.tmp_3')
        block_weights_names[-1].append('conv6_sep_weights')
    elif model_name=='YOLOv6':
        print("run yolov6s")
        data_name_map={'x2paddle_image_arrays':'x2paddle_image_arrays'}
        blocks = [['x2paddle_image_arrays','relu_8.tmp_0'],
                    ['relu_8.tmp_0','relu_15.tmp_0'],
                    ['relu_15.tmp_0','relu_21.tmp_0'],
                    ['concat_1.tmp_0','relu_26.tmp_0'],
                    ['concat_2.tmp_0', 'relu_30.tmp_0'],
                    ['relu_30.tmp_0', 'concat_4.tmp_0'],
                    ['relu_30.tmp_0', 'relu_31.tmp_0'],
                    ['concat_3.tmp_0', 'relu_35.tmp_0'],
                    ['relu_35.tmp_0', 'relu_36.tmp_0'],
                    ['concat_5.tmp_0', 'concat_10.tmp_0'],
                    ['relu_35.tmp_0', 'concat_8.tmp_0']]

        block_weights_names = [['conv2d_0.w_0','conv2d_1.w_0','conv2d_2.w_0','conv2d_3.w_0','conv2d_4.w_0','conv2d_5.w_0','conv2d_6.w_0','conv2d_7.w_0','conv2d_8.w_0'],
                    ['conv2d_9.w_0','conv2d_10.w_0','conv2d_11.w_0','conv2d_12.w_0','conv2d_13.w_0','conv2d_14.w_0','conv2d_15.w_0'],
                    ['conv2d_16.w_0','conv2d_17.w_0','conv2d_18.w_0','conv2d_19.w_0','conv2d_20.w_0','conv2d_21.w_0'],
                    ['conv2d_22.w_0','conv2d_23.w_0','conv2d_24.w_0','conv2d_25.w_0','conv2d_26.w_0'],
                    ['conv2d_27.w_0','conv2d_28.w_0','conv2d_29.w_0','conv2d_30.w_0'],
                    ['conv2d_32.w_0','conv2d_34.w_0','conv2d_35.w_0','conv2d_37.w_0','conv2d_38.w_0','conv2d_39.w_0'],
                    ['conv2d_31.w_0'],
                    ['conv2d_33.w_0','conv2d_36.w_0','conv2d_40.w_0','conv2d_41.w_0'],
                    ['conv2d_42.w_0'],
                    ['conv2d_44.w_0','conv2d_47.w_0','conv2d_51.w_0','conv2d_52.w_0','conv2d_53.w_0','conv2d_54.w_0','conv2d_55.w_0','conv2d_56.w_0','conv2d_57.w_0','conv2d_58.w_0'],
                    ['conv2d_43.w_0','conv2d_45.w_0','conv2d_46.w_0','conv2d_49.w_0','conv2d_48.w_0','conv2d_50.w_0'],] 
    elif model_name=='YOLOv5':
        data_name_map={'x2paddle_images':'x2paddle_images'}
        blocks = [['x2paddle_images','concat_1.tmp_0'],
                ['concat_1.tmp_0', 'sigmoid_74.tmp_0'],
                ['elementwise_mul_14','concat_2.tmp_0'],
                ['concat_2.tmp_0', 'sigmoid_84.tmp_0'],
                ['elementwise_mul_24','concat_4.tmp_0'],
                ['concat_4.tmp_0','sigmoid_92.tmp_0'],
                ['conv2d_93.tmp_0','sigmoid_93.tmp_0'],
                ['concat_5.tmp_0','conv2d_99.tmp_0'],
                ['concat_7.tmp_0', 'concat_8.tmp_0'],
                ['concat_8.tmp_0','sigmoid_104.tmp_0'],
                ['elementwise_mul_44','sigmoid_108.tmp_0'],
                ['elementwise_mul_44','sigmoid_105.tmp_0'],
                ['concat_9.tmp_0','concat_13.tmp_0'],
                ['concat_13.tmp_0','sigmoid_111.tmp_0'],
                ['elementwise_mul_54','sigmoid_115.tmp_0'],
                ['elementwise_mul_54','sigmoid_112.tmp_0'],
                ['concat_14.tmp_0','sigmoid_119.tmp_0']]

        block_weights_names = [['conv2d_{}.w_0'.format(i) for i in range(14)],
                                ['conv2d_14.w_0'],
                            ['conv2d_{}.w_0'.format(i) for i in range(15, 24)],
                            ['conv2d_24.w_0'],
                            ['conv2d_{}.w_0'.format(i) for i in range(25, 32)],
                            ['conv2d_32.w_0'],
                            ['conv2d_33.w_0'],
                            ['conv2d_{}.w_0'.format(i) for i in range(34, 40)],
                            ['conv2d_{}.w_0'.format(i) for i in range(40, 44)],
                            ['conv2d_44.w_0'],
                            ['conv2d_46.w_0'],
                            ['conv2d_45.w_0'],
                            ['conv2d_{}.w_0'.format(i) for i in range(47, 51)],
                            ['conv2d_51.w_0'],
                            ['conv2d_53.w_0'],
                            ['conv2d_52.w_0'],
                            ['conv2d_{}.w_0'.format(i) for i in range(54, 60)]]
    elif model_name=='YOLOv7':
        data_name_map={'x2paddle_images':'x2paddle_images'}
        blocks = [['x2paddle_images','concat_2.tmp_0'],
                 ['concat_2.tmp_0','sigmoid_112.tmp_0'],
                 ['elementwise_mul_20','concat_4.tmp_0'],
                 ['elementwise_mul_20','sigmoid_114.tmp_0'],
                 ['concat_4.tmp_0','sigmoid_123.tmp_0'],
                 ['elementwise_mul_31','concat_8.tmp_0'],
                 ['concat_8.tmp_0','sigmoid_141.tmp_0'],
                 ['elementwise_mul_49','sigmoid_142.tmp_0'],
                 ['concat_9.tmp_0','sigmoid_149.tmp_0'],
                 ['elementwise_mul_57','sigmoid_150.tmp_0'],
                 ['concat_11.tmp_0','sigmoid_157.tmp_0'],
                 ['conv2d_159.tmp_0','sigmoid_162.tmp_0'],
                 ['elementwise_mul_65','sigmoid_160.tmp_0'],
                 ['elementwise_mul_65','sigmoid_161.tmp_0'],
                 ['concat_13.tmp_0','sigmoid_169.tmp_0'],
                 ['conv2d_171.tmp_0','sigmoid_174.tmp_0'],
                 ['elementwise_mul_80','sigmoid_172.tmp_0'],
                 ['elementwise_mul_80','sigmoid_173.tmp_0'],
                 ['concat_16.tmp_0','sigmoid_183.tmp_0']]

        block_weights_names = [['conv2d_{}.w_0'.format(i) for i in range(20)],
                                ['conv2d_20.w_0'],
                                list(set(['conv2d_{}.w_0'.format(i) for i in range(21,31)])-{'conv2d_22.w_0'}),
                                ['conv2d_22.w_0'],
                                ['conv2d_31.w_0'],
                                list(set(['conv2d_{}.w_0'.format(i) for i in range(32,49)])-{'conv2d_33.w_0'}),
                                ['conv2d_49.w_0'],
                                ['conv2d_50.w_0'],
                                ['conv2d_{}.w_0'.format(i) for i in range(51,58)],
                                ['conv2d_58.w_0'],
                                ['conv2d_{}.w_0'.format(i) for i in range(59,66)],
                                ['conv2d_67.w_0','conv2d_70.w_0'],
                                ['conv2d_68.w_0'],
                                ['conv2d_66.w_0','conv2d_69.w_0'],
                                ['conv2d_{}.w_0'.format(i) for i in range(71,78)],
                                ['conv2d_79.w_0','conv2d_82.w_0'],
                                ['conv2d_80.w_0'],
                                ['conv2d_78.w_0','conv2d_81.w_0'],
                                ['conv2d_{}.w_0'.format(i) for i in range(83,92)]]
    else:
        data_name_map = {}
        for name in feed_list:
            data_name_map[name] = name
        blocks = []
        block_weights_names = []
        def get_blocks():
            for name in weight_var_names:
                block_weights_names.append([name])
                block_ = []
                block_.append(input_weight_pairs[name][0])
                block_.append(quantized_op_pairs[name])
                blocks.append(block_)
        get_blocks()


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
