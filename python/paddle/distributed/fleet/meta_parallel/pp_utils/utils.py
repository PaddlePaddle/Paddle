# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib

import paddle
from paddle.fluid import core
from paddle import _C_ops
from paddle.autograd import PyLayer
from paddle.fluid import framework
from ...utils.recompute import check_recompute_necessary, detach_variable
from ..parallel_layers.random import get_rng_state_tracker

__all__ = []


FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
}

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64"
}


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPE_DICT.keys()


def get_tensor_dtype(dtype):
    assert dtype in FLOAT_TYPE_DICT.keys()
    return FLOAT_TYPE_DICT[dtype]


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


def get_tensor_bytes(tensor):
    """Get the bytes a tensor occupied."""
    elem_size = None
    if tensor.dtype == paddle.float32:
        elem_size = 4
    elif tensor.dtype == paddle.float64:
        elem_size = 8
    elif tensor.dtype == paddle.int64:
        elem_size = 8
    elif tensor.dtype == paddle.int32:
        elem_size = 4
    elif tensor.dtype == paddle.float16:
        elem_size = 2
    elif tensor.dtype == paddle.int8:
        elem_size = 1
    else:
        raise ValueError("unknown data type: {}".format(tensor.dtype))
    return tensor.numel() * elem_size


_hcg = None
_recompute_offload = False
_recompute_partition = False


def _initialize_recompute_setting(is_offload, is_partition):
    global _recompute_offload, _recompute_partition

    _recompute_offload = is_offload
    _recompute_partition = is_partition


def _initialize_recompute_hcg(hcg):
    global _hcg
    _hcg = hcg

def _extract_tensors(all_objects):
    tensor_object = [t for t in all_objects if paddle.is_tensor(t)]
    non_tensor_object = [t for t in all_objects if not paddle.is_tensor(t)]
    tensor_flags = [paddle.is_tensor(t) for t in all_objects]
    if type(all_objects) is tuple:
        return tuple(tensor_object), tuple(non_tensor_object), tuple(tensor_flags)
    return tensor_object, non_tensor_object, tensor_flags

def _split_activation(tensors, buffer_params):
    global _hcg
    continuous_data_buffers, continuous_data_offsets = buffer_params[:2]
    _num_checkpoints = buffer_params[-1]
    
    need_partition = _recompute_partition
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    if mp_degree < 2:
        need_partition = False

    tensor_inputs = []
    num_non_tensor = 0

    def get_part_size(tensor):
        tensor_numel = paddle.numel(tensor)
        assert tensor_numel != 0, "can't recompute zero element"
        assert tensor_numel % mp_degree == 0, "The capacity of the activation () cannot be divisible by mp_degree()".format(
            tensor_numel, mp_degree)
        part_size = tensor_numel // mp_degree
        return part_size
    
    for arg_index, tensor in enumerate(tensors):
        if paddle.is_tensor(tensor):
            state = tensor.stop_gradient
            if not need_partition:
                arg_tensor = tensor.cpu() if _recompute_offload else tensor
            else:
                i = arg_index - num_non_tensor
                # use inplace operation to save memory
                data = paddle.flatten(tensor)
                part_size = get_part_size(tensor)
                start = part_size * mp_rank
                end = start + part_size
                part_data = data[start:end].clone()
        
                # to support continuous checkpoints
                if i >= len(continuous_data_buffers):
                    tensor_list = []
                    for _ in range(_num_checkpoints):
                        empty_tensor = paddle.empty(shape=[part_size],dtype=part_data.dtype)
                        tensor_list.append(empty_tensor.cpu() if _recompute_offload else empty_tensor)
                    continuous_data_buffers.append(tensor_list)
                    continuous_data_offsets.append(0)
                elif continuous_data_buffers[i] is None:
                    tensor_list = []
                    for _ in range(_num_checkpoints):
                        empty_tensor = paddle.empty(shape=[part_size],dtype=part_data.dtype)
                        tensor_list.append(empty_tensor.cpu() if _recompute_offload else empty_tensor)
                    continuous_data_buffers[i] = tensor_list
                    continuous_data_offsets[i] = 0
                
                arg_tensor = paddle.assign(part_data,output=continuous_data_buffers[i][continuous_data_offsets[i]])
                continuous_data_offsets[i] = continuous_data_offsets[i] + 1

            arg_tensor.stop_gradient = state 
            tensor_inputs.append(arg_tensor)
        else:
            tensor_inputs.append(tensor)
            num_non_tensor += 1
            continue
 
    return tensor_inputs

def _get_part_for_backward(args, inputs, buffer_params):
    continuous_size_buffers, continuous_size_offsets, _num_checkpoints = buffer_params[2:]
    new_args = []
    num_non_tensor = 0
    
    for arg_idx, (arg, inp) in enumerate(zip(args, inputs)):
        size = paddle.to_tensor(arg.shape) if paddle.is_tensor(arg) else None
        if not paddle.is_tensor(arg):
            num_non_tensor += 1
            num_args.append(arg)
            num_args.append(size)
            continue
        arg = inp
        new_args.append(arg)
        
        i = arg_idx - num_non_tensor
        numel = size.size
        if i >= len(continuous_size_buffers):
            empty_tensor = paddle.empty(shape=[numel * _num_checkpoints],dtype=size.dtype)
            continuous_size_buffers.append(empty_tensor)
            continuous_size_offsets.append(0)
        elif continuous_size_buffers[i] is None:
            empty_tensor = paddle.empty(shape=[numel * _num_checkpoints],dtype=size.dtype)
            continuous_size_buffers[i] = empty_tensor
            continuous_size_offsets[i] = 0

        contiguous_size = paddle.assign(size, output=paddle.slice(continuous_size_buffers[i], [0], [continuous_size_offsets[i]], [continuous_size_offsets[i] + numel]))
        paddle.reshape_(contiguous_size,shape=size.shape)
        continuous_size_offsets[i] = continuous_size_offsets[i] + numel
        new_args.append(contiguous_size)
    
    return new_args     

def _get_cpu_part_for_backward(args, inputs):
    new_args = []
    for arg_idx, (arg, inp) in enumerate(zip(args, inputs)):
        if not paddle.is_tensor(arg):
            new_args.append(arg)
            continue        

        arg = inp
        new_args.append(arg)
    return new_args

def _merge_tensors(ctx, tensor_object):
    non_tensor_objects, tensor_flags = ctx.non_tensor_object, ctx.tensor_flags
    merge_object = []
    tensor_idx = 0
    non_tensor_idx = 0
     
    real_flags = None
    if _recompute_partition:
        real_flags = []
        pre_flag = False
        for flag in tensor_flags:
            if pre_flag:
                pre_flag = False
                continue
            pre_flag = flag
            real_flags.append(flag)
    else:
        real_flags = tensor_flags

    for is_tensor in real_flags:
        if is_tensor:
            merge_object.append(tensor_object[tensor_idx])
            tensor_idx += 1
        else:
            merge_object.append(non_tensor_objects[non_tensor_idx])
            non_tensor_idx += 1
    return merge_object
  
def _merge_activation(ctx, tensors):
    global _hcg
    mp_degree = _hcg.get_model_parallel_world_size()
    mp_rank = _hcg.get_model_parallel_rank()
    mp_group = _hcg.get_model_parallel_group()
    device_id = paddle.distributed.ParallelEnv().device_id
    need_partition = _recompute_partition

    if mp_degree < 2:
        need_partition = False
    
    if not need_partition:
        if _recompute_offload:
            inputs = [t.cuda(device_id) for t in tensors]
        else:
            inputs = tensors
        all_object = _merge_tensors(ctx, inputs)
        return all_object

    assert len(tensors) % 2 == 0,"Error,expected even count of tensors,but get {}".format(len(tensors))

    inputs = []
    num_args = int(len(tensors) / 2)
    for i in range(num_args):
        item = tensors[2 * i]
        size = tensors[2 * i + 1]
      
        #tensors should be all tensors.this judge is for whether differential in the future. 
        if not paddle.is_tensor(item):
            inputs.append(item)
            continue
 
        part_size = item.size
        state = item.stop_gradient
        tensor_size = part_size * mp_degree
        flat_tensor = paddle.zeros(shape=[tensor_size],dtype=item.dtype)
        flat_tensor = flat_tensor.cuda(device_id) if _recompute_offload else flat_tensor
        parts = []
        for idx in range(mp_degree):
            part_i = paddle.slice(flat_tensor,[0],[part_size * i],[part_size * i + part_size]) 
            if idx == mp_rank:
                paddle.assign(item,output=part_i)
            parts.append(part_i)
        
        if mp_group is not None:
            paddle.distributed.all_gather(parts, parts[mp_rank], mp_group)
        input_tensor = flat_tensor.reshape(list(size.numpy()))
        input_tensor.stop_gradient = state
        item = input_tensor
        inputs.append(item)

    all_object = _merge_tensors(ctx, inputs)    

    return all_object


@contextlib.contextmanager
def _swith_rng_state_tracker(rng_state, tracker):
    orig_cuda_rng_state = paddle.get_cuda_rng_state()
    orig_cuda_rng_tracker = get_rng_state_tracker().get_states_tracker()

    paddle.set_cuda_rng_state(rng_state)
    get_rng_state_tracker().set_states_tracker(tracker)
    try:
        yield
    finally:
        paddle.set_cuda_rng_state(orig_cuda_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_cuda_rng_tracker)


class _HPRecomputeFunction(PyLayer):
    """
    Compared with paddle.distributed.fleet.utils.recompute, there are the following differences:
    1. In order to support PipeLineParallel, the input of recompute is modified to ensure that the input can be tuple type.
    2. Offload support for activation
    3. Support MP segmentation of activation to further reduce cuda memory
    4. Adapt to the random state of MP
    """
   
    #to support checkpoint continuous
    _buffer_params = None

    @staticmethod
    def forward(ctx, run_function, all_outputs, *args):
        check_recompute_necessary(args)

        # store for recomputing 
        ctx.run_function = run_function

        # store the rng states
        ctx.fwd_cuda_rng_state = paddle.get_cuda_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker(
        ).get_states_tracker()

        cur_device = paddle.get_device()
        assert 'gpu:' in paddle.get_device(
        ), "Recompute with RNG is not support current device: {}.".format(
            cur_device)

        # TODO support AMP
        tracer = framework._dygraph_tracer()
        if tracer._amp_level == core.AmpLevel.O0:
            ctx.is_fw_autocast = False
        else:
            ctx.is_fw_autocast = True
        ctx.amp_mode = 'O1'
        ctx.amp_white_list, ctx.amp_black_list = tracer._get_amp_op_list()

        with paddle.no_grad():
            outputs = run_function(*args)

        if _recompute_partition:
            tensor_inputs = _split_activation(args, _HPRecomputeFunction._buffer_params)
            new_args = _get_part_for_backward(args, tensor_inputs, _HPRecomputeFunction._buffer_params)
            tensor_object, non_tensor_object, tensor_flags = _extract_tensors(new_args)
            ctx.non_tensor_object = non_tensor_object
            ctx.tensor_flags = tensor_flags
            ctx.save_for_backward(*tensor_object)
        else:
            new_args = []
            for arg in args:
                if paddle.is_tensor(arg):
                    tmp_arg = arg.cpu() if _recompute_offload else arg
                else:
                    tmp_arg = arg
                new_args.append(tmp_arg)
            tensor_object, non_tensor_object, tensor_flags = _extract_tensors(new_args)
            ctx.non_tensor_object = non_tensor_object
            ctx.tensor_flags = tensor_flags
            ctx.save_for_backward(*tensor_object)

        if paddle.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *args):
        with paddle.fluid.dygraph.guard():
            #remove pointers to the continuous buffer.for garbaging the used checkpoints
            continuous_data_buffers, continuous_data_offsets, continuous_size_buffers, continuous_size_offsets = _HPRecomputeFunction._buffer_params[:-1]
            for buffers in continuous_data_buffers:
                buffers = []
                         
            #free all pointers except the store one for backward.
            continuous_data_buffers = []
            continuous_data_offsets = []
            continuous_size_buffers = []
            continuous_size_offsets = [] 
           
            # Restore inputs
            tensors = list(ctx.saved_tensor())
            inputs = _merge_activation(ctx, tensors)

            tracer = framework._dygraph_tracer()
            tracer._has_grad = True

            # need restore auto_cast state as well as w/b list
            with _swith_rng_state_tracker(ctx.fwd_cuda_rng_state,
                                          ctx.fwd_cuda_rng_state_tracker):
                with paddle.amp.auto_cast(
                        enable=ctx.is_fw_autocast,
                        custom_white_list=ctx.amp_white_list,
                        custom_black_list=ctx.amp_black_list,
                        level=ctx.amp_mode):
                    detached_inputs = detach_variable(tuple(inputs))
                    outputs = ctx.run_function(*detached_inputs)

            if isinstance(outputs, core.VarBase):
                outputs = (outputs, )
            assert len(outputs) == len(args)

            forward_outputs_with_grad = []
            backward_inputs = []

            for i in range(len(outputs)):
                if isinstance(outputs[i],
                              core.VarBase) and not outputs[i].stop_gradient:
                    forward_outputs_with_grad.append(outputs[i])
                    backward_inputs.append(args[i])

            if len(forward_outputs_with_grad) == 0:
                raise RuntimeError(
                    "none of output has stop_gradient=False, this recompute() is not necessary"
                )

            # actually backward            
            paddle.autograd.backward(forward_outputs_with_grad, backward_inputs)
            grads = list(inp._grad_ivar() for inp in detached_inputs
                         if isinstance(inp, core.VarBase))
            return grads

    @staticmethod
    def init_continuous_buffer(continuous_buffer):
        if continuous_buffer is not None:        
            assert len(continuous_buffer) == 5, "error to get continuous_buffer params."
            #to support checkpoint continuous
            _HPRecomputeFunction._buffer_params = continuous_buffer


def _hp_recompute(function, continuous_buffer, *args):
    # NODTE(shenliang03)The current hybrid parallel recompute has limitations. 
    # It cannot handle the following situations:
    # 1. The calculation output of recompute, there are tensors that do not require gradients.
    # 2. The forward output tensor has no gradient. This problem can be solved temporarily by detach().
    # 3. Here, we only use float dtype to distinguish whether a gradient is needed in output tensor
    
    all_outputs = []
    _HPRecomputeFunction.init_continuous_buffer(continuous_buffer)
    _HPRecomputeFunction.apply(function, all_outputs, *args)

    if len(all_outputs) == 1:
        return all_outputs[0]
    else:
        for output in all_outputs:
            if paddle.is_tensor(output) and not is_float_tensor(output):
                output.stop_gradient = True

        return tuple(all_outputs)
