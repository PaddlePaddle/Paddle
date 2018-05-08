#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
import numpy as np

import layers
from framework import Program, Variable, program_guard
import unique_name
from layer_helper import LayerHelper
from initializer import Constant

__all__ = [
    'ChunkEvaluator',
    'EditDistance',
    'DetectionMAP',
]


def _clone_var_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


class Evaluator(object):
    """
    Base Class for all evaluators

    Args:
        name(str): The name of evaluator. such as, "accuracy". Used for generate
            temporary variable name.
        main_program(Program, optional): The evaluator should be added to this
            main_program. Default default_main_program()
        startup_program(Program, optional):The parameter should be added to this
            startup_program. Default default_startup_program()

    Attributes:
        states(list): The list of state variables. states will be reset to zero
            when `reset` is invoked.
        metrics(list): The list of metrics variables. They will be calculate
            every mini-batch
    """

    def __init__(self, name, **kwargs):
        warnings.warn(
            "The %s is deprecated, because maintain a modified program inside evaluator cause bug easily, please use fluid.metrics.%s instead."
            % (self.__class__.__name__, self.__class__.__name__), Warning)
        self.states = []
        self.metrics = []
        self.helper = LayerHelper(name, **kwargs)

    def reset(self, executor, reset_program=None):
        """
        reset metric states at the begin of each pass/user specified batch
        """
        if reset_program is None:
            reset_program = Program()

        with program_guard(main_program=reset_program):
            for var in self.states:
                assert isinstance(var, Variable)
                g_var = _clone_var_(reset_program.current_block(), var)
                layers.fill_constant(
                    shape=g_var.shape, value=0.0, dtype=g_var.dtype, out=g_var)

        executor.run(reset_program)

    def eval(self, executor, eval_program=None):
        """
        Evaluate the statistics merged by multiple mini-batches.
        """
        raise NotImplementedError()

    def create_state(self, suffix, dtype, shape):
        """
        Create state variable.

        NOTE: It is not a public API.

        Args:
            suffix(str): the state suffix.
            dtype(str|core.VarDesc.VarType): the state data type
            shape(tuple|list): the shape of state

        Returns: State variable

        """
        state = self.helper.create_variable(
            name="_".join([unique_name.generate(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape)
        self.states.append(state)
        return state


class ChunkEvaluator(Evaluator):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and
    compute the precision recall and F1-score using the accumulated counter
    numbers.
    """

    def __init__(
            self,
            input,
            label,
            chunk_scheme,
            num_chunk_types,
            excluded_chunk_types=None, ):
        super(ChunkEvaluator, self).__init__("chunk_eval")
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.num_infer_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_infer_chunks')
        self.num_label_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_label_chunks')
        self.num_correct_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_correct_chunks')
        precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = layers.chunk_eval(
            input=input,
            label=label,
            chunk_scheme=chunk_scheme,
            num_chunk_types=num_chunk_types,
            excluded_chunk_types=excluded_chunk_types, )
        layers.sums(
            input=[self.num_infer_chunks, num_infer_chunks],
            out=self.num_infer_chunks)
        layers.sums(
            input=[self.num_label_chunks, num_label_chunks],
            out=self.num_label_chunks)
        layers.sums(
            input=[self.num_correct_chunks, num_correct_chunks],
            out=self.num_correct_chunks)

        self.metrics.extend([precision, recall, f1_score])

    def eval(self, executor, eval_program=None):
        if eval_program is None:
            eval_program = Program()
        block = eval_program.current_block()
        num_infer_chunks, num_label_chunks, num_correct_chunks = executor.run(
            eval_program,
            fetch_list=[_clone_var_(block, state) for state in self.states])
        num_infer_chunks = num_infer_chunks[0]
        num_label_chunks = num_label_chunks[0]
        num_correct_chunks = num_correct_chunks[0]
        precision = float(
            num_correct_chunks) / num_infer_chunks if num_infer_chunks else 0
        recall = float(
            num_correct_chunks) / num_label_chunks if num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if num_correct_chunks else 0
        return np.array(
            [precision], dtype='float32'), np.array(
                [recall], dtype='float32'), np.array(
                    [f1_score], dtype='float32')


class EditDistance(Evaluator):
    """
    Accumulate edit distance sum and sequence number from mini-batches and
    compute the average edit_distance and instance error of all batches.

    Args:
        input: the sequences predicted by network.
        label: the target sequences which must has same sequence count
        with input.
        ignored_tokens(list of int): Tokens that should be removed before
        calculating edit distance.

    Example:

        exe = fluid.executor(place)
        distance_evaluator = fluid.Evaluator.EditDistance(input, label)
        for epoch in PASS_NUM:
            distance_evaluator.reset(exe)
            for data in batches:
                loss = exe.run(fetch_list=[cost])
            distance, instance_error = distance_evaluator.eval(exe)

        In the above example:
        'distance' is the average of the edit distance in a pass.
        'instance_error' is the instance error rate in a pass.

    """

    def __init__(self, input, label, ignored_tokens=None, **kwargs):
        super(EditDistance, self).__init__("edit_distance", **kwargs)
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.total_distance = self.create_state(
            dtype='float32', shape=[1], suffix='total_distance')
        self.seq_num = self.create_state(
            dtype='int64', shape=[1], suffix='seq_num')
        self.instance_error = self.create_state(
            dtype='int64', shape=[1], suffix='instance_error')
        distances, seq_num = layers.edit_distance(
            input=input, label=label, ignored_tokens=ignored_tokens)

        zero = layers.fill_constant(shape=[1], value=0.0, dtype='float32')
        compare_result = layers.equal(distances, zero)
        compare_result_int = layers.cast(x=compare_result, dtype='int')
        seq_right_count = layers.reduce_sum(compare_result_int)
        instance_error_count = layers.elementwise_sub(
            x=seq_num, y=seq_right_count)
        total_distance = layers.reduce_sum(distances)
        layers.sums(
            input=[self.total_distance, total_distance],
            out=self.total_distance)
        layers.sums(input=[self.seq_num, seq_num], out=self.seq_num)
        layers.sums(
            input=[self.instance_error, instance_error_count],
            out=self.instance_error)
        self.metrics.append(total_distance)
        self.metrics.append(instance_error_count)

    def eval(self, executor, eval_program=None):
        if eval_program is None:
            eval_program = Program()
        block = eval_program.current_block()
        with program_guard(main_program=eval_program):
            total_distance = _clone_var_(block, self.total_distance)
            seq_num = _clone_var_(block, self.seq_num)
            instance_error = _clone_var_(block, self.instance_error)
            seq_num = layers.cast(x=seq_num, dtype='float32')
            instance_error = layers.cast(x=instance_error, dtype='float32')
            avg_distance = layers.elementwise_div(x=total_distance, y=seq_num)
            avg_instance_error = layers.elementwise_div(
                x=instance_error, y=seq_num)
            result = executor.run(
                eval_program, fetch_list=[avg_distance, avg_instance_error])
        return np.array(result[0]), np.array(result[1])


class DetectionMAP(Evaluator):
    """
    Calculate the detection mean average precision (mAP).

    TODO (Dang Qingqing): update the following doc.
    The general steps are as follows:
    1. calculate the true positive and false positive according to the input
        of detection and labels.
    2. calculate mAP value, support two versions: '11 point' and 'integral'.

    Please get more information from the following articles:
      https://sanchom.wordpress.com/tag/average-precision/
      https://arxiv.org/abs/1512.02325

    Args:
        input (Variable): The detection results, which is a LoDTensor with shape
            [M, 6]. The layout is [label, confidence, xmin, ymin, xmax, ymax].
        gt_label (Variable): The ground truth label index, which is a LoDTensor
            with shape [N, 1].
        gt_difficult (Variable): Whether this ground truth is a difficult
            bounding box (bbox), which is a LoDTensor [N, 1].
        gt_box (Variable): The ground truth bounding box (bbox), which is a
            LoDTensor with shape [N, 6]. The layout is [xmin, ymin, xmax, ymax].
        class_num (int): The class number.
        background_label (int): The index of background label, the background
            label will be ignored. If set to -1, then all categories will be
            considered, 0 by defalut.
        overlap_threshold (float): The threshold for deciding true/false
            positive, 0.5 by defalut.
        evaluate_difficult (bool): Whether to consider difficult ground truth
            for evaluation, True by defalut.
        ap_version (string): The average precision calculation ways, it must be
            'integral' or '11point'. Please check
            https://sanchom.wordpress.com/tag/average-precision/ for details.
            - 11point: the 11-point interpolated average precision.
            - integral: the natural integral of the precision-recall curve.

    Example:

        exe = fluid.executor(place)
        map_evaluator = fluid.Evaluator.DetectionMAP(input,
            gt_label, gt_difficult, gt_box)
        cur_map, accum_map = map_evaluator.get_map_var()
        fetch = [cost, cur_map, accum_map]
        for epoch in PASS_NUM:
            map_evaluator.reset(exe)
            for data in batches:
                loss, cur_map_v, accum_map_v = exe.run(fetch_list=fetch)

        In the above example:

        'cur_map_v' is the mAP of current mini-batch.
        'accum_map_v' is the accumulative mAP of one pass.
    """

    def __init__(self,
                 input,
                 gt_label,
                 gt_box,
                 gt_difficult,
                 class_num,
                 background_label=0,
                 overlap_threshold=0.5,
                 evaluate_difficult=True,
                 ap_version='integral'):
        super(DetectionMAP, self).__init__("map_eval")

        gt_label = layers.cast(x=gt_label, dtype=gt_box.dtype)
        gt_difficult = layers.cast(x=gt_difficult, dtype=gt_box.dtype)
        label = layers.concat([gt_label, gt_difficult, gt_box], axis=1)

        # calculate mean average precision (mAP) of current mini-batch
        map = layers.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            ap_version=ap_version)

        self.create_state(dtype='int32', shape=None, suffix='accum_pos_count')
        self.create_state(dtype='float32', shape=None, suffix='accum_true_pos')
        self.create_state(dtype='float32', shape=None, suffix='accum_false_pos')

        self.has_state = None
        var = self.helper.create_variable(
            persistable=True, dtype='int32', shape=[1])
        self.helper.set_variable_initializer(
            var, initializer=Constant(value=int(0)))
        self.has_state = var

        # calculate accumulative mAP
        accum_map = layers.detection_map(
            input,
            label,
            class_num,
            background_label,
            overlap_threshold=overlap_threshold,
            evaluate_difficult=evaluate_difficult,
            has_state=self.has_state,
            input_states=self.states,
            out_states=self.states,
            ap_version=ap_version)

        layers.fill_constant(
            shape=self.has_state.shape,
            value=1,
            dtype=self.has_state.dtype,
            out=self.has_state)

        self.cur_map = map
        self.accum_map = accum_map

    def get_map_var(self):
        return self.cur_map, self.accum_map

    def reset(self, executor, reset_program=None):
        if reset_program is None:
            reset_program = Program()
        with program_guard(main_program=reset_program):
            var = _clone_var_(reset_program.current_block(), self.has_state)
            layers.fill_constant(
                shape=var.shape, value=0, dtype=var.dtype, out=var)
        executor.run(reset_program)
