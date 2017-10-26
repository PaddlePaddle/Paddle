import unittest
import numpy as np
from op_test import OpTest


def calc_precision(tp_count, fp_count):
    if tp_count > 0.0 or fp_count > 0.0:
        return tp_count / (tp_count + fp_count)
    return 1.0


def calc_recall(tp_count, fn_count):
    if tp_count > 0.0 or fn_count > 0.0:
        return tp_count / (tp_count + fn_count)
    return 1.0


def calc_f1_score(precision, recall):
    if precision > 0.0 or recall > 0.0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def get_states(predictions, labels, weights=None):
    ins_num = predictions.shape[0]
    class_num = predictions.shape[1]
    # TP FP TN FN
    states = np.zeros((class_num, 4)).astype('float32')
    for i in xrange(ins_num):
        w = weights[i] if weights is not None else 1.0
        max_idx = np.argmax(predictions[i])
        if max_idx == labels[i][0]:
            states[max_idx][0] += w
            for j in xrange(class_num):
                states[j][2] += w
            states[max_idx][2] -= w
        else:
            states[labels[i][0]][3] += w
            states[max_idx][1] += w
            for j in xrange(class_num):
                states[j][2] += w
            states[labels[i][0]][2] -= w
            states[max_idx][2] -= w
    return states


def compute_metrics(states):
    class_num = states.shape[0]
    total_tp_count = 0.0
    total_fp_count = 0.0
    total_fn_count = 0.0
    macro_avg_precision = 0.0
    macro_avg_recall = 0.0
    for i in xrange(class_num):
        total_tp_count += states[i][0]
        total_fp_count += states[i][1]
        total_fn_count += states[i][3]
        macro_avg_precision += calc_precision(states[i][0], states[i][1])
        macro_avg_recall += calc_recall(states[i][0], states[i][3])
    metrics = []
    macro_avg_precision /= class_num
    macro_avg_recall /= class_num
    metrics.append(macro_avg_precision)
    metrics.append(macro_avg_recall)
    metrics.append(calc_f1_score(macro_avg_precision, macro_avg_recall))
    micro_avg_precision = calc_precision(total_tp_count, total_fp_count)
    metrics.append(micro_avg_precision)
    micro_avg_recall = calc_recall(total_tp_count, total_fn_count)
    metrics.append(micro_avg_recall)
    metrics.append(calc_f1_score(micro_avg_precision, micro_avg_recall))
    return np.array(metrics).astype('float32')


class TestPrecisionRecallOp_0(OpTest):
    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        class_num = 10
        predictions = np.random.uniform(0, 1.0,
                                        (ins_num, class_num)).astype('float32')
        labels = np.random.choice(xrange(class_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        states = get_states(predictions, labels)
        metrics = compute_metrics(states)

        self.inputs = {'Predictions': predictions, 'Labels': labels}

        self.outputs = {
            'BatchMetrics': metrics,
            'AccumMetrics': metrics,
            'AccumStatesInfo': states
        }

    def test_check_output(self):
        self.check_output()


class TestPrecisionRecallOp_1(OpTest):
    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        class_num = 10
        predictions = np.random.uniform(0, 1.0,
                                        (ins_num, class_num)).astype('float32')
        weights = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        predictions = np.random.random((ins_num, class_num)).astype('float32')
        labels = np.random.choice(xrange(class_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')

        states = get_states(predictions, labels, weights)
        metrics = compute_metrics(states)
        self.inputs = {
            'Predictions': predictions,
            'Labels': labels,
            'Weights': weights
        }

        self.outputs = {
            'BatchMetrics': metrics,
            'AccumMetrics': metrics,
            'AccumStatesInfo': states
        }

    def test_check_output(self):
        self.check_output()


class TestPrecisionRecallOp_2(OpTest):
    def setUp(self):
        self.op_type = "precision_recall"
        ins_num = 64
        class_num = 10
        predictions = np.random.uniform(0, 1.0,
                                        (ins_num, class_num)).astype('float32')
        weights = np.random.uniform(0, 1.0, (ins_num, 1)).astype('float32')
        predictions = np.random.random((ins_num, class_num)).astype('float32')
        labels = np.random.choice(xrange(class_num), ins_num).reshape(
            (ins_num, 1)).astype('int32')
        states = np.random.randint(0, 30, (class_num, 4)).astype('float32')

        accum_states = get_states(predictions, labels, weights)
        batch_metrics = compute_metrics(accum_states)
        accum_states += states
        accum_metrics = compute_metrics(accum_states)

        self.inputs = {
            'Predictions': predictions,
            'Labels': labels,
            'Weights': weights,
            'StatesInfo': states
        }

        self.outputs = {
            'BatchMetrics': batch_metrics,
            'AccumMetrics': accum_metrics,
            'AccumStatesInfo': accum_states
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
