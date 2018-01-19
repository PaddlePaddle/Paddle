# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import sys
import argparse
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('Parse Log')
    parser.add_argument(
        '--file_path', '-f', type=str, help='the path of the log file')
    parser.add_argument(
        '--sample_rate',
        '-s',
        type=float,
        default=1.0,
        help='the rate to take samples from log')
    parser.add_argument(
        '--log_period', '-p', type=int, default=1, help='the period of log')

    args = parser.parse_args()
    return args


def parse_file(file_name):
    loss = []
    error = []
    with open(file_name) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line.startswith('pass'):
                continue
            line_split = line.split(' ')
            if len(line_split) != 5:
                continue

            loss_str = line_split[2][:-1]
            cur_loss = float(loss_str.split('=')[-1])
            loss.append(cur_loss)

            err_str = line_split[3][:-1]
            cur_err = float(err_str.split('=')[-1])
            error.append(cur_err)

    accuracy = [1.0 - err for err in error]

    return loss, accuracy


def sample(metric, sample_rate):
    interval = int(1.0 / sample_rate)
    if interval > len(metric):
        return metric[:1]

    num = len(metric) / interval
    idx = [interval * i for i in range(num)]
    metric_sample = [metric[id] for id in idx]
    return metric_sample


def plot_metric(metric,
                batch_id,
                graph_title,
                line_style='b-',
                line_label='y',
                line_num=1):
    plt.figure()
    plt.title(graph_title)
    if line_num == 1:
        plt.plot(batch_id, metric, line_style, label=line_label)
    else:
        for i in range(line_num):
            plt.plot(batch_id, metric[i], line_style[i], label=line_label[i])
    plt.xlabel('batch')
    plt.ylabel(graph_title)
    plt.legend()
    plt.savefig(graph_title + '.jpg')
    plt.close()


def main():
    args = parse_args()
    assert args.sample_rate > 0. and args.sample_rate <= 1.0, "The sample rate should in the range (0, 1]."

    loss, accuracy = parse_file(args.file_path)
    batch = [args.log_period * i for i in range(len(loss))]

    batch_sample = sample(batch, args.sample_rate)
    loss_sample = sample(loss, args.sample_rate)
    accuracy_sample = sample(accuracy, args.sample_rate)

    plot_metric(loss_sample, batch_sample, 'loss', line_label='loss')
    plot_metric(
        accuracy_sample,
        batch_sample,
        'accuracy',
        line_style='g-',
        line_label='accuracy')


if __name__ == '__main__':
    main()
