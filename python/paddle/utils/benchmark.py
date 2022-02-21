# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import time
import logging
from collections import OrderedDict

__all__ = ['benchmark']


class Hook:
    def begin(self, benchmark):
        pass

    def end(self, benchmark):
        pass

    def before_step(self, benchmark):
        pass

    def after_step(self, benchmark):
        pass


class Benchmark(object):
    def __init__(self):
        self.log_info = dict()
        self.perf_summary = dict()
        self.perf_iter = dict()
        self.iter = 0
        self.num_samples = 1
        self.hooks = self.register_hooks()

    def step(self, num_samples):
        self.num_samples = num_samples
        self.iter += 1
        self.step_end()

    def stats(self):
        pass

    def step_stats(self):
        self.perf_iter.update({
            "reader_cost":
            self.hooks['benchmark_hook'].reader_cost_averager.get_average(),
            "batch_cost":
            self.hooks['benchmark_hook'].batch_cost_averager.get_average(),
            "ips":
            self.hooks['benchmark_hook'].batch_cost_averager.get_ips_average()
        })
        self.hooks['benchmark_hook'].reader_cost_averager.reset()
        self.hooks['benchmark_hook'].batch_cost_averager.reset()

        message = ''
        message += ' reader_cost: %.3f s' % (self.perf_iter['reader_cost'])
        message += ' batch_cost: %.3f s' % (self.perf_iter['batch_cost'])
        message += ' ips: %.3f ' % (self.perf_iter['ips'])
        return message

    def begin(self):
        self.call_hook_begin()

    def step_begin(self):
        self.call_hook_before_step()

    def step_end(self):
        self.call_hook_after_step()

    def end(self):
        self.call_hook_end()

    def register_hooks(self):
        hooks = OrderedDict(benchmark_hook=BenchmarkHook(), log_hook=LogHook())
        return hooks

    def call_hook_begin(self):
        for hook in self.hooks.values():
            hook.begin(self)

    def call_hook_end(self):
        for hook in self.hooks.values():
            hook.end(self)

    def call_hook_before_step(self):
        for hook in self.hooks.values():
            hook.before_step(self)

    def call_hook_after_step(self):
        for hook in self.hooks.values():
            hook.after_step(self)


class LogHook(Hook):
    def end(self):
        benchmark.logger.info("========= Benchmark Perf Summary =========")
        benchmark.logger.info('summary iter range: ' + str(
            benchmark.summary_range))
        benchmark.logger.info('[ips        ]\t' + self.get_message(
            benchmark.perf_summary['ips_summary']))
        benchmark.logger.info('[reader_cost]\t' + self.get_message(
            benchmark.perf_summary['reader_summary']))
        benchmark.logger.info('[batch_cost ]\t' + self.get_message(
            benchmark.perf_summary['batch_summary']))
        benchmark.logger.info('reader_ratio ' + '%3.f' % (
            benchmark.perf_summary['reader_ratio']) + '%')

    def get_message(self, message_dict):
        message = ''
        for k, v in message_dict.items():
            message += '%s: %.3f ' % (k, v)
        return message


class BenchmarkHook(Hook):
    def __init__(self):
        self.reader_cost_averager = TimeAverager()
        self.batch_cost_averager = TimeAverager()
        self.summary_reader_cost_list = []
        self.summary_batch_cost_list = []
        self.summary_ips_list = []
        self.start_time = time.time()

    def begin(self, benchmark):
        self.start_time = time.time()

    def before_step(self, benchmark):
        reader_cost = time.time() - self.start_time
        self.reader_cost_averager.record(reader_cost)
        self.summary_reader_cost_list.append(reader_cost)

    def after_step(self, benchmark):
        batch_cost = time.time() - self.start_time
        self.batch_cost_averager.record(batch_cost, benchmark.num_samples)
        self.summary_batch_cost_list.append(batch_cost)
        self.summary_ips_list.append(float(benchmark.num_samples) / batch_cost)
        self.start_time = time.time()

    def end(self, benchmark):
        self.get_perf_summary(benchmark)

    def get_perf_summary(self, benchmark):
        reader_summary = self.get_stats(self.summary_reader_cost_list)
        batch_summary = self.get_stats(self.summary_batch_cost_list)
        ips_summary = self.get_stats(self.summary_ips_list)

        reader_ratio = reader_summary["avg"] / batch_summary["avg"] * 100

        benchmark.perf_summary.update({
            "reader_summary": reader_summary,
            "batch_summary": batch_summary,
            "ips_summary": ips_summary,
            "reader_ratio": reader_ratio
        })

    def get_stats(self, data_list):
        res = dict(
            avg=sum(data_list) / len(data_list),
            max=max(data_list),
            min=min(data_list))
        return res


class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._total_iters = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._total_iters += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._total_iters == 0:
            return 0
        return self._total_time / float(self._total_iters)

    def get_ips_average(self):
        if not self._total_samples or self._total_iters == 0:
            return 0
        return float(self._total_samples) / self._total_time


_benchmark_ = Benchmark()


def benchmark():
    return _benchmark_
