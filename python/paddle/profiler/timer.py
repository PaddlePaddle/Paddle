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


class Stack(object):
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

    def peek(self):
        if not self.is_empty():
            return self.items[len(self.items) - 1]
        else:
            return None


class Hook:
    def begin(self, benchmark):
        pass

    def end(self, benchmark):
        pass

    def before_reader(self, benchmark):
        pass

    def after_reader(self, benchmark):
        pass

    def after_step(self, benchmark):
        pass


class Benchmark(object):
    def __init__(self):
        self.num_samples = None
        self.hooks = self.register_hooks()
        self.current_event = None
        self.events = Stack()

    def step(self, num_samples=None):
        self.num_samples = num_samples
        self.after_step()

    def step_info(self):
        message = ''
        reader_average = self.current_event.reader_average()
        batch_average = self.current_event.batch_average()
        ips_average = self.current_event.ips_average()
        if reader_average:
            message += ' reader_cost: %.5f s' % (reader_average)
        if batch_average:
            message += ' batch_cost: %.5f s' % (batch_average)
        if ips_average:
            message += ' ips: %.5f samples/s' % (ips_average)
        self.current_event.reset()
        return message

    def begin(self):
        for hook in self.hooks.values():
            hook.begin(self)

    def before_reader(self):
        for hook in self.hooks.values():
            hook.before_reader(self)

    def after_reader(self):
        for hook in self.hooks.values():
            hook.after_reader(self)

    def after_step(self):
        for hook in self.hooks.values():
            hook.after_step(self)

    def end(self):
        for hook in self.hooks.values():
            hook.end(self)

    def register_hooks(self):
        hooks = OrderedDict(timer_hook=TimerHook(), log_hook=LogHook())
        return hooks

    def check_if_need_record(self, reader):
        if self.current_event is None:
            return
        if self.current_event.need_record:
            if self.current_event.reader is None:
                self.current_event.reader = reader
            elif self.current_event.reader is not reader:
                self.current_event.need_record = False
        else:
            if self.current_event.reader is reader:
                self.current_event.need_record = True
                self.hooks['timer_hook'].start_time = time.time()


class LogHook(Hook):
    def end(self, benchmark):
        summary = benchmark.current_event.get_summary()
        if not summary:
            return
        print("========= Benchmark Perf Summary =========")
        print('[ips        ]\t' + self.get_message(summary['ips_summary']))
        print('[reader_cost]\t' + self.get_message(summary['reader_summary']))
        print('[batch_cost ]\t' + self.get_message(summary['batch_summary']))
        print('reader_ratio ' + '%5.f' % (summary['reader_ratio']) + '%')

    def get_message(self, message_dict):
        message = ''
        for k, v in message_dict.items():
            message += '%s: %.3e ' % (k, v)
        return message


class TimerHook(Hook):
    def __init__(self):
        self.start_time = time.time()
        self.start_reader = time.time()

    def begin(self, benchmark):
        benchmark.events.push(Event())
        benchmark.current_event = benchmark.events.peek()
        self.start_time = time.time()

    def before_reader(self, benchmark):
        self.start_reader = time.time()

    def after_reader(self, benchmark):
        if (benchmark.current_event is None) or (
                not benchmark.current_event.need_record):
            return
        reader_cost = time.time() - self.start_reader
        benchmark.current_event.record_reader(reader_cost)
        if benchmark.current_event.total_iters >= benchmark.current_event.skip_iter:
            benchmark.current_event.update_records(
                reader_cost, benchmark.current_event.reader_records)

    def after_step(self, benchmark):
        if (benchmark.current_event is None) or (
                not benchmark.current_event.need_record):
            return
        batch_cost = time.time() - self.start_time
        benchmark.current_event.record_batch(batch_cost, benchmark.num_samples)
        if benchmark.current_event.total_iters >= benchmark.current_event.skip_iter:
            benchmark.current_event.update_records(
                batch_cost, benchmark.current_event.batch_records)
            current_ips = float(benchmark.num_samples) / batch_cost
            benchmark.current_event.update_records(
                current_ips, benchmark.current_event.ips_records)
            if benchmark.num_samples:
                benchmark.current_event.total_samples += benchmark.num_samples
        self.start_time = time.time()

    def end(self, benchmark):
        if benchmark.events.is_empty():
            return
        benchmark.events.pop()
        benchmark.current_event = benchmark.events.peek()
        self.start_time = time.time()


class Event(object):
    def __init__(self):
        self.reader_cost_averager = TimeAverager()
        self.batch_cost_averager = TimeAverager()
        self.total_samples = 0
        self.total_iters = 0
        self.skip_iter = 10
        self.reader_records = dict(max=0, min=float('inf'), total=0)
        self.batch_records = dict(max=0, min=float('inf'), total=0)
        self.ips_records = dict(max=0, min=float('inf'))
        self.reader = None
        self.need_record = True

    def reset(self):
        self.reader_cost_averager.reset()
        self.batch_cost_averager.reset()

    def record_reader(self, usetime):
        self.reader_cost_averager.record(usetime)

    def record_batch(self, usetime, num_samples=None):
        self.batch_cost_averager.record(usetime, num_samples)
        self.total_iters += 1

    def reader_average(self):
        return self.reader_cost_averager.get_average()

    def batch_average(self):
        return self.batch_cost_averager.get_average()

    def ips_average(self):
        return self.batch_cost_averager.get_ips_average()

    def update_records(self, current_record, records):
        if current_record > records['max']:
            records['max'] = current_record
        elif current_record < records['min']:
            records['min'] = current_record
        if 'total' in records.keys():
            records['total'] += current_record

    def get_summary(self):
        if self.total_iters < self.skip_iter:
            return {}
        reader_avg = 0
        batch_avg = 0
        ips_avg = 0
        if self.total_iters:
            reader_avg = self.reader_records['total'] / float(self.total_iters -
                                                              self.skip_iter)
            batch_avg = self.batch_records['total'] / float(self.total_iters -
                                                            self.skip_iter)
            ips_avg = float(self.total_samples) / self.batch_records['total']
        reader_summary = dict(
            max=self.reader_records['max'],
            min=self.reader_records['min'],
            avg=reader_avg)
        batch_summary = dict(
            max=self.batch_records['max'],
            min=self.batch_records['min'],
            avg=batch_avg)
        ips_summary = dict(
            max=self.ips_records['max'],
            min=self.ips_records['min'],
            avg=ips_avg)
        reader_ratio = (reader_avg / batch_avg) * 100
        summary = dict(
            reader_summary=reader_summary,
            batch_summary=batch_summary,
            ips_summary=ips_summary,
            reader_ratio=reader_ratio)

        return summary


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
