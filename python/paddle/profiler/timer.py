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


class Stack(object):
    """
    The stack in a Last-In/First-Out (LIFO) manner. New element is added at
    the end and an element is removed from that end.
    """

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
    """
    As the base class. All types of hooks should inherit from it.
    """

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
    """
    A tool for the statistics of model performance. The `before_reader`
    and `after_reader` are called in the DataLoader to count the cost
    of loading the data. The `begin`, `step` and `end` are called to
    count the cost of a step or total steps.
    """

    def __init__(self):
        self.num_samples = None
        self.hooks = OrderedDict(timer_hook=TimerHook())
        self.current_event = None
        self.events = Stack()

    def step(self, num_samples=None):
        self.num_samples = num_samples
        self.after_step()

    def step_info(self):
        message = ''
        reader_average = self.current_event.reader_average()
        batch_average = self.current_event.batch_average()
        if reader_average:
            message += ' reader_cost: %.5f s' % (reader_average)
        if batch_average:
            info = "step_cost" if self.current_event.speed_mode == "steps/s" else "batch_cost"
            message += ' %s: %.5f s' % (info, batch_average)
        speed_average = self.current_event.speed_average()
        if speed_average:
            message += ' speed: %.3f %s' % (speed_average,
                                            self.current_event.speed_mode)
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

    def check_if_need_record(self, reader):
        if self.current_event is None:
            return
        if self.current_event.need_record:
            # set reader for the current event at the first iter
            if self.current_event.reader is None:
                self.current_event.reader = reader
            elif self.current_event.reader is not reader:
                # enter a new task but not calling beign() to record it.
                # we pause the timer until the end of new task, so that 
                # the cost of new task is not added to the current event.
                # eg. start evaluation in the traing task
                self.current_event.need_record = False
        else:
            # when the new task exits, continue timing for the current event.
            if self.current_event.reader is reader:
                self.current_event.need_record = True
                self.hooks['timer_hook'].start_time = time.time()


class TimerHook(Hook):
    """
    A hook for recording real-time performance and the summary
    performance of total steps. 
    """

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
            if benchmark.current_event.speed_mode == "samples/s":
                current_speed = float(benchmark.num_samples) / batch_cost
                benchmark.current_event.total_samples += benchmark.num_samples
            else:
                current_speed = 1.0 / batch_cost  # steps/s
            benchmark.current_event.update_records(
                current_speed, benchmark.current_event.speed_records)
        self.start_time = time.time()

    def end(self, benchmark):
        if benchmark.events.is_empty():
            return
        self.print_summary(benchmark)
        benchmark.events.pop()
        benchmark.current_event = benchmark.events.peek()
        self.start_time = time.time()

    def print_summary(self, benchmark):
        summary = benchmark.current_event.get_summary()
        if not summary:
            return
        print('Perf Summary'.center(100, '='))
        if summary['reader_ratio'] != 0:
            print('Reader_ratio: ' + '%.3f' % (summary['reader_ratio']) + '%')
        print('Time unit: s, Speed unit: %s' %
              (benchmark.current_event.speed_mode))
        print('|', ''.center(15), '|', 'avg'.center(15), '|', 'max'.center(15),
              '|', 'min'.center(15), '|')
        # if DataLoader is not called, reader_summary is unnecessary.
        if summary['reader_summary']['avg'] != 0:
            self.print_stats('reader_cost', summary['reader_summary'])
        item = 'batch_cost' if benchmark.current_event.speed_mode == 'samples/s' else 'step_cost'
        self.print_stats(item, summary['batch_summary'])
        self.print_stats('Speed', summary['speed_summary'])

    def print_stats(self, item, message_dict):
        avg_str = '%.5f' % (message_dict['avg'])
        max_str = '%.5f' % (message_dict['max'])
        min_str = '%.5f' % (message_dict['min'])
        print('|',
              item.center(15), '|',
              avg_str.center(15), '|',
              max_str.center(15), '|', min_str.center(15), '|')


class Event(object):
    """
    A Event is used to record the cost of every step and the cost of
    the total steps except skipped steps.
    """

    def __init__(self):
        self.reader_cost_averager = TimeAverager()
        self.batch_cost_averager = TimeAverager()
        self.total_samples = 0
        self.total_iters = 0
        self.skip_iter = 10
        self.reader_records = dict(max=0, min=float('inf'), total=0)
        self.batch_records = dict(max=0, min=float('inf'), total=0)
        self.speed_records = dict(max=0, min=float('inf'))
        self.reader = None
        self.need_record = True
        self.speed_mode = "samples/s"

    def reset(self):
        self.reader_cost_averager.reset()
        self.batch_cost_averager.reset()

    def record_reader(self, usetime):
        self.reader_cost_averager.record(usetime)

    def record_batch(self, usetime, num_samples=None):
        if num_samples is None:
            self.speed_mode = "steps/s"
        self.batch_cost_averager.record(usetime, num_samples)
        self.total_iters += 1

    def reader_average(self):
        return self.reader_cost_averager.get_average()

    def batch_average(self):
        return self.batch_cost_averager.get_average()

    def speed_average(self):
        if self.speed_mode == "samples/s":
            return self.batch_cost_averager.get_ips_average()
        else:
            return self.batch_cost_averager.get_step_average()

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
        speed_avg = 0

        self.total_iters -= self.skip_iter
        reader_avg = self.reader_records['total'] / float(self.total_iters)
        batch_avg = self.batch_records['total'] / float(self.total_iters)
        if self.speed_mode == "samples/s":
            speed_avg = float(self.total_samples) / self.batch_records['total']
        else:
            speed_avg = float(self.total_iters) / self.batch_records['total']

        reader_summary = dict(
            max=self.reader_records['max'],
            min=self.reader_records['min'],
            avg=reader_avg)
        batch_summary = dict(
            max=self.batch_records['max'],
            min=self.batch_records['min'],
            avg=batch_avg)
        speed_summary = dict(
            max=self.speed_records['max'],
            min=self.speed_records['min'],
            avg=speed_avg)
        reader_ratio = (reader_avg / batch_avg) * 100
        summary = dict(
            reader_summary=reader_summary,
            batch_summary=batch_summary,
            speed_summary=speed_summary,
            reader_ratio=reader_ratio)

        return summary


class TimeAverager(object):
    """
    Record the cost of every step and count the average.
    """

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

    def get_step_average(self):
        if self._total_iters == 0:
            return 0
        return float(self._total_iters) / self._total_time


_benchmark_ = Benchmark()


def benchmark():
    return _benchmark_
