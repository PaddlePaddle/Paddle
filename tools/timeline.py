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

import argparse
import json
import sys
import unittest

import google.protobuf.text_format as text_format
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--profile_path',
    type=str,
    default='',
    help='Input profile file name. If there are multiple file, the format '
    'should be trainer1=file1,trainer2=file2,ps=file3')
parser.add_argument(
    '--timeline_path', type=str, default='', help='Output timeline file name.')
args = parser.parse_args()


class _ChromeTraceFormatter(object):
    def __init__(self):
        self._events = []
        self._metadata = []

    def _create_event(self, ph, category, name, pid, tid, timestamp):
        """Creates a new Chrome Trace event.

        For details of the file format, see:
        https://github.com/catapult-project/catapult/blob/master/tracing/README.md

        Args:
          ph:  The type of event - usually a single character.
          category: The event category as a string.
          name:  The event name as a string.
          pid:  Identifier of the process generating this event as an integer.
          tid:  Identifier of the thread generating this event as an integer.
          timestamp:  The timestamp of this event as a long integer.

        Returns:
          A JSON compatible event object.
        """
        event = {}
        event['ph'] = ph
        event['cat'] = category
        event['name'] = name
        event['pid'] = pid
        event['tid'] = tid
        event['ts'] = timestamp
        return event

    def emit_pid(self, name, pid):
        """Adds a process metadata event to the trace.

        Args:
          name:  The process name as a string.
          pid:  Identifier of the process as an integer.
        """
        event = {}
        event['name'] = 'process_name'
        event['ph'] = 'M'
        event['pid'] = pid
        event['args'] = {'name': name}
        self._metadata.append(event)

    def emit_region(self, timestamp, duration, pid, tid, category, name, args):
        """Adds a region event to the trace.

        Args:
          timestamp:  The start timestamp of this region as a long integer.
          duration:  The duration of this region as a long integer.
          pid:  Identifier of the process generating this event as an integer.
          tid:  Identifier of the thread generating this event as an integer.
          category: The event category as a string.
          name:  The event name as a string.
          args:  A JSON-compatible dictionary of event arguments.
        """
        event = self._create_event('X', category, name, pid, tid, timestamp)
        event['dur'] = duration
        event['args'] = args
        self._events.append(event)

    def format_to_string(self, pretty=False):
        """Formats the chrome trace to a string.

        Args:
          pretty: (Optional.)  If True, produce human-readable JSON output.

        Returns:
          A JSON-formatted string in Chrome Trace format.
        """
        trace = {}
        trace['traceEvents'] = self._metadata + self._events
        if pretty:
            return json.dumps(trace, indent=4, separators=(',', ': '))
        else:
            return json.dumps(trace, separators=(',', ':'))


class Timeline(object):
    def __init__(self, profile_dict):
        self._profile_dict = profile_dict
        self._pid = 0
        self._devices = dict()
        self._chrome_trace = _ChromeTraceFormatter()

    def _allocate_pid(self):
        cur_pid = self._pid
        self._pid += 1
        return cur_pid

    def _allocate_pids(self):
        for k, profile_pb in self._profile_dict.iteritems():
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    if (k, event.device_id, "CPU") not in self._devices:
                        pid = self._allocate_pid()
                        self._devices[(k, event.device_id, "CPU")] = pid
                        self._chrome_trace.emit_pid("%s:cpu:block:%d" %
                                                    (k, event.device_id), pid)
                elif event.type == profiler_pb2.Event.GPUKernel:
                    if (k, event.device_id, "GPUKernel") not in self._devices:
                        pid = self._allocate_pid()
                        self._devices[(k, event.device_id, "GPUKernel")] = pid
                        self._chrome_trace.emit_pid("%s:gpu:%d" %
                                                    (k, event.device_id), pid)

    def _allocate_events(self):
        for k, profile_pb in self._profile_dict.iteritems():
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    type = "CPU"
                elif event.type == profiler_pb2.Event.GPUKernel:
                    type = "GPUKernel"
                pid = self._devices[(k, event.device_id, type)]
                args = {'name': event.name}
                if event.memcopy.bytes > 0:
                    args = {'mem_bytes': event.memcopy.bytes}
                # TODO(panyx0718): Chrome tracing only handles ms. However, some
                # ops takes micro-seconds. Hence, we keep the ns here.
                self._chrome_trace.emit_region(
                    event.start_ns, (event.end_ns - event.start_ns) / 1.0, pid,
                    event.sub_device_id, 'Op', event.name, args)

    def generate_chrome_trace(self):
        self._allocate_pids()
        self._allocate_events()
        return self._chrome_trace.format_to_string()


profile_path = '/tmp/profile'
if args.profile_path:
    profile_path = args.profile_path
timeline_path = '/tmp/timeline'
if args.timeline_path:
    timeline_path = args.timeline_path

profile_paths = profile_path.split(',')
profile_dict = dict()
if len(profile_paths) == 1:
    with open(profile_path, 'r') as f:
        profile_s = f.read()
        profile_pb = profiler_pb2.Profile()
        profile_pb.ParseFromString(profile_s)
    profile_dict['trainer'] = profile_pb
else:
    for profile_path in profile_paths:
        k, v = profile_path.split('=')
        with open(v, 'r') as f:
            profile_s = f.read()
            profile_pb = profiler_pb2.Profile()
            profile_pb.ParseFromString(profile_s)
        profile_dict[k] = profile_pb

tl = Timeline(profile_dict)
with open(timeline_path, 'w') as f:
    f.write(tl.generate_chrome_trace())
