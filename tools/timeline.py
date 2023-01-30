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
<<<<<<< HEAD

=======
import six
import sys
import unittest

import google.protobuf.text_format as text_format
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--profile_path',
    type=str,
    default='',
    help='Input profile file name. If there are multiple file, the format '
<<<<<<< HEAD
    'should be trainer1=file1,trainer2=file2,ps=file3',
)
parser.add_argument(
    '--timeline_path', type=str, default='', help='Output timeline file name.'
)
args = parser.parse_args()


class _ChromeTraceFormatter:
=======
    'should be trainer1=file1,trainer2=file2,ps=file3')
parser.add_argument('--timeline_path',
                    type=str,
                    default='',
                    help='Output timeline file name.')
args = parser.parse_args()


class _ChromeTraceFormatter(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
        event['name'] = name.replace("ParallelExecutor::Run/", "")
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

    def emit_counter(self, category, name, pid, timestamp, counter, value):
        """Emits a record for a single counter.

        Args:
            category: The event category as string
            name: The event name as string
            pid: Identifier of the process generating this event as integer
            timestamp: The timestamps of this event as long integer
            counter: Name of the counter as string
            value: Value of the counter as integer
            tid: Thread id of the allocation as integer
        """
        event = self._create_event('C', category, name, pid, 0, timestamp)
        event['args'] = {counter: value}
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


<<<<<<< HEAD
class Timeline:
=======
class Timeline(object):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, profile_dict):
        self._profile_dict = profile_dict
        self._pid = 0
        self._devices = dict()
        self._mem_devices = dict()
        self._chrome_trace = _ChromeTraceFormatter()

    def _allocate_pid(self):
        cur_pid = self._pid
        self._pid += 1
        return cur_pid

    def _allocate_pids(self):
<<<<<<< HEAD
        for k, profile_pb in self._profile_dict.items():
=======
        for k, profile_pb in six.iteritems(self._profile_dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    if (k, event.device_id, "CPU") not in self._devices:
                        pid = self._allocate_pid()
                        self._devices[(k, event.device_id, "CPU")] = pid
                        # -1 device id represents CUDA API(RunTime) call.(e.g. cudaLaunch, cudaMemcpy)
                        if event.device_id == -1:
                            self._chrome_trace.emit_pid("%s:cuda_api" % k, pid)
                        else:
                            self._chrome_trace.emit_pid(
<<<<<<< HEAD
                                "%s:cpu:block:%d" % (k, event.device_id), pid
                            )
=======
                                "%s:cpu:block:%d" % (k, event.device_id), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                elif event.type == profiler_pb2.Event.GPUKernel:
                    if (k, event.device_id, "GPUKernel") not in self._devices:
                        pid = self._allocate_pid()
                        self._devices[(k, event.device_id, "GPUKernel")] = pid
                        self._chrome_trace.emit_pid(
<<<<<<< HEAD
                            "%s:gpu:%d" % (k, event.device_id), pid
                        )
=======
                            "%s:gpu:%d" % (k, event.device_id), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            if not hasattr(profile_pb, "mem_events"):
                continue
            for mevent in profile_pb.mem_events:
                if mevent.place == profiler_pb2.MemEvent.CUDAPlace:
                    if (k, mevent.device_id, "GPU") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id, "GPU")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:gpu:%d" % (k, mevent.device_id),
<<<<<<< HEAD
                            pid,
                        )
=======
                            pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                elif mevent.place == profiler_pb2.MemEvent.CPUPlace:
                    if (k, mevent.device_id, "CPU") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id, "CPU")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:cpu:%d" % (k, mevent.device_id),
<<<<<<< HEAD
                            pid,
                        )
                elif mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace:
                    if (
                        k,
                        mevent.device_id,
                        "CUDAPinnedPlace",
                    ) not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[
                            (k, mevent.device_id, "CUDAPinnedPlace")
                        ] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:cudapinnedplace:%d"
                            % (k, mevent.device_id),
                            pid,
                        )
=======
                            pid)
                elif mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace:
                    if (k, mevent.device_id,
                            "CUDAPinnedPlace") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id,
                                           "CUDAPinnedPlace")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:cudapinnedplace:%d" %
                            (k, mevent.device_id), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                elif mevent.place == profiler_pb2.MemEvent.NPUPlace:
                    if (k, mevent.device_id, "NPU") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id, "NPU")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:npu:%d" % (k, mevent.device_id),
<<<<<<< HEAD
                            pid,
                        )
=======
                            pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if (k, 0, "CPU") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "CPU")] = pid
                    self._chrome_trace.emit_pid(
<<<<<<< HEAD
                        "memory usage on %s:cpu:%d" % (k, 0), pid
                    )
=======
                        "memory usage on %s:cpu:%d" % (k, 0), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if (k, 0, "GPU") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "GPU")] = pid
                    self._chrome_trace.emit_pid(
<<<<<<< HEAD
                        "memory usage on %s:gpu:%d" % (k, 0), pid
                    )
=======
                        "memory usage on %s:gpu:%d" % (k, 0), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if (k, 0, "CUDAPinnedPlace") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "CUDAPinnedPlace")] = pid
                    self._chrome_trace.emit_pid(
<<<<<<< HEAD
                        "memory usage on %s:cudapinnedplace:%d" % (k, 0), pid
                    )
=======
                        "memory usage on %s:cudapinnedplace:%d" % (k, 0), pid)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                if (k, 0, "NPU") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "NPU")] = pid
                    self._chrome_trace.emit_pid(
<<<<<<< HEAD
                        "memory usage on %s:npu:%d" % (k, 0), pid
                    )

    def _allocate_events(self):
        for k, profile_pb in self._profile_dict.items():
=======
                        "memory usage on %s:npu:%d" % (k, 0), pid)

    def _allocate_events(self):
        for k, profile_pb in six.iteritems(self._profile_dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    type = "CPU"
                elif event.type == profiler_pb2.Event.GPUKernel:
                    type = "GPUKernel"
                pid = self._devices[(k, event.device_id, type)]
                args = {'name': event.name}
                if event.memcopy.bytes > 0:
                    args['mem_bytes'] = event.memcopy.bytes
                if hasattr(event, "detail_info") and event.detail_info:
                    args['detail_info'] = event.detail_info
                # TODO(panyx0718): Chrome tracing only handles ms. However, some
                # ops takes micro-seconds. Hence, we keep the ns here.
                self._chrome_trace.emit_region(
<<<<<<< HEAD
                    event.start_ns,
                    (event.end_ns - event.start_ns) / 1.0,
                    pid,
                    event.sub_device_id,
                    'Op',
                    event.name,
                    args,
                )
=======
                    event.start_ns, (event.end_ns - event.start_ns) / 1.0, pid,
                    event.sub_device_id, 'Op', event.name, args)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _allocate_memory_event(self):
        if not hasattr(profiler_pb2, "MemEvent"):
            return
        place_to_str = {
            profiler_pb2.MemEvent.CPUPlace: "CPU",
            profiler_pb2.MemEvent.CUDAPlace: "GPU",
            profiler_pb2.MemEvent.CUDAPinnedPlace: "CUDAPinnedPlace",
<<<<<<< HEAD
            profiler_pb2.MemEvent.NPUPlace: "NPU",
        }
        for k, profile_pb in self._profile_dict.items():
=======
            profiler_pb2.MemEvent.NPUPlace: "NPU"
        }
        for k, profile_pb in six.iteritems(self._profile_dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            mem_list = []
            end_profiler = 0
            for mevent in profile_pb.mem_events:
                crt_info = dict()
                crt_info['time'] = mevent.start_ns
                crt_info['size'] = mevent.bytes
                if mevent.place in place_to_str:
                    place = place_to_str[mevent.place]
                else:
                    place = "UnDefine"
                crt_info['place'] = place
                pid = self._mem_devices[(k, mevent.device_id, place)]
                crt_info['pid'] = pid
                crt_info['thread_id'] = mevent.thread_id
                crt_info['device_id'] = mevent.device_id
                mem_list.append(crt_info)
                crt_info = dict()
                crt_info['place'] = place
                crt_info['pid'] = pid
                crt_info['thread_id'] = mevent.thread_id
                crt_info['device_id'] = mevent.device_id
                crt_info['time'] = mevent.end_ns
                crt_info['size'] = -mevent.bytes
                mem_list.append(crt_info)
                end_profiler = max(end_profiler, crt_info['time'])
            mem_list.sort(key=lambda tmp: (tmp.get('time', 0)))
            i = 0
            total_size = 0
            while i < len(mem_list):
                total_size += mem_list[i]['size']
<<<<<<< HEAD
                while (
                    i < len(mem_list) - 1
                    and mem_list[i]['time'] == mem_list[i + 1]['time']
                ):
                    total_size += mem_list[i + 1]['size']
                    i += 1

                self._chrome_trace.emit_counter(
                    "Memory",
                    "Memory",
                    mem_list[i]['pid'],
                    mem_list[i]['time'],
                    0,
                    total_size,
                )
=======
                while i < len(mem_list) - 1 and mem_list[i]['time'] == mem_list[
                        i + 1]['time']:
                    total_size += mem_list[i + 1]['size']
                    i += 1

                self._chrome_trace.emit_counter("Memory", "Memory",
                                                mem_list[i]['pid'],
                                                mem_list[i]['time'], 0,
                                                total_size)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                i += 1

    def generate_chrome_trace(self):
        self._allocate_pids()
        self._allocate_events()
        self._allocate_memory_event()
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
    with open(profile_path, 'rb') as f:
        profile_s = f.read()
        profile_pb = profiler_pb2.Profile()
        profile_pb.ParseFromString(profile_s)
    profile_dict['trainer'] = profile_pb
else:
    for profile_path in profile_paths:
        k, v = profile_path.split('=')
        with open(v, 'rb') as f:
            profile_s = f.read()
            profile_pb = profiler_pb2.Profile()
            profile_pb.ParseFromString(profile_s)
        profile_dict[k] = profile_pb

tl = Timeline(profile_dict)
with open(timeline_path, 'w') as f:
    f.write(tl.generate_chrome_trace())
