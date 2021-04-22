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
import six
import sys
import re
import os
import glob
import unittest

import google.protobuf.text_format as text_format
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2

TIME_PATH="time"
DCGM_PATH="dcgm"
NET_PATH="net"
PROFILE_PATH="profile"

def get_argparse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--profile_path',
        type=str,
        default='.',
        help='Working path that store the monitor data.')

    parser.add_argument(
        '--timeline_path',
        type=str,
        default='.',
        help='Output timeline file name.')

    parser.add_argument(
        '--gpuPerTrainer',
        type=int,
        default=8,
        help='Gpus per trainer.')

    return parser.parse_args()

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

class Timeline(object):
    def __init__(self, args):
        self._args = args
        self._workPath= self._args.profile_path
        self._saveFile = self._args.timeline_path
        self._gpuPerTrainer = self._args.gpuPerTrainer

        self._profile_dict = {}
        self._pid = 0
        self._devices = dict()
        self._mem_devices = dict()
        self._chrome_trace = _ChromeTraceFormatter()

        self._dcgmPath = os.path.join(self._workPath, DCGM_PATH)
        self._timePath = os.path.join(self._workPath, TIME_PATH)
        self._netPath = os.path.join(self._workPath, NET_PATH)
        self._profilePath = os.path.join(self._workPath, PROFILE_PATH)

        self._timeInfo = {}
        self._minTimeStamp = 0

    def _add_profile(self):
        profileFileList = glob.glob(os.path.join(self._profilePath, "*"))
        for profile in profileFileList:
            with open(profile, 'rb') as f:
                profile_s = f.read()
                profile_pb = profiler_pb2.Profile()
                profile_pb.ParseFromString(profile_s)

            self._profile_dict[os.path.basename(profile)] = profile_pb

    def _set_timeInfo(self, timeFileNamePrefix="time.txt", sed="."):
        timeFileNameList = glob.glob(os.path.join(self._timePath, timeFileNamePrefix, sed,"*"))
        for timeFileName in timeFileNameList:
            gpuId = int(timeFileName.split(sed)[-1])
            info={}
            with open(timeFileName, "r") as rf:
                for line in rf:
                    if line.startswith("start time:"):
                        info["start_time"] = int(float(line.split(":")[-1]) * 1e9)

                        self._minTimeStamp  = min(self._minTimeStamp, info["start_time"])

                    if line.startswith("end time:"):
                        info["end_time"] = int(float(line.split(":")[-1]) * 1e9)
            if not info:
                self._timeInfo[gpuId] = info

    def _align_ts(self, ts):
        return ts - self._minTimeStamp
        # return ts

    def _allocate_pid(self):
        cur_pid = self._pid
        self._pid += 1
        # print(cur_pid)
        return cur_pid

    def _allocate_pids(self):
        for k, profile_pb in six.iteritems(self._profile_dict):
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
                                "%s:cpu:block:%d" % (k, event.device_id), pid)
                elif event.type == profiler_pb2.Event.GPUKernel:
                    if (k, event.device_id, "GPUKernel") not in self._devices:
                        pid = self._allocate_pid()
                        self._devices[(k, event.device_id, "GPUKernel")] = pid
                        self._chrome_trace.emit_pid("%s:gpu:%d" %
                                                    (k, event.device_id), pid)
            if not hasattr(profile_pb, "mem_events"):
                continue
            for mevent in profile_pb.mem_events:
                if mevent.place == profiler_pb2.MemEvent.CUDAPlace:
                    if (k, mevent.device_id, "GPU") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id, "GPU")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:gpu:%d" % (k, mevent.device_id),
                            pid)
                elif mevent.place == profiler_pb2.MemEvent.CPUPlace:
                    if (k, mevent.device_id, "CPU") not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id, "CPU")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:cpu:%d" % (k, mevent.device_id),
                            pid)
                elif mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace:
                    if (k, mevent.device_id, "CUDAPinnedPlace"
                        ) not in self._mem_devices:
                        pid = self._allocate_pid()
                        self._mem_devices[(k, mevent.device_id,
                                           "CUDAPinnedPlace")] = pid
                        self._chrome_trace.emit_pid(
                            "memory usage on %s:cudapinnedplace:%d" %
                            (k, mevent.device_id), pid)
                if (k, 0, "CPU") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "CPU")] = pid
                    self._chrome_trace.emit_pid("memory usage on %s:cpu:%d" %
                                                (k, 0), pid)
                if (k, 0, "GPU") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "GPU")] = pid
                    self._chrome_trace.emit_pid("memory usage on %s:gpu:%d" %
                                                (k, 0), pid)
                if (k, 0, "CUDAPinnedPlace") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "CUDAPinnedPlace")] = pid
                    self._chrome_trace.emit_pid(
                        "memory usage on %s:cudapinnedplace:%d" % (k, 0), pid)

    def _allocate_events(self):
        for k, profile_pb in six.iteritems(self._profile_dict):
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
                    self._align_ts(event.start_ns), (event.end_ns - event.start_ns) / 1.0, pid,
                    event.sub_device_id, 'Op', event.name, args)

    def _allocate_memory_event(self):
        if not hasattr(profiler_pb2, "MemEvent"):
            return
        place_to_str = {
            profiler_pb2.MemEvent.CPUPlace: "CPU",
            profiler_pb2.MemEvent.CUDAPlace: "GPU",
            profiler_pb2.MemEvent.CUDAPinnedPlace: "CUDAPinnedPlace"
        }
        for k, profile_pb in six.iteritems(self._profile_dict):
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
                while i < len(mem_list) - 1 and mem_list[i]['time'] == mem_list[
                        i + 1]['time']:
                    total_size += mem_list[i + 1]['size']
                    i += 1

                self._chrome_trace.emit_counter(
                    "Memory", "Memory", mem_list[i]['pid'], self._align_ts(mem_list[i]['time']),
                    0, total_size)
                i += 1

    def _add_mlnx_perf(self, sed="."):
        if self._netPath == None:
            return

        netFileNameList = glob.glob(os.path.join(self._netPath,"*"))

        for netFileName in netFileNameList:
            tx_pid = self._allocate_pid()
            rx_pid = self._allocate_pid()

            trainerId=int(netFileName.split(sed)[-1])

            self._chrome_trace.emit_pid("tx_%d" % trainerId, tx_pid)
            self._chrome_trace.emit_pid("rx_%d" % trainerId, rx_pid)

            with open(netFileName, "r") as rf:
                for line in rf:
                    try:
                        event_str = json.loads(line.strip())
                        event_str["pid"] = tx_pid if event_str["name"]=="tx" else rx_pid
                        # the unit of net is ms, we need ns
                        event_str["ts"] = self._align_ts(event_str["ts"] * 1e6)
                        self._chrome_trace._events.append(event_str)
                    except Exception:
                        print("warning: invalid record [%s] in [%s]. skip it!" % (line[:-1], netFileName))

    def _add_dcgm_perf(self):
        if self._dcgmPath == None:
            return

        dcgmFileNameList = glob.glob(os.path.join(self._dcgmPath,"*"))

        for dcgmFileName in dcgmFileNameList:
            self._add_one_dcgm_perf(dcgmFileName)

    def _add_one_dcgm_perf(self, dcgm_perf_file, sed="."):
        if dcgm_perf_file == None:
            return

        trainerId=int(dcgm_perf_file.split(sed)[-1])
        gpuPerTrainer = 8

        regex_list = [
            (re.compile(r' +'), ','),
            (re.compile(r'^,'), ''),
        ]

        has_header = False
        metric_list = []
        pid_map = {}

        with open(dcgm_perf_file, "r") as rf:
            for line in rf:
                #  skip the unused line
                if 'nv-hostengine' in line or 'dmon' in line or 'Host Engine Listener Started' in line:
                    continue

                # process the head line
                if line.strip().startswith("# Entity"):
                    line = line.strip()[2:]
                    if 'Entity' == line[0:len('Entity')]:
                        if not has_header:
                            has_header = True
                            for r in regex_list:
                                line = r[0].sub(r[1], line)

                            metric_list = line.split(',')
                            for item in metric_list:
                                if item == "Entity" or item == "TIMESTAMP":
                                    continue

                                for gpuId in range(self._gpuPerTrainer):
                                    metric_pid = self._allocate_pid()
                                    name = "%s_%d" % (item, trainerId*self._gpuPerTrainer + gpuId)
                                    self._chrome_trace.emit_pid(name, metric_pid)
                                    pid_map[name] = metric_pid

                # process the data line
                elif line.strip().startswith("GPU"):
                    line = line.strip()[3:]

                    for r in regex_list:
                        line = r[0].sub(r[1], line)

                    record_list = line.split(',')

                    gpuId = int(record_list[metric_list.index("Entity")])
                    # the unit of dcgmn is s, we need ns
                    timestamp = int(float(record_list[metric_list.index(
                        "TIMESTAMP")]) * 1e9)
                    # print(timestamp)
                    if not has_header or len(metric_list) == 0:
                        continue

                    for metric, record in zip(metric_list, record_list):
                        if metric == "Entity" or metric == "TIMESTAMP" or record == "N/A":
                            continue

                        di = {}
                        name = "%s_%d" % (metric, trainerId*self._gpuPerTrainer + gpuId)
                        di['name'] = name
                        di['pid'] = pid_map[name]
                        di['ts'] = self._align_ts(timestamp)
                        di['cat'] = name
                        di['tid'] = 0
                        di['ph'] = "C"
                        di['args'] = {"0": float(record)}
                        self._chrome_trace._events.append(di)

                else:
                    continue

    def generate_chrome_trace(self):
        self._add_profile()
        self._set_timeInfo()

        self._allocate_pids()
        self._allocate_events()
        self._allocate_memory_event()

        self._add_mlnx_perf()
        self._add_dcgm_perf()

        with open(self._saveFile, 'w') as f:
            f.write(self._chrome_trace.format_to_string())

if __name__ == '__main__':
    args = get_argparse()
    tl = Timeline(args)
    tl.generate_chrome_trace()
