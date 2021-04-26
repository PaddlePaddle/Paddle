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
import pandas
import tempfile
import platform
import pandas as pd

import google.protobuf.text_format as text_format
import paddle.fluid.proto.profiler.profiler_pb2 as profiler_pb2

TIME_PATH="time"
DCGM_PATH="dcgm"
NET_PATH="net"
PROFILE_PATH="profile"

dcgmMetricParameterMap = {
                        "gpuUtility"  : [("GPUTL", "GPUTL"), ("GRACT", "GRACT")],
                        "smUtility"   : [("SMACT", "SMACT"), ("SMOCC", "SMOCC")],
                        "memUtility"  : [("FB_USED_RATIO", "FB_USED_RATIO"), ("DRAMA", "DRAMA")],
                        "txUtility"   : [("NVLTX", "NVLTX"), ("NVLRX", "NVLRX"), ("PCITX", "PCITX"), ("PCIRX", "PCIRX")],
                        "calUtility"  : [("FP32A", "FP32A"), ("FP16A", "FP16A"), ("TENSO", "TENSO")]
                    }

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

    parser.add_argument(
        '--groupSize',
        type=int,
        default=8,
        help='Num of trainer in a group.')

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

    def clear(self):
        self._events = []
        self._metadata = []

class Timeline(object):
    def __init__(self, args):
        self._args = args
        self._workPath= self._args.profile_path
        self._saveFilePath = self._args.timeline_path
        self._gpuPerTrainer = self._args.gpuPerTrainer
        self._groupSize = self._args.groupSize

        self._profile_dict = {}
        self._pid = 0
        self._devices = dict()
        self._mem_devices = dict()
        self._chrome_trace = _ChromeTraceFormatter()

        self._dcgmPath = os.path.join(self._workPath, DCGM_PATH)
        self._timePath = os.path.join(self._workPath, TIME_PATH)
        self._netPath = os.path.join(self._workPath, NET_PATH)
        self._profilePath = os.path.join(self._workPath, PROFILE_PATH)

        self._netFileReader = netFileReader(self._netPath)
        self._dcgmFileReader = dcgmFileReader(self._dcgmPath)
        self._profileFileReader = profilFileReader(self._profilePath)

        self._trainerNum = self._dcgmFileReader.getTrainerNum()

        self._timeInfo = {}
        self._minTimeStamp = 0

    def _set_timeInfo(self, timeFileNamePrefix="time.txt", sed="."):
        timeFileNameList = glob.glob(os.path.join(self._timePath, timeFileNamePrefix, sed,"*"))
        for timeFileName in timeFileNameList:
            gpuId = int(timeFileName.split(sed)[-1])
            info={}
            with open(timeFileName, "r") as rf:
                for line in rf:
                    if line.startswith("start time:"):
                        info["start_time"] = int(float(line.split(":")[-1]) * 1e9)

                        self._minTimeStamp = min(self._minTimeStamp, info["start_time"])

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

    def _allocate_pids(self, gpuId):
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
                        if gpuId == event.device_id:
                            pid = self._allocate_pid()
                            self._devices[(k, event.device_id, "GPUKernel")] = pid
                            self._chrome_trace.emit_pid("%s:gpu:%d" %
                                                        (k, event.device_id), pid)
            if not hasattr(profile_pb, "mem_events"):
                continue
            for mevent in profile_pb.mem_events:
                if mevent.place == profiler_pb2.MemEvent.CUDAPlace:
                    if (k, mevent.device_id, "GPU") not in self._mem_devices:
                        if gpuId == mevent.device_id:
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
                        if gpuId == mevent.device_id:
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
                    # if gpuId == mevent.device_id:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "GPU")] = pid
                    self._chrome_trace.emit_pid("memory usage on %s:gpu:%d" %
                                                (k, 0), pid)
                if (k, 0, "CUDAPinnedPlace") not in self._mem_devices:
                    pid = self._allocate_pid()
                    self._mem_devices[(k, 0, "CUDAPinnedPlace")] = pid
                    self._chrome_trace.emit_pid(
                        "memory usage on %s:cudapinnedplace:%d" % (k, 0), pid)

    def _allocate_events(self, gpuId):
        for k, profile_pb in six.iteritems(self._profile_dict):
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    type = "CPU"
                elif event.type == profiler_pb2.Event.GPUKernel:
                    type = "GPUKernel"

                if event.type == profiler_pb2.Event.GPUKernel and event.device_id != gpuId:
                    continue

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

    def _allocate_memory_event(self, gpuId):
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

                if (mevent.place == profiler_pb2.MemEvent.CUDAPlace or mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace) and mevent.device_id != gpuId:
                    continue

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

    def _add_mlnx_perf(self, netFileNameList, sed='.'):
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

    def dumpChromeTraceByGroup(self, groupId):
        netFileNameList = self._netFileReader.getFileListByGroup(groupId)

        if not netFileNameList:
            return

        dcgm_data = self._dcgmFileReader.parseFileByGroup(groupId)
        self._profile_dict = self._profileFileReader.parseFileByGroup(groupId)

        for gpuId in range(self._gpuPerTrainer):
            self._chrome_trace.clear()
            self._pid = 0

            # add profile
            self._devices = dict()
            self._mem_devices = dict()
            self._allocate_pids(gpuId)
            self._allocate_events(gpuId)
            self._allocate_memory_event(gpuId)

            # add net
            self._add_mlnx_perf(netFileNameList)

            # add dcgm
            pid_map = {}

            for i in range(self._groupSize):
                for metric in dcgmMetricParameterMap.keys():

                    metric_pid = self._allocate_pid()
                    name = "%s_%d" % (metric, i)
                    self._chrome_trace.emit_pid(name, metric_pid)

                    if metric not in pid_map.keys():
                        pid_map[metric] = []

                    pid_map[metric].append(metric_pid)

            gpuDcgmData = dcgm_data[dcgm_data['Entity'].isin([gpuId])]
            # gpuDcgmData.info()
            # gpuDcgmData.to_csv("log.log", na_rep='NULL')

            for index, row in gpuDcgmData.iterrows():
                for metric, parameteList in dcgmMetricParameterMap.items():
                    trainerId = int(row['trainerId'])
                    di = {}
                    name = "%s_%d" % (metric, trainerId)
                    di['name'] = name
                    di['pid'] = pid_map[metric][trainerId % self._groupSize]
                    di['ts'] = self._align_ts(int(row['ts']))
                    di['cat'] = metric
                    di['tid'] = 0
                    di['ph'] = "C"

                    args = {}
                    for p in parameteList:
                        args[p[0]] = row[p[1]]
                    di['args'] = args

                    self._chrome_trace._events.append(di)

            if not os.path.exists(self._saveFilePath):
                os.mkdir(self._saveFilePath)

            resultFileName = os.path.join(self._saveFilePath, "result_groupID_%d_gpuID_%d.json" % (groupId, gpuId))
            with open(resultFileName, 'w') as f:
                f.write(self._chrome_trace.format_to_string())
                print("dump %s sucessfully!" % resultFileName)

    def generate_chrome_trace(self):
        self._set_timeInfo()
        groupNum = self._trainerNum / self._groupSize
        for i in range(groupNum):
            self.dumpChromeTraceByGroup(i)

class FileReader(object):
    def __init__(self, dataPath, groupSize=8):
        self._dataPath = dataPath
        self._fileList = []
        self._fileNum = 0
        self._groupSize = groupSize
        self.getFileList()

    def getFileListByGroup(self, groupId):
        if (groupId+1)*self._groupSize > len(self._fileList):
            return []

        return self._fileList[groupId*self._groupSize:(groupId+1)*self._groupSize]

    def getFileList(self):
        self._fileList = glob.glob(os.path.join(self._dataPath, "*.*"))
        self._fileList.sort()
        return self._fileList

    def getTrainerNum(self):
        return len(self._fileList)

    def _getTrainerId(self, fileName, sed="."):
        return int(fileName.split(sed)[-1])

class profilFileReader(FileReader):
    def _parseSingleFile(self, profile):
        with open(profile, 'rb') as f:
            profile_s = f.read()
            profile_pb = profiler_pb2.Profile()
            profile_pb.ParseFromString(profile_s)

            return profile_pb

    def parseFileByGroup(self, groupId):
        fileFist = self.getFileListByGroup(groupId)
        profile_dict = {}

        for file in fileFist:
            profile_dict[os.path.basename(file)] = self._parseSingleFile(file)

        return profile_dict

class netFileReader(FileReader):
    def _parseSingleFile(self, fileName):
        trainerId = self._getTrainerId(fileName)

        if not os.path.exists(fileName):
            WARN(fileName + ' not found')
            return

        with open(fileName, 'r') as fp:
            for line in fp:
                tmp_data = pd.read_json(line)
                if is_first:
                    is_first=False
                    net_data=tmp_data
                else:
                    net_data=pd.concat([net_data,tmp_data],axis=0,join='outer')

        net_data['ts'] = net_data['ts']* 1e6
        net_data['trainerId'] = trainerId

        return net_data

    def parseFileByGroup(self, groupId):
        fileFist = self.getFileListByGroup(groupId)

        for file in fileFist:
            is_first=True
            for fileName in file_list:
                tmp_data=self._parseSingleFile(fileName)
                if tmp_data is None:
                    continue

                if is_first:
                    is_first=False
                    net_data=tmp_data
                else:
                    net_data=pd.concat([net_data,tmp_data],axis=0,join='outer')

        return net_data

class dcgmFileReader(FileReader):
    def parseAllFile(self):
        file_list = glob.glob(os.path.join(self._dataPath, "*.*"))
        print ("total file num is %d !" % len(file_list))
        if not file_list:
            return None

        is_first=True
        for fileName in file_list:
            tmp_data=self._parseSingleFile(fileName)
            if tmp_data is None:
                continue

            if is_first:
                is_first=False
                dcgm_data=tmp_data
            else:
                dcgm_data=pd.concat([dcgm_data,tmp_data],axis=0,join='outer')

        return dcgm_data

    def parseFileByGroup(self, groupId):
        fileFist = self.getFileListByGroup(groupId)

        for file in fileFist:
            is_first=True
            for fileName in fileFist:
                tmp_data=self._parseSingleFile(fileName)
                if tmp_data is None:
                    continue

                if is_first:
                    is_first=False
                    dcgm_data=tmp_data
                else:
                    dcgm_data=pd.concat([dcgm_data,tmp_data],axis=0,join='outer')
        dcgm_data = dcgm_data.dropna()
        return dcgm_data

    def _parseSingleFile(self, fileName):
        trainerId = self._getTrainerId(fileName)

        if not os.path.exists(fileName):
            WARN(fileName + ' not found')
            return

        regex_list = [
            (re.compile(r' +'), ','),
            (re.compile(r'^,'), ''),
        ]

        csv_tempfile = tempfile.TemporaryFile()
        with open(fileName, 'r') as fp:
            has_header = False

            for line in fp:
                # skip `nvidia-dcgm-dmon.sh` init and fini info lines
                if 'nv-hostengine' in line or 'dmon' in line or 'Host Engine Listener Started' in line:
                    continue

                # skip non-needed headers (only the header in 1th line was needed)
                if line.strip().startswith("# Entity"):
                    line = line.strip()[2:]

                if 'Entity' == line[0:len('Entity')]:
                    if has_header:
                        continue
                    else:
                        has_header = True

                if line.strip().startswith("GPU"):
                    line = line.strip()[3:]

                for r in regex_list:
                    line = r[0].sub(r[1], line)
                # print(line)

                # csv_tempfile.write(bytes(line + "\n", encoding='UTF-8'))
                csv_tempfile.write(bytes(line + "\n"))

        csv_tempfile.seek(0)
        dcgm = pd.read_csv(csv_tempfile, header=0, delimiter=',')
        dcgm['FB_USED_RATIO'] = dcgm['FBUSD'] / dcgm['FBTTL']
        dcgm['GPUTL'] = dcgm['GPUTL'] / 100.0
        dcgm['ts'] = dcgm['TIMESTAMP'] * 1e9
        dcgm['trainerId'] = trainerId

        return dcgm

def test_dcgmFileReader():
    args = get_argparse()

    reader = dcgmFileReader(os.path.join(args.profile_path, DCGM_PATH))
    data = reader.parseAllFile()

    print(data)
    data.info()
    data.to_csv("log.log", na_rep='NULL')

if __name__ == '__main__':
    args = get_argparse()
    tl = Timeline(args)
    tl.generate_chrome_trace()

    # test_dcgmFileReader()
