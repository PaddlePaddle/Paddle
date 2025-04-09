# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

import json
import multiprocessing
from multiprocessing import Process

from CspChromeTraceFormatter import ChromeTraceFormatter
from CspFileReader import (
    DCGMINFO_TRACE_NUM,
    FILEORGANIZEFORM_BYRANK,
    NETINFO_TRACE_NUM,
    PIPELINEINFO_TRACE_NUM,
    FileReader,
    getLogger,
)

from paddle.base.proto.profiler import profiler_pb2


class profileFileReader(FileReader):
    def _parseSingleFile(self, profile):
        with open(profile, 'rb') as f:
            profile_s = f.read()
            profile_pb = profiler_pb2.Profile()
            profile_pb.ParseFromString(profile_s)

            return profile_pb

    def _parseTask(self, taskList, q=None):
        profile_dict = {}

        for fileName in taskList:
            rankId = self.getRankId(fileName)
            profile_dict[f"trainerRank.{rankId:03}"] = self._parseSingleFile(
                fileName
            )
            self._logger.info(f"I finish processing {fileName}!")

        if q is not None:
            q.put(profile_dict)

        return profile_dict

    def _is_forwardBackwardInfo(self, items):
        if items["name"] == "marker/compute/MarkerCUDA":
            if "args" in items:
                if isinstance(items["args"], dict):
                    args = items["args"]
                    if "detail_info" in args:
                        if (
                            args["detail_info"] == "marker_forward_B"
                            or args["detail_info"] == "marker_forward_E"
                            or args["detail_info"] == "marker_backward_B"
                            or args["detail_info"] == "marker_backward_E"
                        ):
                            return True
        return False

    def _allocate_forwardBackwardInfo(self, restList, pid, tid):
        def _cmp_ele(items):
            return items["ts"]

        restList.sort(key=_cmp_ele)
        newList = []

        lastEle = {}
        for items in restList:
            if items["args"]["detail_info"].endswith("E"):
                if not lastEle:
                    continue
                else:
                    lastEle["dur"] = items["ts"] - lastEle["ts"]
                    name = lastEle["args"]["detail_info"]
                    name = name[: name.rfind('_')]
                    name = name.split('_')[1]
                    lastEle["name"] = name
                    lastEle["args"]["detail_info"] = name
                    lastEle["args"]["name"] = name
                    if name == "backward":
                        lastEle["cname"] = "good"
                    else:
                        lastEle["cname"] = "bad"

                    lastEle["tid"] = tid
                    lastEle["pid"] = pid

                    newList.append(lastEle)
            else:
                lastEle = items

        return newList

    def _getPipeLineInfo(self, profileList, q=None):
        res = {}
        for profile in profileList:
            rankId = self.getRankId(profile)

            profile_pb = self._parseSingleFile(profile)
            traceEventList = []
            pid = 0
            tid = rankId

            for event in profile_pb.events:
                args = {'name': event.name}
                if event.memcopy.bytes > 0:
                    args['mem_bytes'] = event.memcopy.bytes
                if hasattr(event, "detail_info") and event.detail_info:
                    args['detail_info'] = event.detail_info

                traceEvent = {}
                traceEvent['ph'] = 'X'
                traceEvent['cat'] = 'Op'
                traceEvent['name'] = event.name
                traceEvent['pid'] = pid
                traceEvent['tid'] = tid
                traceEvent['ts'] = self._align_ts(event.start_ns)
                traceEvent['dur'] = (event.end_ns - event.start_ns) / 1.0
                traceEvent['args'] = args

                if self._is_forwardBackwardInfo(traceEvent):
                    traceEventList.append(traceEvent)

            pipeLineList = self._allocate_forwardBackwardInfo(
                traceEventList, pid, tid
            )

            res[str(rankId)] = pipeLineList

        if q is not None:
            q.put(res)

        return res

    def getPipeLineInfo(self, groupId, processNum=8):
        fileFist = self.getFileListByGroup(groupId)

        self._logger.info(
            f"using [{processNum}] process to do this work, total task num is {len(fileFist)}!"
        )
        processPool = []
        pidList = []

        manager = multiprocessing.Manager()
        q = manager.Queue()

        taskList = self._splitTaskListForMultiProcess(fileFist, processNum)
        for task in taskList:
            subproc = Process(
                target=self._getPipeLineInfo,
                args=(
                    task,
                    q,
                ),
            )
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                f"[pipeline info]: process [{subproc.pid}] has been started, total task num is {len(task)} ..."
            )

        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
                f"[pipeline info]: process [{t.pid}] has exited! remained {len(pidList)} process!"
            )

        pipeLineInfo = {}

        metaInfo = {}
        metaInfo['name'] = 'process_name'
        metaInfo['ph'] = 'M'
        metaInfo['pid'] = 0
        metaInfo['args'] = {'name': f"{PIPELINEINFO_TRACE_NUM:02}_pipeLineInfo"}

        for t in processPool:
            for k, v in q.get().items():
                rankId = int(k)
                gpuId = rankId % self._gpuPerTrainer
                if str(gpuId) not in pipeLineInfo.keys():
                    pipeLineInfo[str(gpuId)] = [metaInfo]
                pipeLineInfo[str(gpuId)].extend(v)

        return pipeLineInfo

    def _allocate_pids(self, profile_dict, gpuId, initPid):
        chrome_trace = ChromeTraceFormatter()
        devices = {}
        mem_devices = {}

        initLineNum = initPid + 1
        lineDelta = len(profile_dict.keys())
        i = 0
        for k, profile_pb in profile_dict.items():
            lineNum = initLineNum
            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    if (k, event.device_id, "CPU") not in devices:
                        pid = initPid
                        initPid = initPid + 1
                        devices[(k, event.device_id, "CPU")] = pid
                        # -1 device id represents CUDA API(RunTime) call.(e.g. cudaLaunch, cudaMemcpy)
                        if event.device_id == -1:
                            chrome_trace.emit_pid(
                                f"{lineNum:02}_{k}:cuda_api", pid
                            )
                            lineNum = lineNum + 1
                        else:
                            chrome_trace.emit_pid(
                                f"{lineNum:02}_{k}:cpu:block:{event.device_id}",
                                pid,
                            )
                            lineNum = lineNum + 1
                elif event.type == profiler_pb2.Event.GPUKernel:
                    if (k, event.device_id, "GPUKernel") not in devices:
                        if gpuId == event.device_id:
                            pid = initPid
                            initPid = initPid + 1

                            devices[(k, event.device_id, "GPUKernel")] = pid
                            chrome_trace.emit_pid(
                                f"{lineNum:02}_{k}:gpu:{event.device_id}",
                                pid,
                            )
                            lineNum = lineNum + 1

            if not hasattr(profile_pb, "mem_events"):
                continue
            for mevent in profile_pb.mem_events:
                if mevent.place == profiler_pb2.MemEvent.CUDAPlace:
                    if (k, mevent.device_id, "GPU") not in mem_devices:
                        if gpuId == mevent.device_id:
                            pid = initPid
                            initPid = initPid + 1

                            mem_devices[(k, mevent.device_id, "GPU")] = pid
                            chrome_trace.emit_pid(
                                f"{lineNum:02}_memory usage on {k}:gpu:{mevent.device_id}",
                                pid,
                            )
                            lineNum = lineNum + 1
                elif mevent.place == profiler_pb2.MemEvent.CPUPlace:
                    if (k, mevent.device_id, "CPU") not in mem_devices:
                        pid = initPid
                        initPid = initPid + 1

                        mem_devices[(k, mevent.device_id, "CPU")] = pid
                        chrome_trace.emit_pid(
                            f"{lineNum:02}_memory usage on {k}:cpu:{mevent.device_id}",
                            pid,
                        )
                        lineNum = lineNum + 1
                elif mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace:
                    if (
                        k,
                        mevent.device_id,
                        "CUDAPinnedPlace",
                    ) not in mem_devices:
                        if gpuId == mevent.device_id:
                            pid = initPid
                            initPid = initPid + 1

                            mem_devices[
                                (k, mevent.device_id, "CUDAPinnedPlace")
                            ] = pid
                            chrome_trace.emit_pid(
                                f"{lineNum:02}_memory usage on {k}:cudapinnedplace:{mevent.device_id}",
                                pid,
                            )
                            lineNum = lineNum + 1
                if (k, 0, "CPU") not in mem_devices:
                    pid = initPid
                    initPid = initPid + 1

                    mem_devices[(k, 0, "CPU")] = pid
                    chrome_trace.emit_pid(
                        f"{lineNum:02}_memory usage on {k}:cpu:{0}", pid
                    )
                    lineNum = lineNum + 1
                if (k, 0, "GPU") not in mem_devices:
                    # if gpuId == mevent.device_id:
                    pid = initPid
                    initPid = initPid + 1

                    mem_devices[(k, 0, "GPU")] = pid
                    chrome_trace.emit_pid(
                        f"{lineNum:02}_memory usage on {k}:gpu:{0}", pid
                    )
                    lineNum = lineNum + 1
                if (k, 0, "CUDAPinnedPlace") not in mem_devices:
                    pid = initPid
                    initPid = initPid + 1

                    mem_devices[(k, 0, "CUDAPinnedPlace")] = pid
                    chrome_trace.emit_pid(
                        f"{lineNum:02}_memory usage on {k}:cudapinnedplace:{0}",
                        pid,
                    )
                    lineNum = lineNum + 1
            i = i + 1
        return chrome_trace, devices, mem_devices

    def _allocate_events(self, profile_dict, devices, gpuId):
        chrome_trace = ChromeTraceFormatter()
        for k, profile_pb in profile_dict.items():
            rankId = int(k.split(".")[-1])

            for event in profile_pb.events:
                if event.type == profiler_pb2.Event.CPU:
                    type = "CPU"
                elif event.type == profiler_pb2.Event.GPUKernel:
                    type = "GPUKernel"

                if (
                    event.type == profiler_pb2.Event.GPUKernel
                    and event.device_id != gpuId
                    and rankId % self._gpuPerTrainer != gpuId
                ):
                    continue

                pid = devices[(k, event.device_id, type)]
                args = {'name': event.name}
                if event.memcopy.bytes > 0:
                    args['mem_bytes'] = event.memcopy.bytes
                if hasattr(event, "detail_info") and event.detail_info:
                    args['detail_info'] = event.detail_info
                # TODO(panyx0718): Chrome tracing only handles ms. However, some
                # ops takes micro-seconds. Hence, we keep the ns here.
                chrome_trace.emit_region(
                    self._align_ts(event.start_ns),
                    (event.end_ns - event.start_ns) / 1.0,
                    pid,
                    event.sub_device_id,
                    'Op',
                    event.name,
                    args,
                )
        return chrome_trace

    def _allocate_memory_event(self, profile_dict, mem_devices, gpuId):
        chrome_trace = ChromeTraceFormatter()
        if not hasattr(profiler_pb2, "MemEvent"):
            return
        place_to_str = {
            profiler_pb2.MemEvent.CPUPlace: "CPU",
            profiler_pb2.MemEvent.CUDAPlace: "GPU",
            profiler_pb2.MemEvent.CUDAPinnedPlace: "CUDAPinnedPlace",
        }
        for k, profile_pb in profile_dict.items():
            rankId = int(k.split(".")[-1])

            trainerId = rankId / self._gpuPerTrainer

            if trainerId >= self._displaySize:
                continue

            mem_list = []
            end_profiler = 0
            for mevent in profile_pb.mem_events:
                crt_info = {}
                crt_info['time'] = mevent.start_ns
                crt_info['size'] = mevent.bytes
                if mevent.place in place_to_str:
                    place = place_to_str[mevent.place]
                else:
                    place = "UnDefine"

                if (
                    mevent.place == profiler_pb2.MemEvent.CUDAPlace
                    or mevent.place == profiler_pb2.MemEvent.CUDAPinnedPlace
                ) and mevent.device_id != gpuId:
                    continue

                crt_info['place'] = place
                pid = mem_devices[(k, mevent.device_id, place)]
                crt_info['pid'] = pid
                crt_info['thread_id'] = mevent.thread_id
                crt_info['device_id'] = mevent.device_id
                mem_list.append(crt_info)
                crt_info = {}
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
                while (
                    i < len(mem_list) - 1
                    and mem_list[i]['time'] == mem_list[i + 1]['time']
                ):
                    total_size += mem_list[i + 1]['size']
                    i += 1

                chrome_trace.emit_counter(
                    "Memory",
                    "Memory",
                    mem_list[i]['pid'],
                    self._align_ts(mem_list[i]['time']),
                    0,
                    total_size,
                )
                i += 1
        return chrome_trace

    def _getOPTraceInfoByGpuId(self, groupId, gpuId):
        fileFist = self.getFileListByGroup(groupId)
        newFileList = []
        for file in fileFist:
            rankId = self.getRankId(file)
            localRank = rankId % self._gpuPerTrainer
            if (
                localRank == gpuId
                and (rankId / self._gpuPerTrainer) % self._groupSize
                < self._displaySize
            ):
                newFileList.append(file)

        profile_dict = self._parseTask(newFileList)
        initPid = (
            PIPELINEINFO_TRACE_NUM + DCGMINFO_TRACE_NUM + NETINFO_TRACE_NUM
        )
        metaTrace, devicesPid, mem_devicesPid = self._allocate_pids(
            profile_dict, gpuId, initPid
        )
        eventsTrace = self._allocate_events(profile_dict, devicesPid, gpuId)
        memEventsTrace = self._allocate_memory_event(
            profile_dict, mem_devicesPid, gpuId
        )

        trace = {}
        trace['traceEvents'] = (
            metaTrace._metadata + eventsTrace._events + memEventsTrace._events
        )
        self.dumpOpInfoDict(trace, groupId, gpuId, True)

        return trace

    def getOPTraceInfo(self, groupId):
        manager = multiprocessing.Manager()
        q = manager.Queue()
        processPool = []
        pidList = []

        for gpuId in range(self._gpuPerTrainer):
            subproc = Process(
                target=self._getOPTraceInfoByGpuId,
                args=(
                    groupId,
                    gpuId,
                ),
            )
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                f"[op info]: process [{subproc.pid}] has been started, total task num is {1} ..."
            )

        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
                f"[op info]: process [{t.pid}] has exited! remained {len(pidList)} process!"
            )

        opInfo = {}

        return opInfo

    def parseFileByGroup(self, groupId, processNum=8):
        fileFist = self.getFileListByGroup(groupId)
        return self._parseTask(fileFist)


def test_profileFileReader():
    args = {
        "dataPath": "data/newdata/profile",
        "groupSize": 4,
        "displaySize": 8,
        "gpuPerTrainer": 8,
        "minTimeStamp": 0,
        "organizeForm": FILEORGANIZEFORM_BYRANK,
    }

    testReader = profileFileReader(getLogger(), args)
    testReader.printArgs()
    data = testReader.getOPTraceInfo(0)

    jsObj = json.dumps(data)
    fileObject = open('jsonFile.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()


if __name__ == "__main__":
    test_profileFileReader()
