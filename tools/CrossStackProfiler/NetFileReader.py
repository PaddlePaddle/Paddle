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

<<<<<<< HEAD
import json
import multiprocessing
from multiprocessing import Process

from CspFileReader import (
    FILEORGANIZEFORM_BYTRAINER,
    PIPELINEINFO_TRACE_NUM,
    FileReader,
    getLogger,
)


class netFileReader(FileReader):
=======
import os
import json
import glob
import logging
import pandas as pd
import multiprocessing

from multiprocessing import Process

from CspChromeTraceFormatter import ChromeTraceFormatter

from CspFileReader import FileReader
from CspFileReader import getLogger
from CspFileReader import TIME_PATH, DCGM_PATH, NET_PATH, PROFILE_PATH
from CspFileReader import NETINFO_TRACE_NUM, DCGMINFO_TRACE_NUM, PIPELINEINFO_TRACE_NUM
from CspFileReader import FILEORGANIZEFORM_BYRANK, FILEORGANIZEFORM_BYTRAINER, FILEORGANIZEFORM_BYOTHER, FILEORGANIZEFORM


class netFileReader(FileReader):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _parseSingleFile(self, fileNameList, tx_pid, rx_pid, q=None):

        traceInfo = {}
        traceEventList = []

        metaInfo = {}
        metaInfo['name'] = 'process_name'
        metaInfo['ph'] = 'M'
        metaInfo['pid'] = tx_pid
        metaInfo['args'] = {'name': "%02d_tx" % tx_pid}

        traceEventList.append(metaInfo)
        metaInfo = {}
        metaInfo['name'] = 'process_name'
        metaInfo['ph'] = 'M'
        metaInfo['pid'] = rx_pid
        metaInfo['args'] = {'name': "%02d_rx" % rx_pid}

        traceEventList.append(metaInfo)

        trainerIdList = []
        for fileName in fileNameList:
            trainerId = self.getTrainerId(fileName)
            trainerIdList.append(trainerId)
            with open(fileName, "r") as rf:
                for line in rf:
                    try:
                        event_str = json.loads(line.strip())
<<<<<<< HEAD
                        event_str["pid"] = (
                            tx_pid if event_str["name"] == "tx" else rx_pid
                        )
=======
                        event_str["pid"] = tx_pid if event_str[
                            "name"] == "tx" else rx_pid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        # the unit of net is ms, we need ns
                        event_str["ts"] = self._align_ts(event_str["ts"] * 1e6)
                        event_str["id"] = trainerId
                        traceEventList.append(event_str)

                    except Exception:
                        self._logger.warning(
<<<<<<< HEAD
                            "invalid record [%s] in [%s]. skip it!"
                            % (line[:-1], fileName)
                        )
        traceInfo["traceEvents"] = traceEventList

        if q is not None:
=======
                            "invalid record [%s] in [%s]. skip it!" %
                            (line[:-1], fileName))
        traceInfo["traceEvents"] = traceEventList

        if not q is None:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            q.put(traceInfo)
        else:
            return traceInfo

    def parseFileByGroup(self, groupId, processNum=8):
        fileFist = self.getFileListByGroup(groupId)
<<<<<<< HEAD
        fileFist = fileFist[: min(self._displaySize, len(fileFist))]
=======
        fileFist = fileFist[:min(self._displaySize, len(fileFist))]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        manager = multiprocessing.Manager()
        q = manager.Queue()

        processPool = []
        pidList = []
        tx_pid = PIPELINEINFO_TRACE_NUM
        rx_pid = PIPELINEINFO_TRACE_NUM + 1

        taskList = self._splitTaskListForMultiProcess(fileFist, processNum)
        for task in taskList:
<<<<<<< HEAD
            subproc = Process(
                target=self._parseSingleFile,
                args=(
                    task,
                    tx_pid,
                    rx_pid,
                    q,
                ),
            )
=======
            subproc = Process(target=self._parseSingleFile,
                              args=(
                                  task,
                                  tx_pid,
                                  rx_pid,
                                  q,
                              ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            processPool.append(subproc)
            subproc.start()
            pidList.append(subproc.pid)
            self._logger.info(
                "[Net info]: process [%d] has been started, total task num is %d ..."
<<<<<<< HEAD
                % (subproc.pid, len(processPool))
            )
=======
                % (subproc.pid, len(processPool)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for t in processPool:
            t.join()
            pidList.remove(t.pid)
            self._logger.info(
<<<<<<< HEAD
                "[Net info]: process [%d] has exited! remained %d process!"
                % (t.pid, len(pidList))
            )
=======
                "[Net info]: process [%d] has exited! remained %d process!" %
                (t.pid, len(pidList)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        traceInfo = {}
        isFistProcess = True
        for t in processPool:
            if isFistProcess:
                isFistProcess = False
                traceInfo["traceEvents"] = q.get()["traceEvents"]
            else:
                traceInfo["traceEvents"].extend(q.get()["traceEvents"])

        return traceInfo


def test_netFileReader():
    args = {
        "dataPath": "data/newdata/net",
        "groupSize": 4,
        "displaySize": 2,
        "gpuPerTrainer": 8,
        "minTimeStamp": 0,
        "organizeForm": FILEORGANIZEFORM_BYTRAINER,
    }

    testReader = netFileReader(getLogger(), args)
    testReader.printArgs()
    data = testReader.parseFileByGroup(0, 8)

    jsObj = json.dumps(data, indent=4, separators=(',', ': '))
    fileObject = open('jsonFile.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()


if __name__ == "__main__":
    test_netFileReader()
