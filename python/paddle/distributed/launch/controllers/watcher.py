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

import os
import time
from threading import Thread
import shutil
import subprocess
# from ..utils.nvsmi import get_gpu_info, get_gpu_process, get_gpu_util


class Watcher:
    def __init__(self, ctx):
        self.ctx = ctx

        self.interval = 5

        self.gpu_util = []

        if not self.ctx.args.enable_gpu_log:
            return

        # gpu log file
        self.gpus = self.ctx.args.devices or self.ctx.node.device.labels
        if len(self.gpus) > 0:
            fn = os.path.join(
                self.ctx.args.log_dir, f"{self.ctx.args.job_id}.gpu.log"
            )
            # due to mx-smi do not support stdout with csv format and 
            # default file write option will overwrite the origin file,
            # we use a temporary file to stage the result. 
            self.fn_bck = os.path.join(self.ctx.args.log_dir,
                              "{}.gpu.log.bck".format(self.ctx.args.job_id))
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            self.gpu_fd = open(fn, 'w')
        else:
            return

        # start
        self.proc = Thread(target=self.watch)
        self.proc.daemon = True
        self.proc.start()

    def watch(self):
        if not len(self.gpus) > 0:
            return

        self._print_gpu_info()

        # util_key = "index,utilization_gpu,memory_total,memory_used,memory_free,timestamp"
        util_key = "timestamp,deviceId,deviceName,bdfId,utilization.GPU [%],utilization.VPUE [%],utilization.VPUD [%]"
        self.gpu_fd.write(util_key)
        self.gpu_fd.write('\n')

        while not self.ctx.status.is_done():
            self._save_gpu_log()
            time.sleep(self.interval)

        if hasattr(self, "gpu_fd"):
            self.gpu_fd.close()

    def _print_gpu_info(self):
        try:
            # info_key = "index,uuid,driver_version,name,gpu_serial,display_active,display_mode"
            # self.gpu_fd.write(info_key)
            # self.gpu_fd.write('\n')
            # for line in get_gpu_info(self.gpus):
            #     self.gpu_fd.write(line.str(info_key))
            #     self.gpu_fd.write('\n')
            # self.gpu_fd.write('\n')

            # process_key = "pid,process_name,gpu_uuid,gpu_name,used_memory"
            # self.gpu_fd.write(process_key)
            # self.gpu_fd.write('\n')
            # for line in get_gpu_process(self.gpus):
            #     self.gpu_fd.write(line.str(process_key))
            #     self.gpu_fd.write('\n')
            # self.gpu_fd.write('\n')

            # self.gpu_fd.flush()
            get_gpu_info_cmd = ['mx-smi', '--show-version', '-o', self.fn_bck]
            subprocess.run(get_gpu_info_cmd, timeout=3, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(self.fn_bck, 'r') as gpu_fd_bck:
                stage_content = gpu_fd_bck.read()
                self.gpu_fd.write(stage_content)
                self.gpu_fd.write('\n\n')
            os.remove(self.fn_bck)
            
            get_gpu_memory_cmd = ['mx-smi', '--show-memory', '-o', self.fn_bck]
            subprocess.run(get_gpu_memory_cmd, timeout=3, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(self.fn_bck, 'r') as gpu_fd_bck:
                stage_content = gpu_fd_bck.read()
                self.gpu_fd.write(stage_content)
                self.gpu_fd.write('\n\n')
            os.remove(self.fn_bck)
        except:
            self.ctx.logger.warning("save gpu info failed")

    def _save_gpu_log(self, util_key):
        try:
            # for line in get_gpu_util(self.gpus):
            #     self.gpu_fd.write(line.str(util_key))
            #     self.gpu_fd.write('\n')
            # self.gpu_fd.flush()
            get_gpu_usage_cmd = ['mx-smi', '--show-usage', '-o', self.fn_bck]
            subprocess.run(get_gpu_usage_cmd, timeout=3, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(self.fn_bck, 'r') as gpu_fd_bck:
                stage_content = gpu_fd_bck.readlines()[1]
                self.gpu_fd.write(stage_content)
            os.remove(self.fn_bck)
        except:
            self.ctx.logger.warning("save gpu log failed")

    def stop(self):
        if hasattr(self, "proc"):
            # daemon without join
            # self.proc.join()
            pass
