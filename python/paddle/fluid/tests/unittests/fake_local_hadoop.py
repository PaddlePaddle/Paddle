# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import subprocess
import os
import re
import tempfile


class HadoopClient:
    def __init__(self):
        self.FS_DEFAULT_NAME = tempfile.gettempdir() + '/fake_hadoop_repos'
        self.FS_JOB_UGI = "hadoop,unittest"
        if not os.path.isdir(self.FS_DEFAULT_NAME):
            cmd = "mkdir -p " + self.FS_DEFAULT_NAME
            ret_code = self._run_command(cmd)

    def _run_command(self, cmd, std_in=None, std_out=None, std_err=None):
        proc = subprocess.Popen(
            cmd, stdin=std_in, stdout=std_out, stderr=std_err, shell=True)
        (output, errors) = proc.communicate()
        ret_code = proc.returncode
        return ret_code

    def my_put(self, local_files, hadoop_dir):
        for local_file in local_files:
            if not os.path.exists(local_file):
                self.returncode = 1
                break
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_dir
            if os.path.exists(remote_dir):
                if not os.path.isdir(remote_dir):
                    sys.exit(-1)
            else:
                dir_name = os.path.dirname(remote_dir)
                if not os.path.isdir(dir_name):
                    os.system('mkdir -p ' + dir_name)
            cmd = 'cp -r ' + local_file + ' ' + remote_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def my_download(self, hadoop_files, local_dir):
        for hadoop_file in hadoop_files:
            remote_path = self.FS_DEFAULT_NAME + '/' + hadoop_file
            if not os.path.exists(remote_path):
                self.returncode = 0
            cmd = 'cp -r ' + remote_path + ' ' + local_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def my_mkdir(self, hadoop_dirs):
        for hadoop_dir in hadoop_dirs:
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_dir
            if not os.path.isdir(remote_dir):
                cmd = "mkdir -p " + remote_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def my_ls(self, hadoop_dirs, recursive=False):
        for hadoop_dir in hadoop_dirs:
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_dir
            if not os.path.exists(remote_dir):
                sys.exit(-1)

            dirs = os.listdir(remote_dir)
            total_nums = len(dirs)
            total_results = []

            cmd = "ls -l " + remote_dir
            ls_pipe = ps0_pipe = open(
                os.path.join(tempfile.gettempdir(), "hadoop_ls_pipe.log"),
                "wb+")
            ret = self._run_command(cmd, std_out=ls_pipe)
            if ret:
                sys.exit(ret)
            ret_out = []
            regex = re.compile('\s+')
            with open(
                    os.path.join(tempfile.gettempdir(), "hadoop_ls_pipe.log"),
                    'r') as f:
                ret_out = f.readlines()

            for line_id, line in enumerate(ret_out):
                re_line = regex.split(line.rstrip("\n"))
                if len(re_line) != 9:
                    continue
                re_line[1] = '1'  # copy files
                re_line[5] += re_line[6]
                re_line.remove(re_line[6])
                re_line[-1] = os.path.join(hadoop_dir, re_line[-1])
                total_results.append(' '.join(re_line))
                if recursive and re_line[0][0] == 'd':
                    sub_nums, sub_results = self.my_ls([re_line[-1]], True)
                    total_nums += sub_nums
                    total_results += sub_results
            self.returncode = ret
            return total_nums, total_results

    def my_cat(self, hadoop_files):
        for hadoop_file in hadoop_files:
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_file
            cmd = "cat " + remote_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def my_rm(self, hadoop_dirs):
        for hadoop_dir in hadoop_dirs:
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_dir
            cmd = "rm " + remote_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def my_rmr(self, hadoop_dirs):
        for hadoop_dir in hadoop_dirs:
            remote_dir = self.FS_DEFAULT_NAME + '/' + hadoop_dir
            cmd = "rm -r " + remote_dir
            ret = self._run_command(cmd)
            if ret:
                sys.argv(ret)

    def test_d(self, path_):
        remote_dir = self.FS_DEFAULT_NAME + '/' + path_
        if os.path.isdir(remote_dir):
            return 0
        else:
            return 1

    def test_e(self, path_):
        remote_dir = self.FS_DEFAULT_NAME + '/' + path_
        if os.path.exists(remote_dir):
            return 0
        else:
            return 1

    def my_test(self, args):
        mode = 0  # 0 -e 1 -d 2 -z
        for _ in args:
            if _ == '-d':
                mode = 1
                continue
            elif _ == '-e':
                mode = 0
                continue
            elif _ == '-z':
                mode = 2
                continue
            elif _[0] == '-':
                sys.exit(-1)
                break
            if mode == 1:
                ret = self.test_d(_)
                if ret == 1:
                    sys.exit(ret)
            elif mode == 0:
                ret = self.test_e(_)
                if ret == 1:
                    sys.exit(ret)


def main(argv):
    hadoop_client = HadoopClient()
    for i in range(len(argv)):
        opt = argv[i].split('=')
        if opt[0] == "-Dfs.default.name":
            assert (len(opt) == 2)
        elif opt[0] == "-Dhadoop.job.ugi":
            assert (len(opt) == 2)
        elif opt[0] == 'fs' or opt[0] == 'dfs':
            continue
        else:
            assert (len(opt) == 1)
            if opt[0] == '-put':
                assert (i <= len(argv) - 3)
                hadoop_client.my_put(argv[i + 1:-1], argv[-1])
                break
            elif opt[0] == '-get':
                assert (i <= len(argv) - 3)
                args = argv[i + 1:]
                hadoop_client.my_download(argv[i + 1:-1], argv[-1])
                break
            elif opt[0] == '-ls':
                args = argv[i + 1:]
                nums, results = hadoop_client.my_ls(args)
                print('total ' + str(nums))
                for line_id, line in enumerate(results):
                    print(line)
                break
            elif opt[0] == '-lsr':
                args = argv[i + 1:]
                nums, results = hadoop_client.my_ls(args, True)
                print('total ' + str(nums))
                for line_id, line in enumerate(results):
                    print(line)
                break
            elif opt[0] == '-cat':
                args = argv[i + 1:]
                hadoop_client.my_cat(args)
                break
            elif opt[0] == '-rm':
                args = argv[i + 1:]
                hadoop_client.my_rm(args)
                break
            elif opt[0] == '-rmr':
                args = argv[i + 1:]
                hadoop_client.my_rmr(args)
                break
            elif opt[0] == '-mkdir':
                args = argv[i + 1:]
                hadoop_client.my_mkdir(args)
                break
            elif opt[0] == '-test':
                args = argv[i + 1:]
                hadoop_client.my_test(args)
                break
            else:
                print("Don't support %s command" % opt[0])
                sys.exit(-1)


if __name__ == '__main__':
    opts = sys.argv[1:]
    main(opts)
