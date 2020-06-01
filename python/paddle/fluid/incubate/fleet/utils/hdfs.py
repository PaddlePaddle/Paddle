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
"""HDFS Utils."""

import os
import sys
import subprocess
import multiprocessing
from datetime import datetime

import re
import copy
import errno
import time
import logging

__all__ = ["HDFSClient"]


def get_logger(name, level, fmt):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler('hdfs.log', mode='w')
    formatter = logging.Formatter(fmt=fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class HDFSClient(object):
    """
    A tool of HDFS

    Args:
        hadoop_home (string): hadoop_home
        configs (dict): hadoop config, it is a dict, please contain \
            key "fs.default.name" and "hadoop.job.ugi"
        Can be a float value
    Examples:
        hadoop_home = "/home/client/hadoop-client/hadoop/"

        configs = {
            "fs.default.name": "hdfs://xxx.hadoop.com:54310",
            "hadoop.job.ugi": "hello,hello123"
        }

        client = HDFSClient(hadoop_home, configs)

        client.ls("/user/com/train-25")
        files = client.lsr("/user/com/train-25/models")
    """

    def __init__(self, hadoop_home, configs):
        self.pre_commands = []
        hadoop_bin = '%s/bin/hadoop' % hadoop_home
        self.pre_commands.append(hadoop_bin)
        dfs = 'fs'
        self.pre_commands.append(dfs)

        for k, v in configs.iteritems():
            config_command = '-D%s=%s' % (k, v)
            self.pre_commands.append(config_command)

    def __run_hdfs_cmd(self, commands, retry_times=5):
        whole_commands = copy.deepcopy(self.pre_commands)
        whole_commands.extend(commands)

        ret_code = 0
        ret_out = None
        ret_err = None
        retry_sleep_second = 3
        whole_commands = " ".join(whole_commands)
        for x in range(retry_times + 1):
            proc = subprocess.Popen(
                whole_commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True)
            (output, errors) = proc.communicate()
            ret_code, ret_out, ret_err = proc.returncode, output, errors

            _logger.info(
                'Times: %d, Running command: %s. Return code: %d, Msg: %s' %
                (x, whole_commands, proc.returncode, errors))

            if ret_code == 0:
                break
            time.sleep(retry_sleep_second)

        return ret_code, ret_out, ret_err

    def cat(self, hdfs_path=None):
        """
        cat hdfs file
        Args:
            hdfs_path(str): the hdfs file path
        Returns:
            file content
        """
        if self.is_file(hdfs_path):
            exist_cmd = ['-cat', hdfs_path]
            returncode, output, errors = self.__run_hdfs_cmd(
                exist_cmd, retry_times=1)
            if returncode != 0:
                _logger.error("HDFS cat HDFS path: {} failed".format(hdfs_path))
                return ""
            else:
                _logger.info("HDFS cat HDFS path: {} succeed".format(hdfs_path))
                return output.strip()

        else:
            return ""

    def is_exist(self, hdfs_path=None):
        """
        whether the remote HDFS path exists

        Args:
            hdfs_path(str): the hdfs file path

        Returns:
            True or False
        """
        exist_cmd = ['-test', '-e', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            exist_cmd, retry_times=1)

        if returncode:
            _logger.error("HDFS is_exist HDFS path: {} failed".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS is_exist HDFS path: {} successfully".format(
                hdfs_path))
            return True

    def is_dir(self, hdfs_path=None):
        """
        whether the remote HDFS path is directory

        Args:
            hdfs_path(str): the hdfs file path

        Returns:
            True or False
        """

        if not self.is_exist(hdfs_path):
            return False

        dir_cmd = ['-test', '-d', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(dir_cmd, retry_times=1)

        if returncode:
            _logger.error("HDFS path: {} failed is not a directory".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS path: {} successfully is a directory".format(
                hdfs_path))
            return True

    def is_file(self, hdfs_path=None):
        """
        whether the remote HDFS path is file

        Args:
            hdfs_path(str): the hdfs file path

        Returns:
            True or False
        """

        if not self.is_exist(hdfs_path):
            return False

        dir_cmd = ['-test', '-d', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(dir_cmd, retry_times=1)

        if returncode == 0:
            _logger.error("HDFS path: {} failed is not a file".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS path: {} successfully is a file".format(
                hdfs_path))
            return True

    def delete(self, hdfs_path):
        """
        Remove a file or directory from HDFS.

        whether the remote HDFS path exists

        Args:
            hdfs_path(str): HDFS path.

        Returns:
            True or False
            This function returns `True` if the deletion was successful and `False` if
            no file or directory previously existed at `hdfs_path`.
        """
        _logger.info('Deleting %r.', hdfs_path)

        if not self.is_exist(hdfs_path):
            _logger.warn("HDFS path: {} do not exist".format(hdfs_path))
            return True

        if self.is_dir(hdfs_path):
            del_cmd = ['-rmr', hdfs_path]
        else:
            del_cmd = ['-rm', hdfs_path]

        returncode, output, errors = self.__run_hdfs_cmd(del_cmd, retry_times=0)

        if returncode:
            _logger.error("HDFS path: {} delete files failure".format(
                hdfs_path))
            return False
        else:
            _logger.info("HDFS path: {} delete files successfully".format(
                hdfs_path))
            return True

    def rename(self, hdfs_src_path, hdfs_dst_path, overwrite=False):
        """
        Move a file or folder on HDFS.

        Args:
            hdfs_src_path(str): HDFS path
            hdfs_dst_path(str): HDFS path
            overwrite(bool|False): If the path already exists and overwrite is
                                   False, will return False.
        Returns:
            True or False
        """
        assert hdfs_src_path is not None
        assert hdfs_dst_path is not None

        if not self.is_exist(hdfs_src_path):
            _logger.info("HDFS path do not exist: {}".format(hdfs_src_path))
        if self.is_exist(hdfs_dst_path) and not overwrite:
            _logger.error("HDFS path is exist: {} and overwrite=False".format(
                hdfs_dst_path))

        rename_command = ['-mv', hdfs_src_path, hdfs_dst_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            rename_command, retry_times=1)

        if returncode:
            _logger.error("HDFS rename path: {} to {} failed".format(
                hdfs_src_path, hdfs_dst_path))
            return False
        else:
            _logger.info("HDFS rename path: {} to {} successfully".format(
                hdfs_src_path, hdfs_dst_path))
            return True

    @staticmethod
    def make_local_dirs(local_path):
        """
        create a directory local, is same to mkdir

        Args:
            local_path(str): local path that wants to create a directory.
        """
        try:
            os.makedirs(local_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def makedirs(self, hdfs_path):
        """
        Create a remote directory, recursively if necessary.

        Args:
            hdfs_path(str): Remote path. Intermediate directories will be
                            created appropriately.

        Returns:
            True or False
        """
        _logger.info('Creating directories to %r.', hdfs_path)
        assert hdfs_path is not None

        if self.is_exist(hdfs_path):
            _logger.error("HDFS path is exist: {}".format(hdfs_path))
            return

        mkdirs_commands = ['-mkdir', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            mkdirs_commands, retry_times=1)

        if returncode:
            _logger.error("HDFS mkdir path: {} failed".format(hdfs_path))
            return False
        else:
            _logger.info("HDFS mkdir path: {} successfully".format(hdfs_path))
            return True

    def ls(self, hdfs_path):
        """
        ls directory contents about HDFS hdfs_path

        Args:
            hdfs_path(str): Remote HDFS path will be ls.

        Returns:
            List: a contents list about hdfs_path.
        """
        assert hdfs_path is not None

        if not self.is_exist(hdfs_path):
            return []

        ls_commands = ['-ls', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            ls_commands, retry_times=10)

        if returncode:
            _logger.error("HDFS list path: {} failed".format(hdfs_path))
            return []
        else:
            _logger.info("HDFS list path: {} successfully".format(hdfs_path))

            ret_lines = []
            regex = re.compile('\s+')
            out_lines = output.strip().split("\n")
            for line in out_lines:
                re_line = regex.split(line)
                if len(re_line) == 8:
                    ret_lines.append(re_line[7])
            return ret_lines

    def lsr(self, hdfs_path, excludes=[]):
        """
        list directory contents about HDFS hdfs_path recursively

        Args:
            hdfs_path(str): Remote HDFS path.
            excludes(list): excludes

        Returns:
            List: a contents list about hdfs_path.
        """

        assert hdfs_path is not None

        if not self.is_exist(hdfs_path):
            return []

        ls_commands = ['-lsr', hdfs_path]
        returncode, output, errors = self.__run_hdfs_cmd(
            ls_commands, retry_times=1)

        if returncode:
            _logger.error("HDFS list all files: {} failed".format(hdfs_path))
            return []
        else:
            _logger.info("HDFS list all files: {} successfully".format(
                hdfs_path))
            lines = []
            regex = re.compile('\s+')
            out_lines = output.strip().split("\n")
            for line_id, line in enumerate(out_lines):
                re_line = regex.split(line)
                if len(re_line) == 8:
                    if re_line[0][0] == "d":
                        continue
                    if re_line[7] in excludes:
                        continue
                    else:
                        lines.append((re_line[7], re_line[5] + " " + re_line[6],
                                      line_id))
            lines = sorted(lines, key=lambda line: line[2])
            ret_lines = [ret[0] for ret in lines]
            return ret_lines

    @staticmethod
    def split_files(files, trainer_id, trainers):
        """
        split file list

        Args:
            files(list): file list
            trainer_id(int): trainer mpi rank id
            trainers(int): all trainers num

        Returns:
            fileist(list): file list of current trainer
        """
        remainder = len(files) % trainers
        blocksize = len(files) / trainers

        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1

        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]

        return trainer_files[trainer_id]

    def download(self,
                 hdfs_path,
                 local_path,
                 multi_processes=5,
                 overwrite=False,
                 retry_times=5):
        """
        Download files from HDFS using multi process.

        Args:
            hdfs_path(str): path on hdfs
            local_path(str): path on local
            multi_processes(int|5): the download data process at the same time, default=5
            overwrite(bool): is overwrite
            retry_times(int): retry times

        Returns:
            List:
            Download files in local folder.
        """

        def __subprocess_download(local_path, datas):
            """
            download file from HDFS

            Args:
                hdfs_path(str): the hdfs file path
                local_path(str): the local file path
                overwrite(bool|None): will overwrite the file on HDFS or not
                retry_times(int|5): retry times

            Returns:
                True or False
            """
            for data in datas:
                download_commands = ["-get", data, local_path]

                returncode, output, errors = self.__run_hdfs_cmd(
                    download_commands, retry_times=retry_times)

                if returncode:
                    _logger.error(
                        "Get local path: {} from HDFS path: {} failed".format(
                            local_path, hdfs_path))
                    return False
            return True

        self.make_local_dirs(local_path)

        all_files = self.ls(hdfs_path)

        procs = []
        for i in range(multi_processes):
            process_datas = HDFSClient.split_files(all_files, i,
                                                   multi_processes)
            p = multiprocessing.Process(
                target=__subprocess_download,
                args=(
                    local_path,
                    process_datas, ))
            procs.append(p)
            p.start()

        # complete the processes
        for proc in procs:
            proc.join()

        _logger.info("Finish {} multi process to download datas".format(
            multi_processes))

        local_downloads = []
        for dirname, folder, files in os.walk(local_path):
            for i in files:
                t = os.path.join(dirname, i)
                local_downloads.append(t)
        return local_downloads

    def upload(self,
               hdfs_path,
               local_path,
               multi_processes=5,
               overwrite=False,
               retry_times=5):
        """
        Upload files to HDFS using multi process.

        Args:
            hdfs_path(str): path on hdfs
            local_path(str): path on local
            multi_processes(int|5): the upload data process at the same time, default=5
            overwrite(bool|False): will overwrite file on HDFS or not
            retry_times(int): upload file max retry time.

        Returns:
            None
        """

        def __subprocess_upload(hdfs_path_single, datas):
            for data in datas:
                put_commands = ["-put", data, hdfs_path_single]
                returncode, output, errors = self.__run_hdfs_cmd(put_commands,
                                                                 retry_times)

                if returncode:
                    _logger.error("Put local path: {} to HDFS path: {} failed".
                                  format(data, hdfs_path_single))
                    return False
            return True

        def get_local_files(path):
            """
            get local files

            Args:
                path(str): local path

            Returns:
                list of local files
            """
            rlist = []

            if not os.path.exists(path):
                return rlist

            if os.path.isdir(path):
                for file in os.listdir(path):
                    t = os.path.join(path, file)
                    rlist.append(t)
            else:
                rlist.append(path)
            return rlist

        all_files = get_local_files(local_path)
        if not all_files:
            _logger.info("there are nothing need to upload, exit")
            return

        if self.is_exist(hdfs_path) and overwrite:
            self.delete(hdfs_path)
            self.makedirs(hdfs_path)

        procs = []
        for i in range(multi_processes):
            process_datas = HDFSClient.split_files(all_files, i,
                                                   multi_processes)
            p = multiprocessing.Process(
                target=__subprocess_upload, args=(
                    hdfs_path,
                    process_datas, ))
            procs.append(p)
            p.start()

        # complete the processes
        for proc in procs:
            proc.join()

        _logger.info("Finish upload datas from {} to {}".format(local_path,
                                                                hdfs_path))

    def upload_dir(self, dest_dir, local_dir, overwrite=False):
        """
        upload dir to hdfs
        Args:
            dest_dir(str): hdfs dest dir
            local_dir(str): hdfs local dir
            overwrite(bool): is overwrite
        Returns:
            return code
        """
        local_dir = local_dir.rstrip("/")
        dest_dir = dest_dir.rstrip("/")
        local_basename = os.path.basename(local_dir)
        if self.is_exist(dest_dir + "/" + local_basename) and overwrite:
            self.delete(dest_dir + "/" + local_basename)
        if not self.is_exist(dest_dir):
            self.makedirs(dest_dir)
        put_command = ["-put", local_dir, dest_dir]
        returncode, output, errors = self.__run_hdfs_cmd(put_command)
        if returncode != 0:
            _logger.error("Put local dir: {} to HDFS dir: {} failed".format(
                local_dir, dest_dir))
            return False
        return True


if __name__ == "__main__":
    hadoop_home = "/home/client/hadoop-client/hadoop/"

    configs = {
        "fs.default.name": "hdfs://xxx.hadoop.com:54310",
        "hadoop.job.ugi": "hello,hello123"
    }

    client = HDFSClient(hadoop_home, configs)

    client.ls("/user/com/train-25")
    files = client.lsr("/user/com/train-25/models")
