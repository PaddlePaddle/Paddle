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

from __future__ import print_function

import numpy as np
import contextlib
import six
from .framework import Program, default_main_program, Variable
from . import core
from .executor import global_scope, Executor
from paddle.fluid.proto import data_feed_pb2
from google.protobuf import text_format
from . import io
from .data_feed_desc import DataFeedDesc
from .trainer_desc import TrainerDesc, MultiTrainer, DistMultiTrainer
from .distributed import ps_instance
from .contrib.utils import hdfs_utils as hdfs

__all__ = ['AsyncExecutor']


class AsyncExecutor(object):
    """
    An asynchronous Executor in Python. Through exploiting the power of
    multi-core processor and data queueing, AsyncExecutor makes data reading
    and cosuming decoupled, each run in multiple threads in parallel.

    Instead of reading data in python side, AsyncExecutor accepts a training
    file list, which will be retrieved in C++, then training inputs will be
    read, parsed and fed to training network within C++ code.

    AsyncExecutor is in active development and the API might change in the near
    future.

    Example:
        >>> data_feed = fluid.DataFeedDesc('data.proto')
        >>> startup_program = fluid.default_startup_program()
        >>> main_program = fluid.default_main_program()
        >>> filelist = ["train_data/part-%d" % i for i in range(100)]
        >>> thread_num = len(filelist) / 4
        >>>
        >>> place = fluid.CPUPlace()
        >>> async_executor = fluid.AsyncExecutor(place)
        >>>
        >>> async_executor.run_startup_program(startup_program)
        >>>
        >>> epoch = 10
        >>> for i in range(epoch):
        >>>     async_executor.run(main_program,
        >>>                        data_feed,
        >>>                        filelist,
        >>>                        thread_num,
        >>>                        [acc],
        >>>                        debug=False)

    Args:
        place(fluid.CPUPlace|None): indicate the executor run on which device.
                                   Only CPUPlace supported

    Note:
        For debugging complicated network in parallel-GPUs, you can test it
        on the executor. They has the exactly same arguments, and expected
        the same results.

    Note: Only running on CPUPlace supported.
    """

    def __init__(self, place=None, run_mode=""):
        if place is None:
            place = core.CPUPlace()
        if not isinstance(place, core.CPUPlace):
            raise ValueError("AsyncExecutor only supports CPU device")

        p = core.Place()
        p.set_place(place)

        scope = global_scope()
        self.executor = core.AsyncExecutor(scope, p)
        self.instance = None

    def run(self, program, data_feed, filelist, thread_num, fetch, debug=False):
        if program is None:
            program = default_main_program()
        program_desc = program.desc

        if data_feed is None:
            raise ValueError('ValueError: data_feed should be provided')

        if filelist is None:
            raise ValueError('ValueError: filelist should be provided')

        if isinstance(filelist, str):
            filelist = [filelist]

        if not isinstance(thread_num, int):
            raise TypeError('TypeError: thread_num should be a positive number')

        is_local = self.instance == None
        trainer = None
        if is_local:
            trainer = MultiTrainer()
        else:
            trainer = DistMultiTrainer()
        trainer.gen_trainer_desc(
            dataset=data_feed, fleet_desc=self.dist_desc, worker="downpour")
        trainer.set_thread(thread_num)
        trainer.set_filelist(filelist)
        trainer.set_data_feed(data_feed)
        with open("trainer_desc.proto", "w") as fout:
            fout.write(trainer._desc())
        # define a trainer and a device_worker here
        self.executor.run_from_files(program_desc,
                                     trainer._desc(), debug,
                                     str(id(program_desc)))

    '''
    def run(self,
            program,
            data_feed,
            filelist,
            thread_num,
            fetch,
            mode="",
            debug=False):
        """
        Run program by this AsyncExecutor. Training dataset will be in filelist.
        Users can also inspect certain variables by naming them in parameter
        :code:`fetch`, like in fluid.Executor. Unlike fluid.Executor, however,
        AsyncExecutor doesn't return fetched variables, instead, it will dump
        the values of each fetched variable to stdandard output.

        Running the dataset will be on multiple threads, within each a thread
        local scope will be created, then all OPs also created in that scope.
        Parameters are updated by all the OPs simultaneously.

        Args:
            program(Program): the program that need to run, if not provied,
                              then default_main_program will be used.
            data_feed(DataFeedDesc): A DataFeedDesc object
            filelist(str): a file containing the training dataset file list
            thread_num(int): number of concurrent training threads. See
                             :code:`Note` for how to set this properly
            fetch(str|list): the var name or a list of var names to inspect
            mode(str): run mode of this interface
            debug(bool): When set to True, fetch vars will be printed to
                         standard output after each minibatch

        Note:
            the executor will run all operators in the program but not only
            the operators dependent by the fetch_list.

        Note:
            Running AsyncExecutor will be on multiple threads, each bound to a
            CPU core. To achieve best performance, it's suggested to set thread
            num to be equal or slightly less than that of CPU cores.
        """
        if program is None:
            program = default_main_program()
        program_desc = program.desc

        if data_feed is None:
            raise ValueError('ValueError: data_feed should be provided')

        if filelist is None:
            raise ValueError('ValueError: filelist should be provided')

        if isinstance(filelist, str):
            filelist = [filelist]

        if not isinstance(thread_num, int):
            raise TypeError('TypeError: thread_num should be a positive number')

        if fetch is not None:
            if isinstance(fetch, Variable):
                fetch = [fetch]
            fetch_var_names = [var.name for var in fetch]
            for fetch_var in fetch:
                shape = fetch_var.shape
                if shape[len(shape) - 1] != 1:
                    raise AssertionError(
                        "%s: Fetch variable has wrong shape. Only varibles "
                        "with the last dimension size 1 supported." %
                        (fetch_var.name))

        self.executor.run_from_files(program_desc,
                                     data_feed.desc(), filelist, thread_num,
                                     fetch_var_names, mode, debug, str(id(program_desc)))
    '''

    def download_data(self,
                      afs_path,
                      local_path,
                      fs_default_name,
                      ugi,
                      file_cnt,
                      hadoop_home="$HADOOP_HOME",
                      process_num=12):
        """
        download_data is a default download method for distributed training
        a user download data without this method
        
        Example:
            >>> exe = fluid.AsyncExecutor()
            >>> exe.download_data("/xxx/xxx/xx/",
            >>>                   "./data", "afs://            
            >>>  xxx.xxx.xxx.xxx:9901", "xxx,yyy") 
        Args:
            afs_path(str): afs_path defined by users
            local_path(str): download data path
            fs_default_name(str): file system server address
            ugi(str): hadoop ugi
            file_cn(int): a user can specify file number for debugging
            hadoop_home(str): hadoop home path
            process_num(int): download process num
        """
        if self.instance is None:
            raise ValueError('instance is None, please run'
                             'config_distributed_nodes init instance')

        configs = {"fs.default.name": fs_default_name, "hadoop.job.ugi": ugi}

        client = hdfs.HDFSClient(hadoop_home, configs)
        downloads = hdfs.multi_download(
            client,
            afs_path,
            local_path,
            self.instance.get_worker_index(),
            self.instance.get_node_cnt() / 2,
            multi_processes=process_num)
        self.instance.barrier_worker()  #wait for download_data

    def get_instance(self):
        """
        get current node's instance so that user can do operations
        in distributed setting
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )
        return self.instance

    def config_distributed_nodes(self):
        """
        if a user needs to run distributed async executor
        he or she needs to do a global configuration so that 
        information of current process can be obtained
        """
        self.instance = ps_instance.PaddlePSInstance(1, 2)
        return self.instance

    def stop(self):
        """
        at the end of process, users should call stop to servers
        and barrier all workers
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )
        self.instance.barrier_worker()  #worker do all things
        if self.instance.is_first_worker():
            self.executor.stop_server()
        self.instance.barrier_worker()  #sync
        self.instance.barrier_all()
        self.instance.finalize()

    def init_server(self, dist_desc):
        """
        initialize server of current node if current process is a server
        Args:
        dist_desc(str): a protobuf string that describes 
                        how to init a worker and a server
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )
        self.dist_desc_str = text_format.MessageToString(dist_desc)
        self.dist_desc = dist_desc
        self.executor.init_server(self.dist_desc_str, self.instance._rankid)
        ip = self.executor.start_server()
        self.instance.set_ip(ip)
        self.instance.barrier_all()  #wait all server start
        ips = self.instance.gather_ips()
        self.executor.gather_servers(ips, self.instance.get_node_cnt())
        self.instance.barrier_all()  #wait all worker start

    def init_worker(self, dist_desc, startup_program):
        """
        initialize worker of current node if current process is a worker
        Args:
        dist_desc(str): a protobuf string that describes
                        how to init a worker and a server
        startup_program(fluid.Program): startup program of current process
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )

        self.dist_desc_str = text_format.MessageToString(dist_desc)
        self.dist_desc = dist_desc
        place = core.CPUPlace()
        executor = Executor(place)
        if isinstance(startup_program, list):
            for sp in startup_program:
                executor.run(sp)
        else:
            executor.run(startup_program)

        self.instance.barrier_all()  #wait all server start
        ips = self.instance.gather_ips()
        self.executor.init_worker(self.dist_desc_str, ips,
                                  self.instance.get_node_cnt(),
                                  self.instance._rankid)
        self.instance.barrier_all()  #wait all worker start
        if self.instance.is_first_worker():
            self.executor.init_model()
        self.instance.barrier_worker()  #wait init model

    def init_model(self):
        """
        init_model command that can be invoked from one of the worker
        model parameters are initialized in servers
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )
        self.executor.init_model()

    def save_model(self, save_path):
        """
        save_model command that can be invoked from one of the worker
        model parameters are saved in servers and upload to save_path of file system
        Args:
        save_path(str): save path to file system
        """
        if self.instance is None:
            raise ValueError(
                'instance is None, please run config_distributed_nodes init instance'
            )
        self.executor.save_model(save_path)
