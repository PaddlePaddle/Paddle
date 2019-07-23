#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import logging
import os
import sys
import time
import paddle.fluid as fluid
from paddle.fluid.log_helper import get_logger
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from . import hdfs
from .hdfs import *

__all__ = ["FleetUtil"]

_logger = get_logger(
        __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

class FleetUtil(object):

    def rank0_print(self, s):
        """ 
        Worker of rank 0 print some log.

        Args:
            s(str): string to print

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_print("my log")
        
        """
        if fleet.worker_index() != 0:
            return
        print s
        sys.stdout.flush()

    def rank0_info(self, s):
        """
        Worker of rank 0 print some log info.

        Args:
            s(str): string to log

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_info("my log info")

        """
        if fleet.worker_index() != 0:
            return
        _logger.info(s)

    def rank0_error(self, s):
        """
        Worker of rank 0 print some log error.

        Args:
            s(str): string to log

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.rank0_error("my log error")

        """
        if fleet.worker_index() != 0:
            return
        _logger.error(s)

    def set_zero(self,
                 var_name,
                 scope=fluid.global_scope(),
                 place=fluid.CPUPlace(),
                 param_type="int64"):
        """
        Set tensor of a Variable to zero.

        Args:
            var_name(str): name of Variable
            scope(Scope): Scope object, default is fluid.global_scope()
            place(Place): Place object, default is fluid.CPUPlace()
            param_type(str): param data type, default is int64

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.set_zero(myvar.name, myscope)
            
        """
        param = scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype(param_type)
        param.set(param_array, place)

    def print_global_auc(self,
                         scope=fluid.global_scope(),
                         stat_pos="_generated_var_2",
                         stat_neg="_generated_var_3",
                         print_prefix=""):
        """
        Print global auc of all distributed workers.

        Args:
            scope(Scope): Scope object, default is fluid.global_scope()
            stat_pos(str): name of auc pos bucket Variable
            stat_neg(str): name of auc neg bucket Variable
        
        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.print_global_auc(myscope, stat_pos=stat_pos,
                                         stat_neg=stat_neg)

              # below is part of model
              emb = my_slot_net(slots, label) # emb can be fc layer of size 1
              similarity_norm = fluid.layers.sigmoid(fluid.layers.clip(\
                  emb, min=-15.0, max=15.0), name="similarity_norm")\
              binary_predict = fluid.layers.concat(input=[\
                  fluid.layers.elementwise_sub(\
                      fluid.layers.ceil(similarity_norm), similarity_norm),\
                  similarity_norm], axis=1)
              auc, batch_auc, [batch_stat_pos, batch_stat_neg, stat_pos, \
                  stat_neg] = fluid.layers.auc(input=binary_predict,\
                                               label=label, curve='ROC',\
                                               num_thresholds=4096)

        """
        if scope.find_var(stat_pos) is None or scope.find_var(stat_neg) is None:
            rank0_print("not found auc bucket")
            return
        fleet._role_maker._barrier_worker()
        # auc pos bucket
        pos = np.array(scope.find_var(stat_pos).get_tensor())
        # auc pos bucket shape
        old_pos_shape = np.array(pos.shape)
        # reshape to one dim
        pos = pos.reshape(-1)
        global_pos = np.copy(pos) * 0
        # mpi allreduce
        fleet._role_maker._node_type_comm.Allreduce(pos, global_pos)
        # reshape to its original shape
        global_pos = global_pos.reshape(old_pos_shape)

        # auc neg bucket
        neg = np.array(scope.find_var(stat_neg).get_tensor())
        old_neg_shape = np.array(neg.shape)
        neg = neg.reshape(-1)
        global_neg = np.copy(neg) * 0
        fleet._role_maker._node_type_comm.Allreduce(neg, global_neg)
        global_neg = global_neg.reshape(old_neg_shape)

        # calculate auc
        num_bucket = len(global_pos[0])
        area = 0.0
        pos = 0.0
        neg = 0.0
        new_pos = 0.0
        new_neg = 0.0
        for i in xrange(num_bucket):
            index = num_bucket - 1 - i
            new_pos = pos + global_pos[0][index]
            new_neg = neg + global_neg[0][index]
            area += (new_neg - neg) * (pos + new_pos) / 2;
            pos = new_pos
            neg = new_neg
        if pos * neg == 0 or all_ins_num == 0:
            rank0_print(print_prefix + " global auc = 0.5")
        else:
            rank0_print(\
                    print_prefix + " global auc = %s" % (area / (pos * neg)))
        fleet._role_maker._barrier_worker()

    def load_fleet_model_one_table(self, table_id, path):
        """
        load pslib model to one table

        Args:
            table_id(int): load model to one table, default is None, which mean
                           load all table.
            path(str): model path

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.load_fleet_model("hdfs:/my/model/path", table_id=1)
        """
        fleet.load_one_table(table_id, path)

    def load_fleet_model(self, path, mode=0):
        """
        load pslib model

        Args:
            path(str): model path
            mode(str): 0 or 1, which means load checkpoint or delta model,
                       default is 0
        
        Examples:
            .. code-block:: python
              
              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()

              fleet_util.load_fleet_model("hdfs:/my/model/path")

              fleet_util.load_fleet_model("hdfs:/my/model/path", mode=0)

        """
        fleet.init_server(path, mode=mode)

    def save_fleet_model(self, path, mode=0):
        """
        save pslib model

        Args:
            path(str): model path
            mode(str): 0 or 1, which means save checkpoint or delta model,
                       default is 0

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_fleet_model("hdfs:/my/model/path")
                       
        """
        fleet.save_persistables(None, path, mode=mode)

    def _get_xbox_str(self, output_path, day, model_path, xbox_base_key,
                      data_path, monitor_data={}):
        xbox_dict = collections.OrderedDict()
        xbox_dict["id"] = int(time.time())
        xbox_dict["key"] = xbox_base_key
        xbox_dict["input"] = model_path + "000"
        xbox_dict["record_count"] = 111111
        xbox_dict["job_name"] = "default_job_name"
        xbox_dict["ins_tag"] = "feasign"
        if isinstance(data_path, list):
            xbox_dict["ins_path"] = ",".join(data_path)
        else:
            xbox_dict["ins_path"] = data_path
        job_id_with_host = os.popen("echo -n ${JOB_ID}").read().strip()
        instance_id = os.popen("echo -n ${INSTANCE_ID}").read().strip()
        start_pos = instance_id.find(job_id_with_host)
        end_pos = instance_id.find("--")
        if start_pos != -1 and end_pos != -1:
            job_id_with_host = instance_id[start_pos : end_pos]
        xbox_dict["job_id"] = job_id_with_host

    def write_model_donefile(self, output_path, day, model_path, pass_id,
                             xbox_base_key, hadoop_fs_name, hadoop_fs_ugi,
                             hadoop_home="$HADOOP_HOME",
                             donefile_name="donefile.txt"):
        """
        write donefile when save model

        Args:
            output_path(str): output path
            day(str|int): training day
            model_path(str): save model path
            pass_id(str|int): training pass id
            xbox_base_key(str|int): xbox base key
            hadoop_fs_name(str): hdfs/afs fs name
            hadoop_fs_ugi(str): hdfs/afs fs ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"
            donefile_name(str): donefile name, default is "donefile.txt"

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.write_model_donefile(output_path="hdfs:/my/output",
                                              day=20190723,
                                              model_path="hdfs:/my/model",
                                              pass_id=66,
                                              xbox_base_key=int(time.time()),
                                              hadoop_fs_name="hdfs://xxx",
                                              hadoop_fs_ugi="user,passwd")

        """
        day = str(day)
        pass_id = str(pass_id)
        xbox_base_key = str(xbox_base_key)

        if fleet.worker_index() == 0:
            donefile_path = output_path + "/" + donefile_name
            content  = "%s\t%lu\t%s\t%s\t%d" % (day, xbox_base_key,\
                                                model_path, pass_id, 0)
            configs = {
                "fs.default.name": hadoop_fs_name,
                "hadoop.job.ugi": hadoop_fs_ugi
            }
            client = HDFSClient(hadoop_home, configs)
            if client.is_exist(donefile_path):
                pre_content = client.cat(donefile_path)
                pre_content_list = pre_content.split("\n")
                day_list = [i.split("\t")[0] for i in pre_content_list]
                pass_list = [i.split("\t")[3] for i in pre_content_list]
                exist = False
                for i in range(len(day_list)):
                    if day == day_list[i] and pass_id == pass_list[i]:
                        exist = True
                        break
                if not exist:
                    with open(donefile_name, "w") as f:
                        f.write(pre_content + "\n")
                        f.write(content + "\n")
                    client.upload(donefile_path, donefile_name,
                                  multi_processes=1, overwrite=True)
                    rank0_error("write model %s/%s donefile succeed" % \
                                  (day, pass_id))
                else:
                    rank0_error("not write model donefile because model "
                                  "%s/%s already exists" % (day, pass_id))
            else:
                with open(donefile_name, "w") as f:
                    f.write(pre_content + "\n")
                    f.write(content + "\n")
                client.upload(donefile_path, donefile_name, multi_processes=1,
                              overwrite=True)
                rank0_error("write model %s/%s donefile succeed" % \
                               (day, pass_id))
        fleet._role_maker._barrier_worker()

    def write_delta_donefile(self, output_path, day, model_path, pass_id,
                             xbox_base_key, data_path, hadoop_fs_name,
                             hadoop_fs_ugi, monitor_data = {},
                             hadoop_home="$HADOOP_HOME",
                             donefile_name="xbox_patch_done.txt"):
        if fleet.worker_index() == 0:
            donefile_path = output_path + "/" + donefile_name
            xbox_str = self._get_xbox_str(output_path, day, model_path, \
                    xbox_base_key, data_path, monitor_data={})
                                          
            
    
        fleet._role_maker._barrier_worker()

    def load_model(self, output_path, day, pass_id):
        """
        load pslib model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.load_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/%s/" % (day, pass_id)
        load_path = output_path + suffix_name
        rank0_error("going to load_model %s" % load_path)
        load_fleet_model(load_path)
        rank0_error("load_model done")

    def save_model(self, output_path, day, pass_id):
        """
        save pslib model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/%s/" % (day, pass_id)
        model_path = output_path + suffix_name
        rank0_error("going to save_model %s" % model_path)
        save_fleet_model(model_path)
        rank0_error("save_model done")

    def save_batch_model(self, output_path, day):
        """
        save batch model

        Args:
            output_path(str): output path
            day(str|int): training day

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_batch_model("hdfs:/my/path", 20190722)

        """
        day = str(day)
        suffix_name = "/%s/batch_model/" % day
        model_path = output_path + suffix_name
        rank0_error("going to save_model %s" % model_path)
        fleet.save_persistables(None, model_path, mode=3)
        rank0_error("save_batch_model done")

    def save_delta_model(self, output_path, day, pass_id):
        """
        save delta model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_batch_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/delta-%s/" % (day, pass_id)
        model_path = output_path + suffix_name
        rank0_error("going to save_delta_model %s" % model_path)
        fleet.save_persistables(None, model_path, mode=1)
        rank0_error("save_delta_model done")

    def save_xbox_base_model(self, output_path, day, pass_id):
        """
        save xbox base model

        Args:
            output_path(str): output path
            day(str|int): training day
            pass_id(str|int): training pass id

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              fleet_util.save_xbox_base_model("hdfs:/my/path", 20190722, 88)

        """
        day = str(day)
        pass_id = str(pass_id)
        suffix_name = "/%s/base/" % day
        model_path = output_path + suffix_name
        rank0_error("going to save_xbox_base_model " + model_path)
        fleet.save_persistables(None, model_path, mode=2)
        rank0_error("save_xbox_base_model done")

    def get_last_save_model(self, output_path, hadoop_fs_name,
                            hadoop_fs_ugi, hadoop_home="$HADOOP_HOME"):
        """
        get last saved model info from donefile.txt

        Args:
            output_path(str): output path
            hadoop_fs_name(str): hdfs/afs fs_name
            hadoop_fs_ugi(str): hdfs/afs fs_ugi
            hadoop_home(str): hadoop home, default is "$HADOOP_HOME"

        Returns:
            [last_save_day, last_save_pass, last_path]
            last_save_day(int): day of saved model
            last_save_pass(int): pass id of saved
            last_path(str): model path

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              last_save_day, last_save_pass, last_path = \
                  fleet_util.save_xbox_base_model("hdfs:/my/path", 20190722, 88)
            
        """
        last_save_day = -1
        last_save_pass = -1
        last_path = ""
        donefile_path = output_path + "/donefile.txt"
        configs = {
            "fs.default.name": hadoop_fs_name,
            "hadoop.job.ugi": hadoop_fs_ugi
        }
        client = HDFSClient(hadoop_home, configs)
        if not client.is_file(donefile_path):
            return [-1, -1, ""]
        content = client.cat(donefile_path)
        content = content.split("\n")[-1].split("\t")
        last_save_day = int(content[0])
        last_save_pass = int(content[3])
        last_path = content[2]
        return [last_save_day, last_save_pass, last_path]

    def get_online_pass_interval(self, days, hours, split_interval,
                                 split_per_pass, is_data_hourly_placed):
        """
        get online pass interval

        Args:
            days(str): days to train
            hours(str): hours to train
            split_interval(int|str): split interval
            split_per_pass(int}str): split per pass
            is_data_hourly_placed(bool): is data hourly placed

        Returns:
            online_pass_interval(list)

        Examples:
            .. code-block:: python

              from paddle.fluid.incubate.fleet.utils.fleet_util import FleetUtil
              fleet_util = FleetUtil()
              online_pass_interval = fleet_util.get_online_pass_interval(
                  days="{20190720..20190729}",
                  hours="{0..23}",
                  split_interval=5,
                  split_per_pass=2,
                  is_data_hourly_placed=False) 
    
        """
        days = os.popen("echo -n " + days).read().split(" ")
        hours = os.popen("echo -n " + hours).read().split(" ")
        split_interval = int(split_interval)
        split_per_pass = int(split_per_pass)
        splits_per_day = 24 * 60 / split_interval
        pass_per_day = splits_per_day / split_per_pass
        left_train_hour = int(hours[0])
        right_train_hour = int(hours[-1])

        start = 0
        split_path = []
        for i in range(splits_per_day):
            h = start / 60
            m = start % 60
            if h < left_train_hour or h > right_train_hour:
                start += split_interval
                continue
            if is_data_hourly_placed:
                split_path.append("%02d" % h)
            else:
                split_path.append("%02d%02d" % (h, m))
            start += split_interval

        start = 0    
        online_pass_interval = []
        for i in range(pass_per_day):
            online_pass_interval.append([])
            for j in range(start, start + split_per_pass):
                online_pass_interval[i].append(split_path[j])
            start += split_per_pass

        return online_pass_interval

         
