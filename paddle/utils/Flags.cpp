/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Flags.h"

#ifdef PADDLE_ONLY_CPU
P_DEFINE_bool(use_gpu, false, "Only support CPU training");
#else
P_DEFINE_bool(use_gpu, true, "Whether to use GPU for training");
#endif

P_DEFINE_bool(
    parallel_nn,
    false,
    "Whether to use multi-threads to calculate one neural network."
    "If it was set false, use gpu_id specify which gpu core to use"
    "(the device property in the trainer config file will be ingored)."
    "If it was set true, the gpu core is specified by the trainer"
    "  config file(gpu_id will be ignored).");
P_DEFINE_int32(trainer_count, 1, "Defined how many trainers to train");
P_DEFINE_int32(gpu_id, 0, "Which gpu core to use");
P_DEFINE_int32(port, 20134, "Listening port for pserver");
P_DEFINE_int32(data_server_port, 21134, "Listening port for dserver");
P_DEFINE_int32(ports_num,
               1,
               "The ports number for parameter send,"
               " increment based on default port number");
P_DEFINE_int32(ports_num_for_sparse,
               0,
               "The ports number for parameter send,"
               " increment based on default (port + ports_num)");
P_DEFINE_string(nics, "xgbe0,xgbe1", "network device name for pservers");
P_DEFINE_string(rdma_tcp, "tcp", "use rdma or tcp rdma transport protocol");
P_DEFINE_int32(
    trainer_id,
    0,
    "For distributed training, each trainer must be given an unique id"
    " ranging from 0 to num_trainers-1. Trainer 0 is the master"
    " trainer");
P_DEFINE_int32(num_gradient_servers, 1, "number of gradient servers");
P_DEFINE_string(comment, "", "A string for commenting this training task");
P_DEFINE_string(load_missing_parameter_strategy,
                "fail",
                "which operation to take on load model fails. support "
                "fail/rand/zero only.");
P_DEFINE_int32(log_period, 100, "Log progress every so many batches");
P_DEFINE_int32(log_period_server,
               500,
               "Log progress every so many batches at pserver end");
P_DEFINE_double(checkgrad_eps, 1e-5, "parameter change size for checkgrad");
P_DEFINE_int32(enable_parallel_vector,
               0,
               "threshold for enable parallel vector");
P_DEFINE_bool(loadsave_parameters_in_pserver,
              false,
              "load and save parameters in pserver. "
              "only work while parameter set sparse_remote_update.");
P_DEFINE_int32(beam_size,
               1,
               "Beam size used in generating most probable output sequences.");

P_DEFINE_bool(show_layer_stat, false, "show the statistics of each layer");
P_DEFINE_string(predict_file, "", "File name for saving predict result");
P_DEFINE_bool(prev_batch_state, false, "batch is continue with next batch");
P_DEFINE_string(init_model_path,
                "",
                "Path of the initial model parameters."
                "If it was set, start_pass will be ignored.");
