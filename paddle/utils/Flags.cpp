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
DEFINE_bool(use_gpu, false, "Only support CPU training");
#else
DEFINE_bool(use_gpu, true, "Whether to use GPU for training");
#endif

DEFINE_bool(parallel_nn,
            false,
            "Whether to use multi-threads to calculate one neural network."
            "If it was set false, use gpu_id specify which gpu core to use"
            "(the device property in the trainer config file will be ingored)."
            "If it was set true, the gpu core is specified by the trainer"
            "  config file(gpu_id will be ignored).");
DEFINE_int32(trainer_count, 1, "Defined how many trainers to train");
DEFINE_int32(gpu_id, 0, "Which gpu core to use");
DEFINE_int32(port, 20134, "Listening port for pserver");
DEFINE_int32(ports_num,
             1,
             "Number of ports for sending dense parameter,"
             " following ports on parameter server will be visited"
             " for sending dense parameter: [port, port+ports_num-1]");
DEFINE_int32(ports_num_for_sparse,
             0,
             "Number of ports for sending sparse parameter,"
             " following ports on parameter server will be visited"
             " for sending sparse parameter:"
             " [port+ports_num, port+ports_num+ports_num_for_sparse-1]");
DEFINE_string(nics, "xgbe0,xgbe1", "network device name for pservers");
DEFINE_string(rdma_tcp, "tcp", "use rdma or tcp rdma transport protocol");
DEFINE_int32(trainer_id,
             0,
             "For distributed training, each trainer must be given an unique id"
             " ranging from 0 to num_trainers-1. Trainer 0 is the master"
             " trainer");
DEFINE_int32(num_gradient_servers, 1, "number of gradient servers");
DEFINE_string(comment, "", "A string for commenting this training task");
DEFINE_string(load_missing_parameter_strategy,
              "fail",
              "which operation to take on load model fails. support "
              "fail/rand/zero only.");
DEFINE_int32(log_period, 100, "Log progress every so many batches");
DEFINE_int32(log_period_server,
             500,
             "Log progress every so many batches at pserver end");
DEFINE_double(checkgrad_eps, 1e-5, "parameter change size for checkgrad");
DEFINE_int32(enable_parallel_vector, 0, "threshold for enable parallel vector");
DEFINE_bool(loadsave_parameters_in_pserver,
            false,
            "load and save parameters in pserver. "
            "only work while parameter set sparse_remote_update.");
DEFINE_int32(beam_size,
             1,
             "Beam size used in generating most probable output sequences.");

DEFINE_bool(show_layer_stat, false, "show the statistics of each layer");
DEFINE_string(predict_file, "", "File name for saving predict result");
DEFINE_bool(prev_batch_state, false, "batch is continue with next batch");
DEFINE_string(init_model_path,
              "",
              "Path of the initial model parameters."
              "If it was set, start_pass will be ignored.");
