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

#pragma once

#include "CommandLineParser.h"

P_DECLARE_bool(parallel_nn);
P_DECLARE_int32(async_count);
P_DECLARE_int32(port);
P_DECLARE_int32(data_server_port);
P_DECLARE_bool(use_gpu);
P_DECLARE_int32(gpu_id);
P_DECLARE_int32(trainer_count);
P_DECLARE_int32(ports_num);
P_DECLARE_int32(ports_num_for_sparse);
P_DECLARE_string(nics);
P_DECLARE_string(rdma_tcp);
P_DECLARE_int32(trainer_id);
P_DECLARE_int32(num_gradient_servers);
P_DECLARE_string(comment);
P_DECLARE_string(load_missing_parameter_strategy);
P_DECLARE_int32(log_period);
P_DECLARE_int32(log_period_server);
P_DECLARE_double(checkgrad_eps);
P_DECLARE_int32(enable_parallel_vector);
P_DECLARE_bool(loadsave_parameters_in_pserver);
P_DECLARE_int32(beam_size);
P_DECLARE_bool(show_layer_stat);
P_DECLARE_string(predict_file);
P_DECLARE_bool(prev_batch_state);
P_DECLARE_string(init_model_path);
