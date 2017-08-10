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

#include <gflags/gflags.h>

DECLARE_bool(parallel_nn);
DECLARE_int32(async_count);
DECLARE_int32(port);
DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_int32(trainer_count);
DECLARE_int32(ports_num);
DECLARE_int32(ports_num_for_sparse);
DECLARE_string(nics);
DECLARE_string(rdma_tcp);
DECLARE_int32(trainer_id);
DECLARE_int32(num_gradient_servers);
DECLARE_string(comment);
DECLARE_string(load_missing_parameter_strategy);
DECLARE_int32(log_period);
DECLARE_int32(log_period_server);
DECLARE_double(checkgrad_eps);
DECLARE_int32(enable_parallel_vector);
DECLARE_bool(loadsave_parameters_in_pserver);
DECLARE_int32(beam_size);
DECLARE_bool(show_layer_stat);
DECLARE_string(predict_file);
DECLARE_bool(prev_batch_state);
DECLARE_string(init_model_path);
