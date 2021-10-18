// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "glog/logging.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/tcmpt/hapi/all.h"

paddle::experimental::Tensor matmul_v2_dygraph_function(
    const paddle::experimental::Tensor& X,
    const paddle::experimental::Tensor& Y, const bool trans_x,
    const bool trans_y, const bool use_mkldnn,
    const std::string& mkldnn_data_type, const int op_role,
    const std::vector<std::string>& op_role_var,
    const std::string& op_namescope,
    const std::vector<std::string>& op_callstack, const std::string& op_device,
    const bool with_quant_attr, bool trace_backward);
paddle::experimental::Tensor sigmoid_dygraph_function(
    const paddle::experimental::Tensor& X, const bool use_mkldnn,
    const bool use_cudnn, const int op_role,
    const std::vector<std::string>& op_role_var,
    const std::string& op_namescope,
    const std::vector<std::string>& op_callstack, const std::string& op_device,
    const bool with_quant_attr, bool trace_backward);
