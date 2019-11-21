/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <glog/logging.h>

#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/operator.h"

DECLARE_bool(enable_unused_var_check);

namespace paddle {
namespace framework {

const std::unordered_set<std::string> op_has_unsed_vars_white_list = {
    "batch_norm",
    "batch_norm_grad",
    "crop",
    "cvm",
    "dgc_momentum",
    "fake_quantize_range_abs_max",
    "fill_zeros_like",
    "reshape2_grad_grad",
    "reshape2_grad",
    "gru_grad",
    "op_with_kernel"};

std::unordered_set<std::string>* GetThreadLocalUsedVarNameSet();
void CheckUnusedVar(const OperatorBase& op, const Scope& scope);

}  // namespace framework
}  // namespace paddle
