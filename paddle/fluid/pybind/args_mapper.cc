// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// custom arg mapper function.
// The function here will be called by the functions in
// paddle/fluid/pybind/static_op_function.cc and
// paddle/fluid/pybind/eager_op_function.cc. Mainly used to customize the args
// parser from PyObject *args and PyObject *kwargs

#include "paddle/fluid/pybind/args_mapper.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/op_function_common.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
namespace paddle {
namespace pybind {}  // namespace pybind

}  // namespace paddle
