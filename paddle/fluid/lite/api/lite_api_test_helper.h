// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <gflags/gflags.h>
#include "paddle/fluid/lite/api/cxx_api.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/op_registry.h"

DECLARE_string(model_dir);
DECLARE_string(optimized_model);

namespace paddle {
namespace lite {

const lite::Tensor* RunHvyModel();

}  // namespace lite
}  // namespace paddle
