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

<<<<<<< HEAD
#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/ddim.h"
=======
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/mixed_vector.h"
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace paddle {
namespace framework {

using LoD = std::vector<paddle::framework::Vector<size_t>>;

}  // namespace framework
}  // namespace paddle
