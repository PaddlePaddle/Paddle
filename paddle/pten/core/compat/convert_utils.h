/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/common/backend.h"
#include "paddle/pten/common/data_type.h"
#include "paddle/pten/common/layout.h"
#include "paddle/pten/common/place.h"
#include "paddle/pten/core/tensor_meta.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/data_type.h"

// TODO(chenweihang): this file may need to be removed

namespace pten {

std::string TransToPtenKernelName(const std::string& fluid_op_name);
const std::string& TransToFluidOpName(const std::string& pten_kernel_name);

Backend TransToPtenBackend(const pten::Place& place);
pten::Place TransToPtenPlace(const Backend& backend, bool set_device_id = true);

}  // namespace pten
