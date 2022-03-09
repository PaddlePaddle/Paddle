// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/infrt/dialect/infrt/common_type.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_factory.h"

namespace infrt {

phi::Backend cvtTarget2Phi(TargetType target);
TargetType cvtTargetFromPhi(phi::Backend backend);

phi::DataType cvtPrecision2Phi(PrecisionType precision);
PrecisionType cvtPrecisionFromPhi(phi::DataType datatype);

phi::DataLayout cvtLayout2Phi(LayoutType layout);
LayoutType cvtLayoutFromPhi(phi::DataLayout layout);

phi::KernelKey cvtPlace2Phi(const Place& place);
Place cvtPlaceFromPhi(phi::TensorArgDef tensor_arg);

}  // namespace infrt
