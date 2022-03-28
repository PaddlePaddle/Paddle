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

#include "paddle/infrt/dialect/infrt/common/types.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_factory.h"

namespace infrt {

::phi::Backend ConvertTargetToPhi(TargetType target);
TargetType ConvertTargetFromPhi(::phi::Backend backend);

::phi::DataType ConvertPrecisionToPhi(PrecisionType precision);
PrecisionType ConvertPrecisionFromPhi(::phi::DataType datatype);

::phi::DataLayout ConvertLayoutToPhi(LayoutType layout);
LayoutType ConvertLayoutFromPhi(::phi::DataLayout layout);

::phi::KernelKey ConvertPlaceToPhi(const Place& place);
Place ConvertPlaceFromPhi(::phi::TensorArgDef tensor_arg);

}  // namespace infrt
