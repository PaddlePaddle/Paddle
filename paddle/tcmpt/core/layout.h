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

#include <ostream>

namespace pt {

/**
 * We need to ensure that the operator library is relatively independent
 * and does not depend on the framework. Therefore, before calling the kernel
 * in the Tensor Compute library inside the framework, the internal
 * layout needs to be converted to the data type in the Tensor Compute
 * library.
 *
 * Here we also can use the DataLayout in framework, they are all enum classes.
 */
enum class DataLayout {
  kUndef = 0,
  kAny,
  kNHWC,
  kNCHW,
  kMKLDNN,
  kNumLayouts,
};

std::ostream& operator<<(std::ostream& os, DataLayout dtype);

}  // namespace pt
