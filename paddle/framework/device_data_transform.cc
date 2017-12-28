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

#include "paddle/framework/data_transform.h"
#include "paddle/framework/tensor_util.h"

namespace paddle {
namespace framework {

static OpKernelType k0(proto::DataType::FP32, platform::CPUPlace(),
                       DataLayout::kAnyLayout, LibraryType::kPlain);
static OpKernelType k1(proto::DataType::FP32, platform::CUDAPlace(0),
                       DataLayout::kAnyLayout, LibraryType::kPlain);

void CPU_fromto_GPU(std::vector<platform::DeviceContext*> ctx,
                    const KernelTypePair& pair, const Variable& in,
                    Variable* out) {
  CopyFrom(in.Get<Tensor>(), pair.second.place_, *ctx[0],
           out->GetMutable<Tensor>());
}

}  // namespace framework
}  // namespace paddle

namespace frw = paddle::framework;

REGISTER_DATA_TRANSFORM_FN(frw::k0, frw::k1, frw::CPU_fromto_GPU);
REGISTER_DATA_TRANSFORM_FN(frw::k1, frw::k0, frw::CPU_fromto_GPU);
