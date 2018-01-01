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
#include "paddle/framework/tensor.h"
#include "paddle/framework/lod_tensor.h"

namespace paddle {
namespace framework {

OpKernelType k0(proto::DataType::FP32, platform::CPUPlace(),
                       DataLayout::kAnyLayout, LibraryType::kPlain);
OpKernelType k1(proto::DataType::FP32, platform::CUDAPlace(0),
                       DataLayout::kAnyLayout, LibraryType::kPlain);

void CPU_fromto_GPU(const platform::DeviceContext* ctx,
                    const KernelTypePair& pair, const Variable& in,
                    Variable* out) {
  std::cout << "CPU_fromto_GPU in" << std::endl;
  std::cout << "src_place: " << in.Get<LoDTensor>().place() << std::endl;
  std::cout << "dst_place: " << pair.second.place_ << std::endl;
  CopyFrom(in.Get<LoDTensor>(), pair.second.place_, *ctx,
           out->GetMutable<LoDTensor>());
  std::cout << "CPU_fromto_GPU out" << std::endl;
}

}  // namespace framework
}  // namespace paddle

namespace frw = paddle::framework;


REGISTER_DATA_TRANSFORM_MODEULE(device_data_transform);
REGISTER_DATA_TRANSFORM_FN(frw::k0, frw::k1, frw::CPU_fromto_GPU);
REGISTER_DATA_TRANSFORM_FN(frw::k1, frw::k1, frw::CPU_fromto_GPU);

