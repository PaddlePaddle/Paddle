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
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/tensor.h"
#include "paddle/framework/tensor_util.h"

namespace paddle {
namespace framework {

OpKernelType k0(proto::DataType::FP32, platform::CPUPlace(),
                DataLayout::kAnyLayout, LibraryType::kPlain);
OpKernelType k1(proto::DataType::FP32, platform::CUDAPlace(0),
                DataLayout::kAnyLayout, LibraryType::kPlain);
OpKernelType k3(proto::DataType::INT64, platform::CPUPlace(),
                DataLayout::kAnyLayout, LibraryType::kPlain);
OpKernelType k4(proto::DataType::INT64, platform::CUDAPlace(0),
                DataLayout::kAnyLayout, LibraryType::kPlain);

void CPU_fromto_GPU(const platform::DeviceContext* ctx,
                    const KernelTypePair& pair, const Variable& in,
                    Variable* out) {
  auto& in_tensor = in.Get<LoDTensor>();
  auto* out_tensor = out->GetMutable<LoDTensor>();
  VLOG(3) << "do data copy from " << in_tensor.place() << " to "
          << pair.second.place_;
  VLOG(3) << "ctx place:" << ctx->GetPlace();
  out_tensor->set_lod(in_tensor.lod());
  out_tensor->set_layout(in_tensor.layout());
  CopyFrom(in_tensor, pair.second.place_, *ctx, out_tensor);
  ctx->Wait();
  VLOG(3) << "copy done";
}

}  // namespace framework
}  // namespace paddle

namespace frw = paddle::framework;

REGISTER_DATA_TRANSFORM_MODEULE(device_data_transform);
REGISTER_DATA_TRANSFORM_FN(frw::k0, frw::k1, frw::CPU_fromto_GPU);
REGISTER_DATA_TRANSFORM_FN(frw::k1, frw::k0, frw::CPU_fromto_GPU);
REGISTER_DATA_TRANSFORM_FN(frw::k3, frw::k4, frw::CPU_fromto_GPU);
REGISTER_DATA_TRANSFORM_FN(frw::k4, frw::k3, frw::CPU_fromto_GPU);
