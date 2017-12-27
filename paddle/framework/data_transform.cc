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

namespace paddle {
namespace framework {

DataTransformFnMap& DataTransformFnMap::Instance() {
  static DataTransformFnMap data_transform_map;
  return data_transform_map;
}

// auto kernel_FP32 = OpKernelType(proto::DataType::FP32, platform::CPUPlace(),
//                                 DataLayout::kNHWC, LibraryType::kPlain);

// auto kernel_FP64 = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
//                                 DataLayout::kNHWC, LibraryType::kPlain);

// void TransDataType(const OpKernelType& kernel_out,
//                    const std::vector<platform::DeviceContext*> ctx,
//                    const Variable& in, Variable* out) {

//   PADDLE_ENFORCE(in.IsType<LoDTensor>(),
//                  "Only Support LoDTensor transform!.");
//   auto src = in->Get<LoDTensor>();
//   auto* dst = out->GetMutable<LoDTensor>();
//   auto dims = in.Get<LoDTensor>().dims();
//   dst->Resize(dims);
//   auto context = *ctx[0];

//   VisitDataType(kernel_out.data_type_, CastDataType<DeviceContext>(out,
//   context));
//   CopyFrom(src, kernel_out.place_, context, dst);
// }

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
// REGISTER_DATA_TRANSFORM_FN(f::kernel_FP32, f::kernel_FP64, f::TransDataType);
