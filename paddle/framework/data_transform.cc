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
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

DataTransformFnMap& DataTransformFnMap::Instance() {
  static DataTransformFnMap data_transform_map;
  return data_transform_map;
}

auto kernel_FP32 = OpKernelType(proto::DataType::FP32, platform::CPUPlace(),
                                DataLayout::kNHWC, LibraryType::kPlain);

auto kernel_FP64 = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                                DataLayout::kNHWC, LibraryType::kPlain);

void TransDataType(const platform::DeviceContext* ctx,
                   const KernelTypePair& kernel_pair, const Variable& in,
                   Variable* out) {
  PADDLE_ENFORCE(in.IsType<LoDTensor>(), "Only Support LoDTensor transform!.");
  auto src = in.Get<LoDTensor>();
  auto* dst = out->GetMutable<LoDTensor>();
  auto dims = src.dims();
  dst->Resize(dims);
  auto dst_type = kernel_pair.second.data_type_;
  auto src_type = kernel_pair.first.data_type_;

  switch (src_type) {
    case proto::DataType::FP32:
      framework::VisitDataType(dst_type, CastDataType<float>(src, dst, ctx));
    // case proto::DataType::FP64:
    //     framework::VisitDataType(
    //         dst_type,
    //         CastDataType<platform::DeviceContext, double>(src, dst, ctx));
    // case proto::DataType::INT32:
    //     framework::VisitDataType(
    //         dst_type, CastDataType<platform::DeviceContext, int>(src, dst,
    //         ctx));
    // case proto::DataType::INT64:
    //     framework::VisitDataType(
    //         dst_type,
    //         CastDataType<platform::DeviceContext, int64_t>(src, dst, ctx));
    // case proto::DataType::BOOL:
    //     framework::VisitDataType(
    //         dst_type, CastDataType<platform::DeviceContext, bool>(src, dst,
    //         ctx));
    default:
      PADDLE_THROW("Not support type %d", src_type);
  }
}

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;
REGISTER_DATA_TRANSFORM_FN(f::kernel_FP32, f::kernel_FP64, f::TransDataType);
