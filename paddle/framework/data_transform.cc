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
#include <functional>

#include "paddle/framework/data_transform.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

DataTransformFnMap& DataTransformFnMap::Instance() {
  static DataTransformFnMap data_transform_map;
  return data_transform_map;
}

auto KernelFP32 = OpKernelType(proto::DataType::FP32, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelFP64 = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelNHWC = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNHWC, LibraryType::kPlain);

auto KernelNCHW = OpKernelType(proto::DataType::FP64, platform::CPUPlace(),
                               DataLayout::kNCHW, LibraryType::kPlain);

void TransDataType(const platform::DeviceContext* ctx,
                   const KernelTypePair& kernel_pair, const Variable& in,
                   Variable* out) {
  PADDLE_ENFORCE(in.IsType<Tensor>(), "Only Support Tensor transform!.");
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_pair.first.place_,
                                      kernel_pair.second.place_),
      "TransDataType Only Support DataType transform on same place!");

  auto src = in.Get<Tensor>();
  auto* dst = out->GetMutable<Tensor>();

  auto dims = src.dims();
  dst->Resize(dims);
  auto dst_type = kernel_pair.second.data_type_;
  auto src_type = kernel_pair.first.data_type_;

  switch (src_type) {
    case proto::DataType::FP32:
      framework::VisitDataType(dst_type, CastDataType<float>(src, dst, ctx));
      break;
    case proto::DataType::FP64:
      framework::VisitDataType(dst_type, CastDataType<double>(src, dst, ctx));
      break;
    case proto::DataType::INT32:
      framework::VisitDataType(dst_type, CastDataType<int>(src, dst, ctx));
      break;
    case proto::DataType::INT64:
      framework::VisitDataType(dst_type, CastDataType<int>(src, dst, ctx));
      break;
    case proto::DataType::BOOL:
      framework::VisitDataType(dst_type, CastDataType<bool>(src, dst, ctx));
      break;
    default:
      PADDLE_THROW("Not support type %d", src_type);
  }
}

void TransDataLayout(const platform::DeviceContext* ctx,
                     const KernelTypePair& kernel_pair, const Variable& in,
                     Variable* out, const std::vector<int>& axis) {
  PADDLE_ENFORCE(in.IsType<Tensor>(), "Only Support Tensor transform!.");
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_pair.first.place_,
                                      kernel_pair.second.place_),
      "TransDataLayout Only Support DataType transform on same place!");

  auto src = in.Get<Tensor>();
  auto* dst = out->GetMutable<Tensor>();
  PADDLE_ENFORCE(arity(src.dims()) == 4, "Input Arity Only Suppport 4!");

  auto src_dim = src.dims();
  dst->Resize(src_dim);
  auto place = kernel_pair.second.place_;
  CopyFrom(src, place, *ctx, dst);

  std::vector<int> dst_dim;
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  dst->Resize(make_ddim(dst_dim));

  auto src_type = kernel_pair.first.data_type_;
  framework::VisitDataType(src_type, CastDataLayout(src, dst, ctx, axis));

  dst->set_layout(kernel_pair.second.data_layout_);
}

}  // namespace framework
}  // namespace paddle

namespace f = paddle::framework;

std::vector<int> NHWC2NCHW = {0, 3, 1, 2};
std::vector<int> NCHW2NHWC = {0, 2, 3, 1};

REGISTER_DATA_TRANSFORM_FN(f::KernelFP32, f::KernelFP64, f::TransDataType);
REGISTER_DATA_TRANSFORM_FN(f::KernelNHWC, f::KernelNCHW,
                           std::bind(f::TransDataLayout, std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4, NHWC2NCHW));
REGISTER_DATA_TRANSFORM_FN(f::KernelNCHW, f::KernelNHWC,
                           std::bind(f::TransDataLayout, std::placeholders::_1,
                                     std::placeholders::_2,
                                     std::placeholders::_3,
                                     std::placeholders::_4, NCHW2NHWC));
