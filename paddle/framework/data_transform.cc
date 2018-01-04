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
#include "paddle/framework/device_data_transform.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {

DataTransformFnMap& DataTransformFnMap::Instance() {
  static DataTransformFnMap data_transform_map;
  return data_transform_map;
}

const VariableAttr GetVariableAttr(const Variable& var) {
  VariableAttr attr;
  const Tensor* tensor;

  if (var.IsType<LoDTensor>()) {
    tensor = &var.Get<LoDTensor>();
  } else if (var.IsType<SelectedRows>()) {
    tensor = &var.Get<SelectedRows>().value();
  } else {
    PADDLE_THROW("unknown var type");
  }

  attr.place = tensor->place();
  attr.data_layout = tensor->layout();
  attr.data_type = ToDataType(tensor->type());
  attr.tensor = tensor;
  return attr;
}

Tensor* DataTransform(const VarAttrMatch& match,
                      const VariableAttr& input_var_attr,
                      const OpKernelType& expected_kernel_type) {
  Tensor* out = nullptr;
  if (!match.place) {
    out = DeviceTransform(input_var_attr.place, expected_kernel_type.place_,
                          *input_var_attr.tensor);
  }
  return out;
}

void CopyVariableWithTensor(const Variable& in_var, Variable& out_var,
                            const Tensor& tensor) {
  if (in_var.IsType<LoDTensor>()) {
    auto& in_lod_tensor = in_var.Get<LoDTensor>();
    auto* tran_lod_tensor = out_var.GetMutable<LoDTensor>();
    tran_lod_tensor->set_lod(in_lod_tensor.lod());
    tran_lod_tensor->set_layout(in_lod_tensor.layout());
    tran_lod_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<SelectedRows>()) {
    auto& in_selected_rows = in_var.Get<SelectedRows>();
    auto* trans_selected_rows = out_var.GetMutable<SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW("unknown var type");
  }
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
      framework::VisitDataType(dst_type, CastDataType<int64_t>(src, dst, ctx));
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
                     Variable* out) {
  PADDLE_ENFORCE(in.IsType<Tensor>(), "Only Support Tensor transform!.");
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_pair.first.place_,
                                      kernel_pair.second.place_),
      "TransDataType Only Support DataType transform on same place!");

  auto src = in.Get<Tensor>();
  auto* dst = out->GetMutable<Tensor>();
  PADDLE_ENFORCE(arity(src.dims()) == 4, "Input Arity Only Suppport 4!");

  auto src_dim = src.dims();
  dst->Resize(src_dim);
  auto place = kernel_pair.second.place_;
  CopyFrom(src, place, *ctx, dst);
  const std::vector<int> axis = {0, 2, 3, 1};

  std::vector<int64_t> dst_dim;
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
REGISTER_DATA_TRANSFORM_FN(f::KernelFP32, f::KernelFP64, f::TransDataType);
REGISTER_DATA_TRANSFORM_FN(f::KernelNHWC, f::KernelNCHW, f::TransDataLayout);
