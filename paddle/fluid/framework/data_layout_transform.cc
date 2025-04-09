//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/op_kernel_type.h"

#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle::framework {

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to) {
  PADDLE_ENFORCE_NE(
      from,
      to,
      common::errors::InvalidArgument(
          "Layout transform should transform between different layout."));
  if (from == DataLayout::kNCHW && to == DataLayout::kNHWC) {
    return {0, 2, 3, 1};
  } else if (from == DataLayout::kNHWC && to == DataLayout::kNCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW(
        common::errors::InvalidArgument("Unsupported layout transform."));
  }
}

template <typename T>
void CastDataLayout::apply() {
  auto place = ctx_->GetPlace();

  if (phi::is_cpu_place(place)) {
    phi::funcs::Transpose<phi::CPUContext, T, 4> trans4;
    auto* context = static_cast<const phi::CPUContext*>(ctx_);
    trans4(*context, in_, out_, axis_);
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "Unsupported data layout cast from CPU to GPU."));
  }
}

void TransDataLayout(const phi::KernelKey& kernel_type_for_var,
                     const phi::KernelKey& expected_kernel_type,
                     const phi::DenseTensor& in,
                     phi::DenseTensor* out,
                     const phi::Place& place) {
  PADDLE_ENFORCE(
      backends_are_same_class(kernel_type_for_var.backend(),
                              expected_kernel_type.backend()),
      common::errors::PreconditionNotMet(
          "TransDataLayout only support DataLayout transform on same place."));

  TransDataLayout(kernel_type_for_var.layout(),
                  expected_kernel_type.layout(),
                  place,
                  in,
                  out);
}

void TransDataLayout(DataLayout from_layout,
                     DataLayout to_layout,
                     phi::Place place,
                     const phi::DenseTensor& in,
                     phi::DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      arity(in.dims()),
      4,
      common::errors::InvalidArgument(
          "Input dimension arity only can be 4, the input dimension is %s.",
          in.dims()));

  auto& pool = phi::DeviceContextPool::Instance();

  auto src_dim = in.dims();
  std::vector<int64_t> dst_dim;

  auto axis = GetAxis(from_layout, to_layout);
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->Resize(common::make_ddim(dst_dim));
  out->mutable_data(place, in.dtype());

  framework::VisitDataType(
      static_cast<proto::VarType::Type>(phi::TransToProtoVarType(in.dtype())),
      CastDataLayout(pool.Get(place), axis, in, out));

  out->set_layout(to_layout);
}

}  // namespace paddle::framework
