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
#include "paddle/phi/kernels/funcs/data_layout_transform.h"

#include "paddle/phi/kernels/funcs/math_function.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_reuse.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to) {
  PADDLE_ENFORCE_NE(
      from,
      to,
      platform::errors::InvalidArgument(
          "Layout transform should transform between different layout."));
  if (from == DataLayout::kNCHW && to == DataLayout::kNHWC) {
    return {0, 2, 3, 1};
  } else if (from == DataLayout::kNHWC && to == DataLayout::kNCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Unsupported layout transform."));
  }
}

template <typename T>
void CastDataLayout::apply() {
  auto place = ctx_->GetPlace();

  if (platform::is_cpu_place(place)) {
    phi::funcs::Transpose<phi::CPUContext, T, 4> trans4;
    auto* context = static_cast<const phi::CPUContext*>(ctx_);
    trans4(*context, in_, out_, axis_);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Unsupported data layout cast from CPU to GPU."));
  }
}

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type,
                     const Tensor& in,
                     Tensor* out) {
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_type_for_var.place_,
                                      expected_kernel_type.place_),
      platform::errors::PreconditionNotMet(
          "TransDataLayout only support DataLayout transform on same place."));

  PADDLE_ENFORCE_EQ(
      arity(in.dims()),
      4,
      platform::errors::InvalidArgument(
          "Input dimension arity only can be 4, the input dimension is %s.",
          in.dims()));

  auto& pool = platform::DeviceContextPool::Instance();

  auto src_dim = in.dims();
  std::vector<int64_t> dst_dim;

  auto axis = GetAxis(kernel_type_for_var.data_layout_,
                      expected_kernel_type.data_layout_);
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->Resize(phi::make_ddim(dst_dim));
  out->mutable_data(expected_kernel_type.place_, in.dtype());

  framework::VisitDataType(
      framework::TransToProtoVarType(in.dtype()),
      CastDataLayout(pool.Get(expected_kernel_type.place_), axis, in, out));

  out->set_layout(expected_kernel_type.data_layout_);
}

#ifdef PADDLE_WITH_MKLDNN
using dnnl::memory;
using dnnl::primitive;
using dnnl::reorder;

void TransDataLayoutFromMKLDNN(const OpKernelType& kernel_type_for_var,
                               const OpKernelType& expected_kernel_type,
                               const Tensor& in,
                               Tensor* out) {
  auto in_layout = kernel_type_for_var.data_layout_;
  auto out_layout = expected_kernel_type.data_layout_;
  auto place = expected_kernel_type.place_;

  PADDLE_ENFORCE(
      in_layout == DataLayout::kMKLDNN && out_layout != DataLayout::kMKLDNN,
      platform::errors::InvalidArgument(
          "TransDataLayoutFromMKLDNN only supports transform from MKLDNN to "
          "non-MKLDNN"));

  phi::funcs::innerTransDataLayoutFromOneDNN(
      in_layout,
      paddle::platform::MKLDNNDeviceContext::tls().get_cur_paddle_data_layout(),
      in,
      out,
      place);
}

#endif

}  // namespace framework
}  // namespace paddle
