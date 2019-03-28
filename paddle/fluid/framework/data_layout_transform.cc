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
#include <vector>

#include "paddle/fluid/operators/math/math_function.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {

std::vector<int> GetAxis(const DataLayout& from, const DataLayout& to) {
  PADDLE_ENFORCE_NE(from, to,
                    "layout transform should transform different layout");
  if (from == DataLayout::kNCHW && to == DataLayout::kNHWC) {
    return {0, 2, 3, 1};
  } else if (from == DataLayout::kNHWC && to == DataLayout::kNCHW) {
    return {0, 3, 1, 2};
  } else {
    PADDLE_THROW("unsupported transform");
  }
}

struct CastDataLayout {
  CastDataLayout(const platform::DeviceContext* ctx,
                 const std::vector<int>& axis, const framework::Tensor& in,
                 framework::Tensor* out)
      : in_(in), out_(out), ctx_(ctx), axis_(axis) {}
  const framework::Tensor in_;
  framework::Tensor* out_;
  const platform::DeviceContext* ctx_;
  const std::vector<int> axis_;

  template <typename T>
  void apply() {
    auto place = ctx_->GetPlace();

    if (platform::is_cpu_place(place)) {
      operators::math::Transpose<platform::CPUDeviceContext, T, 4> trans4;
      auto* context = static_cast<const platform::CPUDeviceContext*>(ctx_);
      trans4(*context, in_, out_, axis_);
    } else {
      PADDLE_THROW("Unsupport CPU <-> GPU!");
    }
  }
};

void TransDataLayout(const OpKernelType& kernel_type_for_var,
                     const OpKernelType& expected_kernel_type, const Tensor& in,
                     Tensor* out) {
  PADDLE_ENFORCE(
      platform::places_are_same_class(kernel_type_for_var.place_,
                                      expected_kernel_type.place_),
      "TransDataLayout only support DataLayout transform on same place!");

  PADDLE_ENFORCE(arity(in.dims()) == 4, "Input Arity only support 4!");

  auto& pool = platform::DeviceContextPool::Instance();

  auto src_dim = in.dims();
  std::vector<int64_t> dst_dim;

  auto axis = GetAxis(kernel_type_for_var.data_layout_,
                      expected_kernel_type.data_layout_);
  dst_dim.resize(axis.size());
  for (size_t i = 0; i < axis.size(); i++) {
    dst_dim[i] = src_dim[axis[i]];
  }

  out->Resize(make_ddim(dst_dim));
  out->mutable_data(expected_kernel_type.place_, in.type());

  framework::VisitDataType(
      in.type(),
      CastDataLayout(pool.Get(expected_kernel_type.place_), axis, in, out));

  out->set_layout(expected_kernel_type.data_layout_);
}

#ifdef PADDLE_WITH_MKLDNN
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;

void* GetDataFromTensor(const Tensor& tensor, mkldnn::memory::data_type type) {
  switch (type) {
    case mkldnn::memory::data_type::f32:
      return platform::to_void_cast(tensor.data<float>());
    case mkldnn::memory::data_type::s8:
      return platform::to_void_cast(tensor.data<int8_t>());
    case mkldnn::memory::data_type::u8:
      return platform::to_void_cast(tensor.data<unsigned char>());
    case mkldnn::memory::data_type::s16:
      return platform::to_void_cast(tensor.data<int16_t>());
    case mkldnn::memory::data_type::s32:
      return platform::to_void_cast(tensor.data<int32_t>());
    default:
      PADDLE_THROW("wrong mkldnn type provided");
  }
}
#endif

void TransDataLayoutFromMKLDNN(const OpKernelType& kernel_type_for_var,
                               const OpKernelType& expected_kernel_type,
                               const Tensor& in, Tensor* out) {
  auto in_layout = kernel_type_for_var.data_layout_;
  auto out_layout = expected_kernel_type.data_layout_;

  PADDLE_ENFORCE(
      in_layout == DataLayout::kMKLDNN && out_layout != DataLayout::kMKLDNN,
      "TransDataLayoutFromMKLDNN only supports transform from MKLDNN to "
      "non-MKLDNN");

#ifdef PADDLE_WITH_MKLDNN
  PADDLE_ENFORCE(in.format() != memory::format::format_undef &&
                     in.format() != memory::format::any,
                 "Input tensor should have specified memory format");

  // Set default as NCHW in case not specified
  out_layout =
      out_layout == DataLayout::kAnyLayout ? DataLayout::kNCHW : out_layout;

  std::vector<int> in_tz = paddle::framework::vectorize2int(in.dims());
  std::vector<int> out_tz = in_tz;

  memory::data_type in_type = ToMKLDNNDataType(in.type());
  PADDLE_ENFORCE(in_type != memory::data_type::data_undef,
                 "Input tensor type is not supported: %s", in.type());
  memory::data_type out_type = in_type;

  // output tensor has the same dims as input. Reorder don't change dims
  out->Resize(in.dims());

  // tempory mem pd fr out , to make reorder
  auto out_mem_pd = paddle::platform::create_prim_desc_from_dims(
      paddle::framework::vectorize2int(out->dims()),
      mkldnn::memory::format::blocked, out_type);
  if (in.get_mkldnn_prim_desc() != out_mem_pd) {
    void* in_data = GetDataFromTensor(in, in_type);
    auto out_data = out->mutable_data(expected_kernel_type.place_, in.type());

    auto in_memory = memory(in.get_mkldnn_prim_desc(), in_data);
    auto out_memory = memory(out_mem_pd, out_data);

    platform::Reorder(in_memory, out_memory);
  } else {
    out->ShareDataWith(in);
  }
  out->set_layout(out_layout);
#endif
}

}  // namespace framework
}  // namespace paddle
