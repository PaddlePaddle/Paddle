// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <string>
#include <unordered_map>
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;

template <typename T, dnnl::algorithm BINARY_OP>
class EltwiseMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    const auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");

    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    float scale_o = ctx.Attr<float>("Scale_out");
    int axis = ctx.Attr<int>("axis");

    platform::BinaryMKLDNNHandler<T> handler(BINARY_OP, axis, mkldnn_engine,
                                             ctx.GetPlace(), x, y, z, scale_x,
                                             scale_y, scale_o);

    const auto src_x_memory = handler.AcquireSrcMemory(x);
    const auto src_y_memory = handler.AcquireSecondSrcMemory(y);
    // (jczaja) For Inplace src and dst should be the same memory object.
    // So x should share buffer with z. But UT mechanics is testing inplace
    // execution for this op not checking that x can be bradcasted to match in
    // shape y tensor.
    // This is wrong as when x is to be broadcasted then z(out) will match the
    // shape of y which is bigger than x. Hence if x is smaller in shape than z
    // and they share a buffer (of
    // shape x) then this buffer is not big enough to hold result of elementwise
    // operation.
    const bool reuse_x_memopry =
        x->numel() == z->numel() && x->IsSharedBufferWith(*z);
    std::shared_ptr<dnnl::memory> dst_memory = nullptr;
    if (reuse_x_memopry) {
      dst_memory = src_x_memory;
      // NOTE(chenfeiyu): when the output reuses memory from other tensor rather
      // than allocate its own, it's still need to take care of its data type.
      // Unfortunately, paddle's operator only infers the output' shape, but not
      // the data type. mutable_data<T> takes care of allocation and data type
      // normally, but if the memory is already allocated and there is no need
      // to re-allocate, it just set the data type. So this it added there to
      // get the right data type.
      z->mutable_data<T>(ctx.GetPlace());
    } else {
      dst_memory = handler.AcquireDstMemory(z);
    }

    const auto binary_prim = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *src_x_memory},
        {DNNL_ARG_SRC_1, *src_y_memory},
        {DNNL_ARG_DST, *dst_memory}};

    binary_prim->execute(astream, args);
    astream.wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

inline std::vector<int64_t> CalculateBroadcastedDims(const Tensor* x,
                                                     const Tensor* y) {
  const auto src_tz = phi::vectorize(x->dims());
  const auto dst_tz = phi::vectorize(y->dims());

  size_t j = 0;
  std::vector<int64_t> dst_tz_ex(src_tz.size(), 1);
  for (size_t i = 0; i < src_tz.size(); ++i) {
    dst_tz_ex[i] = (src_tz[i] != dst_tz[j]) ? 1 : dst_tz[j++];
    if (j == dst_tz.size()) break;
  }

  return dst_tz_ex;
}
}  // namespace operators
}  // namespace paddle
