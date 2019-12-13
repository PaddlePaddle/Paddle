/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn/include/mkldnn.hpp>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

#ifdef PADDLE_WITH_XBYAK
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#endif

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using platform::StringToMKLDNNFormat;

template <typename T>
static void ComputeBroadcastedMultiply(const T* x_data, const T* y_data,
                                       T* z_data, int64_t n, int64_t c,
                                       int64_t h, int64_t w, int simd_width,
                                       void (*multiply)(const T*, const T*, T*,
                                                        int, int)) {
  const int64_t C = c / simd_width;
#pragma omp parallel for collapse(2)
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < C; ci++) {
      auto ptr_x =
          x_data + ni * C * h * w * simd_width + ci * h * w * simd_width;

      auto ptr_y = y_data + ni * C * simd_width + ci * simd_width;
      auto ptr_z =
          z_data + ni * C * h * w * simd_width + ci * h * w * simd_width;

      multiply(ptr_x, ptr_y, ptr_z, h, w);
    }
  }
}

template <typename T>
class ElementwiseMulMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = framework::Tensor;

    int axis = ctx.Attr<int>("axis");
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* z_data = z->mutable_data<T>(ctx.GetPlace());

    auto x_dims = x->dims();
    auto y_dims_untrimmed = y->dims();
    auto x_int_dims = paddle::framework::vectorize<int64_t>(x_dims);

    int pre, num, post, is_run_common_broadcast;
    get_mid_dims(x_dims, y_dims_untrimmed, axis, &pre, &num, &post,
                 &is_run_common_broadcast);

    if (post == 1) PADDLE_THROW("Not implemented when post is 1");

    const int64_t n = x_dims[0];
    const int64_t c = x_dims[1];
    const int64_t h = x_dims[2];
    const int64_t w = x_dims[3];

    const int simd_width = 16;
    auto multiply =
        jit::KernelFuncs<jit::NCHW16CMulNCTuple<T>, platform::CPUPlace>::Cache()
            .At(0);
    ComputeBroadcastedMultiply(x_data, y_data, z_data, n, c, h, w, simd_width,
                               multiply);

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(x->format());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ElementwiseMulMKLDNNKernel<float>)
