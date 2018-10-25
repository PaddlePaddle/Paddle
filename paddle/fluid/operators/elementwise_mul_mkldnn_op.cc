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

#include "paddle/fluid/operators/elementwise_op.h"
#include "paddle/fluid/operators/elementwise_op_function.h"

#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using framework::DataLayout;

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

    if (x_dims != y_dims_untrimmed) {
      int pre, n, post;
      get_mid_dims(x_dims, y_dims_untrimmed, axis, &pre, &n, &post);

      if (post == 1) {
        PADDLE_THROW("Not implemented when post is 1");
      } else {
        // Just check whether it works for RE-Resnext.

        PADDLE_ENFORCE_EQ(x_dims.size(), 4, "X should have 4 dimensions");

        int n = x_dims[0];
        int c = x_dims[1];
        int h = x_dims[2];
        int w = x_dims[3];

        PADDLE_ENFORCE(y_dims_untrimmed[0] == n && y_dims_untrimmed[1] == c,
                       "Y should be in nc format");

        constexpr int simd_width = 16;
        int C = c / simd_width;

        for (int ni = 0; ni < n; ni++) {
          for (int ci = 0; ci < C; ci++) {
            for (int hi = 0; hi < h; hi++) {
              for (int wi = 0; wi < w; wi++) {
                auto ptr_x = x_data + ni * C * h * w * simd_width +
                             ci * h * w * simd_width + hi * w * simd_width +
                             wi * simd_width;
                auto ptr_y = y_data + ni * C * simd_width + ci * simd_width;

                auto ptr_z = z_data + ni * C * h * w * simd_width +
                             ci * h * w * simd_width + hi * w * simd_width +
                             wi * simd_width;

                for (int i = 0; i < simd_width; i++) {
                  ptr_z[i] = ptr_x[i] * ptr_y[i];
                }
              }
            }
          }
        }
      }

      z->set_layout(DataLayout::kMKLDNN);
      z->set_format(x->format());
    } else {
      PADDLE_THROW("Not implemented when dims are equal");
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ElementwiseMulMKLDNNKernel<float>)
