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
#include "paddle/fluid/operators/elementwise_op.h"
#include "paddle/fluid/operators/elementwise_op_function.h"

#include "paddle/fluid/platform/mkldnn_helper.h"

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;

struct vector_mul : public Xbyak::CodeGenerator {
  vector_mul() {
    // RDI is ptr X
    // RSI is ptr Y
    // RDX is ptr Z
    // RCX is h
    // r8 is w

    push(rbx);

    xor_(rax, rax);
    xor_(r10, r10);
    vmovups(zmm3, ptr[rsi]);

    L("h_loop");
    xor_(rbx, rbx);
    L("w_loop");
    vmovups(zmm2, ptr[rdi + rax]);
    vmulps(zmm1, zmm2, zmm3);
    vmovups(ptr[rdx + rax], zmm1);
    add(rax, 64);
    inc(rbx);
    cmp(r8, rbx);
    jnz("w_loop");
    inc(r10);
    cmp(r10, rcx);
    jnz("h_loop");

    pop(rbx);
    ret();
  }
};

void check(const float* x, const float* y, float* z, int w) {
  for (int wi = 0; wi < w; wi++) {
    for (int i = 0; i < 16; i++) {
      z[wi * 16 + i] = x[wi * 16 + i] * y[i];
    }
  }
}

static mkldnn::memory::format StringToMKLDNNFormat(std::string& format) {
  std::transform(format.begin(), format.end(), format.begin(), ::tolower);

  if(!format.compare("nchw")) {
    return memory::format::nchw;
  } else if(!format.compare("nchw16c")) {
    return memory::format::nChw16c;
  } else if(!format.compare("nchw8c")) {
    return memory::format::nChw8c;
  } else if(!format.compare("nhwc")) {
    return memory::format::nhwc;
  } else {
    return memory::format::any;
  }
}

static void UpdateDataFormat(const framework::ExecutionContext& ctx,
  framework::Tensor* tensor, const char* attribute) {
  if(ctx.op().HasAttr(attribute)) {
    auto format_as_string = ctx.Attr<std::string>(attribute);
    auto format = StringToMKLDNNFormat(format_as_string);
    if (format != memory::format::any) {
      tensor->set_format(format);
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

    UpdateDataFormat(ctx, (Tensor*)x, "x_data_format");
    UpdateDataFormat(ctx, (Tensor*)y, "y_data_format");

    if (x->format() == memory::format::nChw16c && y->format() == memory::format::nc) {
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

          vector_mul mul;

          using mul_func_t =
          void (*)(const float *, const float *, float *, int, int);

          mul_func_t mul_func = (mul_func_t) mul.getCode();

          #pragma omp parallel for collapse(2)
          for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < C; ci++) {
              auto ptr_x =
                      x_data + ni * C * h * w * simd_width +
                      ci * h * w * simd_width;

              auto ptr_y = y_data + ni * C * simd_width + ci * simd_width;
              auto ptr_z =
                      z_data + ni * C * h * w * simd_width +
                      ci * h * w * simd_width;

              mul_func(ptr_x, ptr_y, ptr_z, h, w);
            }
          }
        }

        z->set_layout(DataLayout::kMKLDNN);
        z->set_format(x->format());
      } else {
        PADDLE_THROW("Not implemented when dims are equal");
      }
    } else {
      // Fallback to naive version:
      auto mul_func = [](T a, T b) -> T { return a * b; };

      TransformFunctor<decltype(mul_func), T,
                       paddle::platform::CPUDeviceContext, T>
          functor(
              x, y, z,
              ctx.template device_context<paddle::platform::CPUDeviceContext>(),
              mul_func);

      axis = (axis == -1 ? x_dims.size() - y_dims_untrimmed.size() : axis);
      PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                     "Axis should be in range [0, x_dims)");

      auto y_dims = trim_trailing_singular_dims(y_dims_untrimmed);
      axis = (y_dims.size() == 0) ? x_dims.size() : axis;

      int pre, n, post;
      get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

      if (post == 1) {
        functor.RunRowWise(n, pre);
      } else {
        functor.RunMidWise(n, pre, post);
      }
      z->set_layout(DataLayout::kMKLDNN);
      z->set_format(x->format());
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ElementwiseMulMKLDNNKernel<float>)
