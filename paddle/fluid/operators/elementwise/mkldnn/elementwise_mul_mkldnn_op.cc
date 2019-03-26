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

static void UpdateDataFormat(const framework::ExecutionContext& ctx,
                             framework::Tensor* tensor, const char* attribute) {
  if (ctx.op().HasAttr(attribute)) {
    auto format_as_string = ctx.Attr<std::string>(attribute);
    auto format = StringToMKLDNNFormat(&format_as_string);
    if (format != memory::format::any) {
      tensor->set_format(format);
    }
  }
}

template <typename T>
static void ReorderInput(framework::Tensor* tensor,
                         const platform::Place& place,
                         const mkldnn::engine& engine, bool isFourDim) {
  using platform::to_void_cast;
  auto dims = paddle::framework::vectorize2int(tensor->dims());
  framework::Tensor out_tensor;
  out_tensor.Resize(tensor->dims());
  out_tensor.set_format(isFourDim ? memory::format::nchw : memory::format::nc);
  out_tensor.set_layout(tensor->layout());
  mkldnn::memory input_memory = {
      {{dims, platform::MKLDNNGetDataType<T>(), tensor->format()}, engine},
      to_void_cast<T>(tensor->data<T>())};
  mkldnn::memory output_memory = {
      {{dims, platform::MKLDNNGetDataType<T>(), out_tensor.format()}, engine},
      to_void_cast<T>(out_tensor.mutable_data<T>(place))};
  platform::Reorder(input_memory, output_memory);
  tensor->ShareDataWith(out_tensor);
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
    auto x_int_dims = paddle::framework::vectorize2int(x_dims);

    UpdateDataFormat(ctx, const_cast<Tensor*>(x), "x_data_format");
    UpdateDataFormat(ctx, const_cast<Tensor*>(y), "y_data_format");

    const bool is_avx512_enabled = platform::MayIUse(platform::avx512f);
    const bool are_dims_divisable = !(x_int_dims[1] % 16);
    const bool is_x_format_correct = x->format() == memory::format::nChw16c;
    const bool is_y_format_correct = y->format() == memory::format::nc;
    if (is_x_format_correct && is_y_format_correct && are_dims_divisable &&
        is_avx512_enabled) {
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

        auto multiply = jit::KernelFuncs<jit::NCHW16CMulNCTuple<T>,
                                         platform::CPUPlace>::Cache()
                            .At(0);
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

      z->set_layout(DataLayout::kMKLDNN);
      z->set_format(x->format());
    } else {
      // Fallback to naive version:
      const bool are_inputs_in_same_format = x->format() == y->format();
      const bool is_x_nchw = x->format() == memory::format::nchw;
      const bool is_x_nc = x->format() == memory::format::nc;
      const bool is_x_x = x->format() == memory::format::x;
      const bool is_y_nchw = y->format() == memory::format::nchw;
      const bool is_y_nc = y->format() == memory::format::nc;
      const bool is_y_x = y->format() == memory::format::x;
      if (!are_inputs_in_same_format) {
        using platform::MKLDNNDeviceContext;
        auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
        const auto& mkldnn_engine = dev_ctx.GetEngine();
        if (!(is_x_nchw || is_x_nc || is_x_x))
          ReorderInput<T>(const_cast<Tensor*>(x), ctx.GetPlace(), mkldnn_engine,
                          x->dims().size() == 4);
        if (!(is_y_nchw || is_y_nc || is_y_x))
          ReorderInput<T>(const_cast<Tensor*>(y), ctx.GetPlace(), mkldnn_engine,
                          y->dims().size() == 4);
      }

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
