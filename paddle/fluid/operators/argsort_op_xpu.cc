/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

const int XPU_SORT_MAX_SIZE = 16384;

template <typename T, typename TID>
static inline void xpu_argsort(xpu::Context* ctx, const T* input_data,
                               T* output_data, TID* indices_data, int m, int n,
                               bool descending) {
  int ret =
      xpu::sort(ctx, input_data, output_data, indices_data, m, n, descending);
  PADDLE_ENFORCE_EQ(
      ret, XPU_SUCCESS,
      platform::errors::External("XPU sort kernel return wrong value[%d %s].",
                                 ret, XPUAPIErrorMsg[ret]));
}

template <typename T>
static inline void xpu_transpose(xpu::Context* ctx, const T* x, T* y,
                                 const std::vector<int>& xshape,
                                 const std::vector<int>& permute) {
  int ret = xpu::transpose(ctx, x, y, xshape, permute);
  PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU transpose kernel return wrong value[%d %s]", ret,
                        XPUAPIErrorMsg[ret]));
}

template <typename TX, typename TY>
static inline void xpu_cast(xpu::Context* ctx, const TX* x, TY* y, int len) {
  int ret = xpu::cast_v2(ctx, x, y, len);
  PADDLE_ENFORCE_EQ(
      ret, XPU_SUCCESS,
      platform::errors::External("XPU cast kernel return wrong value[%d %s]",
                                 ret, XPUAPIErrorMsg[ret]));
}

template <typename T, bool VALUE_NEED_CAST = false,
          bool INDEX_NEED_CAST = false>
struct XPUArgsort {
  void operator()(xpu::Context* ctx, const T* input_data, T* output_data,
                  int64_t* indices_data, const std::vector<int>& data_shape,
                  const std::vector<int>& permute, bool descending) {
    xpu::ctx_guard RAII_GUARD(ctx);
    int m = data_shape[0] * data_shape[2];
    int n = data_shape[1];
    int len = data_shape[0] * data_shape[1] * data_shape[2];
    std::vector<int> trans_data_shape{data_shape[0], data_shape[2],
                                      data_shape[1]};

    T* input_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    T* output_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    int64_t* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);

    xpu_transpose(ctx, input_data, input_data_trans, data_shape, permute);
    xpu_argsort(ctx, input_data_trans, output_data_trans, indices_data_trans, m,
                n, descending);
    xpu_transpose(ctx, output_data_trans, output_data, trans_data_shape,
                  permute);
    xpu_transpose(ctx, indices_data_trans, indices_data, trans_data_shape,
                  permute);
  }
};

template <typename T>
struct XPUArgsort<T, false, true> {
  void operator()(xpu::Context* ctx, const T* input_data, T* output_data,
                  int64_t* indices_data, const std::vector<int>& data_shape,
                  const std::vector<int>& permute, bool descending) {
    xpu::ctx_guard RAII_GUARD(ctx);
    int m = data_shape[0] * data_shape[2];
    int n = data_shape[1];
    int len = data_shape[0] * data_shape[1] * data_shape[2];
    std::vector<int> trans_data_shape{data_shape[0], data_shape[2],
                                      data_shape[1]};

    T* input_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    T* output_data_trans = RAII_GUARD.alloc_l3_or_gm<T>(len);
    int* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int>(len);
    int64_t* cast_data_int64 = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);

    xpu_transpose(ctx, input_data, input_data_trans, data_shape, permute);
    xpu_argsort(ctx, input_data_trans, output_data_trans, indices_data_trans, m,
                n, descending);
    xpu_transpose(ctx, output_data_trans, output_data, trans_data_shape,
                  permute);
    xpu_cast(ctx, indices_data_trans, cast_data_int64, len);
    xpu_transpose(ctx, cast_data_int64, indices_data, trans_data_shape,
                  permute);
  }
};

template <>
struct XPUArgsort<int64_t, true, true> {
  void operator()(xpu::Context* ctx, const int64_t* input_data,
                  int64_t* output_data, int64_t* indices_data,
                  const std::vector<int>& data_shape,
                  const std::vector<int>& permute, bool descending) {
    xpu::ctx_guard RAII_GUARD(ctx);
    int m = data_shape[0] * data_shape[2];
    int n = data_shape[1];
    int len = data_shape[0] * data_shape[1] * data_shape[2];
    std::vector<int> trans_data_shape{data_shape[0], data_shape[2],
                                      data_shape[1]};

    int* input_data_trans = RAII_GUARD.alloc_l3_or_gm<int>(len);
    int* output_data_trans = RAII_GUARD.alloc_l3_or_gm<int>(len);
    int* indices_data_trans = RAII_GUARD.alloc_l3_or_gm<int>(len);
    int* cast_data_int = RAII_GUARD.alloc_l3_or_gm<int>(len);
    int64_t* cast_data_int64 = RAII_GUARD.alloc_l3_or_gm<int64_t>(len);

    xpu_cast(ctx, input_data, cast_data_int, len);
    xpu_transpose(ctx, cast_data_int, input_data_trans, data_shape, permute);
    xpu_argsort(ctx, input_data_trans, output_data_trans, indices_data_trans, m,
                n, descending);

    xpu_cast(ctx, output_data_trans, cast_data_int64, len);
    xpu_transpose(ctx, cast_data_int64, output_data, trans_data_shape, permute);
    xpu_cast(ctx, indices_data_trans, cast_data_int64, len);
    xpu_transpose(ctx, cast_data_int64, indices_data, trans_data_shape,
                  permute);
  }
};

template <typename T>
class ArgsortXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int axis = ctx.Attr<int>("axis");
    bool descending = ctx.Attr<bool>("descending");

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    int n = in_dims[axis];

    PADDLE_ENFORCE_LT(
        n, XPU_SORT_MAX_SIZE,
        platform::errors::InvalidArgument(
            "The axis dimension of Input should less than %d, but got %d.",
            XPU_SORT_MAX_SIZE, in_dims[axis]));

    auto input_data = input->data<T>();
    auto output_data = output->mutable_data<T>(ctx.GetPlace());
    auto indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::XPUDeviceContext>();
    int len_before = phi::product(phi::slice_ddim(in_dims, 0, axis));
    int len_after =
        phi::product(phi::slice_ddim(in_dims, axis + 1, in_dims.size()));
    bool int64_need_cast =
        (std::is_same<T, int64_t>::value && n > (XPU_SORT_MAX_SIZE / 2))
            ? true
            : false;
    bool index_need_cast = (n > (XPU_SORT_MAX_SIZE / 2)) ? true : false;
    std::vector<int> permute_vec{0, 2, 1};
    std::vector<int> data_shape{len_before, n, len_after};

    if (int64_need_cast) {
      XPUArgsort<T, true, true>()(dev_ctx.x_context(), input_data, output_data,
                                  indices_data, data_shape, permute_vec,
                                  descending);
    } else if (index_need_cast) {
      XPUArgsort<T, false, true>()(dev_ctx.x_context(), input_data, output_data,
                                   indices_data, data_shape, permute_vec,
                                   descending);
    } else {
      XPUArgsort<T, false, false>()(dev_ctx.x_context(), input_data,
                                    output_data, indices_data, data_shape,
                                    permute_vec, descending);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(argsort, ops::ArgsortXPUKernel<float>,
                       ops::ArgsortXPUKernel<int>,
                       ops::ArgsortXPUKernel<int64_t>);

#endif
