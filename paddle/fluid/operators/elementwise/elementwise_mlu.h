// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_MLU
#include <vector>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

inline void GetReduceAxes(const int axis,
                          const framework::DDim& src_ddims,
                          const framework::DDim& target_ddims,
                          std::vector<int>* axes) {
  int64_t src_dim_size = src_ddims.size();
  int64_t target_dim_size = target_ddims.size();
  for (int64_t i = 0; i < src_dim_size; ++i) {
    if (i < axis || i >= target_dim_size + axis) {
      axes->push_back(i);
      continue;
    }
    if (src_ddims[i] > target_ddims[i - axis]) {
      axes->push_back(i);
    }
  }
}

inline void GetReduceAxesAndDstDims(const int axis,
                                    const framework::DDim& src_ddims,
                                    const framework::DDim& target_ddims,
                                    std::vector<int>* reduce_axes,
                                    std::vector<int>* dst_dims_vec) {
  int64_t src_dim_size = src_ddims.size();
  int64_t target_dim_size = target_ddims.size();

  int src_axis = (target_dim_size < src_dim_size ? axis : 0);
  for (int ax = 0; ax < src_dim_size; ++ax) {
    if ((ax < src_axis || ax >= src_axis + target_dim_size) ||
        (src_ddims[ax] > 1 && target_ddims[ax - src_axis] == 1)) {
      reduce_axes->push_back(ax);
    } else {
      dst_dims_vec->push_back(src_ddims[ax]);
    }
  }
  if (dst_dims_vec->size() == 0) {
    // target_var is scalar
    dst_dims_vec->push_back(1);
  }
}

template <typename T>
void MLUOpTensorKernel(const framework::ExecutionContext& ctx,
                       const cnnlOpTensorDesc_t op_tensor_op) {
  PADDLE_ENFORCE_EQ(
      platform::is_mlu_place(ctx.GetPlace()),
      true,
      platform::errors::Unavailable("This kernel only runs on MLU."));
  PADDLE_ENFORCE_EQ((op_tensor_op == CNNL_OP_TENSOR_ADD) ||
                        (op_tensor_op == CNNL_OP_TENSOR_SUB) ||
                        (op_tensor_op == CNNL_OP_TENSOR_MUL),
                    true,
                    platform::errors::Unavailable(
                        "This kernel of MLU only support ADD, SUB, MUL."));

  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* out = ctx.Output<phi::DenseTensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());

  int axis = ctx.Attr<int>("axis");
  const auto& x_dims = x->dims();
  const auto& y_dims = y->dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnlOpTensorDesc op_tensor_desc(
      op_tensor_op, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  MLUCnnl::OpTensor(ctx,
                    op_tensor_desc.get(),
                    x_desc.get(),
                    GetBasePtr(x),
                    y_desc.get(),
                    GetBasePtr(y),
                    out_desc.get(),
                    GetBasePtr(out),
                    ToCnnlDataType<T>());
}

// ------------------ BinaryOp -----------------
enum BINARY_FUNCTOR {
  DIV,
  DIVNONAN,
  MAXIMUM,
  MINIMUM,
  POW,
};

template <BINARY_FUNCTOR func>
void MLUBinary(const framework::ExecutionContext& ctx,
               cnnlComputationPreference_t prefer,
               const cnnlTensorDescriptor_t x_desc,
               const void* x,
               const cnnlTensorDescriptor_t y_desc,
               const void* y,
               const cnnlTensorDescriptor_t out_desc,
               void* out);

template <>
inline void MLUBinary<DIV>(const framework::ExecutionContext& ctx,
                           cnnlComputationPreference_t prefer,
                           const cnnlTensorDescriptor_t x_desc,
                           const void* x,
                           const cnnlTensorDescriptor_t y_desc,
                           const void* y,
                           const cnnlTensorDescriptor_t out_desc,
                           void* out) {
  MLUCnnl::Div(ctx, prefer, x_desc, x, y_desc, y, out_desc, out);
}

template <>
inline void MLUBinary<MAXIMUM>(
    const framework::ExecutionContext& ctx,
    cnnlComputationPreference_t prefer,  // useless, only for compatible
    const cnnlTensorDescriptor_t x_desc,
    const void* x,
    const cnnlTensorDescriptor_t y_desc,
    const void* y,
    const cnnlTensorDescriptor_t out_desc,
    void* out) {
  MLUCnnl::Maximum(ctx, x_desc, x, y_desc, y, out_desc, out);
}

template <>
inline void MLUBinary<MINIMUM>(const framework::ExecutionContext& ctx,
                               cnnlComputationPreference_t prefer,
                               const cnnlTensorDescriptor_t in1_desc,
                               const void* in1,
                               const cnnlTensorDescriptor_t in2_desc,
                               const void* in2,
                               const cnnlTensorDescriptor_t out_desc,
                               void* out) {
  MLUCnnl::Minimum(ctx, in1_desc, in1, in2_desc, in2, out_desc, out);
}

template <>
inline void MLUBinary<POW>(const framework::ExecutionContext& ctx,
                           cnnlComputationPreference_t prefer,
                           const cnnlTensorDescriptor_t x_desc,
                           const void* x,
                           const cnnlTensorDescriptor_t y_desc,
                           const void* y,
                           const cnnlTensorDescriptor_t out_desc,
                           void* out) {
  MLUCnnl::Pow(ctx, prefer, x_desc, x, y_desc, y, out_desc, out);
}

template <BINARY_FUNCTOR Functor, typename T>
void MLUBinaryOp(const framework::ExecutionContext& ctx) {
  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* out = ctx.Output<phi::DenseTensor>("Out");
  out->mutable_data<T>(ctx.GetPlace());

  int axis = ctx.Attr<int>("axis");
  const auto& x_dims = x->dims();
  const auto& y_dims = y->dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<T>());

  cnnlComputationPreference_t prefer_type = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUBinary<Functor>(ctx,
                     prefer_type,
                     x_desc.get(),
                     GetBasePtr(x),
                     y_desc.get(),
                     GetBasePtr(y),
                     out_desc.get(),
                     GetBasePtr(out));
}

// ------------------ UnaryOp -----------------
enum UNARY_FUNCTOR {
  NEG,
  RECIPROCAL,
};

template <UNARY_FUNCTOR func>
void MLUUnary(const framework::ExecutionContext& ctx,
              cnnlComputationPreference_t prefer,
              const cnnlTensorDescriptor_t input_desc,
              const void* input,
              const cnnlTensorDescriptor_t output_desc,
              void* output);

template <>
inline void MLUUnary<NEG>(const framework::ExecutionContext& ctx,
                          cnnlComputationPreference_t prefer,
                          const cnnlTensorDescriptor_t input_desc,
                          const void* input,
                          const cnnlTensorDescriptor_t output_desc,
                          void* output) {
  MLUCnnl::Neg(ctx, input_desc, input, output_desc, output);
}

template <>
inline void MLUUnary<RECIPROCAL>(const framework::ExecutionContext& ctx,
                                 cnnlComputationPreference_t prefer,
                                 const cnnlTensorDescriptor_t input_desc,
                                 const void* input,
                                 const cnnlTensorDescriptor_t output_desc,
                                 void* output) {
  MLUCnnl::Reciprocal(ctx, input_desc, input, output_desc, output);
}

template <UNARY_FUNCTOR Functor, typename Tin, typename Tout = Tin>
void MLUUnaryOp(const framework::ExecutionContext& ctx) {
  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* out = ctx.Output<phi::DenseTensor>("Out");

  out->mutable_data<Tout>(ctx.GetPlace());

  MLUCnnlTensorDesc x_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType<Tin>());
  MLUCnnlTensorDesc out_desc(*out, CNNL_LAYOUT_ARRAY, ToCnnlDataType<Tout>());

  cnnlComputationPreference_t prefer_type = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUUnary<Functor>(ctx,
                    prefer_type,
                    x_desc.get(),
                    GetBasePtr(x),
                    out_desc.get(),
                    GetBasePtr(out));
}

// ------------------ MLUElementwiseGradOp -----------------
enum MINMAX_GRAD_FUNCTOR {
  MAXIMUM_GRAD,
  MINIMUM_GRAD,
};
template <MINMAX_GRAD_FUNCTOR Functor, typename Tin, typename Tout = Tin>
void MLUMinMaxGradHelper(const framework::ExecutionContext& ctx) {
  auto* x = ctx.Input<phi::DenseTensor>("X");
  auto* y = ctx.Input<phi::DenseTensor>("Y");
  auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
  auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
  int axis = ctx.Attr<int>("axis");

  const auto& x_dims = x->dims();
  const auto& y_dims = y->dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);

  // mask = Logic(x, y) only support min & max
  cnnlLogicOp_t logic =
      Functor == MAXIMUM_GRAD ? CNNL_LOGIC_OP_GE : CNNL_LOGIC_OP_LE;
  Tensor mask(x->dtype());
  mask.Resize(phi::make_ddim(out_dims_array));
  mask.mutable_data<Tin>(ctx.GetPlace());

  cnnlDataType_t data_type = ToCnnlDataType<Tin>();
  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), data_type);
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), data_type);
  MLUCnnlTensorDesc mask_desc(max_dim, out_dims_array.data(), data_type);
  MLUCnnl::Logic(ctx,
                 logic,
                 x_desc.get(),
                 GetBasePtr(x),
                 y_desc.get(),
                 GetBasePtr(y),
                 mask_desc.get(),
                 GetBasePtr(&mask));

  // dx = Mul(dz, mask)
  Tensor dx_temp(x->dtype());
  dx_temp.Resize(dout->dims());
  dx_temp.mutable_data<Tout>(ctx.GetPlace());
  MLUCnnlTensorDesc dout_desc(*dout);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(ctx,
                    mul_op_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(dout),
                    dout_desc.get(),
                    GetBasePtr(&mask),
                    dout_desc.get(),
                    GetBasePtr(&dx_temp),
                    data_type);

  // dy = Sub(dz, dx)
  Tensor dy_temp(y->dtype());
  dy_temp.Resize(dout->dims());
  dy_temp.mutable_data<Tout>(ctx.GetPlace());
  MLUCnnlOpTensorDesc sub_op_desc(
      CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(ctx,
                    sub_op_desc.get(),
                    dout_desc.get(),
                    GetBasePtr(dout),
                    dout_desc.get(),
                    GetBasePtr(&dx_temp),
                    dout_desc.get(),
                    GetBasePtr(&dy_temp),
                    data_type);

  if (dx) {
    if (dx->dims() != dout->dims()) {
      dx->mutable_data<Tout>(ctx.GetPlace());
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dx_temp.dims(), dx->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       data_type,
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dx_desc(*dx);
      MLUCnnl::Reduce(ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dx_temp),
                      0,
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
    } else {
      dx->ShareDataWith(dx_temp);
    }
  }

  if (dy) {
    if (dy->dims() != dout->dims()) {
      dy->mutable_data<Tout>(ctx.GetPlace());
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dy_temp.dims(), dy->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       data_type,
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dy_desc(*dy);
      MLUCnnl::Reduce(ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dy_temp),
                      0,
                      nullptr,
                      nullptr,
                      dy_desc.get(),
                      GetBasePtr(dy));
    } else {
      dy->ShareDataWith(dy_temp);
    }
  }
}

}  // namespace operators
}  // namespace paddle
#endif
