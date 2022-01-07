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

#include "paddle/fluid/operators/mkldnn/matmul_mkldnn_op.h"

namespace {

using dnnl::memory;
using dnnl::primitive;
using paddle::framework::DataLayout;
using paddle::framework::ExecutionContext;
using paddle::platform::GetMKLDNNFormat;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::to_void_cast;
using Tensor = paddle::framework::Tensor;
using paddle::framework::DDim;
using paddle::framework::GradVarName;
using paddle::framework::make_ddim;
using paddle::framework::vectorize;

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static DDim RowMatrixDimsFromVector(const DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : paddle::framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static DDim ColumnMatrixDimsFromVector(const DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : paddle::framework::make_ddim({y_dim[0], 1});
}

static std::vector<int64_t> Transpose(const std::vector<int64_t>& x,
                                      const std::vector<int>& axis) {
  size_t in_rank = x.size();
  size_t axis_size = axis.size();

  auto axis_set = std::set<int>(axis.begin(), axis.end());
  PADDLE_ENFORCE_EQ(axis_set.size(), axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "In an axis array, elements must be unique."));

  PADDLE_ENFORCE_EQ(in_rank, axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "The input dimension's size "
                        "should be equal to the axis's size. "
                        "But received dimension is %d, "
                        "axis's size is %d",
                        in_rank, axis_size));

  PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()), axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "Axis values must be ranging from 0 to (dims - 1)."));

  std::vector<int64_t> new_x(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    new_x[i] = x[axis[i]];
  }
  return new_x;
}

std::vector<int64_t> GetInputStrides(const ExecutionContext& ctx,
                                     const std::string input_name) {
  auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
  auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
  auto input_dims = ctx.Input<Tensor>(input_name)->dims();
  auto new_dims = input_dims;
  if (!shape.empty() && !axis.empty()) {
    new_dims = input_dims.reshape(shape).transpose(axis);
  }

  auto& MatrixDimsFromVector =
      input_name == "X" ? RowMatrixDimsFromVector : ColumnMatrixDimsFromVector;
  paddle::operators::math::MatDescriptor mat_dim =
      paddle::operators::math::CreateMatrixDescriptor(
          MatrixDimsFromVector(new_dims), 0,
          ctx.Attr<bool>(std::string("trans_") +
                         static_cast<char>(std::tolower(input_name[0]))));

  std::vector<int64_t> strides;
  if (!shape.empty()) {
    auto shape2 = input_dims.reshape(shape);
    strides.push_back(1);
    for (auto i = shape2.size() - 1; i > 0; --i) {
      strides.insert(strides.begin(),
                     strides.front() * static_cast<int64_t>(shape2[i]));
    }
    strides = Transpose(strides, axis);
    if (shape.size() == 2)
      strides.insert(strides.begin(),
                     static_cast<int64_t>(shape[0] * shape[1]));
    mat_dim.stride_ = strides[0];
    if (mat_dim.trans_) std::swap(*strides.rbegin(), *(++strides.rbegin()));
  }
  return strides;
}

template <typename T>
class MatMulV2MKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::matmul> {
 public:
  MatMulV2MKLDNNHandler(const dnnl::engine engine,
                        paddle::platform::Place cpu_place,
                        const std::vector<int64_t>& x_org_dims, bool trans_x,
                        const std::vector<int64_t>& y_org_dims, bool trans_y,
                        bool is_output_fused,
                        const std::vector<int64_t>& x_strides_override,
                        const std::vector<int64_t>& y_strides_override)
      : paddle::platform::MKLDNNHandlerNoCachingT<T, dnnl::matmul>(engine,
                                                                   cpu_place) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = x_dims.size() - 3;
    const int H_idx = x_dims.size() - 2;
    const int W_idx = x_dims.size() - 1;

    if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
    if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

    const memory::dim M = x_dims[H_idx];
    const memory::dim K = x_dims[W_idx];
    const memory::dim N = y_dims[W_idx];

    std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
    std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

    x_strides.reserve(x_dims.size());
    y_strides.reserve(x_dims.size());
    out_strides.reserve(x_dims.size());

    if (!x_strides_override.empty()) {
      x_strides = x_strides_override;
    } else {
      if (!trans_x) {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      } else {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      }
    }

    if (!y_strides_override.empty()) {
      y_strides = y_strides_override;
    } else {
      if (!trans_y) {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      } else {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      }
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = x_dims.size() - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      if (x_strides_override.empty()) {
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      }
      if (y_strides_override.empty()) {
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
      }
      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    if (is_output_fused) {
      out_strides = FakeTransposeStrides(out_ddims);
    }

    auto x_md = memory::desc(x_dims, MKLDNNGetDataType<T>(), x_strides);
    auto y_md = memory::desc(y_dims, MKLDNNGetDataType<T>(), y_strides);
    auto out_md = memory::desc(out_ddims, MKLDNNGetDataType<T>(), out_strides);

    this->AcquireForwardPrimitiveDescriptor(x_md, y_md, out_md);
  }

  std::vector<int64_t> FakeTransposeStrides(
      const std::vector<int64_t>& matmul_out_dims) const {
    // fuse matmul_v2 + transpose + reshape guarantees that output is 4D and
    // transpose axis are: {0, 2, 1, 3}
    std::vector<int64_t> transpose_axis = {0, 2, 1, 3};
    std::vector<int64_t> fake_strides(transpose_axis.size());
    int ndims = static_cast<int>(transpose_axis.size());

    int total_stride = 1;

    for (int i = ndims - 1; i >= 0; --i) {
      fake_strides[transpose_axis[i]] = total_stride;
      total_stride *= matmul_out_dims[transpose_axis[i]];
    }

    return fake_strides;
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<T>(input_data));
  }
};

bool IsOutputFused(const ExecutionContext& ctx) {
  auto& fused_reshape_Out = ctx.Attr<std::vector<int>>("fused_reshape_Out");
  auto& fused_transpose_Out = ctx.Attr<std::vector<int>>("fused_transpose_Out");
  return !fused_reshape_Out.empty() && !fused_transpose_Out.empty();
}

float ComputeOutputScale(const ExecutionContext& ctx) {
  float scale_x = ctx.Attr<float>("Scale_x");
  float scale_y = ctx.Attr<float>("Scale_y");
  bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
  float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
  return scale_out / (scale_x * scale_y);
}

template <typename T>
void ExecuteMatMulV2(const ExecutionContext& ctx,
                     const MKLDNNDeviceContext& dev_ctx,
                     const dnnl::engine onednn_engine,
                     paddle::platform::Place cpu_place, const Tensor* x,
                     std::vector<int64_t>& x_dims, bool trans_x,
                     const Tensor* y, std::vector<int64_t>& y_dims,
                     bool trans_y, Tensor* out, std::vector<int64_t>& out_dims,
                     int execution_number = 0) {
  std::vector<int64_t> x_strides_override = GetInputStrides(ctx, "X");
  std::vector<int64_t> y_strides_override = GetInputStrides(ctx, "Y");
  MatMulV2MKLDNNHandler<T> handler(onednn_engine, ctx.GetPlace(), x_dims,
                                   trans_x, y_dims, trans_y, IsOutputFused(ctx),
                                   x_strides_override, y_strides_override);

  const auto src_memory_p = handler.AcquireSrcMemory(x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(y);
  const auto dst_memory_p = handler.AcquireDstMemory(out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto& astream = MKLDNNDeviceContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  auto format = paddle::platform::MKLDNNFormatForSize(
      out->dims().size(), dnnl::memory::format_tag::nchw);
  out->set_layout(paddle::framework::DataLayout::kMKLDNN);
  out->set_format(format);
}

DDim GetDimForInput(const paddle::framework::ExecutionContext& ctx,
                    const std::string& input_name) {
  auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
  auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.Input<paddle::framework::Tensor>(input_name)->dims();
  if (!shape.empty() && !axis.empty()) {
    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}

template <typename T>
class MatMulV2MKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 private:
  void CalculateMatrixDims(const ExecutionContext& ctx,
                           const std::vector<int64_t>& x_dims,
                           const std::vector<int64_t>& y_dims,
                           std::vector<int64_t>& x_bd_dims,
                           std::vector<int64_t>& y_bd_dims,
                           std::vector<int64_t>& out_dims, Tensor* out) const {
    if (x_dims.size() == 1) {
      x_bd_dims[x_bd_dims.size() - 1] = x_dims[0];
    } else if (x_dims.size() == 2) {
      x_bd_dims[x_bd_dims.size() - 1] = x_dims[1];
      x_bd_dims[x_bd_dims.size() - 2] = x_dims[0];
    } else {
      for (size_t i = 0; i < x_dims.size(); ++i) {
        x_bd_dims[x_bd_dims.size() - x_dims.size() + i] = x_dims[i];
      }
    }
    if (y_dims.size() == 1) {
      y_bd_dims[x_bd_dims.size() - 2] = y_dims[0];
    } else if (y_dims.size() == 2) {
      y_bd_dims[y_bd_dims.size() - 1] = y_dims[1];
      y_bd_dims[y_bd_dims.size() - 2] = y_dims[0];
    } else {
      for (size_t i = 0; i < y_dims.size(); ++i) {
        y_bd_dims[y_bd_dims.size() - y_dims.size() + i] = y_dims[i];
      }
    }

    if (!IsOutputFused(ctx) && x_dims.size() > 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < x_bd_dims.size() - 2; ++i) {
        PADDLE_ENFORCE_EQ(
            x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] == 1 ||
                y_bd_dims[i] == 1,
            true, paddle::platform::errors::InvalidArgument(
                      "Tensor dimensions are incorrect for broadcasting."
                      "Dimensions in X and Y must be same or equal to 1, but "
                      "received x_dim[%d]=%d and y_dims[%d]= %d",
                      i, x_bd_dims[i], i, y_bd_dims[i]));
        out_dims[i] = std::max(x_bd_dims[i], y_bd_dims[i]);
      }
      out->Resize(make_ddim(out_dims));
    }
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = vectorize(GetDimForInput(ctx, "X"));
    auto y_dims = vectorize(GetDimForInput(ctx, "Y"));
    auto out_dims = vectorize(out->dims());

    int ndims = std::max(x_dims.size(), y_dims.size());
    ndims = std::max(ndims, 3);

    std::vector<int64_t> x_bd_dims(ndims, 1);
    std::vector<int64_t> y_bd_dims(ndims, 1);

    CalculateMatrixDims(ctx, x_dims, y_dims, x_bd_dims, y_bd_dims, out_dims,
                        out);

    ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x,
                       x_bd_dims, trans_x, y, y_bd_dims, trans_y, out,
                       out_dims);
  }
};

template <typename T>
class MatMulV2GradMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 private:
  void CalculateGradMatrixDims(const ExecutionContext& ctx, Tensor* dx_tmp,
                               Tensor* dy_tmp,
                               const std::vector<int64_t>& dx_dims,
                               const std::vector<int64_t>& dy_dims,
                               std::vector<int64_t>& dx_bd_dims,
                               std::vector<int64_t>& dy_bd_dims) const {
    for (size_t i = 0; i < dx_dims.size() - 2; ++i) {
      if (dx_dims[i] != dy_dims[i]) {
        if (dx_dims[i] == 1) {
          dx_bd_dims[i] = dy_dims[i];
        } else {
          dy_bd_dims[i] = dx_dims[i];
        }
      }
    }

    dx_tmp->Resize(make_ddim(dx_bd_dims));
    dx_tmp->mutable_data<T>(ctx.GetPlace());
    dy_tmp->Resize(make_ddim(dy_bd_dims));
    dy_tmp->mutable_data<T>(ctx.GetPlace());
  }

  void ReduceSumForMatmulGradOutput(
      const ExecutionContext& ctx, const MKLDNNDeviceContext& dev_ctx,
      const dnnl::engine onednn_engine, const Tensor* dx_tmp, Tensor* dx,
      std::vector<int64_t>& dx_dims,
      const std::vector<int64_t>& squeezed_dims) const {
    paddle::platform::ReductionMKLDNNHandler<T> handler(
        dnnl::algorithm::reduction_sum, 0.0f, 0.0f, onednn_engine,
        ctx.GetPlace(), dx_tmp, dx, dx_dims);

    auto src_memory_p = handler.AcquireSrcMemory(dx_tmp);
    auto dst_memory_p = handler.AcquireDstMemory(dx);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    auto reduction_p = handler.AcquireForwardPrimitive();

    reduction_p->execute(astream, reduction_args);
    astream.wait();

    dx->set_format(paddle::platform::GetMKLDNNFormat(
        dst_memory_p->get_desc().reshape(squeezed_dims)));
  }

  std::vector<int64_t> ExtendDimsWithOnes(const std::vector<int64_t>& dims,
                                          int new_size) const {
    std::vector<int64_t> new_dims(new_size, 1);
    for (size_t i = 0; i < dims.size(); ++i) {
      new_dims[new_size - dims.size() + i] = dims[i];
    }

    return new_dims;
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto x_dims = vectorize(x->dims());
    auto y_dims = vectorize(y->dims());

    bool is_broadcast = true;
    if (x_dims.size() <= 2 || y_dims.size() <= 2) {
      is_broadcast = false;
    } else if (x_dims.size() != y_dims.size()) {
      is_broadcast = true;
    } else {
      is_broadcast =
          !std::equal(x_dims.cbegin(), x_dims.cbegin() + x_dims.size() - 2,
                      y_dims.cbegin());
    }

    // if no broadcasting is needed, we can simply use matmul's grad and avoid
    // using reduce_sum
    if (!is_broadcast) {
      matmul_v1_grad_mkldnn_kernel.Compute(ctx);
      return;
    }

    auto* dout = ctx.Input<Tensor>(GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(GradVarName("Y"));

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    auto dout_dims = vectorize(dout->dims());

    size_t ndims = std::max(x->dims().size(), y->dims().size());
    ndims = std::max<size_t>(ndims, 3);

    if (x_dims.size() != ndims) {
      x_dims = ExtendDimsWithOnes(x_dims, ndims);
    } else if (y_dims.size() != ndims) {
      y_dims = ExtendDimsWithOnes(y_dims, ndims);
    }

    // in broadcasting scenario new memory is required because
    // reduce sum must be calculated upon broadcasted dims
    Tensor dx_tmp, dy_tmp;

    std::vector<int64_t> dx_bd_dims(x_dims);
    std::vector<int64_t> dy_bd_dims(y_dims);

    CalculateGradMatrixDims(ctx, &dx_tmp, &dy_tmp, x_dims, y_dims, dx_bd_dims,
                            dy_bd_dims);

    if (trans_x && trans_y) {
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), y, y_dims,
                         true, dout, dout_dims, true, &dx_tmp, dx_bd_dims, 1);
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                         dout_dims, true, x, x_dims, true, &dy_tmp, dy_bd_dims,
                         2);
    } else if (trans_x) {
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), y, y_dims,
                         false, dout, dout_dims, true, &dx_tmp, dx_bd_dims, 1);
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x, x_dims,
                         false, dout, dout_dims, false, &dy_tmp, dy_bd_dims, 2);
    } else if (trans_y) {
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                         dout_dims, false, y, y_dims, false, &dx_tmp,
                         dx_bd_dims, 1);
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                         dout_dims, true, x, x_dims, false, &dy_tmp, dy_bd_dims,
                         2);
    } else {
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                         dout_dims, false, y, y_dims, true, &dx_tmp, dx_bd_dims,
                         1);
      ExecuteMatMulV2<T>(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x, x_dims,
                         true, dout, dout_dims, false, &dy_tmp, dy_bd_dims, 2);
    }

    if (x_dims != dx_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx, dev_ctx, onednn_engine, &dx_tmp, dx,
                                   x_dims,
                                   paddle::framework::vectorize(x->dims()));
    } else {
      *dx = std::move(dx_tmp);
    }
    if (y_dims != dy_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx, dev_ctx, onednn_engine, &dy_tmp, dy,
                                   y_dims,
                                   paddle::framework::vectorize(y->dims()));
    } else {
      *dy = std::move(dy_tmp);
    }

    dx->Resize(x->dims());
    dy->Resize(y->dims());
  }

 private:
  paddle::operators::MatMulGradMKLDNNKernel<T> matmul_v1_grad_mkldnn_kernel;
};
}  // anonymous namespace

REGISTER_OP_KERNEL(matmul_v2, MKLDNN, ::paddle::platform::CPUPlace,
                   MatMulV2MKLDNNKernel<float>,
                   MatMulV2MKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(matmul_v2_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   MatMulV2GradMKLDNNKernel<float>,
                   MatMulV2GradMKLDNNKernel<paddle::platform::bfloat16>);
