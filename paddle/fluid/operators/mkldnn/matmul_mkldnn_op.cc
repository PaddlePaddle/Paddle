/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "dnnl.hpp"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using platform::MKLDNNDeviceContext;
using framework::ExecutionContext;

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static framework::DDim RowMatrixFromVector(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static framework::DDim ColumnMatrixFromVector(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

template <typename XT, typename YT, typename OT>
class MatMulFactory {
 public:
  void CreateAndExecute(const ExecutionContext& ctx) {
    SetDNNLEngine(ctx);
    if (IsInitialized()) {
      UpdateDataPointers(ctx);
      Execute();
      SetOutputFormat(ctx);
      return;
    }
    CreateMemories(ctx);
    CreatePrimitive(ctx);
    Execute();
    SetOutputFormat(ctx);
    SetInitialized();
  }

 private:
  struct MatMulDims {
    const memory::dim BS, M, N, K;
  };

  void SetDNNLEngine(const ExecutionContext& ctx) {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    engine_ = dev_ctx.GetEngine();
  }

  template <typename T>
  dnnl::memory CreateMemory(const memory::dims& dims,
                            const memory::dims& strides, const T* data) {
    auto md = memory::desc(dims, MKLDNNGetDataType<T>(), strides);
    return dnnl::memory(md, engine_, to_void_cast(data));
  }

  MatMulDims GetMatmulDims(const ExecutionContext& ctx) {
    auto dim_x = math::CreateMatrixDescriptor(
        RowMatrixFromVector(ctx.Input<Tensor>("X")->dims()), 0,
        ctx.Attr<bool>("transpose_X"));
    auto dim_y = math::CreateMatrixDescriptor(
        ColumnMatrixFromVector(ctx.Input<Tensor>("Y")->dims()), 0,
        ctx.Attr<bool>("transpose_Y"));

    const auto x_bs = dim_x.batch_size_;
    const auto y_bs = dim_y.batch_size_;
    PADDLE_ENFORCE_EQ(x_bs > 0 && y_bs > 0 && x_bs != y_bs, false,
                      platform::errors::InvalidArgument(
                          "If batch sizes of X and Y are positive,"
                          "they have to be equal."));

    // Store 1 if both batches are zero, otherwise save the nonzero batch
    const memory::dim BS = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = dim_x.height_;
    const memory::dim N = dim_y.width_;
    const memory::dim K = dim_x.width_;
    return {BS, M, N, K};
  }

  void CreateMemories(const ExecutionContext& ctx) {
    auto matmul_dims = GetMatmulDims(ctx);
    auto BS = matmul_dims.BS;
    auto M = matmul_dims.M;
    auto N = matmul_dims.N;
    auto K = matmul_dims.K;
    bool x_trans = ctx.Attr<bool>("transpose_X");
    bool y_trans = ctx.Attr<bool>("transpose_Y");

    typedef memory::dims dims;
    dims x_dims = {BS, M, K};
    dims y_dims = {BS, K, N};
    dims out_dims = {BS, M, N};

    // Translate transA and transB
    dims x_strides = !x_trans ? dims{M * K, K, 1} : dims{M * K, 1, M};
    dims y_strides = !y_trans ? dims{N * K, N, 1} : dims{N * K, 1, K};
    dims out_strides = {M * N, N, 1};

    x_mem_ =
        CreateMemory<XT>(x_dims, x_strides, ctx.Input<Tensor>("X")->data<XT>());
    y_mem_ =
        CreateMemory<YT>(y_dims, y_strides, ctx.Input<Tensor>("Y")->data<YT>());
    out_mem_ = CreateMemory<OT>(
        out_dims, out_strides,
        ctx.Output<Tensor>("Out")->mutable_data<OT>(ctx.GetPlace()));
  }

  float ComputeOutputScale(const ExecutionContext& ctx) {
    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
    float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
    float alpha = ctx.Attr<float>("alpha");
    return alpha * scale_out / (scale_x * scale_y);
  }

  void CreatePrimitive(const ExecutionContext& ctx) {
    dnnl::primitive_attr attr;
    float scale_out = ComputeOutputScale(ctx);
    if (scale_out != 1.0f) {
      constexpr unsigned tensor_wide_scale = 0;
      attr.set_output_scales(tensor_wide_scale, {scale_out});
    }

    auto matmul_d = dnnl::matmul::desc(x_mem_.get_desc(), y_mem_.get_desc(),
                                       out_mem_.get_desc());
    auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, attr, engine_);
    matmul_prim_ = dnnl::matmul(matmul_pd);
  }

  void Execute() {
    dnnl::stream stream(engine_);
    matmul_prim_.execute(stream, {
                                     {MKLDNN_ARG_SRC, x_mem_},
                                     {MKLDNN_ARG_WEIGHTS, y_mem_},
                                     {MKLDNN_ARG_DST, out_mem_},
                                 });
    stream.wait();
  }

  void SetOutputFormat(const ExecutionContext& ctx) {
    using platform::MKLDNNFormatForSize;
    auto* out = ctx.Output<Tensor>("Out");
    auto format =
        MKLDNNFormatForSize(out->dims().size(), MKLDNNMemoryFormat::nchw);
    out->set_format(format);
    out->set_layout(DataLayout::kMKLDNN);
  }

  void UpdateDataPointers(const ExecutionContext& ctx) {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    x_mem_.set_data_handle(to_void_cast(x->data<XT>()));
    y_mem_.set_data_handle(to_void_cast(y->data<YT>()));
    out_mem_.set_data_handle(out->mutable_data<OT>(ctx.GetPlace()));
  }

  // If initialized, x memory should've been already initialized
  bool IsInitialized() { return initialized_; }

  void SetInitialized() { initialized_ = true; }

 private:
  dnnl::engine engine_;
  dnnl::memory x_mem_;
  dnnl::memory y_mem_;
  dnnl::memory out_mem_;
  dnnl::matmul matmul_prim_;
  bool initialized_ = false;
};

template <typename XT, typename YT, typename OT>
static std::shared_ptr<MatMulFactory<XT, YT, OT>> GetPrimitiveFactory(
    const ExecutionContext& ctx) {
  const auto x_dims = framework::vectorize<int>(ctx.Input<Tensor>("X")->dims());
  const auto y_dims = framework::vectorize<int>(ctx.Input<Tensor>("Y")->dims());
  const auto& out_name = ctx.OutputName("Out");
  const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

  const std::string key =
      platform::CreateKey(platform::ThreadIDasStr(), x_dims, y_dims, out_name);

  auto factory =
      std::static_pointer_cast<MatMulFactory<XT, YT, OT>>(dev_ctx.GetBlob(key));
  if (factory == nullptr) {
    factory = std::make_shared<MatMulFactory<XT, YT, OT>>();
    dev_ctx.SetBlob(key, factory);
  }

  return factory;
}

template <typename T>
constexpr bool IsInt8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}
// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename XT, typename YT>
static void ExecuteMatMul(const ExecutionContext& ctx) {
  constexpr bool is_int8 = IsInt8<XT>();
  const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
  constexpr bool fuse_relu = false;  // TODO(intel): Enable eltwise fuses
  if (!is_int8 || force_fp32_output) {
    GetPrimitiveFactory<XT, YT, float>(ctx)->CreateAndExecute(ctx);
  } else if (fuse_relu) {
    GetPrimitiveFactory<XT, YT, uint8_t>(ctx)->CreateAndExecute(ctx);
  } else {
    GetPrimitiveFactory<XT, YT, int8_t>(ctx)->CreateAndExecute(ctx);
  }
}

template <typename T>
class DNNLMatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(ctx.Attr<int>("head_number"), 1,
                      platform::errors::Unimplemented(
                          "DNNL matmul doesn't support multiple heads."));
    ExecuteMatMul<T, T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::DNNLMatMulKernel<float>, ops::DNNLMatMulKernel<int8_t>,
                   ops::DNNLMatMulKernel<uint8_t>);
