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

#include "mkldnn.hpp"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using platform::to_void_cast;
using framework::DataLayout;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using platform::MKLDNNDeviceContext;
using framework::ExecutionContext;
using Tensor = framework::Tensor;

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static framework::DDim RowMatrixDimsFromVector(const framework::DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static framework::DDim ColumnMatrixDimsFromVector(
    const framework::DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : framework::make_ddim({y_dim[0], 1});
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
    const memory::dims x_dims, y_dims, out_dims, x_strides, y_strides,
        out_strides;
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

  bool IsOutputFused(const ExecutionContext& ctx) const {
    auto& reshape_Out = ctx.Attr<std::vector<int64_t>>("reshape_Out");
    auto& axis_Out = ctx.Attr<std::vector<int64_t>>("axis_Out");
    return !reshape_Out.empty() && !axis_Out.empty();
  }

  void correctStridesWhenOutputFused(const ExecutionContext& ctx,
                                     const memory::dim N, memory::dim b,
                                     memory::dims* out_strides) const {
    if (IsOutputFused(ctx)) *out_strides = {N, b * N, 1};
  }

  MatMulDims GetMatmulDims(const ExecutionContext& ctx) {
    auto mat_dim_x = math::CreateMatrixDescriptor(
        RowMatrixDimsFromVector(ctx.Input<Tensor>("X")->dims()), 0,
        ctx.Attr<bool>("transpose_X"));
    auto mat_dim_y = math::CreateMatrixDescriptor(
        ColumnMatrixDimsFromVector(ctx.Input<Tensor>("Y")->dims()), 0,
        ctx.Attr<bool>("transpose_Y"));

    const auto x_bs = mat_dim_x.batch_size_;
    const auto y_bs = mat_dim_y.batch_size_;
    PADDLE_ENFORCE_EQ(x_bs > 0 && y_bs > 0 && x_bs != y_bs, false,
                      platform::errors::InvalidArgument(
                          "If batch sizes of X and Y are positive,"
                          "they have to be equal."));

    // Store 1 if both batches are zero, otherwise save the nonzero batch
    const memory::dim BS = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = mat_dim_x.height_;
    const memory::dim N = mat_dim_y.width_;
    const memory::dim K = mat_dim_x.width_;

    batch_size_ = 1;
    auto b = BS;
    if (BS > 1 && IsOutputFused(ctx)) {
      batch_size_ = ctx.Input<Tensor>("X")->dims()[0];
      b = BS / batch_size_;
    }
    memory::dims x_dims = {b, M, K};
    memory::dims y_dims = {b, K, N};
    memory::dims out_dims = {b, M, N};

    size_t x_size = b * M * K * sizeof(XT);
    size_t y_size = b * K * N * sizeof(YT);
    size_t out_size = b * M * N * sizeof(OT);
    offsets_ = {x_size, y_size, out_size};

    // Translate transA and transB
    memory::dims strides_x = !ctx.Attr<bool>("transpose_X")
                                 ? memory::dims{M * K, K, 1}
                                 : memory::dims{M * K, 1, M};
    memory::dims strides_y = !ctx.Attr<bool>("transpose_Y")
                                 ? memory::dims{N * K, N, 1}
                                 : memory::dims{N * K, 1, K};
    memory::dims out_strides = memory::dims{M * N, N, 1};

    correctStridesWhenOutputFused(ctx, N, b, &out_strides);

    return {x_dims, y_dims, out_dims, strides_x, strides_y, out_strides};
  }

  void CreateMemories(const ExecutionContext& ctx) {
    auto matmul_dims = GetMatmulDims(ctx);

    x_mem_ = CreateMemory<XT>(matmul_dims.x_dims, matmul_dims.x_strides,
                              ctx.Input<Tensor>("X")->data<XT>());
    y_mem_ = CreateMemory<YT>(matmul_dims.y_dims, matmul_dims.y_strides,
                              ctx.Input<Tensor>("Y")->data<YT>());
    out_mem_ = CreateMemory<OT>(
        matmul_dims.out_dims, matmul_dims.out_strides,
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

    auto offsets = offsets_;
    unsigned bs = batch_size_;
    void* x_ptr = x_mem_.get_data_handle();
    void* y_ptr = y_mem_.get_data_handle();
    void* out_ptr = out_mem_.get_data_handle();
    for (unsigned i = 0; i < bs; i++) {
      x_mem_.set_data_handle(x_ptr);
      y_mem_.set_data_handle(y_ptr);
      out_mem_.set_data_handle(out_ptr);
      matmul_prim_.execute(stream, {
                                       {MKLDNN_ARG_SRC, x_mem_},
                                       {MKLDNN_ARG_WEIGHTS, y_mem_},
                                       {MKLDNN_ARG_DST, out_mem_},
                                   });
      x_ptr = static_cast<char*>(x_ptr) + offsets.x_offset;
      y_ptr = static_cast<char*>(y_ptr) + offsets.y_offset;
      out_ptr = static_cast<char*>(out_ptr) + offsets.out_offset;
    }
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
  struct memory_offsets {
    size_t x_offset;
    size_t y_offset;
    size_t out_offset;
  };

  dnnl::engine engine_;
  dnnl::memory x_mem_;
  dnnl::memory y_mem_;
  dnnl::memory out_mem_;
  dnnl::matmul matmul_prim_;
  memory_offsets offsets_;
  unsigned batch_size_;
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
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(ctx.Attr<int>("head_number"), 1,
                        platform::errors::Unimplemented(
                            "DNNL matmul doesn't support multiple heads."));
    }
    ExecuteMatMul<T, T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::DNNLMatMulKernel<float>, ops::DNNLMatMulKernel<int8_t>,
                   ops::DNNLMatMulKernel<uint8_t>);
