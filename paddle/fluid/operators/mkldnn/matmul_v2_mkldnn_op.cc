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
using paddle::framework::DataLayout;
using paddle::framework::ExecutionContext;
using paddle::platform::GetMKLDNNFormat;
using paddle::platform::MatMulV2MKLDNNHandler;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNFormatForSize;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::to_void_cast;
using phi::vectorize;
using Tensor = phi::DenseTensor;
using paddle::framework::GradVarName;
using phi::make_ddim;

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static Tensor FoldOuterDims(const Tensor &input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename T>
static Tensor FoldFirstAndLastDims(const MKLDNNDeviceContext &dev_ctx,
                                   const Tensor *input) {
  auto input_dims = vectorize(input->dims());
  if (input_dims.size() != 3) {
    return *input;
  }

  Tensor output;
  output.Resize({input_dims[1], input_dims[0], input_dims[2]});

  auto output_dims = vectorize(output.dims());

  memory::data_type input_type = paddle::framework::ToMKLDNNDataType(
      paddle::framework::TransToProtoVarType(input->dtype()));
  paddle::platform::ReorderMKLDNNHandler reorder_handler(
      output_dims,
      paddle::framework::TransToProtoVarType(input->dtype()),
      input_type,
      dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      memory::format_tag::abc,
      paddle::platform::to_void_cast(input->data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      &output, memory::format_tag::bac, dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                  reorder_dst_memory_p);

  auto &astream = MKLDNNDeviceContext::tls().get_stream();
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();

  output.Resize({input_dims[1], input_dims[0] * input_dims[2]});
  return output;
}

template <typename T>
constexpr bool IsInt8() {
  return std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
}

template <typename T>
constexpr bool IsBfloat16() {
  return std::is_same<T, paddle::platform::bfloat16>::value;
}

// Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
// original x_dim is returned.
static paddle::framework::DDim RowMatrixDimsFromVector(
    const paddle::framework::DDim &x_dim) {
  return x_dim.size() > 1 ? x_dim : phi::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static paddle::framework::DDim ColumnMatrixDimsFromVector(
    const paddle::framework::DDim &y_dim) {
  return y_dim.size() > 1 ? y_dim : phi::make_ddim({y_dim[0], 1});
}

phi::DDim GetDimForInput(const ExecutionContext &ctx, std::string input_name) {
  auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
  auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
  auto input_dims = ctx.Input<phi::DenseTensor>(input_name)->dims();
  if (!shape.empty() && !axis.empty()) {
    return input_dims.reshape(shape).transpose(axis);
  }
  return input_dims;
}

template <typename XT, typename YT, typename OT>
class MatMulMKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatMulMKLDNNHandler(const dnnl::engine engine,
                      paddle::platform::Place cpu_place,
                      Tensor *x,
                      bool trans_x,
                      Tensor *y,
                      bool trans_y,
                      Tensor *out,
                      float scale)
      : paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul>(engine,
                                                                    cpu_place) {
    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x->dims(), 0, trans_x);
    auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y->dims(), 0, trans_y);

    memory::dim x_bs = mat_dim_x.batch_size_;
    memory::dim y_bs = mat_dim_y.batch_size_;

    memory::dim out_bs = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = mat_dim_x.height_;
    const memory::dim N = mat_dim_y.width_;
    const memory::dim K = mat_dim_x.width_;

    memory::dims x_dims = {x_bs > 0 ? x_bs : 1, M, K};
    memory::dims y_dims = {y_bs > 0 ? y_bs : 1, K, N};
    memory::dims out_dims = {out_bs, M, N};

    memory::dims x_strides =
        !trans_x ? memory::dims{M * K, K, 1} : memory::dims{M * K, 1, M};

    memory::dims y_strides =
        !trans_y ? memory::dims{N * K, N, 1} : memory::dims{N * K, 1, K};
    memory::dims out_strides = memory::dims{M * N, N, 1};

    auto x_md = memory::desc(x_dims, MKLDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, MKLDNNGetDataType<YT>(), y_strides);
    auto out_md = memory::desc(out_dims, MKLDNNGetDataType<OT>(), out_strides);

    dnnl::primitive_attr attrs;
    if (scale != 1.0f) attrs.set_output_scales(0, {scale});

    this->AcquireForwardPrimitiveDescriptor(attrs, x_md, y_md, out_md);
  }
  // Constructor for FWD MatMul
  MatMulMKLDNNHandler(const dnnl::engine engine, const ExecutionContext &ctx)
      : paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul>(
            engine, ctx.GetPlace()) {
    const dnnl::primitive_attr matmul_attrs = CreateMatmulAttrs(ctx);

    auto matmul_dims_ = GetMatmulDims(ctx);
    auto x_md = memory::desc(
        matmul_dims_.x_dims, MKLDNNGetDataType<XT>(), matmul_dims_.x_strides);
    auto y_md = memory::desc(
        matmul_dims_.y_dims, MKLDNNGetDataType<YT>(), matmul_dims_.y_strides);
    auto out_md = memory::desc(matmul_dims_.out_dims,
                               MKLDNNGetDataType<OT>(),
                               matmul_dims_.out_strides);
    this->AcquireForwardPrimitiveDescriptor(matmul_attrs, x_md, y_md, out_md);
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const Tensor *input) {
    const YT *input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<YT>(input_data));
  }

 public:
  void Execute(const phi::DenseTensor *x,
               const phi::DenseTensor *y,
               phi::DenseTensor *out) {
    const auto src_memory_p = this->AcquireSrcMemory(x);
    const auto weights_memory_p = this->AcquireWeightsMemory(y);
    const auto dst_memory_p = this->AcquireDstMemory(out);

    auto matmul_p = this->AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto &astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();

    // Simulate batch matmul by processing in loop
    void *x_ptr = src_memory_p->get_data_handle();
    void *y_ptr = weights_memory_p->get_data_handle();
    void *out_ptr = dst_memory_p->get_data_handle();
    auto offsets = this->GetOffsets();
    for (uint16_t i = 0; i < this->GetBatchSize(); ++i) {
      src_memory_p->set_data_handle(x_ptr);
      weights_memory_p->set_data_handle(y_ptr);
      dst_memory_p->set_data_handle(out_ptr);
      matmul_p->execute(astream, matmul_args);
      x_ptr = static_cast<char *>(x_ptr) + std::get<0>(offsets);
      y_ptr = static_cast<char *>(y_ptr) + std::get<1>(offsets);
      out_ptr = static_cast<char *>(out_ptr) + std::get<2>(offsets);
    }
    astream.wait();

    out->set_mem_desc(dst_memory_p->get_desc().reshape(out->dims()));
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(phi::DenseTensor *output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence Tensor size is bigger that dst memory
    // primitive size. So would we request less memory that is there and it
    // triggers an
    // assertion.  So as there is no 'any' format here we can leave default size
    // of Tensor as computed in ComputeInferShape
    OT *ptr = output->mutable_data<OT>(this->place_);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }

 private:
  struct MatMulDims {
    const memory::dims x_dims, y_dims, out_dims, x_strides, y_strides,
        out_strides;
  };

  std::pair<phi::funcs::MatDescriptor, memory::dims> GetInputDimsAndStrides(
      const ExecutionContext &ctx, std::string input_name) {
    auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
    auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
    auto input_dims = ctx.Input<phi::DenseTensor>(input_name)->dims();
    auto new_dims = input_dims;
    if (!shape.empty() && !axis.empty()) {
      new_dims = input_dims.reshape(shape).transpose(axis);
    }

    auto &MatrixDimsFromVector = input_name == "X" ? RowMatrixDimsFromVector
                                                   : ColumnMatrixDimsFromVector;
    phi::funcs::MatDescriptor mat_dim = phi::funcs::CreateMatrixDescriptor(
        MatrixDimsFromVector(new_dims),
        0,
        ctx.Attr<bool>("transpose_" + input_name));

    memory::dims strides;
    if (!shape.empty()) {
      auto shape2 = input_dims.reshape(shape);
      strides.push_back(1);
      for (auto i = shape2.size() - 1; i > 0; --i) {
        strides.insert(strides.begin(), strides.front() * shape2[i]);
      }
      strides = Transpose(strides, axis);
      if (shape.size() == 4)
        strides.erase(strides.begin());
      else if (shape.size() == 2)
        strides.insert(strides.begin(), shape[0] * shape[1]);
      mat_dim.stride_ = strides[0];
      if (mat_dim.trans_) std::swap(*strides.rbegin(), *(++strides.rbegin()));
    }
    return std::make_pair(mat_dim, strides);
  }

  float ComputeOutputScale(const ExecutionContext &ctx) {
    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
    float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
    float alpha = ctx.Attr<float>("alpha");
    return alpha * scale_out / (scale_x * scale_y);
  }

  bool IsInputFused(const ExecutionContext &ctx) const {
    return !(ctx.Attr<std::vector<int>>("fused_reshape_X").empty() &&
             ctx.Attr<std::vector<int>>("fused_reshape_Y").empty());
  }

  bool IsOutputFused(const ExecutionContext &ctx) const {
    auto &fused_reshape_Out = ctx.Attr<std::vector<int>>("fused_reshape_Out");
    auto &fused_transpose_Out =
        ctx.Attr<std::vector<int>>("fused_transpose_Out");
    return !fused_reshape_Out.empty() && !fused_transpose_Out.empty();
  }

  MatMulDims GetMatmulDims(const ExecutionContext &ctx) {
    phi::funcs::MatDescriptor mat_dim_x;
    memory::dims strides_x;
    std::tie(mat_dim_x, strides_x) = GetInputDimsAndStrides(ctx, "X");
    phi::funcs::MatDescriptor mat_dim_y;
    memory::dims strides_y;
    std::tie(mat_dim_y, strides_y) = GetInputDimsAndStrides(ctx, "Y");

    auto x_bs = mat_dim_x.batch_size_;
    auto y_bs = mat_dim_y.batch_size_;
    PADDLE_ENFORCE_EQ(x_bs > 0 && y_bs > 0 && x_bs != y_bs,
                      false,
                      paddle::platform::errors::InvalidArgument(
                          "If batch sizes of X and Y are positive,"
                          "they have to be equal."));

    memory::dim out_bs = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = mat_dim_x.height_;
    const memory::dim N = mat_dim_y.width_;
    const memory::dim K = mat_dim_x.width_;

    batch_size_ = 1;
    if (out_bs > 1 && (IsOutputFused(ctx) || IsInputFused(ctx))) {
      auto x_dims = GetDimForInput(ctx, "X");
      auto y_dims = GetDimForInput(ctx, "Y");
      batch_size_ = x_bs > y_bs ? x_dims[0] : y_dims[0];
      x_bs /= batch_size_;
      y_bs /= batch_size_;
      out_bs /= batch_size_;
    }
    memory::dims x_dims = {x_bs > 0 ? x_bs : 1, M, K};
    memory::dims y_dims = {y_bs > 0 ? y_bs : 1, K, N};
    memory::dims out_dims = {out_bs, M, N};

    x_offset_ = x_bs * M * K * sizeof(XT);
    y_offset_ = y_bs * K * N * sizeof(YT);
    out_offset_ = out_bs * M * N * sizeof(OT);

    // Translate transA and transB
    if (strides_x.empty())
      strides_x = !ctx.Attr<bool>("transpose_X") ? memory::dims{M * K, K, 1}
                                                 : memory::dims{M * K, 1, M};
    if (strides_y.empty())
      strides_y = !ctx.Attr<bool>("transpose_Y") ? memory::dims{N * K, N, 1}
                                                 : memory::dims{N * K, 1, K};
    memory::dims out_strides = memory::dims{M * N, N, 1};

    CorrectStridesWhenFloatOutputFused(ctx, N, out_bs, &out_strides);

    return {x_dims, y_dims, out_dims, strides_x, strides_y, out_strides};
  }

  std::vector<int64_t> Transpose(const std::vector<int64_t> &x,
                                 const std::vector<int> &axis) {
    size_t in_rank = x.size();
    size_t axis_size = axis.size();

    auto axis_set = std::set<int>(axis.begin(), axis.end());
    PADDLE_ENFORCE_EQ(axis_set.size(),
                      axis_size,
                      paddle::platform::errors::InvalidArgument(
                          "In an axis array, elements must be unique."));

    PADDLE_ENFORCE_EQ(in_rank,
                      axis_size,
                      paddle::platform::errors::InvalidArgument(
                          "The input dimension's size "
                          "should be equal to the axis's size. "
                          "But received dimension is %d, "
                          "axis's size is %d",
                          in_rank,
                          axis_size));

    PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()),
                      axis_size,
                      paddle::platform::errors::InvalidArgument(
                          "Axis values must be ranging from 0 to (dims - 1)."));

    std::vector<int64_t> new_x(x.size());
    for (size_t i = 0; i < x.size(); i++) {
      new_x[i] = x[axis[i]];
    }
    return new_x;
  }

  void CorrectStridesWhenFloatOutputFused(const ExecutionContext &ctx,
                                          const memory::dim N,
                                          memory::dim b,
                                          memory::dims *out_strides) const {
    if (!IsInt8<OT>() && !IsBfloat16<OT>() && IsOutputFused(ctx)) {
      *out_strides = {N, b * N, 1};
    }
  }

  uint16_t GetBatchSize(void) const { return batch_size_; }

  std::tuple<uint32_t, uint32_t, uint32_t> GetOffsets() const {
    return std::make_tuple(x_offset_, y_offset_, out_offset_);
  }

  dnnl::primitive_attr CreateMatmulAttrs(const ExecutionContext &ctx) {
    dnnl::primitive_attr matmul_attrs;
    dnnl::post_ops post_operations;

    float scale_out = ComputeOutputScale(ctx);
    if (scale_out != 1.0f) {
      matmul_attrs.set_output_scales(0, {scale_out});
    }
    paddle::platform::AppendActivation(ctx, post_operations);

    matmul_attrs.set_post_ops(post_operations);
    return matmul_attrs;
  }

 private:
  uint32_t x_offset_;
  uint32_t y_offset_;
  uint32_t out_offset_;
  uint16_t batch_size_;
};

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorToMatrixSequence(
    Tensor *x, const phi::funcs::MatDescriptor &descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

/**
 * Reshape the x,y,out tensor to 3-D or 2-D tensor by matrix descriptor
 * Out = matmul(x, y)
 *
 * This method will first calculate X,Y matrix sequence, and then calculate
 * the out shape.
 *
 * Assume X = [BatchSize, H1, W1], Y = [BatchSize, H2, W2]
 * The out = [BatchSize, H1, W2]
 *
 * If there is no batch size in `X` and `Y`, the out will be [H1, W2]
 * If any of `X` and `Y` has batch size BatchSize, the out will have the
 * BatchSize.
 */
static void ReshapeXYOutToMatrixSequence(
    Tensor *x, Tensor *y, Tensor *out, bool trans_x, bool trans_y) {
  auto x_dim = RowMatrixDimsFromVector(x->dims());
  auto y_dim = ColumnMatrixDimsFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorToMatrixSequence(x, mat_dim_x);
  ReshapeTensorToMatrixSequence(y, mat_dim_y);
}

// Choose appropriate Handler instances based on inferred
// output type (uint8, int8 or float).
template <typename XT, typename YT>
static void ExecuteMatMul(const ExecutionContext &ctx) {
  constexpr bool is_int8 = IsInt8<XT>();
  constexpr bool is_bfloat16 = IsBfloat16<XT>();
  const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
  const bool fuse_relu =
      ctx.HasAttr("fuse_activation")
          ? ctx.Attr<std::string>("fuse_activation") == "relu"
          : false;
  auto *x = ctx.Input<phi::DenseTensor>("X");
  auto *y = ctx.Input<phi::DenseTensor>("Y");
  auto *out = ctx.Output<phi::DenseTensor>("Out");
  const auto &dev_ctx =
      ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
  const auto &onednn_engine = dev_ctx.GetEngine();

  if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
    MatMulMKLDNNHandler<XT, YT, float>(onednn_engine, ctx).Execute(x, y, out);
  } else if (is_bfloat16) {
    MatMulMKLDNNHandler<XT, YT, paddle::platform::bfloat16>(onednn_engine, ctx)
        .Execute(x, y, out);
  } else if (fuse_relu) {
    MatMulMKLDNNHandler<XT, YT, uint8_t>(onednn_engine, ctx).Execute(x, y, out);
  } else {
    MatMulMKLDNNHandler<XT, YT, int8_t>(onednn_engine, ctx).Execute(x, y, out);
  }
}

template <typename T>
class MatMulMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<int>("head_number"),
          1,
          paddle::platform::errors::Unimplemented(
              "oneDNN matmul doesn't support multiple heads. Expected "
              "head_number=1. But received `head_number` is %d",
              ctx.Attr<int>("head_number")));
    }
    ExecuteMatMul<T, T>(ctx);
  }
};

static std::vector<int64_t> Transpose(const std::vector<int64_t> &x,
                                      const std::vector<int> &axis) {
  size_t in_rank = x.size();
  size_t axis_size = axis.size();

  auto axis_set = std::set<int>(axis.begin(), axis.end());
  PADDLE_ENFORCE_EQ(axis_set.size(),
                    axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "In an axis array, elements must be unique."));

  PADDLE_ENFORCE_EQ(in_rank,
                    axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "The input dimension's size "
                        "should be equal to the axis's size. "
                        "But received dimension is %d, "
                        "axis's size is %d",
                        in_rank,
                        axis_size));

  PADDLE_ENFORCE_LT(*std::max_element(axis.begin(), axis.end()),
                    axis_size,
                    paddle::platform::errors::InvalidArgument(
                        "Axis values must be ranging from 0 to (dims - 1)."));

  std::vector<int64_t> new_x(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    new_x[i] = x[axis[i]];
  }
  return new_x;
}

std::vector<int64_t> GetInputStrides(const ExecutionContext &ctx,
                                     const std::string input_name) {
  auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
  auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
  auto input_dims = ctx.Input<phi::DenseTensor>(input_name)->dims();
  auto new_dims = input_dims;
  if (!shape.empty() && !axis.empty()) {
    new_dims = input_dims.reshape(shape).transpose(axis);
  }

  auto &MatrixDimsFromVector =
      input_name == "X" ? RowMatrixDimsFromVector : ColumnMatrixDimsFromVector;
  phi::funcs::MatDescriptor mat_dim = phi::funcs::CreateMatrixDescriptor(
      MatrixDimsFromVector(new_dims),
      0,
      ctx.HasAttr("trans_x")
          ? ctx.Attr<bool>(std::string("trans_") +
                           static_cast<char>(std::tolower(input_name[0])))
          : ctx.Attr<bool>(std::string("transpose_") + input_name[0]));

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

bool IsOutputFused(const ExecutionContext &ctx) {
  auto &fused_reshape_Out = ctx.Attr<std::vector<int>>("fused_reshape_Out");
  auto &fused_transpose_Out = ctx.Attr<std::vector<int>>("fused_transpose_Out");
  return !fused_reshape_Out.empty() && !fused_transpose_Out.empty();
}

float ComputeOutputScale(const ExecutionContext &ctx) {
  float scale_x = ctx.Attr<float>("Scale_x");
  float scale_y = ctx.Attr<float>("Scale_y");
  bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
  float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
  float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;
  return alpha * scale_out / (scale_x * scale_y);
}

template <typename T, typename T_out>
void ExecuteMatMulV2(const ExecutionContext &ctx,
                     const MKLDNNDeviceContext &dev_ctx,
                     const dnnl::engine onednn_engine,
                     paddle::platform::Place cpu_place,
                     const Tensor *x,
                     const std::vector<int64_t> &x_dims,
                     bool trans_x,
                     const Tensor *y,
                     const std::vector<int64_t> &y_dims,
                     bool trans_y,
                     Tensor *out,
                     const std::vector<int64_t> &out_dims,
                     int execution_number = 0) {
  std::vector<int64_t> x_strides_override = GetInputStrides(ctx, "X");
  std::vector<int64_t> y_strides_override = GetInputStrides(ctx, "Y");
  MatMulV2MKLDNNHandler<T, T, T_out> handler(ctx,
                                             onednn_engine,
                                             ctx.GetPlace(),
                                             x_dims,
                                             trans_x,
                                             y_dims,
                                             trans_y,
                                             IsOutputFused(ctx),
                                             x_strides_override,
                                             y_strides_override);

  const auto src_memory_p = handler.AcquireSrcMemory(x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(y);
  const auto dst_memory_p = handler.AcquireDstMemory(out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (ctx.HasInput("ResidualData")) {
    auto *residual_data = ctx.Input<phi::DenseTensor>("ResidualData");
    const auto residual_data_memory_p = handler.AcquireSrcMemory(residual_data);
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        *residual_data_memory_p});
  }

  auto &astream = MKLDNNDeviceContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  // TODO(jczaja): Explain why int8 format of dst is ABCD and do not need
  // permute
  if (IsOutputFused(ctx) && !IsInt8<T_out>()) {
    auto axis = ctx.Attr<std::vector<int>>("fused_transpose_Out");
    auto permuted_md = dst_memory_p->get_desc().permute_axes(axis);
    out->set_mem_desc(
        permuted_md.reshape(phi::vectorize<int64_t>(out->dims())));
  } else {
    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(phi::vectorize<int64_t>(out->dims())));
  }
}

template <typename T>
class MatMulV2MKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<int>("head_number"),
          1,
          paddle::platform::errors::Unimplemented(
              "oneDNN matmul doesn't support multiple heads. Expected "
              "head_number=1. But received `head_number` is %d",
              ctx.Attr<int>("head_number")));
    }
    constexpr bool is_int8 = IsInt8<T>();
    constexpr bool is_bfloat16 = IsBfloat16<T>();
    const bool force_fp32_output = ctx.HasAttr("force_fp32_output")
                                       ? ctx.Attr<bool>("force_fp32_output")
                                       : false;
    constexpr bool fuse_relu = false;  // TODO(intel): Enable eltwise fuses
    if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
      RunKernel<float>(ctx);
    } else if (is_bfloat16) {
      RunKernel<paddle::platform::bfloat16>(ctx);
    } else if (fuse_relu) {
      RunKernel<uint8_t>(ctx);
    } else {
      RunKernel<int8_t>(ctx);
    }
  }

 private:
  void CalculateMatrixDims(const ExecutionContext &ctx,
                           const std::vector<int64_t> &x_dims,
                           const std::vector<int64_t> &y_dims,
                           std::vector<int64_t> *x_bd_dims,
                           std::vector<int64_t> *y_bd_dims,
                           std::vector<int64_t> *out_dims,
                           Tensor *out) const {
    if (x_dims.size() == 1) {
      (*x_bd_dims)[(*x_bd_dims).size() - 1] = x_dims[0];
    } else if (x_dims.size() == 2) {
      (*x_bd_dims)[(*x_bd_dims).size() - 1] = x_dims[1];
      (*x_bd_dims)[(*x_bd_dims).size() - 2] = x_dims[0];
    } else {
      for (size_t i = 0; i < x_dims.size(); ++i) {
        (*x_bd_dims)[(*x_bd_dims).size() - x_dims.size() + i] = x_dims[i];
      }
    }
    if (y_dims.size() == 1) {
      (*y_bd_dims)[(*x_bd_dims).size() - 2] = y_dims[0];
    } else if (y_dims.size() == 2) {
      (*y_bd_dims)[(*y_bd_dims).size() - 1] = y_dims[1];
      (*y_bd_dims)[(*y_bd_dims).size() - 2] = y_dims[0];
    } else {
      for (size_t i = 0; i < y_dims.size(); ++i) {
        (*y_bd_dims)[(*y_bd_dims).size() - y_dims.size() + i] = y_dims[i];
      }
    }

    if (!IsOutputFused(ctx) && x_dims.size() > 2 && y_dims.size() > 2) {
      for (size_t i = 0; i < (*x_bd_dims).size() - 2; ++i) {
        PADDLE_ENFORCE_EQ(
            (*x_bd_dims)[i] == (*y_bd_dims)[i] || (*x_bd_dims)[i] == 1 ||
                (*y_bd_dims)[i] == 1,
            true,
            paddle::platform::errors::InvalidArgument(
                "Tensor dimensions are incorrect for broadcasting."
                "Dimensions in X and Y must be same or equal to 1, but "
                "received x_dim[%d]=%d and y_dims[%d]= %d",
                i,
                (*x_bd_dims)[i],
                i,
                (*y_bd_dims)[i]));
        (*out_dims)[i] = std::max((*x_bd_dims)[i], (*y_bd_dims)[i]);
      }
      out->Resize(phi::make_ddim((*out_dims)));
    }
  }

  template <typename T_out>
  void RunKernel(const ExecutionContext &ctx) const {
    const auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *y = ctx.Input<phi::DenseTensor>("Y");
    auto *out = ctx.Output<phi::DenseTensor>("Out");
    bool trans_x = ctx.HasAttr("trans_x") ? ctx.Attr<bool>("trans_x")
                                          : ctx.Attr<bool>("transpose_X");
    bool trans_y = ctx.HasAttr("trans_y") ? ctx.Attr<bool>("trans_y")
                                          : ctx.Attr<bool>("transpose_Y");

    auto x_dims = vectorize(GetDimForInput(ctx, "X"));
    auto y_dims = vectorize(GetDimForInput(ctx, "Y"));
    auto out_dims = vectorize(out->dims());

    int ndims = std::max(x_dims.size(), y_dims.size());
    ndims = std::max(ndims, 3);

    std::vector<int64_t> x_bd_dims(ndims, 1);
    std::vector<int64_t> y_bd_dims(ndims, 1);

    CalculateMatrixDims(
        ctx, x_dims, y_dims, &x_bd_dims, &y_bd_dims, &out_dims, out);

    ExecuteMatMulV2<T, T_out>(ctx,
                              dev_ctx,
                              onednn_engine,
                              ctx.GetPlace(),
                              x,
                              x_bd_dims,
                              trans_x,
                              y,
                              y_bd_dims,
                              trans_y,
                              out,
                              out_dims);
  }
};

template <typename T>
class MatMulV2GradMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override { RunKernel(ctx); }

 private:
  void CalculateGradMatrixDims(const ExecutionContext &ctx,
                               Tensor *dx_tmp,
                               Tensor *dy_tmp,
                               const std::vector<int64_t> &dx_dims,
                               const std::vector<int64_t> &dy_dims,
                               std::vector<int64_t> *dx_bd_dims,
                               std::vector<int64_t> *dy_bd_dims) const {
    for (size_t i = 0; i < dx_dims.size() - 2; ++i) {
      if (dx_dims[i] != dy_dims[i]) {
        if (dx_dims[i] == 1) {
          (*dx_bd_dims)[i] = dy_dims[i];
        } else {
          (*dy_bd_dims)[i] = dx_dims[i];
        }
      }
    }

    dx_tmp->Resize(phi::make_ddim((*dx_bd_dims)));
    dx_tmp->mutable_data<T>(ctx.GetPlace());
    dy_tmp->Resize(phi::make_ddim((*dy_bd_dims)));
    dy_tmp->mutable_data<T>(ctx.GetPlace());
  }

  void ReduceSumForMatmulGradOutput(
      const ExecutionContext &ctx,
      const MKLDNNDeviceContext &dev_ctx,
      const dnnl::engine onednn_engine,
      const Tensor *dx_tmp,
      Tensor *dx,
      const std::vector<int64_t> &dx_dims,
      const std::vector<int64_t> &squeezed_dims) const {
    paddle::platform::ReductionMKLDNNHandler<T> handler(
        dnnl::algorithm::reduction_sum,
        0.0f,
        0.0f,
        onednn_engine,
        ctx.GetPlace(),
        dx_tmp,
        dx,
        dx_dims);

    auto src_memory_p = handler.AcquireSrcMemory(dx_tmp);
    auto dst_memory_p = handler.AcquireDstMemory(dx);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

    auto &astream = MKLDNNDeviceContext::tls().get_stream();
    auto reduction_p = handler.AcquireForwardPrimitive();

    reduction_p->execute(astream, reduction_args);
    astream.wait();

    dx->set_mem_desc(dst_memory_p->get_desc().reshape(squeezed_dims));
  }

  std::vector<int64_t> ExtendDimsWithOnes(const std::vector<int64_t> &dims,
                                          int new_size) const {
    std::vector<int64_t> new_dims(new_size, 1);
    for (size_t i = 0; i < dims.size(); ++i) {
      new_dims[new_size - dims.size() + i] = dims[i];
    }

    return new_dims;
  }

  void RunKernel(const ExecutionContext &ctx) const {
    const auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *y = ctx.Input<phi::DenseTensor>("Y");

    auto x_dims = vectorize(x->dims());
    auto y_dims = vectorize(y->dims());

    bool is_broadcast = true;
    if (x_dims.size() <= 2 || y_dims.size() <= 2) {
      is_broadcast = false;
    } else if (x_dims.size() != y_dims.size()) {
      is_broadcast = true;
    } else {
      is_broadcast = !std::equal(x_dims.cbegin(),
                                 x_dims.cbegin() + x_dims.size() - 2,
                                 y_dims.cbegin());
    }

    // if no broadcasting is needed, we can simply use matmul's grad and avoid
    // using reduce_sum
    if (!is_broadcast) {
      matmul_v1_grad_mkldnn_kernel.Compute(ctx);
      return;
    }

    auto *dout = ctx.Input<phi::DenseTensor>(GradVarName("Out"));
    auto *dx = ctx.Output<phi::DenseTensor>(GradVarName("X"));
    auto *dy = ctx.Output<phi::DenseTensor>(GradVarName("Y"));

    bool trans_x = ctx.HasAttr("trans_x") ? ctx.Attr<bool>("trans_x")
                                          : ctx.Attr<bool>("transpose_X");
    bool trans_y = ctx.HasAttr("trans_y") ? ctx.Attr<bool>("trans_y")
                                          : ctx.Attr<bool>("transpose_Y");
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

    CalculateGradMatrixDims(
        ctx, &dx_tmp, &dy_tmp, x_dims, y_dims, &dx_bd_dims, &dy_bd_dims);

    if (trans_x && trans_y) {
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            y,
                            y_dims,
                            true,
                            dout,
                            dout_dims,
                            true,
                            &dx_tmp,
                            dx_bd_dims,
                            1);
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            dout,
                            dout_dims,
                            true,
                            x,
                            x_dims,
                            true,
                            &dy_tmp,
                            dy_bd_dims,
                            2);
    } else if (trans_x) {
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            y,
                            y_dims,
                            false,
                            dout,
                            dout_dims,
                            true,
                            &dx_tmp,
                            dx_bd_dims,
                            1);
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            x,
                            x_dims,
                            false,
                            dout,
                            dout_dims,
                            false,
                            &dy_tmp,
                            dy_bd_dims,
                            2);
    } else if (trans_y) {
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            dout,
                            dout_dims,
                            false,
                            y,
                            y_dims,
                            false,
                            &dx_tmp,
                            dx_bd_dims,
                            1);
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            dout,
                            dout_dims,
                            true,
                            x,
                            x_dims,
                            false,
                            &dy_tmp,
                            dy_bd_dims,
                            2);
    } else {
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            dout,
                            dout_dims,
                            false,
                            y,
                            y_dims,
                            true,
                            &dx_tmp,
                            dx_bd_dims,
                            1);
      ExecuteMatMulV2<T, T>(ctx,
                            dev_ctx,
                            onednn_engine,
                            ctx.GetPlace(),
                            x,
                            x_dims,
                            true,
                            dout,
                            dout_dims,
                            false,
                            &dy_tmp,
                            dy_bd_dims,
                            2);
    }

    if (x_dims != dx_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx,
                                   dev_ctx,
                                   onednn_engine,
                                   &dx_tmp,
                                   dx,
                                   x_dims,
                                   phi::vectorize(x->dims()));
    } else {
      *dx = std::move(dx_tmp);
    }
    if (y_dims != dy_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx,
                                   dev_ctx,
                                   onednn_engine,
                                   &dy_tmp,
                                   dy,
                                   y_dims,
                                   phi::vectorize(y->dims()));
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

namespace paddle {
namespace operators {

template <typename T>
void MatMulGradMKLDNNKernel<T>::Compute(const ExecutionContext &ctx) const {
  if (ctx.HasAttr("head_number")) {
    PADDLE_ENFORCE_EQ(
        ctx.Attr<int>("head_number"),
        1,
        platform::errors::Unimplemented(
            "oneDNN matmul doesn't support multiple heads. Expected "
            "head_number=1. But received `head_number` is %d",
            ctx.Attr<int>("head_number")));
  }
  RunKernel(ctx);
}

template <typename T>
void MatMulGradMKLDNNKernel<T>::ExecuteMatMulGrad(
    const ExecutionContext &ctx,
    const MKLDNNDeviceContext &dev_ctx,
    const dnnl::engine &engine,
    Tensor *x,
    bool trans_x,
    bool is_fold_init_dims_x,
    Tensor *y,
    bool trans_y,
    bool is_fold_init_dims_y,
    Tensor *out) const {
  // gradient is calculated in a different way when broadcasting is used
  bool need_combine = (x->dims().size() == 3 || y->dims().size() == 3) &&
                      out->dims().size() == 2;

  Tensor x_combined, y_combined;
  if (!need_combine) {
    x_combined = *x;
    y_combined = *y;
  } else {
    x_combined = is_fold_init_dims_x ? FoldOuterDims(*x)
                                     : FoldFirstAndLastDims<T>(dev_ctx, x);
    y_combined = is_fold_init_dims_y ? FoldOuterDims(*y)
                                     : FoldFirstAndLastDims<T>(dev_ctx, y);
  }

  float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

  MatMulMKLDNNHandler<T, T, T> handler(engine,
                                       ctx.GetPlace(),
                                       &x_combined,
                                       trans_x,
                                       &y_combined,
                                       trans_y,
                                       out,
                                       alpha);

  const auto src_memory_p = handler.AcquireSrcMemory(&x_combined);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y_combined);
  const auto dst_memory_p = handler.AcquireDstMemory(out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto &astream = platform::MKLDNNDeviceContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  out->set_mem_desc(
      dst_memory_p->get_desc().reshape(vectorize<int64_t>(out->dims())));
}

template <typename T>
void MatMulGradMKLDNNKernel<T>::RunKernel(const ExecutionContext &ctx) const {
  const auto &dev_ctx =
      ctx.template device_context<platform::MKLDNNDeviceContext>();
  const auto &onednn_engine = dev_ctx.GetEngine();

  auto x = *ctx.Input<phi::DenseTensor>("X");
  auto y = *ctx.Input<phi::DenseTensor>("Y");
  auto dout = *ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
  auto *dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
  auto *dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));

  bool transpose_x = ctx.HasAttr("transpose_X") ? ctx.Attr<bool>("transpose_X")
                                                : ctx.Attr<bool>("trans_x");
  bool transpose_y = ctx.HasAttr("transpose_Y") ? ctx.Attr<bool>("transpose_Y")
                                                : ctx.Attr<bool>("trans_y");

  ReshapeXYOutToMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);

  framework::DDim dx_dims;
  if (dx) {
    dx_dims = dx->dims();
    if (dx_dims != x.dims()) {
      dx->Resize(x.dims());
    }
  }

  framework::DDim dy_dims;
  if (dy) {
    dy_dims = dy->dims();
    if (dy_dims != y.dims()) {
      dy->Resize(y.dims());
    }
  }

  if (transpose_x && transpose_y) {
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &y, true, true, &dout, true, false, dx);
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &dout, true, true, &x, true, false, dy);
  } else if (transpose_x) {
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &y, false, false, &dout, true, false, dx);
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &x, false, false, &dout, false, true, dy);
  } else if (transpose_y) {
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &dout, false, false, &y, false, true, dx);
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &dout, true, true, &x, false, true, dy);
  } else {
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &dout, false, false, &y, true, false, dx);
    this->ExecuteMatMulGrad(
        ctx, dev_ctx, onednn_engine, &x, true, true, &dout, false, true, dy);
  }

  if (dx) {
    if (dx_dims != x.dims()) {
      dx->Resize(dx_dims);
      dx->set_mem_desc(x.mem_desc());
    }
  }
  if (dy) {
    if (dy_dims != y.dims()) {
      dy->Resize(dy_dims);
      dy->set_mem_desc(y.mem_desc());
    }
  }
}

template class MatMulGradMKLDNNKernel<float>;
template class MatMulGradMKLDNNKernel<paddle::platform::bfloat16>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   MatMulV2MKLDNNKernel<float>,
                   MatMulV2MKLDNNKernel<paddle::platform::bfloat16>,
                   MatMulV2MKLDNNKernel<int8_t>,
                   MatMulV2MKLDNNKernel<uint8_t>);

REGISTER_OP_KERNEL(matmul_grad,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::MatMulGradMKLDNNKernel<float>,
                   ops::MatMulGradMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(matmul_v2,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   MatMulV2MKLDNNKernel<float>,
                   MatMulV2MKLDNNKernel<paddle::platform::bfloat16>,
                   MatMulV2MKLDNNKernel<int8_t>,
                   MatMulV2MKLDNNKernel<uint8_t>);

REGISTER_OP_KERNEL(matmul_v2_grad,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   MatMulV2GradMKLDNNKernel<float>,
                   MatMulV2GradMKLDNNKernel<paddle::platform::bfloat16>);
