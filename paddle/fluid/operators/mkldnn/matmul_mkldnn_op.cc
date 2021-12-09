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
#include <tuple>

using dnnl::memory;
using dnnl::primitive;
using paddle::framework::DataLayout;
using paddle::framework::ExecutionContext;
using paddle::framework::vectorize;
using paddle::platform::GetMKLDNNFormat;
using paddle::platform::MKLDNNFormatForSize;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::to_void_cast;
using Tensor = paddle::framework::Tensor;

namespace {

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static Tensor FoldOuterDims(const Tensor& input) {
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
static Tensor FoldFirstAndLastDims(const MKLDNNDeviceContext& dev_ctx,
                                   const Tensor* input) {
  auto input_dims = vectorize(input->dims());
  if (input_dims.size() != 3) {
    return *input;
  }

  Tensor output;
  output.Resize({input_dims[1], input_dims[0], input_dims[2]});

  auto output_dims = vectorize(output.dims());

  memory::data_type input_type =
      paddle::framework::ToMKLDNNDataType(input->type());
  paddle::platform::ReorderMKLDNNHandler reorder_handler(
      output_dims, input->type(), input_type, dev_ctx.GetEngine());

  auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
      memory::format_tag::abc,
      paddle::platform::to_void_cast(input->data<T>()));
  auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
      &output, memory::format_tag::bac, dev_ctx.GetPlace());
  auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                  reorder_dst_memory_p);

  paddle::platform::RecordEvent record_reorder(
      "int_reorder", paddle::platform::EventRole::kUniqueOp);

  auto& astream = MKLDNNDeviceContext::tls().get_stream();
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
    const paddle::framework::DDim& x_dim) {
  return x_dim.size() > 1 ? x_dim : paddle::framework::make_ddim({1, x_dim[0]});
}

// Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
// original y_dim is returned.
static paddle::framework::DDim ColumnMatrixDimsFromVector(
    const paddle::framework::DDim& y_dim) {
  return y_dim.size() > 1 ? y_dim : paddle::framework::make_ddim({y_dim[0], 1});
}

template <typename XT, typename YT, typename OT>
class MatMulMKLDNNHandler
    : public paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  MatMulMKLDNNHandler(const dnnl::engine engine,
                      paddle::platform::Place cpu_place, Tensor* x,
                      bool trans_x, Tensor* y, bool trans_y, Tensor* out,
                      float scale)
      : paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul>(engine,
                                                                    cpu_place) {
    auto mat_dim_x =
        paddle::operators::math::CreateMatrixDescriptor(x->dims(), 0, trans_x);
    auto mat_dim_y =
        paddle::operators::math::CreateMatrixDescriptor(y->dims(), 0, trans_y);

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
  MatMulMKLDNNHandler(const dnnl::engine engine, const ExecutionContext& ctx,
                      float scale)
      : paddle::platform::MKLDNNHandlerNoCachingT<XT, dnnl::matmul>(
            engine, ctx.GetPlace()),
        matmul_dims_(GetMatmulDims(ctx)) {
    dnnl::primitive_attr attr;
    float scale_out = ComputeOutputScale(ctx);
    if (scale_out != 1.0f) {
      constexpr unsigned tensor_wide_scale = 0;
      attr.set_output_scales(tensor_wide_scale, {scale_out});
    }

    auto x_md = memory::desc(matmul_dims_.x_dims, MKLDNNGetDataType<XT>(),
                             matmul_dims_.x_strides);
    auto y_md = memory::desc(matmul_dims_.y_dims, MKLDNNGetDataType<YT>(),
                             matmul_dims_.y_strides);
    auto out_md = memory::desc(matmul_dims_.out_dims, MKLDNNGetDataType<OT>(),
                               matmul_dims_.out_strides);
    this->AcquireForwardPrimitiveDescriptor(attr, x_md, y_md, out_md);
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const Tensor* input) {
    const YT* input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<YT>(input_data));
  }

 public:
  void Execute(const paddle::framework::Tensor* x,
               const paddle::framework::Tensor* y,
               paddle::framework::Tensor* out) {
    const auto src_memory_p = this->AcquireSrcMemory(x);
    const auto weights_memory_p = this->AcquireWeightsMemory(y);
    const auto dst_memory_p = this->AcquireDstMemory(out);

    auto matmul_p = this->AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();

    // Simulate batch matmul by processing in loop
    void* x_ptr = src_memory_p->get_data_handle();
    void* y_ptr = weights_memory_p->get_data_handle();
    void* out_ptr = dst_memory_p->get_data_handle();
    auto offsets = this->GetOffsets();
    for (uint16_t i = 0; i < this->GetBatchSize(); ++i) {
      src_memory_p->set_data_handle(x_ptr);
      weights_memory_p->set_data_handle(y_ptr);
      dst_memory_p->set_data_handle(out_ptr);
      matmul_p->execute(astream, {
                                     {DNNL_ARG_SRC, *src_memory_p},
                                     {DNNL_ARG_WEIGHTS, *weights_memory_p},
                                     {DNNL_ARG_DST, *dst_memory_p},
                                 });
      x_ptr = static_cast<char*>(x_ptr) + std::get<0>(offsets);
      y_ptr = static_cast<char*>(y_ptr) + std::get<1>(offsets);
      out_ptr = static_cast<char*>(out_ptr) + std::get<2>(offsets);
    }
    astream.wait();

    auto format =
        MKLDNNFormatForSize(out->dims().size(), dnnl::memory::format_tag::nchw);
    out->set_format(format);
    out->set_layout(DataLayout::kMKLDNN);
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(
      paddle::framework::Tensor* output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence Tensor size is bigger that dst memory
    // primitive size. So would we request less memory that is there and it
    // triggers an
    // assertion.  So as there is no 'any' format here we can leave default size
    // of Tensor as computed in ComputeInferShape
    OT* ptr = output->mutable_data<OT>(this->place_);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }

 private:
  struct MatMulDims {
    const memory::dims x_dims, y_dims, out_dims, x_strides, y_strides,
        out_strides;
  };

  std::pair<paddle::operators::math::MatDescriptor, memory::dims>
  GetInputDimsAndStrides(const ExecutionContext& ctx, std::string input_name) {
    auto shape = ctx.Attr<std::vector<int>>("fused_reshape_" + input_name);
    auto axis = ctx.Attr<std::vector<int>>("fused_transpose_" + input_name);
    auto input_dims = ctx.Input<Tensor>(input_name)->dims();
    auto new_dims = input_dims;
    if (!shape.empty() && !axis.empty()) {
      auto it_zero = std::find(shape.begin(), shape.end(), 0);
      if (it_zero != shape.end()) {
        for (uint64_t i = 0; i < shape.size(); i++) {
          if (shape[i] == 0) {
            PADDLE_ENFORCE_LT(
                i, input_dims.size(),
                paddle::platform::errors::InvalidArgument(
                    "The index of 0 in fused_reshape_%s ",
                    "should be less than output dim size, ",
                    "but the index is %d and output dim size is %d", input_name,
                    i, input_dims.size()));
            shape[i] = input_dims.at(i);
          }
        }
      }

      // if "-1" is present then one of reshape dims must be infered
      auto it_negative = std::find(shape.begin(), shape.end(), -1);
      if (it_negative != shape.end()) {
        int64_t dim_product = 1;
        for (int i = 0; i < input_dims.size(); i++) {
          dim_product *= input_dims.at(i);
        }

        int64_t shape_product = std::accumulate(shape.begin(), shape.end(), -1,
                                                std::multiplies<int>());
        int index = std::distance(shape.begin(), it_negative);
        shape[index] = dim_product / shape_product;
      }

      new_dims = input_dims.reshape(shape).transpose(axis);
    }

    auto& MatrixDimsFromVector = input_name == "X" ? RowMatrixDimsFromVector
                                                   : ColumnMatrixDimsFromVector;
    paddle::operators::math::MatDescriptor mat_dim =
        paddle::operators::math::CreateMatrixDescriptor(
            MatrixDimsFromVector(new_dims), 0,
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

  float ComputeOutputScale(const ExecutionContext& ctx) {
    float scale_x = ctx.Attr<float>("Scale_x");
    float scale_y = ctx.Attr<float>("Scale_y");
    bool force_fp32_out = ctx.Attr<bool>("force_fp32_output");
    float scale_out = force_fp32_out ? 1.f : ctx.Attr<float>("Scale_out");
    float alpha = ctx.Attr<float>("alpha");
    return alpha * scale_out / (scale_x * scale_y);
  }

  bool IsInputFused(const ExecutionContext& ctx) const {
    return !(ctx.Attr<std::vector<int>>("fused_reshape_X").empty() &&
             ctx.Attr<std::vector<int>>("fused_reshape_Y").empty());
  }

  bool IsOutputFused(const ExecutionContext& ctx) const {
    auto& fused_reshape_Out = ctx.Attr<std::vector<int>>("fused_reshape_Out");
    auto& fused_transpose_Out =
        ctx.Attr<std::vector<int>>("fused_transpose_Out");
    return !fused_reshape_Out.empty() && !fused_transpose_Out.empty();
  }

  MatMulDims GetMatmulDims(const ExecutionContext& ctx) {
    paddle::operators::math::MatDescriptor mat_dim_x;
    memory::dims strides_x;
    std::tie(mat_dim_x, strides_x) = GetInputDimsAndStrides(ctx, "X");
    paddle::operators::math::MatDescriptor mat_dim_y;
    memory::dims strides_y;
    std::tie(mat_dim_y, strides_y) = GetInputDimsAndStrides(ctx, "Y");

    auto x_bs = mat_dim_x.batch_size_;
    auto y_bs = mat_dim_y.batch_size_;
    PADDLE_ENFORCE_EQ(x_bs > 0 && y_bs > 0 && x_bs != y_bs, false,
                      paddle::platform::errors::InvalidArgument(
                          "If batch sizes of X and Y are positive,"
                          "they have to be equal."));

    memory::dim out_bs = x_bs || y_bs ? std::max(x_bs, y_bs) : 1;
    const memory::dim M = mat_dim_x.height_;
    const memory::dim N = mat_dim_y.width_;
    const memory::dim K = mat_dim_x.width_;

    batch_size_ = 1;
    if (out_bs > 1 && (IsOutputFused(ctx) || IsInputFused(ctx))) {
      auto& x_dims = ctx.Input<Tensor>("X")->dims();
      auto& y_dims = ctx.Input<Tensor>("Y")->dims();
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

  std::vector<int64_t> Transpose(const std::vector<int64_t>& x,
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

  void CorrectStridesWhenFloatOutputFused(const ExecutionContext& ctx,
                                          const memory::dim N, memory::dim b,
                                          memory::dims* out_strides) const {
    if (!IsInt8<OT>() && !IsBfloat16<OT>() && IsOutputFused(ctx)) {
      *out_strides = {N, b * N, 1};
    }
  }

  uint16_t GetBatchSize(void) const { return batch_size_; }

  std::tuple<uint32_t, uint32_t, uint32_t> GetOffsets() const {
    return std::make_tuple(x_offset_, y_offset_, out_offset_);
  }

 private:
  MatMulDims matmul_dims_;
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
    Tensor* x, const paddle::operators::math::MatDescriptor& descriptor) {
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
static void ReshapeXYOutToMatrixSequence(Tensor* x, Tensor* y, Tensor* out,
                                         bool trans_x, bool trans_y) {
  auto x_dim = RowMatrixDimsFromVector(x->dims());
  auto y_dim = ColumnMatrixDimsFromVector(y->dims());
  auto mat_dim_x =
      paddle::operators::math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y =
      paddle::operators::math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  ReshapeTensorToMatrixSequence(x, mat_dim_x);
  ReshapeTensorToMatrixSequence(y, mat_dim_y);
}

// Choose appropriate Handler instances based on inferred
// output type (uint8, int8 or float).
template <typename XT, typename YT>
static void ExecuteMatMul(const ExecutionContext& ctx) {
  constexpr bool is_int8 = IsInt8<XT>();
  constexpr bool is_bfloat16 = IsBfloat16<XT>();
  const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
  constexpr bool fuse_relu = false;  // TODO(intel): Enable eltwise fuses
  auto* x = ctx.Input<Tensor>("X");
  auto* y = ctx.Input<Tensor>("Y");
  auto* out = ctx.Output<Tensor>("Out");
  float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;
  const auto& dev_ctx =
      ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();

  if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
    MatMulMKLDNNHandler<XT, YT, float>(dev_ctx.GetEngine(), ctx, alpha)
        .Execute(x, y, out);
  } else if (is_bfloat16) {
    MatMulMKLDNNHandler<XT, YT, paddle::platform::bfloat16>(dev_ctx.GetEngine(),
                                                            ctx, alpha)
        .Execute(x, y, out);
  } else if (fuse_relu) {
    MatMulMKLDNNHandler<XT, YT, uint8_t>(dev_ctx.GetEngine(), ctx, alpha)
        .Execute(x, y, out);
  } else {
    MatMulMKLDNNHandler<XT, YT, int8_t>(dev_ctx.GetEngine(), ctx, alpha)
        .Execute(x, y, out);
  }
}

template <typename T>
class MatMulMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override {
    if (ctx.HasAttr("head_number")) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<int>("head_number"), 1,
          paddle::platform::errors::Unimplemented(
              "oneDNN matmul doesn't support multiple heads. Expected "
              "head_number=1. But received `head_number` is %d",
              ctx.Attr<int>("head_number")));
    }
    ExecuteMatMul<T, T>(ctx);
  }
};

}  // anonymous namespace

namespace paddle {
namespace operators {

template <typename T>
void MatMulGradMKLDNNKernel<T>::Compute(const ExecutionContext& ctx) const {
  if (ctx.HasAttr("head_number")) {
    PADDLE_ENFORCE_EQ(
        ctx.Attr<int>("head_number"), 1,
        platform::errors::Unimplemented(
            "oneDNN matmul doesn't support multiple heads. Expected "
            "head_number=1. But received `head_number` is %d",
            ctx.Attr<int>("head_number")));
  }
  RunKernel(ctx);
}

template <typename T>
void MatMulGradMKLDNNKernel<T>::ExecuteMatMulGrad(
    const ExecutionContext& ctx, const MKLDNNDeviceContext& dev_ctx,
    const dnnl::engine& engine, Tensor* x, bool trans_x,
    bool is_fold_init_dims_x, Tensor* y, bool trans_y, bool is_fold_init_dims_y,
    Tensor* out) const {
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

  MatMulMKLDNNHandler<T, T, T> handler(engine, ctx.GetPlace(), &x_combined,
                                       trans_x, &y_combined, trans_y, out,
                                       alpha);

  const auto src_memory_p = handler.AcquireSrcMemory(&x_combined);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y_combined);
  const auto dst_memory_p = handler.AcquireDstMemory(out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, dnnl::memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  out->set_layout(framework::DataLayout::kMKLDNN);
  out->set_format(platform::GetMKLDNNFormat(
      dst_memory_p->get_desc().reshape(vectorize<int64_t>(out->dims()))));
}

template <typename T>
void MatMulGradMKLDNNKernel<T>::RunKernel(const ExecutionContext& ctx) const {
  const auto& dev_ctx =
      ctx.template device_context<platform::MKLDNNDeviceContext>();
  const auto& onednn_engine = dev_ctx.GetEngine();

  auto x = *ctx.Input<Tensor>("X");
  auto y = *ctx.Input<Tensor>("Y");
  auto dout = *ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
  auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

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
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &y, true, true, &dout,
                            true, false, dx);
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &dout, true, true, &x,
                            true, false, dy);
  } else if (transpose_x) {
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &y, false, false,
                            &dout, true, false, dx);
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &x, false, false,
                            &dout, false, true, dy);
  } else if (transpose_y) {
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &dout, false, false,
                            &y, false, true, dx);
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &dout, true, true, &x,
                            false, true, dy);
  } else {
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &dout, false, false,
                            &y, true, false, dx);
    this->ExecuteMatMulGrad(ctx, dev_ctx, onednn_engine, &x, true, true, &dout,
                            false, true, dy);
  }

  if (dx) {
    if (dx_dims != x.dims()) {
      dx->Resize(dx_dims);
      dx->set_format(x.format());
    }
  }
  if (dy) {
    if (dy_dims != y.dims()) {
      dy->Resize(dy_dims);
      dy->set_format(y.format());
    }
  }
}

template class MatMulGradMKLDNNKernel<float>;
template class MatMulGradMKLDNNKernel<paddle::platform::bfloat16>;

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul, MKLDNN, ::paddle::platform::CPUPlace,
                   MatMulMKLDNNKernel<float>,
                   MatMulMKLDNNKernel<paddle::platform::bfloat16>,
                   MatMulMKLDNNKernel<int8_t>, MatMulMKLDNNKernel<uint8_t>);

REGISTER_OP_KERNEL(matmul_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MatMulGradMKLDNNKernel<float>,
                   ops::MatMulGradMKLDNNKernel<paddle::platform::bfloat16>);
