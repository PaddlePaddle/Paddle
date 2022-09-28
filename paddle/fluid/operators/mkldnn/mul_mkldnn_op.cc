/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::DDim;
using framework::ExecutionContext;
using framework::LoDTensor;
using framework::Tensor;

using platform::MatMulV2MKLDNNHandler;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;

constexpr int kMULMKLDNNINT8 = 1;
constexpr int kMULMKLDNNFP32 = 2;

template <typename XT, typename YT, typename OT>
class MulPrimitiveFactory {
 public:
  explicit MulPrimitiveFactory(const dnnl::engine &engine) : engine_(engine) {}

  inner_product_forward CreateMulPrimitive(const Tensor *x_input,
                                           const Tensor *y_input,
                                           Tensor *output,
                                           const ExecutionContext &ctx) {
    /* check data format and reorder if need */
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    // TODO(intel-minghui) : Remove the restriction that only supports Input(Y)
    // as weights
    PADDLE_ENFORCE_EQ(
        (std::is_same<YT, float>::value),
        true,
        platform::errors::InvalidArgument(
            "Input(Y) must be fp32 data type since only fp32 data type is "
            "supported in the current design of MKLDNN INT8."));

    auto x_matrix = UpdateDataFormat<XT>(x_input, x_num_col_dims, ctx);
    auto y_matrix = UpdateDataFormat<YT>(y_input, y_num_col_dims, ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (mul_) {
      UpdateDataPointers(ctx, output, &x_matrix);
      Execute();
      return *(mul_);
    }

    auto src_desc = CreateMemDescriptor<XT>(&x_matrix, MKLDNNMemoryFormat::nc);
    x_input_ = CreateMemory<XT>(src_desc, &x_matrix);

    if (is_int8_) {
      const auto trans_y = TransposeInputY(&y_matrix);
      auto scale_y = ctx.Attr<std::vector<float>>("scale_y");
      y_input_ = QuantInputY(trans_y, scale_y);
    } else {
      y_input_ = TransposeInputY(&y_matrix);
    }

    auto dst_desc = CreateMemDescriptor<OT>(output, MKLDNNMemoryFormat::any);

    mul_ = CreateMulPrimitive(*x_input_, *y_input_, dst_desc, output, ctx);
    Execute();
    return *(mul_);
  }

 private:
  memory ReorderWithScale(const memory::desc &src_desc,
                          const memory::desc &dst_desc,
                          void *src_data,
                          const std::vector<float> &scale) {
    auto mask = scale.size() > 1 ? 1 : 0;
    dnnl::primitive_attr attr;
    attr.set_output_scales(mask, scale);

    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = memory(dst_desc, engine_);

    auto reorder_pd = dnnl::reorder::primitive_desc(src_mem, dst_mem, attr);

    auto reorder = dnnl::reorder(reorder_pd);

    auto &astream = platform::MKLDNNDeviceContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder(
          "int_reorder",
          platform::TracerEventType::UserDefined,
          2,
          platform::EventRole::kUniqueOp);
      reorder.execute(astream, src_mem, dst_mem);
      astream.wait();
    }

    return dst_mem;
  }

  memory QuantInputY(memory input_y, const std::vector<float> &scale_y) {
    const auto &dims = input_y.get_desc().data.dims;
    auto ndims = input_y.get_desc().data.ndims;
    auto y_dims = std::vector<int64_t>(dims, dims + ndims);

    auto user_y_desc = CreateMemDescriptor<YT>(y_dims, MKLDNNMemoryFormat::oi);
    auto y_desc = CreateMemDescriptor<int8_t>(y_dims, MKLDNNMemoryFormat::oi);

    return ReorderWithScale(
        user_y_desc, y_desc, input_y.get_data_handle(), scale_y);
  }

  dnnl::primitive_attr CreateMulAttr(const ExecutionContext &ctx,
                                     bool force_fp32_output) {
    dnnl::primitive_attr mul_attr;

    auto scale_y_data = ctx.Attr<std::vector<float>>("scale_y");
    auto scale_x_data = ctx.Attr<float>("scale_x");
    auto scale_out_data =
        force_fp32_output ? 1.0f : ctx.Attr<float>("scale_out");

    bool is_multi_channel = scale_y_data.size() > 1;
    int count = is_multi_channel ? scale_y_data.size() : 1;
    std::vector<float> output_shift_scale(count);
    for (int i = 0; i < count; i++) {
      if (scale_y_data[i] == 0.0)
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            scale_out_data / (scale_x_data * scale_y_data[i]);
    }
    int mul_mask = is_multi_channel ? 1 : 0;
    mul_attr.set_output_scales(mul_mask, output_shift_scale);

    return mul_attr;
  }

  inner_product_forward CreateMulPrimitive(const memory &x_memory,
                                           const memory &y_memory,
                                           const memory::desc &dst_desc,
                                           Tensor *output,
                                           const ExecutionContext &ctx) {
    const auto x_desc = x_memory.get_desc();
    const auto y_desc = y_memory.get_desc();
    inner_product_forward::primitive_desc mul_prim_desc;

    const auto &mul_desc = inner_product_forward::desc(
        prop_kind::forward, x_desc, y_desc, dst_desc);

    if (is_int8_) {
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
      auto mul_attr = CreateMulAttr(ctx, force_fp32_output);
      mul_prim_desc =
          inner_product_forward::primitive_desc(mul_desc, mul_attr, engine_);
    } else {
      mul_prim_desc = inner_product_forward::primitive_desc(mul_desc, engine_);
    }

    output_ = CreateDstMemory(mul_prim_desc, ctx, output);

    return inner_product_forward(mul_prim_desc);
  }

  void Execute() {
    auto &astream = platform::MKLDNNDeviceContext::tls().get_stream();
    (*mul_).execute(astream,
                    {{DNNL_ARG_SRC, *x_input_},
                     {DNNL_ARG_WEIGHTS, *y_input_},
                     {DNNL_ARG_DST, *output_}});
    astream.wait();
  }

  template <typename T>
  Tensor UpdateDataFormat(const Tensor *data,
                          int num_col_dims,
                          const ExecutionContext &ctx) {
    Tensor x_tmp;
    Tensor data_matrix;
    MKLDNNMemoryFormat src_fmt = data->format();
    MKLDNNMemoryFormat dst_fmt;
    auto src_mdesc = CreateMemDescriptor<T>(data, src_fmt);

    if ((data->dims().size() == 4 &&
         src_fmt != (dst_fmt = MKLDNNMemoryFormat::nchw)) ||
        (data->dims().size() == 5 &&
         src_fmt != (dst_fmt = MKLDNNMemoryFormat::ncdhw))) {
      auto dst_mdesc = CreateMemDescriptor<T>(data, dst_fmt);
      x_tmp.mutable_data<T>(ctx.GetPlace(), data->memory_size());

      Reorder(src_mdesc,
              dst_mdesc,
              to_void_cast<T>(data->data<T>()),
              to_void_cast<T>(x_tmp.data<T>()));

      x_tmp.Resize(data->dims());
      x_tmp.set_mem_desc(dst_mdesc);
      data_matrix = framework::ReshapeToMatrix(x_tmp, num_col_dims);
    } else {
      data_matrix = framework::ReshapeToMatrix(*data, num_col_dims);
    }

    return data_matrix;
  }

  void UpdateDataPointers(const ExecutionContext &ctx,
                          Tensor *out,
                          const Tensor *in) {
    x_input_->set_data_handle(to_void_cast<XT>(in->data<XT>()));
    output_->set_data_handle(out->mutable_data<OT>(ctx.GetPlace()));
    out->set_mem_desc(output_->get_desc());
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const Tensor *tensor,
      MKLDNNMemoryFormat format,
      memory::data_type type = platform::MKLDNNGetDataType<T>()) {
    auto dims = phi::vectorize<int64_t>(tensor->dims());
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const std::vector<int64_t> &dims,
      MKLDNNMemoryFormat format,
      memory::data_type type = platform::MKLDNNGetDataType<T>()) {
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory CreateMemory(const memory::desc &desc, const Tensor *tensor) {
    return memory(desc, engine_, to_void_cast<T>(tensor->data<T>()));
  }

  memory CreateDstMemory(
      const inner_product_forward::primitive_desc &mul_prim_desc,
      const ExecutionContext &ctx,
      Tensor *output) {
    auto dst_desc = mul_prim_desc.dst_desc();
    auto buffer_size = dst_desc.get_size();

    OT *output_data = output->mutable_data<OT>(ctx.GetPlace(), buffer_size);
    output->set_mem_desc(dst_desc);
    return memory(dst_desc, engine_, to_void_cast<OT>(output_data));
  }

  memory Reorder(const memory::desc &src_desc,
                 const memory::desc &dst_desc,
                 void *src_data,
                 void *dst_data = NULL) {
    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = dst_data ? memory(dst_desc, engine_, dst_data)
                            : memory(dst_desc, engine_);

    auto reorder = dnnl::reorder(src_mem, dst_mem);

    auto &astream = platform::MKLDNNDeviceContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder(
          "int_reorder",
          platform::TracerEventType::UserDefined,
          2,
          platform::EventRole::kUniqueOp);
      reorder.execute(astream, src_mem, dst_mem);
      astream.wait();
    }

    return dst_mem;
  }

  memory TransposeInputY(const Tensor *input_y) {
    auto dims = phi::vectorize<int64_t>(input_y->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<YT>(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor<YT>(dims, MKLDNNMemoryFormat::oi);
    return Reorder(src_desc, dst_desc, to_void_cast<YT>(input_y->data<YT>()));
  }

  const dnnl::engine &engine_;
  paddle::optional<memory> x_input_;
  paddle::optional<memory> y_input_;
  paddle::optional<memory> output_;
  paddle::optional<inner_product_forward> mul_;
  static constexpr bool is_int8_ =
      std::is_same<XT, int8_t>::value || std::is_same<XT, uint8_t>::value;
};

/* OT: output data type */
template <typename XT, typename YT, typename OT>
std::shared_ptr<MulPrimitiveFactory<XT, YT, OT>> GetPrimitiveFactory(
    const MKLDNNDeviceContext &dev_ctx,
    const ExecutionContext &ctx,
    const Tensor *input_x,
    const Tensor *input_y,
    const dnnl::engine &mkldnn_engine) {
  std::string key =
      platform::CreateKey(dev_ctx,
                          framework::TransToProtoVarType(input_x->dtype()),
                          phi::vectorize(input_x->dims()),
                          framework::TransToProtoVarType(input_y->dtype()),
                          phi::vectorize(input_y->dims()),
                          ctx.OutputName("Out"));
  key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

  auto prim_creator = std::static_pointer_cast<MulPrimitiveFactory<XT, YT, OT>>(
      dev_ctx.GetBlob(key));

  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<MulPrimitiveFactory<XT, YT, OT>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename XT, typename YT>
inner_product_forward GetMulPrimitive(const MKLDNNDeviceContext &dev_ctx,
                                      const ExecutionContext &ctx,
                                      const Tensor *input_x,
                                      const Tensor *input_y,
                                      Tensor *output,
                                      const dnnl::engine &mkldnn_engine) {
  constexpr bool is_int8 =
      std::is_same<XT, int8_t>::value || std::is_same<XT, uint8_t>::value;
  bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

  if (is_int8 && !force_fp32_output) {
    return GetPrimitiveFactory<XT, YT, int8_t>(
               dev_ctx, ctx, input_x, input_y, mkldnn_engine)
        ->CreateMulPrimitive(input_x, input_y, output, ctx);

  } else {
    return GetPrimitiveFactory<XT, YT, float>(
               dev_ctx, ctx, input_x, input_y, mkldnn_engine)
        ->CreateMulPrimitive(input_x, input_y, output, ctx);
  }
}

/* XT: input x data type, YT: input y data type */
template <typename XT, typename YT>
class MulMKLDNNINT8Kernel : public framework::OpKernel<XT> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Mul must use CPUPlace"));
    platform::MKLDNNDeviceContext::tls().log_lib_version();
    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto &mkldnn_engine = dev_ctx.GetEngine();

    const Tensor *x = ctx.Input<Tensor>("X");
    const Tensor *y = ctx.Input<Tensor>("Y");
    Tensor *out = ctx.Output<Tensor>("Out");
    auto out_dims = out->dims();

    auto mul = GetMulPrimitive<XT, YT>(dev_ctx, ctx, x, y, out, mkldnn_engine);

    if (out_dims.size() != 2) {
      out->Resize(out_dims);
    }

    auto in_md = dnnl::memory::desc(*dnnl_primitive_desc_query_md(
        mul.get_primitive_desc(), dnnl_query_dst_md, 0));
    out->set_mem_desc(in_md.reshape(phi::vectorize<int64_t>(out->dims())));
  }
};

template <typename XT, typename YT>
class MulMKLDNNKernel : public framework::OpKernel<XT> {
 public:
  void Compute(const ExecutionContext &ctx) const override { RunKernel(ctx); }

 protected:
  void ExecuteMatMul(const ExecutionContext &ctx,
                     const MKLDNNDeviceContext &dev_ctx,
                     const dnnl::engine &onednn_engine,
                     const platform::Place &cpu_place,
                     const Tensor *x,
                     const std::vector<int64_t> &x_dims,
                     bool trans_x,
                     const Tensor *y,
                     const std::vector<int64_t> &y_dims,
                     bool trans_y,
                     Tensor *out) const {
    static const std::vector<int64_t> vec_placeholder;
    MatMulV2MKLDNNHandler<XT, YT, XT> handler(ctx,
                                              onednn_engine,
                                              ctx.GetPlace(),
                                              x_dims,
                                              trans_x,
                                              y_dims,
                                              trans_y,
                                              false,
                                              vec_placeholder,
                                              vec_placeholder);

    const auto src_memory_p = handler.AcquireSrcMemory(x);
    const auto weights_memory_p = handler.AcquireWeightsMemory(y);
    const auto dst_memory_p = handler.AcquireDstMemory(out);

    auto matmul_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto &astream = MKLDNNDeviceContext::tls().get_stream();
    matmul_p->execute(astream, matmul_args);
    astream.wait();

    // This kernel is flattening dims so then we need to unflattened version
    // that should be set in out reshape require plain layout, but
    // MatmulV2MKLDNNHanlder enforces one so it should work
    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(phi::vectorize<int64_t>(out->dims())));
  }

 private:
  void RunKernel(const ExecutionContext &ctx) const {
    const auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    const auto *x = ctx.Input<Tensor>("X");
    const auto *y = ctx.Input<Tensor>("Y");
    auto *out = ctx.Output<Tensor>("Out");

    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    const Tensor x_matrix = x->dims().size() > 2
                                ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                                : *x;
    const Tensor y_matrix = y->dims().size() > 2
                                ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                                : *y;

    // adding mb dim because MatMulV2 handler needs it
    std::vector<int64_t> y_dims(3, 1);
    std::vector<int64_t> x_dims(3, 1);

    y_dims[1] = y_matrix.dims()[0];
    y_dims[2] = y_matrix.dims()[1];

    x_dims[1] = x_matrix.dims()[0];
    x_dims[2] = x_matrix.dims()[1];

    ExecuteMatMul(ctx,
                  dev_ctx,
                  onednn_engine,
                  ctx.GetPlace(),
                  &x_matrix,
                  x_dims,
                  false,
                  &y_matrix,
                  y_dims,
                  false,
                  out);
  }
};

template <typename XT, typename YT>
class MulGradMKLDNNKernel : public MulMKLDNNKernel<XT, YT> {
 public:
  void Compute(const ExecutionContext &ctx) const override { RunKernel(ctx); }

 private:
  template <typename OT = XT>
  void RunKernel(const ExecutionContext &ctx) const {
    const auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &onednn_engine = dev_ctx.GetEngine();

    const auto *x = ctx.Input<LoDTensor>("X");
    const auto *y = ctx.Input<LoDTensor>("Y");
    const auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto *dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<LoDTensor>(framework::GradVarName("Y"));

    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    const Tensor x_matrix = x->dims().size() > 2
                                ? framework::ReshapeToMatrix(*x, x_num_col_dims)
                                : static_cast<const Tensor &>(*x);
    const Tensor y_matrix = y->dims().size() > 2
                                ? framework::ReshapeToMatrix(*y, y_num_col_dims)
                                : static_cast<const Tensor &>(*y);

    Tensor dout_matrix = *dout;
    dout_matrix.Resize({phi::flatten_to_2d(x->dims(), x_num_col_dims)[0],
                        phi::flatten_to_2d(y->dims(), y_num_col_dims)[1]});

    // adding mb dim because MatMulV2 handler needs it
    std::vector<int64_t> x_dims(3, 1);
    std::vector<int64_t> y_dims(3, 1);
    std::vector<int64_t> dout_dims(3, 1);

    x_dims[1] = x_matrix.dims()[0];
    x_dims[2] = x_matrix.dims()[1];

    y_dims[1] = y_matrix.dims()[0];
    y_dims[2] = y_matrix.dims()[1];

    dout_dims[1] = dout_matrix.dims()[0];
    dout_dims[2] = dout_matrix.dims()[1];

    if (dx != nullptr) {
      dx->set_lod(x->lod());
      this->ExecuteMatMul(ctx,
                          dev_ctx,
                          onednn_engine,
                          ctx.GetPlace(),
                          &dout_matrix,
                          dout_dims,
                          false,
                          &y_matrix,
                          y_dims,
                          true,
                          static_cast<Tensor *>(dx));
    }
    if (dy != nullptr) {
      dy->set_lod(y->lod());
      this->ExecuteMatMul(ctx,
                          dev_ctx,
                          onednn_engine,
                          ctx.GetPlace(),
                          &x_matrix,
                          x_dims,
                          true,
                          &dout_matrix,
                          dout_dims,
                          false,
                          static_cast<Tensor *>(dy));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    U8,
                                    ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNINT8Kernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    S8,
                                    ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNINT8Kernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kMULMKLDNNFP32,
                                    ops::MulMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    mul,
    MKLDNN,
    ::paddle::platform::CPUPlace,
    BF16,
    ops::kMULMKLDNNFP32,
    ops::MulMKLDNNKernel<paddle::platform::bfloat16,
                         paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(mul,
                   MKLDNN,
                   ::paddle::platform::CPUPlace,
                   ops::MulMKLDNNINT8Kernel<uint8_t, float>,
                   ops::MulMKLDNNKernel<paddle::platform::bfloat16,
                                        paddle::platform::bfloat16>,
                   ops::MulMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul_grad,
                                    MKLDNN,
                                    ::paddle::platform::CPUPlace,
                                    FP32,
                                    ops::kMULMKLDNNFP32,
                                    ops::MulGradMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    mul_grad,
    MKLDNN,
    ::paddle::platform::CPUPlace,
    BF16,
    ops::kMULMKLDNNFP32,
    ops::MulGradMKLDNNKernel<paddle::platform::bfloat16,
                             paddle::platform::bfloat16>,
    ops::MulGradMKLDNNKernel<float, float>);
