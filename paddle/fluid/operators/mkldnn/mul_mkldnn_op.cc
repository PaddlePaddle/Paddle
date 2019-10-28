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
#include <vector>
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::DDim;
using framework::ExecutionContext;
using framework::Tensor;
using mkldnn::inner_product_forward;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename XT, typename YT, typename OT>
class MulPrimitiveFactory {
 public:
  explicit MulPrimitiveFactory(const mkldnn::engine &engine)
      : engine_(engine) {}

  virtual ~MulPrimitiveFactory() {}

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor *input_x, const Tensor *input_y, Tensor *output,
      const ExecutionContext &ctx) {
    /* check format and reorder if need */
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    auto x_matrix = UpdateDataFormat<XT>(input_x, x_num_col_dims, ctx);
    auto y_matrix = UpdateDataFormat<YT>(input_y, y_num_col_dims, ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (mul_) {
      UpdateDataPointers(ctx, output, &x_matrix);
      return *mul_;
    }

    auto src_desc = CreateMemDescriptor<XT>(&x_matrix, MKLDNNMemoryFormat::nc);
    x_input_ = CreateMemory<XT>(src_desc, &x_matrix);
    y_input_ = TransposeInputY(&y_matrix);
    auto dst_desc = CreateMemDescriptor<OT>(output, MKLDNNMemoryFormat::any);

    mul_ = CreateMulPrimitive(*x_input_, *y_input_, dst_desc, output, ctx);
    return *mul_;
  }

 protected:
  template <typename T>
  Tensor UpdateDataFormat(const Tensor *data, int num_col_dims,
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

      Reorder(src_mdesc, dst_mdesc, to_void_cast<T>(data->data<T>()),
              to_void_cast<T>(x_tmp.data<T>()));

      x_tmp.Resize(data->dims());
      x_tmp.set_format((MKLDNNMemoryFormat)dst_mdesc.data.format);
      data_matrix = framework::ReshapeToMatrix(x_tmp, num_col_dims);
    } else {
      data_matrix = framework::ReshapeToMatrix(*data, num_col_dims);
    }

    return data_matrix;
  }

  void UpdateDataPointers(const ExecutionContext &ctx, Tensor *out,
                          const Tensor *in) {
    x_input_->set_data_handle(to_void_cast<XT>(in->data<XT>()));
    output_->set_data_handle(out->mutable_data<OT>(ctx.GetPlace()));

    if (out->format() == MKLDNNMemoryFormat::format_undef) {
      auto output_format = platform::GetMKLDNNFormat(*output_);
      out->set_format((MKLDNNMemoryFormat)output_format);
    }
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const Tensor *tensor, MKLDNNMemoryFormat format,
      memory::data_type type = platform::MKLDNNGetDataType<T>()) {
    auto dims = framework::vectorize<int>(tensor->dims());
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const std::vector<int> &dims, MKLDNNMemoryFormat format,
      memory::data_type type = platform::MKLDNNGetDataType<T>()) {
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory CreateMemory(const memory::desc &desc, const Tensor *tensor) {
    return memory({desc, engine_}, to_void_cast<T>(tensor->data<T>()));
  }

  memory CreateDstMemory(
      const inner_product_forward::primitive_desc &mul_prim_desc,
      const ExecutionContext &ctx, Tensor *output) {
    auto dst_prim_desc = mul_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_prim_desc.get_size();

    OT *output_data = output->mutable_data<OT>(ctx.GetPlace(), buffer_size);
    memory dst_mem(dst_prim_desc, to_void_cast<OT>(output_data));
    output->set_format(platform::GetMKLDNNFormat(dst_mem));
    return dst_mem;
  }

  memory Reorder(const memory::desc &src_desc, const memory::desc &dst_desc,
                 void *src_data, void *dst_data = NULL) {
    auto src_mem = memory({src_desc, engine_}, src_data);
    auto dst_mem = dst_data ? memory({dst_desc, engine_}, dst_data)
                            : memory({dst_desc, engine_});

    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  memory TransposeInputY(const Tensor *input_y) {
    auto dims = framework::vectorize<int>(input_y->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<YT>(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor<YT>(dims, MKLDNNMemoryFormat::oi);
    return Reorder(src_desc, dst_desc, to_void_cast<YT>(input_y->data<YT>()));
  }

  inner_product_forward CreateMulPrimitive(const memory &x_memory,
                                           const memory &y_memory,
                                           const memory::desc &dst_desc,
                                           Tensor *output,
                                           const ExecutionContext &ctx) {
    const auto y_desc = y_memory.get_primitive_desc().desc();
    const auto x_desc = x_memory.get_primitive_desc().desc();

    auto mul_prim_desc = CreateMulPrimDesc(x_desc, y_desc, dst_desc);
    output_ = CreateDstMemory(mul_prim_desc, ctx, output);

    return inner_product_forward(mul_prim_desc, x_memory, y_memory, *output_);
  }

  inner_product_forward::primitive_desc CreateMulPrimDesc(
      const memory::desc &x_desc, const memory::desc &y_desc,
      const memory::desc &dst_desc) {
    auto mul_desc = inner_product_forward::desc(prop_kind::forward, x_desc,
                                                y_desc, dst_desc);

    return inner_product_forward::primitive_desc(mul_desc, engine_);
  }

 protected:
  const mkldnn::engine &engine_;
  boost::optional<memory> x_input_;
  boost::optional<memory> y_input_;
  boost::optional<memory> output_;
  boost::optional<inner_product_forward> mul_;
};  // namespace operators

template <typename XT, typename YT, typename OT>
class QuantMulPrimitiveFactory : public MulPrimitiveFactory<XT, YT, OT> {
 public:
  using MulPrimitiveFactory<XT, YT, OT>::MulPrimitiveFactory;

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor *x_input, const Tensor *y_input, Tensor *output,
      const ExecutionContext &ctx) {
    /* check data format and reorder if need */
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto scale_y = ctx.Attr<std::vector<float>>("scale_y");

    // TODO(intel-minghui) : Remove the restriction that only supports Input(Y)
    // as weights
    bool enforce = std::is_same<YT, float>::value;
    PADDLE_ENFORCE(
        enforce == true,
        "Input(Y) supposed to be fp32 data type since only fp32 data type is "
        "supported in the current design of MKLDNN INT8.");

    auto x_matrix =
        this->template UpdateDataFormat<XT>(x_input, x_num_col_dims, ctx);
    auto y_matrix =
        this->template UpdateDataFormat<YT>(y_input, y_num_col_dims, ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (this->mul_) {
      this->UpdateDataPointers(ctx, output, &x_matrix);
      return *(this->mul_);
    }

    auto src_desc = this->template CreateMemDescriptor<XT>(
        &x_matrix, MKLDNNMemoryFormat::nc);
    this->x_input_ = this->template CreateMemory<XT>(src_desc, &x_matrix);

    const auto trans_y = this->TransposeInputY(&y_matrix);
    this->y_input_ = QuantInputY(trans_y, scale_y);

    auto dst_desc =
        this->template CreateMemDescriptor<OT>(output, MKLDNNMemoryFormat::any);

    this->mul_ = CreateMulPrimitive(*(this->x_input_), *(this->y_input_),
                                    dst_desc, output, ctx);
    return *(this->mul_);
  }

  memory ReorderWithScale(const memory::desc &src_desc,
                          const memory::desc &dst_desc, void *src_data,
                          const std::vector<float> &scale) {
    auto mask = scale.size() > 1 ? 1 : 0;
    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scale);

    auto src_mem = memory({src_desc, this->engine_}, src_data);
    auto dst_mem = memory({dst_desc, this->engine_});

    auto reorder_pd = mkldnn::reorder::primitive_desc(
        src_mem.get_primitive_desc(), dst_mem.get_primitive_desc(), attr);

    auto reorder = mkldnn::reorder(reorder_pd, src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  memory QuantInputY(memory input_y, const std::vector<float> &scale_y) {
    const auto &dims = input_y.get_primitive_desc().desc().data.dims;
    auto ndims = input_y.get_primitive_desc().desc().data.ndims;
    auto y_dims = std::vector<int>(dims, dims + ndims);

    auto user_y_desc =
        this->template CreateMemDescriptor<YT>(y_dims, MKLDNNMemoryFormat::oi);
    auto y_desc = this->template CreateMemDescriptor<int8_t>(
        y_dims, MKLDNNMemoryFormat::oi);

    return ReorderWithScale(user_y_desc, y_desc, input_y.get_data_handle(),
                            scale_y);
  }

  mkldnn::primitive_attr CreateMulAttr(const ExecutionContext &ctx,
                                       bool force_fp32_output) {
    mkldnn::primitive_attr mul_attr;

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
    const auto x_desc = x_memory.get_primitive_desc().desc();
    const auto y_desc = y_memory.get_primitive_desc().desc();
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    mkldnn::primitive_attr mul_attr = CreateMulAttr(ctx, force_fp32_output);
    auto mul_prim_desc = CreateMulPrimDesc(x_desc, y_desc, dst_desc, mul_attr);

    this->output_ = this->CreateDstMemory(mul_prim_desc, ctx, output);

    return inner_product_forward(mul_prim_desc, x_memory, y_memory,
                                 *(this->output_));
  }

  inner_product_forward::primitive_desc CreateMulPrimDesc(
      const memory::desc &x_desc, const memory::desc &y_desc,
      const memory::desc &dst_desc, const mkldnn::primitive_attr &mul_attr) {
    const auto &mul_desc = inner_product_forward::desc(
        prop_kind::forward, x_desc, y_desc, dst_desc);

    return inner_product_forward::primitive_desc(mul_desc, mul_attr,
                                                 this->engine_);
  }
};

/* OT: output data type */
template <typename XT, typename YT, typename OT>
std::shared_ptr<MulPrimitiveFactory<XT, YT, OT>> GetPrimitiveFactory(
    const MKLDNNDeviceContext &dev_ctx, const ExecutionContext &ctx,
    const Tensor *input_x, const Tensor *input_y,
    const mkldnn::engine &mkldnn_engine, bool enable_quant) {
  const std::string key = platform::CreateKey(
      input_x->type(), framework::vectorize<int>(input_x->dims()),
      input_y->type(), framework::vectorize<int>(input_y->dims()),
      ctx.op().Output("Out"));

  auto prim_creator = std::static_pointer_cast<MulPrimitiveFactory<XT, YT, OT>>(
      dev_ctx.GetBlob(key));

  if (prim_creator == nullptr) {
    prim_creator =
        enable_quant
            ? std::make_shared<QuantMulPrimitiveFactory<XT, YT, OT>>(
                  mkldnn_engine)
            : std::make_shared<MulPrimitiveFactory<XT, YT, OT>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename XT, typename YT>
inner_product_forward GetMulPrimitive(const MKLDNNDeviceContext &dev_ctx,
                                      const ExecutionContext &ctx,
                                      const Tensor *input_x,
                                      const Tensor *input_y, Tensor *output,
                                      const mkldnn::engine &mkldnn_engine) {
  bool enable_quant =
      std::is_same<XT, int8_t>::value || std::is_same<XT, uint8_t>::value;
  bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

  if (enable_quant && !force_fp32_output) {
    return GetPrimitiveFactory<XT, YT, int8_t>(dev_ctx, ctx, input_x, input_y,
                                               mkldnn_engine, enable_quant)
        ->CreateMulPrimitive(input_x, input_y, output, ctx);

  } else {
    return GetPrimitiveFactory<XT, YT, float>(dev_ctx, ctx, input_x, input_y,
                                              mkldnn_engine, enable_quant)
        ->CreateMulPrimitive(input_x, input_y, output, ctx);
  }
}

/* XT: input x data type, YT: input y data type */
template <typename XT, typename YT>
class MulMKLDNNKernel : public framework::OpKernel<XT> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &mkldnn_engine = dev_ctx.GetEngine();

    const Tensor *x = ctx.Input<Tensor>("X");
    const Tensor *y = ctx.Input<Tensor>("Y");
    Tensor *out = ctx.Output<Tensor>("Out");
    auto out_dims = out->dims();

    auto mul = GetMulPrimitive<XT, YT>(dev_ctx, ctx, x, y, out, mkldnn_engine);

    stream(stream::kind::eager).submit({mul}).wait();

    if (out_dims.size() != 2) {
      out->Resize(out_dims);
    }
    out->set_layout(DataLayout::kMKLDNN);
    out->set_format(platform::MKLDNNFormatForSize(
        out_dims.size(), mkldnn::memory::format::nchw));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kMULMKLDNNINT8,
                                    ops::MulMKLDNNKernel<int8_t, float>);

REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MulMKLDNNKernel<uint8_t, float>);
