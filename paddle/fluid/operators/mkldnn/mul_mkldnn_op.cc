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

using framework::DDim;
using framework::DataLayout;
using framework::ExecutionContext;
using framework::Tensor;
using mkldnn::inner_product_forward;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename T, typename K>
class MulPrimitiveFactory {
 public:
  explicit MulPrimitiveFactory(const mkldnn::engine &engine)
      : engine_(engine) {}

  virtual ~MulPrimitiveFactory() {}

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor *input, const Tensor *weights, Tensor *output,
      const ExecutionContext &ctx) {
    /* check format and reorder if need */
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");

    memory::data_type x_dt = paddle::framework::ToMKLDNNDataType(input->type());
    memory::data_type y_dt =
        paddle::framework::ToMKLDNNDataType(weights->type());

    auto x_matrix = UpdateDataFormat(input, false,
                                     /*is_weights*/ x_num_col_dims, x_dt, ctx);
    auto y_matrix = UpdateDataFormat(weights, true,
                                     /*is_weights*/ y_num_col_dims, y_dt, ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (fc_) {
      UpdateDataPointers(ctx, output, &x_matrix);
      return *fc_;
    }

    auto src_desc = CreateMemDescriptor(&x_matrix, x_dt, memory::format::nc);
    input_ = CreateMemory(src_desc, &x_matrix, false /*is_weights*/);
    weights_ = TransposeWeights(&y_matrix);

    /* for non-quantize mul, output type supposed to be fp32 */
    auto dst_desc = CreateMemDescriptor<float>(output, memory::format::any);

    fc_ = CreateFcPrimitive(*input_, *weights_, dst_desc, output, ctx);
    return *fc_;
  }

 protected:
  void UpdateDataPointers(const ExecutionContext &ctx, Tensor *out,
                          const Tensor *in) {
    input_->set_data_handle(const_cast<T *>(in->data<T>()));
    output_->set_data_handle(out->mutable_data<float>(ctx.GetPlace()));
    if (out->format() == memory::format::format_undef) {
      auto output_format = output_->get_primitive_desc().desc().data.format;
      out->set_format((memory::format)output_format);
    }
  }

  Tensor UpdateDataFormat(const Tensor *data, bool is_weights, int num_col_dims,
                          memory::data_type type, const ExecutionContext &ctx) {
    Tensor x_tmp;
    Tensor data_matrix;

    memory::format src_fmt = data->format();
    memory::format dst_fmt;
    auto dims = framework::vectorize2int(data->dims());
    auto src_mdesc = CreateMemDescriptor(dims, type, src_fmt);

    if ((data->dims().size() == 4 &&
         src_fmt != (dst_fmt = memory::format::nchw)) ||
        (data->dims().size() == 5 &&
         dst_fmt != (dst_fmt = memory::format::ncdhw))) {
      auto dst_mdesc = CreateMemDescriptor(dims, type, dst_fmt);

      if (is_weights) {
        x_tmp.mutable_data<K>(ctx.GetPlace(),
                              ::paddle::memory::Allocator::kDefault,
                              data->memory_size());

        auto src_mem =
            memory({src_mdesc, engine_}, to_void_cast<K>(data->data<K>()));
        auto dst_mem =
            memory({dst_mdesc, engine_}, to_void_cast<K>(x_tmp.data<K>()));

        auto reorder = mkldnn::reorder(src_mem, dst_mem);
        stream(stream::kind::eager).submit({reorder}).wait();
      } else {
        x_tmp.mutable_data<T>(ctx.GetPlace(),
                              ::paddle::memory::Allocator::kDefault,
                              data->memory_size());

        auto src_mem =
            memory({src_mdesc, engine_}, to_void_cast<T>(data->data<T>()));
        auto dst_mem =
            memory({dst_mdesc, engine_}, to_void_cast<T>(x_tmp.data<T>()));

        auto reorder = mkldnn::reorder(src_mem, dst_mem);
        stream(stream::kind::eager).submit({reorder}).wait();
      }

      x_tmp.Resize(data->dims());
      x_tmp.set_format((memory::format)dst_mdesc.data.format);
      data_matrix = framework::ReshapeToMatrix(x_tmp, num_col_dims);
    } else {
      data_matrix = framework::ReshapeToMatrix(*data, num_col_dims);
    }

    return data_matrix;
  }

  memory Reorder(const memory::desc &src_desc, const memory::desc &dst_desc,
                 const void *src_data) {
    auto src_mem = memory({src_desc, engine_}, const_cast<void *>(src_data));
    auto dst_mem = memory({dst_desc, engine_});

    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  memory TransposeWeights(const Tensor *weights) {
    auto dims = framework::vectorize2int(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<K>(dims, memory::format::io);
    auto dst_desc = CreateMemDescriptor<K>(dims, memory::format::oi);
    return Reorder(src_desc, dst_desc, weights->data<K>());
  }

  template <typename DT>
  memory::desc CreateMemDescriptor(const std::vector<int> &dims,
                                   memory::format format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<DT>(),
                                   format);
  }

  template <typename DT>
  memory::desc CreateMemDescriptor(const Tensor *tensor,
                                   memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor<DT>(dims, format);
  }

  memory::desc CreateMemDescriptor(const std::vector<int> &dims,
                                   memory::data_type type,
                                   memory::format format) {
    return platform::MKLDNNMemDesc(dims, type, format);
  }

  memory::desc CreateMemDescriptor(const Tensor *tensor, memory::data_type type,
                                   memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor(dims, type, format);
  }

  memory CreateMemory(const memory::desc &desc, const Tensor *tensor,
                      bool is_weights) {
    if (is_weights)
      return CreateMemory(desc, tensor->data<K>());
    else
      return CreateMemory(desc, tensor->data<T>());
  }

  memory CreateMemory(const memory::desc &desc, const void *data) {
    return memory({desc, engine_}, const_cast<void *>(data));
  }

  inner_product_forward CreateFcPrimitive(const memory &src_memory,
                                          const memory &weights_memory,
                                          const memory::desc &dst_desc,
                                          Tensor *output,
                                          const ExecutionContext &ctx) {
    const auto weights_desc = weights_memory.get_primitive_desc().desc();
    const auto src_desc = src_memory.get_primitive_desc().desc();

    auto fc_prim_desc = CreateFcPrimDesc(src_desc, weights_desc, dst_desc);
    output_ = this->CreateDstMemory(fc_prim_desc, ctx, output);

    return inner_product_forward(fc_prim_desc, src_memory, weights_memory,
                                 *output_);
  }

  inner_product_forward::primitive_desc CreateFcPrimDesc(
      const memory::desc &input_desc, const memory::desc &weights_desc,
      const memory::desc &dst_desc) {
    auto fc_desc = inner_product_forward::desc(prop_kind::forward, input_desc,
                                               weights_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, engine_);
  }

  memory CreateDstMemory(
      const inner_product_forward::primitive_desc &fc_prim_desc,
      const ExecutionContext &ctx, Tensor *output) {
    auto dst_prim_desc = fc_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_prim_desc.get_size();

    float *output_data = output->mutable_data<float>(
        ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
    output->set_format((memory::format)dst_prim_desc.desc().data.format);
    return memory(dst_prim_desc, to_void_cast<float>(output_data));
  }

 protected:
  const mkldnn::engine &engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};  // namespace operators

template <typename T, typename K>
class QuantMulPrimitiveFactory : public MulPrimitiveFactory<T, K> {
 public:
  using MulPrimitiveFactory<T, K>::MulPrimitiveFactory;

  virtual inner_product_forward CreateMulPrimitive(
      const Tensor *input, const Tensor *weights, Tensor *output,
      const ExecutionContext &ctx) {
    /* check data format and reorder if need */
    int x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int y_num_col_dims = ctx.Attr<int>("y_num_col_dims");
    auto scale_weight = ctx.Attr<std::vector<float>>("scale_y");

    memory::data_type x_dt = framework::ToMKLDNNDataType(input->type());
    memory::data_type y_dt = framework::ToMKLDNNDataType(weights->type());
    auto x_matrix = this->UpdateDataFormat(input, false, /*is_weights*/
                                           x_num_col_dims, x_dt, ctx);
    auto y_matrix = this->UpdateDataFormat(weights, true, /*is_weights*/
                                           y_num_col_dims, y_dt, ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (this->fc_) {
      UpdateDataPointers(ctx, output, &x_matrix);
      return *(this->fc_);
    }

    auto src_desc =
        this->CreateMemDescriptor(&x_matrix, x_dt, memory::format::nc);
    this->input_ =
        this->CreateMemory(src_desc, &x_matrix, false /*is_weights*/);

    if (std::is_same<K, float>::value) {
      const auto trans_weights_ = TransposeWeights(&y_matrix, y_dt);
      this->weights_ = QuantWeights(trans_weights_, scale_weight);
    } else if (std::is_same<K, int8_t>::value) {
      this->weights_ = this->TransposeWeights(&y_matrix, y_dt);
    } /* else { something goes wrong } */

    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    memory::data_type dst_dt;
    if (force_fp32_output) {
      dst_dt = framework::ToMKLDNNDataType(
          framework::DataTypeTrait<float>::DataType);
    } else {
      dst_dt = framework::ToMKLDNNDataType(
          framework::DataTypeTrait<int8_t>::DataType);
    }

    auto dst_desc =
        this->CreateMemDescriptor(output, dst_dt, memory::format::any);

    this->fc_ = CreateFcPrimitive(*(this->input_), *(this->weights_), dst_desc,
                                  output, ctx);

    return *(this->fc_);
  }

  void UpdateDataPointers(const ExecutionContext &ctx, Tensor *out,
                          const Tensor *in) {
    this->input_->set_data_handle(const_cast<T *>(in->data<T>()));

    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    if (force_fp32_output) {
      this->output_->set_data_handle(out->mutable_data<float>(ctx.GetPlace()));
    } else {
      this->output_->set_data_handle(out->mutable_data<int8_t>(ctx.GetPlace()));
    }

    if (out->format() == memory::format::format_undef) {
      auto output_format =
          this->output_->get_primitive_desc().desc().data.format;
      out->set_format((memory::format)output_format);
    }
  }

  memory ReorderWithScale(const memory::desc &src_desc,
                          const memory::desc &dst_desc, const void *src_data,
                          const std::vector<float> &scale) {
    auto mask = scale.size() > 1 ? 1 : 0;
    mkldnn::primitive_attr attr;
    attr.set_output_scales(mask, scale);

    auto src_mem =
        memory({src_desc, this->engine_}, const_cast<void *>(src_data));
    auto dst_mem = memory({dst_desc, this->engine_});

    auto reorder_pd = mkldnn::reorder::primitive_desc(
        src_mem.get_primitive_desc(), dst_mem.get_primitive_desc(), attr);

    auto reorder = mkldnn::reorder(reorder_pd, src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();

    return dst_mem;
  }

  memory TransposeWeights(const Tensor *weights, memory::data_type type) {
    auto dims = framework::vectorize2int(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = this->CreateMemDescriptor(dims, type, memory::format::io);
    auto dst_desc = this->CreateMemDescriptor(dims, type, memory::format::oi);
    return this->Reorder(src_desc, dst_desc, weights->data<K>());
  }

  memory QuantWeights(memory weights, const std::vector<float> &scale_weight) {
    const auto &dims = weights.get_primitive_desc().desc().data.dims;
    auto ndims = weights.get_primitive_desc().desc().data.ndims;
    auto weights_dims = std::vector<int>(dims, dims + ndims);
    auto weights_dt = platform::MKLDNNGetDataType<K>();

    auto weights_desc = this->CreateMemDescriptor(
        weights_dims, memory::data_type::s8, memory::format::oi);

    auto user_weights_desc =
        this->CreateMemDescriptor(weights_dims, weights_dt, memory::format::oi);

    return ReorderWithScale(user_weights_desc, weights_desc,
                            weights.get_data_handle(), scale_weight);
  }

  inner_product_forward CreateFcPrimitive(const memory &src_memory,
                                          const memory &weights_memory,
                                          const memory::desc &dst_desc,
                                          Tensor *output,
                                          const ExecutionContext &ctx) {
    const auto weights_desc = weights_memory.get_primitive_desc().desc();
    const auto src_desc = src_memory.get_primitive_desc().desc();

    mkldnn::primitive_attr fc_attr;

    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("scale_y");
    auto scale_in_data = ctx.Attr<float>("scale_x");
    auto scale_out_data =
        force_fp32_output ? 1.0f : ctx.Attr<float>("scale_out");

    bool is_multi_channel = scale_weights_data.size() > 1;
    int count = is_multi_channel ? scale_weights_data.size() : 1;
    std::vector<float> output_shift_scale(count);
    for (int i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0)
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            scale_out_data / (scale_in_data * scale_weights_data[i]);
    }
    int fc_mask = is_multi_channel ? 1 : 0;
    fc_attr.set_output_scales(fc_mask, output_shift_scale);

    auto fc_prim_desc =
        CreateFcPrimDesc(src_desc, weights_desc, dst_desc, fc_attr);

    this->output_ =
        CreateDstMemory(fc_prim_desc, ctx, output, force_fp32_output);

    return inner_product_forward(fc_prim_desc, src_memory, weights_memory,
                                 *(this->output_));
  }

  inner_product_forward::primitive_desc CreateFcPrimDesc(
      const memory::desc &input_desc, const memory::desc &weights_desc,
      const memory::desc &dst_desc, const mkldnn::primitive_attr &fc_attr) {
    const auto &fc_desc = inner_product_forward::desc(
        prop_kind::forward, input_desc, weights_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, fc_attr,
                                                 this->engine_);
  }

  memory CreateDstMemory(
      const inner_product_forward::primitive_desc &fc_prim_desc,
      const ExecutionContext &ctx, Tensor *output, bool force_fp32_output) {
    auto dst_prim_desc = fc_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_prim_desc.get_size();

    if (force_fp32_output) {
      float *output_data = output->mutable_data<float>(
          ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
      output->set_format((memory::format)dst_prim_desc.desc().data.format);

      return memory(dst_prim_desc, to_void_cast<float>(output_data));
    } else {
      int8_t *output_data = output->mutable_data<int8_t>(
          ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
      output->set_format((memory::format)dst_prim_desc.desc().data.format);

      return memory(dst_prim_desc, to_void_cast<int8_t>(output_data));
    }
  }
};

static std::string GetHash(const Tensor *input, const Tensor *weights,
                           const std::string &suffix) {
  auto dim2str = [](const DDim &operand_dims) {
    std::string str = "";
    for (int i = 0; i < operand_dims.size(); ++i) {
      str += std::to_string(operand_dims[i]) + "-";
    }
    return str;
  };

  std::string hash = std::to_string((unsigned)input->format()) +
                     std::to_string((unsigned)input->type()) +
                     dim2str(input->dims()) +
                     std::to_string((unsigned)weights->format()) +
                     std::to_string((unsigned)weights->type()) +
                     dim2str(weights->dims()) + suffix;

  return hash;
}

template <typename T, typename K>
std::shared_ptr<MulPrimitiveFactory<T, K>> GetPrimitiveFactory(
    const MKLDNNDeviceContext &dev_ctx, const ExecutionContext &ctx,
    const Tensor *input, const Tensor *weights,
    const mkldnn::engine &mkldnn_engine) {
  const std::string key = GetHash(input, weights, ctx.op().Output("Out"));

  auto prim_creator =
      std::static_pointer_cast<MulPrimitiveFactory<T, K>>(dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    bool enable_quant =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    if (enable_quant) {
      prim_creator =
          std::make_shared<QuantMulPrimitiveFactory<T, K>>(mkldnn_engine);
    } else {
      prim_creator = std::make_shared<MulPrimitiveFactory<T, K>>(mkldnn_engine);
    }
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename T, typename K>
class MulMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto &mkldnn_engine = dev_ctx.GetEngine();

    const Tensor *x = ctx.Input<Tensor>("X");
    const Tensor *y = ctx.Input<Tensor>("Y");
    Tensor *z = ctx.Output<Tensor>("Out");

    auto prim_creator =
        GetPrimitiveFactory<T, K>(dev_ctx, ctx, x, y, mkldnn_engine);
    auto fc = prim_creator->CreateMulPrimitive(x, y, z, ctx);

    stream(stream::kind::eager).submit({fc}).wait();

    z->set_layout(DataLayout::kMKLDNN);
    z->set_format(z->format());
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

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(mul, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kMULMKLDNNFP32,
                                    ops::MulMKLDNNKernel<float, float>);

REGISTER_OP_KERNEL(mul, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MulMKLDNNKernel<float, float>);
