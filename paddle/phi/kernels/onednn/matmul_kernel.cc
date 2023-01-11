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

#include <string>

#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

using dnnl::engine;
using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;
using paddle::framework::ReshapeToMatrix;

namespace phi {

DDim GetDimsForInput(const OneDNNContext &dev_ctx,
                     DDim input_dims,
                     std::string input_name) {
  auto shape =
      dev_ctx.HasDnnAttr("fused_reshape_" + input_name)
          ? PADDLE_GET_CONST(std::vector<int>,
                             dev_ctx.GetDnnAttr("fused_reshape_" + input_name))
          : std::vector<int>();
  auto axis = dev_ctx.HasDnnAttr("fused_transpose_" + input_name)
                  ? PADDLE_GET_CONST(
                        std::vector<int>,
                        dev_ctx.GetDnnAttr("fused_transpose_" + input_name))
                  : std::vector<int>();
  if (!shape.empty() && !axis.empty()) {
    return input_dims.reshape(shape).transpose(axis);
  }
  return input_dims;
}

void CalculateMatrixDims(const std::vector<int64_t> &x_dims,
                         const std::vector<int64_t> &y_dims,
                         std::vector<int64_t> *x_bd_dims,
                         std::vector<int64_t> *y_bd_dims,
                         DenseTensor *out,
                         const bool is_output_fused) {
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

  if (!is_output_fused && x_dims.size() > 2 && y_dims.size() > 2) {
    auto out_dims = vectorize(out->dims());
    for (size_t i = 0; i < (*x_bd_dims).size() - 2; ++i) {
      PADDLE_ENFORCE_EQ(
          (*x_bd_dims)[i] == (*y_bd_dims)[i] || (*x_bd_dims)[i] == 1 ||
              (*y_bd_dims)[i] == 1,
          true,
          errors::InvalidArgument(
              "Tensor dimensions are incorrect for broadcasting."
              "Dimensions in X and Y must be same or equal to 1, but "
              "received x_dim[%d]=%d and y_dims[%d]= %d",
              i,
              (*x_bd_dims)[i],
              i,
              (*y_bd_dims)[i]));
      (out_dims)[i] = std::max((*x_bd_dims)[i], (*y_bd_dims)[i]);
    }
    out->Resize(make_ddim((out_dims)));
  }
}

template <typename T, typename Context>
void MatmulKernel(const Context &dev_ctx,
                  const DenseTensor &x,
                  const DenseTensor &y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor *out) {
  if (dev_ctx.HasDnnAttr("head_number")) {
    const auto head_number =
        PADDLE_GET_CONST(int, dev_ctx.GetDnnAttr("head_number"));
    PADDLE_ENFORCE_EQ(
        head_number,
        1,
        errors::Unimplemented(
            "oneDNN matmul doesn't support multiple heads. Expected "
            "head_number=1. But received `head_number` is %d",
            head_number));
  }

  constexpr bool is_int8 = funcs::is_int8<T>();
  constexpr bool is_bfloat16 = funcs::is_bfloat16<T>();
  const bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;

  bool fuse_relu = false;
  if (dev_ctx.HasDnnAttr("fuse_activation")) {
    auto act_type =
        PADDLE_GET_CONST(std::string, dev_ctx.GetDnnAttr("fuse_activation"));
    if (act_type == "relu" || act_type == "relu6") {
      fuse_relu = true;
    }
  }

  auto x_dims = vectorize(GetDimsForInput(dev_ctx, x.dims(), "X"));
  auto y_dims = vectorize(GetDimsForInput(dev_ctx, y.dims(), "Y"));

  int ndims = std::max(x_dims.size(), y_dims.size());
  ndims = std::max(ndims, 3);

  std::vector<int64_t> x_bd_dims(ndims, 1);
  std::vector<int64_t> y_bd_dims(ndims, 1);

  CalculateMatrixDims(x_dims,
                      y_dims,
                      &x_bd_dims,
                      &y_bd_dims,
                      out,
                      funcs::IsOutputFused(dev_ctx));

  if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
    funcs::ExecuteMatmul<T, float>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else if (is_bfloat16) {
    funcs::ExecuteMatmul<T, paddle::platform::bfloat16>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else if (fuse_relu) {
    funcs::ExecuteMatmul<T, uint8_t>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else {
    funcs::ExecuteMatmul<T, int8_t>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  }
}

template <typename XT, typename YT, typename OT>
class MulPrimitiveFactory {
 public:
  explicit MulPrimitiveFactory(const engine &engine) : engine_(engine) {}

  inner_product_forward CreateMulPrimitive(const DenseTensor *x_input,
                                           const DenseTensor *y_input,
                                           DenseTensor *output,
                                           int x_num_col_dims,
                                           int y_num_col_dims,
                                           const OneDNNContext &dev_ctx) {
    // TODO(intel-minghui) : Remove the restriction that only supports Input(Y)
    // as weights
    PADDLE_ENFORCE_EQ(
        (std::is_same<YT, float>::value),
        true,
        errors::InvalidArgument(
            "Input(Y) must be fp32 data type since only fp32 data type is "
            "supported in the current design of OneDNN INT8."));

    /* check data format and reorder if need */
    auto x_matrix = UpdateDataFormat<XT>(x_input, x_num_col_dims, dev_ctx);
    auto y_matrix = UpdateDataFormat<YT>(y_input, y_num_col_dims, dev_ctx);

    auto output_dim = output->dims();
    if (output_dim.size() != 2) {
      output->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
    }

    if (mul_) {
      UpdateDataPointers(dev_ctx, output, &x_matrix);
      Execute();
      return *(mul_);
    }

    auto src_desc =
        CreateMemDescriptor<XT>(&x_matrix, funcs::OneDNNMemoryFormat::nc);
    x_input_ = CreateMemory<XT>(src_desc, &x_matrix);

    if (is_int8_) {
      const auto trans_y = TransposeInputY(&y_matrix);
      auto scale_y = dev_ctx.HasDnnAttr("scale_y")
                         ? PADDLE_GET_CONST(std::vector<float>,
                                            dev_ctx.GetDnnAttr("scale_y"))
                         : std::vector<float>();
      y_input_ = QuantInputY(trans_y, scale_y);
    } else {
      y_input_ = TransposeInputY(&y_matrix);
    }

    auto dst_desc =
        CreateMemDescriptor<OT>(output, funcs::OneDNNMemoryFormat::any);

    mul_ = CreateMulPrimitive(*x_input_, *y_input_, dst_desc, output, dev_ctx);
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

    auto &astream = OneDNNContext::tls().get_stream();
    {
      paddle::platform::RecordEvent record_reorder(
          "int_reorder",
          paddle::platform::TracerEventType::UserDefined,
          2,
          paddle::platform::EventRole::kUniqueOp);
      reorder.execute(astream, src_mem, dst_mem);
      astream.wait();
    }

    return dst_mem;
  }

  memory QuantInputY(memory input_y, const std::vector<float> &scale_y) {
    const auto &dims = input_y.get_desc().data.dims;
    auto ndims = input_y.get_desc().data.ndims;
    auto y_dims = std::vector<int64_t>(dims, dims + ndims);

    auto user_y_desc =
        CreateMemDescriptor<YT>(y_dims, funcs::OneDNNMemoryFormat::oi);
    auto y_desc =
        CreateMemDescriptor<int8_t>(y_dims, funcs::OneDNNMemoryFormat::oi);

    return ReorderWithScale(
        user_y_desc, y_desc, input_y.get_data_handle(), scale_y);
  }

  dnnl::primitive_attr CreateMulAttr(const OneDNNContext &dev_ctx,
                                     bool force_fp32_output) {
    dnnl::primitive_attr mul_attr;

    auto scale_y_data = dev_ctx.HasDnnAttr("scale_y")
                            ? PADDLE_GET_CONST(std::vector<float>,
                                               dev_ctx.GetDnnAttr("scale_y"))
                            : std::vector<float>{1.0};
    auto scale_x_data =
        dev_ctx.HasDnnAttr("scale_x")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("scale_x"))
            : 1.0f;
    auto scale_out =
        dev_ctx.HasDnnAttr("scale_out")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("scale_out"))
            : 1.0f;
    auto scale_out_data = force_fp32_output ? 1.0f : scale_out;

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
                                           DenseTensor *output,
                                           const OneDNNContext &dev_ctx) {
    const auto x_desc = x_memory.get_desc();
    const auto y_desc = y_memory.get_desc();
    inner_product_forward::primitive_desc mul_prim_desc;

    const auto &mul_desc = inner_product_forward::desc(
        prop_kind::forward, x_desc, y_desc, dst_desc);

    if (is_int8_) {
      bool force_fp32_output =
          dev_ctx.HasDnnAttr("force_fp32_output")
              ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
              : false;
      auto mul_attr = CreateMulAttr(dev_ctx, force_fp32_output);
      mul_prim_desc =
          inner_product_forward::primitive_desc(mul_desc, mul_attr, engine_);
    } else {
      mul_prim_desc = inner_product_forward::primitive_desc(mul_desc, engine_);
    }

    output_ = CreateDstMemory(mul_prim_desc, dev_ctx, output);

    return inner_product_forward(mul_prim_desc);
  }

  void Execute() {
    auto &astream = OneDNNContext::tls().get_stream();
    (*mul_).execute(astream,
                    {{DNNL_ARG_SRC, *x_input_},
                     {DNNL_ARG_WEIGHTS, *y_input_},
                     {DNNL_ARG_DST, *output_}});
    astream.wait();
  }

  template <typename T>
  DenseTensor UpdateDataFormat(const DenseTensor *data,
                               int num_col_dims,
                               const OneDNNContext &dev_ctx) {
    DenseTensor x_tmp;
    DenseTensor data_matrix;
    // This code is enforcing plain (non-blocked) memory arrangement
    // in order to flatten (reduce dimensionality) of DenseTensor later
    auto src_mdesc = data->mem_desc();
    auto dst_mdesc = data->dims().size() >= 4
                         ? (data->dims().size() == 5
                                ? CreateMemDescriptor<T>(
                                      data, funcs::OneDNNMemoryFormat::ncdhw)
                                : CreateMemDescriptor<T>(
                                      data, funcs::OneDNNMemoryFormat::nchw))
                         : src_mdesc;

    if (src_mdesc != dst_mdesc) {
      dev_ctx.template Alloc<T>(&x_tmp, data->memory_size());

      Reorder(src_mdesc,
              dst_mdesc,
              funcs::to_void_cast<T>(data->data<T>()),
              funcs::to_void_cast<T>(x_tmp.data<T>()));

      x_tmp.Resize(data->dims());
      x_tmp.set_mem_desc(dst_mdesc);
      data_matrix = ReshapeToMatrix(x_tmp, num_col_dims);
    } else {
      data_matrix = ReshapeToMatrix(*data, num_col_dims);
    }

    return data_matrix;
  }

  void UpdateDataPointers(const OneDNNContext &dev_ctx,
                          DenseTensor *out,
                          const DenseTensor *in) {
    x_input_->set_data_handle(funcs::to_void_cast<XT>(in->data<XT>()));
    output_->set_data_handle(dev_ctx.template Alloc<OT>(out));
    out->set_mem_desc(output_->get_desc());
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const DenseTensor *tensor,
      funcs::OneDNNMemoryFormat format,
      memory::data_type type = funcs::OneDNNGetDataType<T>()) {
    auto dims = vectorize<int64_t>(tensor->dims());
    return funcs::OneDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory::desc CreateMemDescriptor(
      const std::vector<int64_t> &dims,
      funcs::OneDNNMemoryFormat format,
      memory::data_type type = funcs::OneDNNGetDataType<T>()) {
    return funcs::OneDNNMemDesc(dims, type, format);
  }

  template <typename T>
  memory CreateMemory(const memory::desc &desc, const DenseTensor *tensor) {
    return memory(desc, engine_, funcs::to_void_cast<T>(tensor->data<T>()));
  }

  memory CreateDstMemory(
      const inner_product_forward::primitive_desc &mul_prim_desc,
      const OneDNNContext &dev_ctx,
      DenseTensor *output) {
    auto dst_desc = mul_prim_desc.dst_desc();
    auto buffer_size = dst_desc.get_size();

    OT *output_data = dev_ctx.template Alloc<OT>(output, buffer_size);
    output->set_mem_desc(dst_desc);
    return memory(dst_desc, engine_, funcs::to_void_cast<OT>(output_data));
  }

  memory Reorder(const memory::desc &src_desc,
                 const memory::desc &dst_desc,
                 void *src_data,
                 void *dst_data = NULL) {
    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = dst_data ? memory(dst_desc, engine_, dst_data)
                            : memory(dst_desc, engine_);

    auto reorder = dnnl::reorder(src_mem, dst_mem);

    auto &astream = OneDNNContext::tls().get_stream();
    {
      paddle::platform::RecordEvent record_reorder(
          "int_reorder",
          paddle::platform::TracerEventType::UserDefined,
          2,
          paddle::platform::EventRole::kUniqueOp);
      reorder.execute(astream, src_mem, dst_mem);
      astream.wait();
    }

    return dst_mem;
  }

  memory TransposeInputY(const DenseTensor *input_y) {
    auto dims = vectorize<int64_t>(input_y->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc =
        CreateMemDescriptor<YT>(dims, funcs::OneDNNMemoryFormat::io);
    auto dst_desc =
        CreateMemDescriptor<YT>(dims, funcs::OneDNNMemoryFormat::oi);
    return Reorder(
        src_desc, dst_desc, funcs::to_void_cast<YT>(input_y->data<YT>()));
  }

  const engine &engine_;
  paddle::optional<memory> x_input_;
  paddle::optional<memory> y_input_;
  paddle::optional<memory> output_;
  paddle::optional<inner_product_forward> mul_;
  static constexpr bool is_int8_ = funcs::is_int8<XT>();
};

/* OT: output data type */
template <typename XT, typename YT, typename OT>
std::shared_ptr<MulPrimitiveFactory<XT, YT, OT>> GetPrimitiveFactory(
    const OneDNNContext &dev_ctx,
    const DenseTensor *input_x,
    const DenseTensor *input_y,
    const engine &onednn_engine) {
  std::string key = funcs::CreateKey(dev_ctx,
                                     TransToProtoVarType(input_x->dtype()),
                                     vectorize(input_x->dims()),
                                     TransToProtoVarType(input_y->dtype()),
                                     vectorize(input_y->dims()),
                                     dev_ctx.GetOutputsName("Out")[0]);
  key = funcs::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

  auto prim_creator = std::static_pointer_cast<MulPrimitiveFactory<XT, YT, OT>>(
      dev_ctx.GetBlob(key));

  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<MulPrimitiveFactory<XT, YT, OT>>(onednn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

/* XT: input x data type, YT: input y data type */
template <typename XT, typename YT>
inner_product_forward GetMulPrimitive(const OneDNNContext &dev_ctx,
                                      const DenseTensor *input_x,
                                      const DenseTensor *input_y,
                                      DenseTensor *output,
                                      int x_num_col_dims,
                                      int y_num_col_dims,
                                      const engine &onednn_engine) {
  constexpr bool is_int8 = funcs::is_int8<XT>();
  bool force_fp32_output =
      dev_ctx.HasDnnAttr("force_fp32_output")
          ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
          : false;

  if (is_int8 && !force_fp32_output) {
    return GetPrimitiveFactory<XT, YT, int8_t>(
               dev_ctx, input_x, input_y, onednn_engine)
        ->CreateMulPrimitive(
            input_x, input_y, output, x_num_col_dims, y_num_col_dims, dev_ctx);

  } else {
    return GetPrimitiveFactory<XT, YT, float>(
               dev_ctx, input_x, input_y, onednn_engine)
        ->CreateMulPrimitive(
            input_x, input_y, output, x_num_col_dims, y_num_col_dims, dev_ctx);
  }
}

/* XT: input x data type */
template <typename XT, typename Context>
void MatmulWithFlattenKernelINT8(const Context &dev_ctx,
                                 const DenseTensor &x,
                                 const DenseTensor &y,
                                 int x_num_col_dims,
                                 int y_num_col_dims,
                                 DenseTensor *out) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == AllocationType::CPU,
                    true,
                    errors::PreconditionNotMet(
                        "oneDNN MatmulWithFlatten kernel must use CPUPlace"));

  OneDNNContext::tls().log_lib_version();
  auto &onednn_engine = dev_ctx.GetEngine();

  auto out_dims = out->dims();

  auto mul = GetMulPrimitive<XT, float>(
      dev_ctx, &x, &y, out, x_num_col_dims, y_num_col_dims, onednn_engine);

  if (out_dims.size() != 2) {
    out->Resize(out_dims);
  }

  auto in_md = memory::desc(*dnnl_primitive_desc_query_md(
      mul.get_primitive_desc(), dnnl_query_dst_md, 0));
  out->set_mem_desc(in_md.reshape(vectorize<int64_t>(out->dims())));
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context &dev_ctx,
                             const DenseTensor &x,
                             const DenseTensor &y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             DenseTensor *out) {
  constexpr bool is_int8 = funcs::is_int8<T>();
  if (is_int8) {
    MatmulWithFlattenKernelINT8<T, Context>(
        dev_ctx, x, y, x_num_col_dims, y_num_col_dims, out);
    return;
  }

  const DenseTensor x_matrix =
      x.dims().size() > 2 ? ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? ReshapeToMatrix(y, y_num_col_dims) : y;

  // adding mb dim because MatMulV2 handler needs it
  std::vector<int64_t> x_dims(3, 1);
  std::vector<int64_t> y_dims(3, 1);

  x_dims[1] = x_matrix.dims()[0];
  x_dims[2] = x_matrix.dims()[1];
  y_dims[1] = y_matrix.dims()[0];
  y_dims[2] = y_matrix.dims()[1];

  funcs::ExecuteMul<T>(
      dev_ctx, x_matrix, y_matrix, x_dims, y_dims, false, false, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(matmul,
                   OneDNN,
                   ONEDNN,
                   phi::MatmulKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}

PD_REGISTER_KERNEL(matmul_with_flatten,
                   OneDNN,
                   ONEDNN,
                   phi::MatmulWithFlattenKernel,
                   float,
                   phi::dtype::bfloat16,
                   uint8_t,
                   int8_t) {}
