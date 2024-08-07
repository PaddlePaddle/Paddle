// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/onednn/matmul_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

using dnnl::memory;

namespace phi {
namespace fusion {

template <typename XT, typename YT, typename OT>
class FusedMatmulOneDNNHandler
    : public funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  FusedMatmulOneDNNHandler(const OneDNNContext &dev_ctx,
                           const DenseTensor *residual_data,
                           const std::vector<int64_t> &x_org_dims,
                           const std::vector<int64_t> &y_org_dims,
                           bool trans_x,
                           bool trans_y,
                           const float matmul_alpha,
                           const std::vector<int64_t> &x_strides_override,
                           const std::vector<int64_t> &y_strides_override,
                           bool is_output_fused,
                           const std::string &fuse_activation,
                           const float fuse_alpha,
                           const float fuse_beta,
                           const float fused_output_scale,
                           const float scale_x,
                           const float scale_y,
                           const float scale_in_eltwise,
                           const float scale_out,
                           const bool force_fp32_output)
      : funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul>(dev_ctx.GetEngine(),
                                                         dev_ctx.GetPlace()) {
    // M X K * K X N
    std::vector<int64_t> x_dims(x_org_dims);
    std::vector<int64_t> y_dims(y_org_dims);

    const int MB_idx = static_cast<int>(x_dims.size()) - 3;
    const int H_idx = static_cast<int>(x_dims.size()) - 2;
    const int W_idx = static_cast<int>(x_dims.size()) - 1;

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

    if (x_strides_override.empty()) {
      if (trans_x) {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      } else {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      }
    } else {
      x_strides = x_strides_override;
    }

    if (y_strides_override.empty()) {
      if (trans_y) {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      } else {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      }
    } else {
      y_strides = y_strides_override;
    }

    out_strides.insert(out_strides.end(), {M * N, N, 1});
    out_ddims.insert(out_ddims.end(),
                     {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

    for (int i = static_cast<int>(x_dims.size()) - 4; i >= 0; --i) {
      out_ddims[i] = std::max(x_dims[i], y_dims[i]);
      if (x_strides_override.empty()) {
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
      }
      if (y_strides_override.empty()) {
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
      }
      out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
    }

    // TODO(jczaja): Why not for int8??
    if (!funcs::is_int8<OT>() && is_output_fused) {
      std::vector<int> transpose_axis = {0, 2, 1, 3};
      out_strides = phi::funcs::FakeTransposeStrides(out_ddims, transpose_axis);
    }

    auto x_md = memory::desc(x_dims, funcs::OneDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, funcs::OneDNNGetDataType<YT>(), y_strides);
    auto out_md =
        memory::desc(out_ddims, funcs::OneDNNGetDataType<OT>(), out_strides);

    const auto matmul_attrs = CreateMatmulAttrs(dev_ctx,
                                                residual_data,
                                                matmul_alpha,
                                                fuse_activation,
                                                fuse_alpha,
                                                fuse_beta,
                                                fused_output_scale,
                                                scale_x,
                                                scale_y,
                                                scale_in_eltwise,
                                                scale_out,
                                                force_fp32_output,
                                                out_ddims);

    this->AcquireForwardPrimitiveDescriptor(matmul_attrs, x_md, y_md, out_md);
  }

  dnnl::primitive_attr CreateMatmulAttrs(
      const OneDNNContext &dev_ctx,
      const DenseTensor *residual_data,
      const float matmul_alpha,
      const std::string &fuse_activation,
      const float fuse_alpha,
      const float fuse_beta,
      const float fused_output_scale,
      const float scale_x,
      const float scale_y,
      const float scale_in_eltwise,
      const float scale_out,
      const bool force_fp32_output,
      const std::vector<int64_t> &out_ddims) {
    dnnl::primitive_attr matmul_attrs;
    dnnl::post_ops post_operations;

    if (scale_x != 1.0f) {
      matmul_attrs.set_scales_mask(DNNL_ARG_SRC, 0);
    }

    // alpha can be folded to weight scale
    if (scale_y != 1.0f || matmul_alpha != 1.0f) {
      matmul_attrs.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    }

    if (!force_fp32_output && scale_out != 1.0f) {
      matmul_attrs.set_scales_mask(DNNL_ARG_DST, 0);
    }

    if (residual_data) {
      // fill 1 in the front of adesc, to make residual ndims to be same as dst
      // dims
      int dst_size = out_ddims.size();
      int origin_size = residual_data->mem_desc().get_ndims();
      auto reshaped_md = residual_data->mem_desc();
      dnnl::memory::dims expanded_dims = residual_data->mem_desc().get_dims();
      if (origin_size < dst_size) {
        expanded_dims.insert(expanded_dims.begin(), dst_size - origin_size, 1);
        reshaped_md = residual_data->mem_desc().reshape(expanded_dims);
      }

      auto residual_data_tz = vectorize(residual_data->dims());
      auto chosen_memory_format = funcs::OneDNNMemoryFormat::any;
      dnnl::memory::desc residual_data_md;
      if (!out_ddims.empty() && out_ddims[0] > 1 &&
          residual_data_tz.size() == 4 && residual_data_tz[0] == 1 &&
          residual_data_tz[1] > 1 && residual_data_tz[2] > 1 &&
          residual_data_tz[3] > 1) {
        chosen_memory_format = funcs::OneDNNMemoryFormat::nchw;
        residual_data_md = memory::desc(
            out_ddims, funcs::OneDNNGetDataType<OT>(), chosen_memory_format);
      } else {
        residual_data_md = reshaped_md;
      }

      post_operations.append_binary(dnnl::algorithm::binary_add,
                                    residual_data_md);
      if (scale_in_eltwise != 0.0f) {
        float sum_scale = 1.f / scale_in_eltwise;
        post_operations.append_sum(sum_scale);
      }
    }

    funcs::AppendActivation(
        dev_ctx, post_operations, fuse_activation, fuse_alpha, fuse_beta);

    if (fused_output_scale != 1.0f) {
      post_operations.append_eltwise(
          dnnl::algorithm::eltwise_linear, fused_output_scale, 0.0f);
    }

    matmul_attrs.set_post_ops(post_operations);
    return matmul_attrs;
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const DenseTensor *input) {
    const YT *input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(), funcs::to_void_cast<YT>(input_data));
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryResidual(
      const DenseTensor *input) {
    const XT *input_data = input->data<XT>();
    auto residual_memory_p = this->AcquireMemoryFromPrimitive(
        input->mem_desc(), phi::funcs::to_void_cast<XT>(input_data));
    return residual_memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemoryStride(
      const DenseTensor *input) {
    const XT *input_data = input->data<XT>();
    std::shared_ptr<dnnl::memory> src_mem =
        this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc());
    auto residual_vec = vectorize(input->dims());
    int IC = residual_vec[1];
    int IH = residual_vec[2];
    int IW = residual_vec[3];
    size_t size = this->fwd_pd_->dst_desc().get_size() / sizeof(XT);
    XT *dst = static_cast<XT *>(src_mem->get_data_handle());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (size_t i = 0; i < size; ++i) {
      auto mod_i =
          static_cast<int>(i - floor(i / (IC * IH * IW)) * (IC * IH * IW));
      // Make 1*C*H*W to N*C*H*W to avoid broadcast overhead
      dst[i] = input_data[mod_i];
    }
    return src_mem;
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(const OneDNNContext &dev_ctx,
                                                 DenseTensor *output) {
    // We cannot use base AcquireDstMemory as it makes an allocation request
    // base on DST memory primitive size. This is fine in general, but in MatMul
    // we have primitive that covers only one batch of Data and then shift
    // pointer for every new batch. Hence DenseTensor size is bigger that
    // dst memory primitive size. So would we request less memory that is there
    // and it triggers an assertion.  So as there is no 'any' format here we can
    // leave default size of DenseTensor as computed in ComputeInferShape
    OT *ptr = dev_ctx.template Alloc<OT>(output);
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr);
  }
};

template <typename T, typename T_out>
void ExecuteFusedMatmul(const OneDNNContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const DenseTensor *residual_data,
                        const std::vector<int64_t> &x_dims,
                        const std::vector<int64_t> &y_dims,
                        bool trans_x,
                        bool trans_y,
                        const float matmul_alpha,
                        const std::vector<int64_t> &x_strides_override,
                        const std::vector<int64_t> &y_strides_override,
                        const bool is_output_fused,
                        const std::vector<int> &fused_transpose_Out,
                        const std::string &fuse_activation,
                        const float fuse_alpha,
                        const float fuse_beta,
                        const float fused_output_scale,
                        const float scale_x,
                        const float scale_y,
                        const float scale_in_eltwise,
                        const float scale_out,
                        const bool force_fp32_output,
                        DenseTensor *out) {
  FusedMatmulOneDNNHandler<T, T, T_out> handler(dev_ctx,
                                                residual_data,
                                                x_dims,
                                                y_dims,
                                                trans_x,
                                                trans_y,
                                                matmul_alpha,
                                                x_strides_override,
                                                y_strides_override,
                                                is_output_fused,
                                                fuse_activation,
                                                fuse_alpha,
                                                fuse_beta,
                                                fused_output_scale,
                                                scale_x,
                                                scale_y,
                                                scale_in_eltwise,
                                                scale_out,
                                                force_fp32_output);

  const auto src_memory_p = handler.AcquireSrcMemory(&x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y);
  const auto dst_memory_p = handler.AcquireDstMemory(dev_ctx, out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  if (residual_data) {
    auto residual_data_vec = vectorize(residual_data->dims());
    std::shared_ptr<dnnl::memory> residual_data_memory_p;
    if (std::max((x_dims)[0], (y_dims)[0]) > 1 &&
        residual_data_vec.size() == 4 && residual_data_vec[0] == 1 &&
        residual_data_vec[1] > 1 && residual_data_vec[2] > 1 &&
        residual_data_vec[3] > 1) {
      residual_data_memory_p = handler.AcquireSrcMemoryStride(residual_data);
    } else {
      residual_data_memory_p = handler.AcquireSrcMemoryResidual(residual_data);
    }
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        *residual_data_memory_p});
  }

  if (scale_x != 1.0f) {
    dnnl::memory::desc src_scales_md(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto src_scales_mem =
        std::make_shared<dnnl::memory>(src_scales_md, dev_ctx.GetEngine());
    *reinterpret_cast<float *>(src_scales_mem->get_data_handle()) =
        1.f / scale_x;
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, *src_scales_mem});
  }

  if (scale_y != 1.0f || matmul_alpha != 1.0f) {
    dnnl::memory::desc wei_scales_md(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto wei_scales_mem =
        std::make_shared<dnnl::memory>(wei_scales_md, dev_ctx.GetEngine());
    *reinterpret_cast<float *>(wei_scales_mem->get_data_handle()) =
        matmul_alpha / scale_y;
    matmul_args.insert(
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, *wei_scales_mem});
  }

  if (!force_fp32_output && scale_out != 1.0f) {
    dnnl::memory::desc dst_scales_md(
        {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    auto dst_scales_mem =
        std::make_shared<dnnl::memory>(dst_scales_md, dev_ctx.GetEngine());
    *reinterpret_cast<float *>(dst_scales_mem->get_data_handle()) =
        1.f / scale_out;
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, *dst_scales_mem});
  }

  auto &astream = OneDNNContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  if (is_output_fused && !funcs::is_int8<T_out>()) {
    auto permuted_md =
        dst_memory_p->get_desc().permute_axes(fused_transpose_Out);
    out->set_mem_desc(
        permuted_md.reshape(common::vectorize<int64_t>(out->dims())));
  } else {
    out->set_mem_desc(dst_memory_p->get_desc().reshape(
        common::vectorize<int64_t>(out->dims())));
  }
}

std::vector<int64_t> GetInputShape(DDim input_dims,
                                   std::vector<int> shape,
                                   std::vector<int> axis) {
  if (!shape.empty() && !axis.empty()) {
    return common::vectorize(input_dims.reshape(shape).transpose(axis));
  }
  return common::vectorize(input_dims);
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
    auto out_dims = common::vectorize(out->dims());
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
    out->Resize(common::make_ddim((out_dims)));
  }
}

template <typename T, typename Context>
void FusedMatmulKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &y,
                       const paddle::optional<DenseTensor> &residual_data,
                       bool transpose_x,
                       bool transpose_y,
                       const float matmul_alpha,
                       const std::string &fuse_activation,
                       const float fuse_alpha,
                       const float fuse_beta,
                       const float fused_output_scale,
                       const std::vector<int> &fused_reshape_X,
                       const std::vector<int> &fused_transpose_X,
                       const std::vector<int> &fused_reshape_Y,
                       const std::vector<int> &fused_transpose_Y,
                       const std::vector<int> &fused_reshape_Out,
                       const std::vector<int> &fused_transpose_Out,
                       const std::string &mkldnn_data_type UNUSED,
                       const float scale_x,
                       const float scale_y,
                       const float scale_in_eltwise,
                       const float scale_out,
                       const bool force_fp32_output,
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

  bool fuse_relu = false;
  if (fuse_activation == "relu" || fuse_activation == "relu6") {
    fuse_relu = true;
  }

  auto x_dims = GetInputShape(x.dims(), fused_reshape_X, fused_transpose_X);
  auto y_dims = GetInputShape(y.dims(), fused_reshape_Y, fused_transpose_Y);
  auto is_output_fused =
      !fused_reshape_Out.empty() && !fused_transpose_Out.empty();

  auto x_strides_override = funcs::GetInputStrides(
      "X", x.dims(), transpose_x, fused_reshape_X, fused_transpose_X);
  auto y_strides_override = funcs::GetInputStrides(
      "Y", y.dims(), transpose_y, fused_reshape_Y, fused_transpose_Y);

  int ndims = static_cast<int>(std::max(x_dims.size(), y_dims.size()));
  ndims = std::max(ndims, 3);

  std::vector<int64_t> x_bd_dims(ndims, 1);
  std::vector<int64_t> y_bd_dims(ndims, 1);

  CalculateMatrixDims(
      x_dims, y_dims, &x_bd_dims, &y_bd_dims, out, is_output_fused);

  if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
    ExecuteFusedMatmul<T, float>(dev_ctx,
                                 x,
                                 y,
                                 residual_data.get_ptr(),
                                 x_bd_dims,
                                 y_bd_dims,
                                 transpose_x,
                                 transpose_y,
                                 matmul_alpha,
                                 x_strides_override,
                                 y_strides_override,
                                 is_output_fused,
                                 fused_transpose_Out,
                                 fuse_activation,
                                 fuse_alpha,
                                 fuse_beta,
                                 fused_output_scale,
                                 scale_x,
                                 scale_y,
                                 scale_in_eltwise,
                                 scale_out,
                                 force_fp32_output,
                                 out);
  } else if (is_bfloat16) {
    ExecuteFusedMatmul<T, phi::dtype::bfloat16>(dev_ctx,
                                                x,
                                                y,
                                                residual_data.get_ptr(),
                                                x_bd_dims,
                                                y_bd_dims,
                                                transpose_x,
                                                transpose_y,
                                                matmul_alpha,
                                                x_strides_override,
                                                y_strides_override,
                                                is_output_fused,
                                                fused_transpose_Out,
                                                fuse_activation,
                                                fuse_alpha,
                                                fuse_beta,
                                                fused_output_scale,
                                                scale_x,
                                                scale_y,
                                                scale_in_eltwise,
                                                scale_out,
                                                force_fp32_output,
                                                out);
  } else if (fuse_relu) {
    ExecuteFusedMatmul<T, uint8_t>(dev_ctx,
                                   x,
                                   y,
                                   residual_data.get_ptr(),
                                   x_bd_dims,
                                   y_bd_dims,
                                   transpose_x,
                                   transpose_y,
                                   matmul_alpha,
                                   x_strides_override,
                                   y_strides_override,
                                   is_output_fused,
                                   fused_transpose_Out,
                                   fuse_activation,
                                   fuse_alpha,
                                   fuse_beta,
                                   fused_output_scale,
                                   scale_x,
                                   scale_y,
                                   scale_in_eltwise,
                                   scale_out,
                                   force_fp32_output,
                                   out);
  } else {
    ExecuteFusedMatmul<T, int8_t>(dev_ctx,
                                  x,
                                  y,
                                  residual_data.get_ptr(),
                                  x_bd_dims,
                                  y_bd_dims,
                                  transpose_x,
                                  transpose_y,
                                  matmul_alpha,
                                  x_strides_override,
                                  y_strides_override,
                                  is_output_fused,
                                  fused_transpose_Out,
                                  fuse_activation,
                                  fuse_alpha,
                                  fuse_beta,
                                  fused_output_scale,
                                  scale_x,
                                  scale_y,
                                  scale_in_eltwise,
                                  scale_out,
                                  force_fp32_output,
                                  out);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_matmul,
                   OneDNN,
                   ONEDNN,
                   phi::fusion::FusedMatmulKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
