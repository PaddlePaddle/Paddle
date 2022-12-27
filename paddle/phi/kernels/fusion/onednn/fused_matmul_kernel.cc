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

#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

using dnnl::engine;
using dnnl::inner_product_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;
using paddle::framework::ReshapeToMatrix;

namespace phi {

bool IsOutputFused(const OneDNNContext &dev_ctx) {
  const auto shape =
      dev_ctx.HasDnnAttr("fused_reshape_Out")
          ? PADDLE_GET_CONST(std::vector<int>,
                             dev_ctx.GetDnnAttr("fused_reshape_Out"))
          : std::vector<int>();
  const auto axis =
      dev_ctx.HasDnnAttr("fused_transpose_Out")
          ? PADDLE_GET_CONST(std::vector<int>,
                             dev_ctx.GetDnnAttr("fused_transpose_Out"))
          : std::vector<int>();
  return !shape.empty() && !axis.empty();
}

template <typename XT, typename YT, typename OT>
class FusedMatmulOneDNNHandler
    : public funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul> {
 public:
  FusedMatmulOneDNNHandler(const OneDNNContext &dev_ctx,
                           const std::vector<int64_t> &x_org_dims,
                           const std::vector<int64_t> &y_org_dims,
                           bool trans_x,
                           bool trans_y,
                           const std::vector<int64_t> &x_strides_override,
                           const std::vector<int64_t> &y_strides_override,
                           bool is_output_fused)
      : funcs::OneDNNHandlerNoCachingT<XT, dnnl::matmul>(dev_ctx.GetEngine(),
                                                         dev_ctx.GetPlace()) {
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

    // TODO(jczaja): Why not for int8??
    if (!funcs::is_int8<OT>() && is_output_fused) {
      out_strides = FakeTransposeStrides(out_ddims);
    }

    auto x_md = memory::desc(x_dims, funcs::OneDNNGetDataType<XT>(), x_strides);
    auto y_md = memory::desc(y_dims, funcs::OneDNNGetDataType<YT>(), y_strides);
    auto out_md =
        memory::desc(out_ddims, funcs::OneDNNGetDataType<OT>(), out_strides);

    const auto matmul_attrs = CreateMatmulAttrs(dev_ctx);

    this->AcquireForwardPrimitiveDescriptor(matmul_attrs, x_md, y_md, out_md);
  }

  float ComputeOutputScale(const OneDNNContext &dev_ctx) {
    float alpha = dev_ctx.HasDnnAttr("alpha")
                      ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("alpha"))
                      : 1.0f;

    if (dev_ctx.HasDnnAttr("Scale_x") && dev_ctx.HasDnnAttr("Scale_y") &&
        dev_ctx.HasDnnAttr("Scale_out")) {
      float scale_x = PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_x"));
      float scale_y = PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_y"));
      bool force_fp32_out =
          dev_ctx.HasDnnAttr("force_fp32_output")
              ? PADDLE_GET_CONST(bool, dev_ctx.GetDnnAttr("force_fp32_output"))
              : false;
      float scale_out =
          force_fp32_out
              ? 1.f
              : PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_out"));
      alpha *= scale_out / (scale_x * scale_y);
    }
    return alpha;
  }

  dnnl::primitive_attr CreateMatmulAttrs(const OneDNNContext &dev_ctx) {
    dnnl::primitive_attr matmul_attrs;
    dnnl::post_ops post_operations;

    float scale_out = ComputeOutputScale(dev_ctx);
    if (scale_out != 1.0f) {
      matmul_attrs.set_output_scales(0, {scale_out});
    }
    const auto *residual_data = dev_ctx.HasDnnInput("ResidualData")
                                    ? dev_ctx.GetDnnInput("ResidualData")
                                    : nullptr;

    if (residual_data) {
      auto residual_data_tz = vectorize(residual_data->dims());
      auto residual_data_md = memory::desc(residual_data_tz,
                                           funcs::OneDNNGetDataType<OT>(),
                                           dnnl::memory::format_tag::any);
      post_operations.append_binary(dnnl::algorithm::binary_add,
                                    residual_data_md);
      if (dev_ctx.HasDnnAttr("Scale_in_eltwise")) {
        float scale_in_eltwise =
            PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("Scale_in_eltwise"));
        float sum_scale = scale_out / scale_in_eltwise;
        post_operations.append_sum(sum_scale);
      }
    }

    funcs::AppendActivation(dev_ctx, post_operations);

    const float scale_alpha =
        dev_ctx.HasDnnAttr("fused_output_scale")
            ? PADDLE_GET_CONST(float, dev_ctx.GetDnnAttr("fused_output_scale"))
            : 1.0f;
    if (scale_alpha != 1.0f) {
      post_operations.append_eltwise(
          1.0, dnnl::algorithm::eltwise_linear, scale_alpha, 0.0f);
    }

    matmul_attrs.set_post_ops(post_operations);
    return matmul_attrs;
  }

  std::vector<int64_t> FakeTransposeStrides(
      const std::vector<int64_t> &matmul_out_dims) const {
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

  std::shared_ptr<memory> AcquireWeightsMemory(const DenseTensor *input) {
    const YT *input_data = input->data<YT>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->weights_desc(), funcs::to_void_cast<YT>(input_data));
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

static DDim RowMatrixDimsFromVector(const DDim &x_dim) {
  return x_dim.size() > 1 ? x_dim : make_ddim({1, x_dim[0]});
}

static DDim ColumnMatrixDimsFromVector(const DDim &y_dim) {
  return y_dim.size() > 1 ? y_dim : make_ddim({y_dim[0], 1});
}

static std::vector<int64_t> TransposeAxis(const std::vector<int64_t> &x,
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

static std::vector<int64_t> GetInputStrides(const OneDNNContext &dev_ctx,
                                            const DDim &input_dims,
                                            const std::string input_name,
                                            const bool transpose_input) {
  auto new_dims = input_dims;
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
    new_dims = input_dims.reshape(shape).transpose(axis);
  }

  auto &MatrixDimsFromVector =
      input_name == "X" ? RowMatrixDimsFromVector : ColumnMatrixDimsFromVector;
  funcs::MatDescriptor mat_dim = funcs::CreateMatrixDescriptor(
      MatrixDimsFromVector(new_dims), 0, transpose_input);

  std::vector<int64_t> strides;
  if (!shape.empty()) {
    auto shape2 = input_dims.reshape(shape);
    strides.push_back(1);
    for (auto i = shape2.size() - 1; i > 0; --i) {
      strides.insert(strides.begin(),
                     strides.front() * static_cast<int64_t>(shape2[i]));
    }
    strides = TransposeAxis(strides, axis);
    if (shape.size() == 2)
      strides.insert(strides.begin(),
                     static_cast<int64_t>(shape[0] * shape[1]));
    mat_dim.stride_ = strides[0];
    if (mat_dim.trans_) std::swap(*strides.rbegin(), *(++strides.rbegin()));
  }
  return strides;
}

template <typename T, typename T_out>
void ExecuteFusedMatmul(const OneDNNContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        const std::vector<int64_t> &x_dims,
                        const std::vector<int64_t> &y_dims,
                        bool trans_x,
                        bool trans_y,
                        DenseTensor *out) {
  auto x_strides_override = GetInputStrides(dev_ctx, x.dims(), "X", trans_x);
  auto y_strides_override = GetInputStrides(dev_ctx, y.dims(), "Y", trans_y);
  FusedMatmulOneDNNHandler<T, T, T_out> handler(dev_ctx,
                                                x_dims,
                                                y_dims,
                                                trans_x,
                                                trans_y,
                                                x_strides_override,
                                                y_strides_override,
                                                IsOutputFused(dev_ctx));

  const auto src_memory_p = handler.AcquireSrcMemory(&x);
  const auto weights_memory_p = handler.AcquireWeightsMemory(&y);
  const auto dst_memory_p = handler.AcquireDstMemory(dev_ctx, out);

  auto matmul_p = handler.AcquireForwardPrimitive();

  std::unordered_map<int, memory> matmul_args = {
      {DNNL_ARG_SRC, *src_memory_p},
      {DNNL_ARG_WEIGHTS, *weights_memory_p},
      {DNNL_ARG_DST, *dst_memory_p}};

  const auto *residual_data = dev_ctx.HasDnnInput("ResidualData")
                                  ? dev_ctx.GetDnnInput("ResidualData")
                                  : nullptr;

  if (residual_data) {
    const auto residual_data_memory_p = handler.AcquireSrcMemory(residual_data);
    matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                        *residual_data_memory_p});
  }

  auto &astream = OneDNNContext::tls().get_stream();
  matmul_p->execute(astream, matmul_args);
  astream.wait();

  // TODO(jczaja): Explain why int8 format of dst is ABCD and do not need
  // permute
  if (IsOutputFused(dev_ctx) && !funcs::is_int8<T_out>()) {
    const auto axis =
        dev_ctx.HasDnnAttr("fused_transpose_Out")
            ? PADDLE_GET_CONST(std::vector<int>,
                               dev_ctx.GetDnnAttr("fused_transpose_Out"))
            : std::vector<int>();
    auto permuted_md = dst_memory_p->get_desc().permute_axes(axis);
    out->set_mem_desc(permuted_md.reshape(vectorize<int64_t>(out->dims())));
  } else {
    out->set_mem_desc(
        dst_memory_p->get_desc().reshape(vectorize<int64_t>(out->dims())));
  }
}

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
void FusedMatmulKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &y,
                       bool transpose_x,
                       bool transpose_y,
                       const std::string &fuse_activation,
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
  if (fuse_activation == "relu" || fuse_activation == "relu6") {
    fuse_relu = true;
  }

  auto x_dims = vectorize(GetDimsForInput(dev_ctx, x.dims(), "X"));
  auto y_dims = vectorize(GetDimsForInput(dev_ctx, y.dims(), "Y"));

  int ndims = std::max(x_dims.size(), y_dims.size());
  ndims = std::max(ndims, 3);

  std::vector<int64_t> x_bd_dims(ndims, 1);
  std::vector<int64_t> y_bd_dims(ndims, 1);

  CalculateMatrixDims(
      x_dims, y_dims, &x_bd_dims, &y_bd_dims, out, IsOutputFused(dev_ctx));

  if (force_fp32_output || ((!is_int8) && (!is_bfloat16))) {
    ExecuteFusedMatmul<T, float>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else if (is_bfloat16) {
    ExecuteFusedMatmul<T, paddle::platform::bfloat16>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else if (fuse_relu) {
    ExecuteFusedMatmul<T, uint8_t>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  } else {
    ExecuteFusedMatmul<T, int8_t>(
        dev_ctx, x, y, x_bd_dims, y_bd_dims, transpose_x, transpose_y, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fused_matmul,
                   OneDNN,
                   ONEDNN,
                   phi::FusedMatmulKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
