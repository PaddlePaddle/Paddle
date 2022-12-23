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

DDim GetDimsForInputS(const OneDNNContext &dev_ctx,
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

void CalculateMatrixDimsS(const std::vector<int64_t> &x_dims,
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

  auto x_dims = vectorize(GetDimsForInputS(dev_ctx, x.dims(), "X"));
  auto y_dims = vectorize(GetDimsForInputS(dev_ctx, y.dims(), "Y"));

  int ndims = std::max(x_dims.size(), y_dims.size());
  ndims = std::max(ndims, 3);

  std::vector<int64_t> x_bd_dims(ndims, 1);
  std::vector<int64_t> y_bd_dims(ndims, 1);

  CalculateMatrixDimsS(x_dims,
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

}  // namespace phi

PD_REGISTER_KERNEL(fused_matmul,
                   OneDNN,
                   ONEDNN,
                   phi::FusedMatmulKernel,
                   float,
                   phi::dtype::bfloat16,
                   int8_t,
                   uint8_t) {}
