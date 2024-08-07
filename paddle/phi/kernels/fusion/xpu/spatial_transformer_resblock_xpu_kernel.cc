// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"

#include "glog/logging.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

#ifdef PADDLE_WITH_XPU_XFT
#include "layers/spatial_transformer.h"
namespace xft = baidu::xpu::xft;
#endif

namespace phi {
namespace fusion {

static std::vector<std::vector<int>> IntVec1DTo2D(const std::vector<int>& vec,
                                                  int dim) {
  std::vector<std::vector<int>> res;
  int size = static_cast<int>(vec.size());
  for (auto i = 0; i < size; i += dim) {
    std::vector<int> tmp;
    for (auto j = 0; j < dim; j++) {
      tmp.push_back(vec[i + j]);
    }
    res.emplace_back(std::move(tmp));
  }
  return res;
}

template <typename T, typename Context>
void SpatialTransformerResblockXPUKernel(
    const Context& ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& x_max,
    const std::vector<const DenseTensor*>& conv_bias,
    const std::vector<const DenseTensor*>& conv_filter,
    const std::vector<const DenseTensor*>& conv_filter_max,
    const std::vector<const DenseTensor*>& gn_bias,
    const std::vector<const DenseTensor*>& gn_scale,
    const std::vector<int>& dilations,
    const std::vector<int>& paddings,
    const std::vector<int>& strides,
    const std::vector<float>& gn_eps,
    const std::vector<int>& gn_groups,
    const std::vector<int>& groups,
    bool conv_fix,
    bool has_silu_fc_input,
    bool include_silu,
    DenseTensor* out,
    DenseTensor* out_max) {
#ifdef PADDLE_WITH_XPU_XFT
  using XPUType = typename XPUTypeTrait<T>::Type;

  auto* in1 = reinterpret_cast<const XPUType*>(x.data<T>());
  const XPUType* in2 = nullptr;
  auto* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  int batch = static_cast<int>(x.dims()[0]);
  int channel = static_cast<int>(x.dims()[1]);
  int nh = static_cast<int>(x.dims()[2]);
  int nw = static_cast<int>(x.dims()[3]);
  int input2_dim = -1;

  if (has_silu_fc_input) {
    PADDLE_ENFORCE_XDNN_SUCCESS(-1, "has_silu_fc_input unsupported yet!!!");
  }
  if (include_silu) {
    PADDLE_ENFORCE_XDNN_SUCCESS(-1, "include_silu unsupported yet!!!");
  }

  std::vector<xft::xftVec<float>> xft_gn_weight_;
  std::vector<xft::xftVec<float>> xft_gn_bias_;
  std::vector<xft::xftMat<int16_t>> xft_fc_weights_;
  std::vector<xft::xftVec<float>> xft_fc_bias_;
  std::vector<xft::xftTensor<int16_t, 4>> xft_conv_weights_;
  std::vector<xft::xftVec<float>> xft_conv_bias_;
  std::vector<const float*> input_max_{nullptr};

  // prepare gn_scale
  for (auto* gn_scale : gn_scale) {
    xft_gn_weight_.emplace_back(const_cast<float*>(gn_scale->data<float>()),
                                xft::xftVec<float>::dim_t{gn_scale->dims()[0]});
  }

  // prepare gn_bias
  for (auto* gn_bias : gn_bias) {
    xft_gn_bias_.emplace_back(const_cast<float*>(gn_bias->data<float>()),
                              xft::xftVec<float>::dim_t{gn_bias->dims()[0]});
  }

  // prepare input_max
  for (auto* input_max : x_max) {
    input_max_.emplace_back(const_cast<float*>(input_max->data<float>()));
  }
  if (x_max.size() == 0) {
    input_max_.emplace_back(nullptr);
  }

  std::vector<std::vector<int>> kernel_dims_2d;
  // prepare conv params
  for (size_t i = 0; i < conv_filter.size(); i++) {
    int xn = conv_filter[i]->dims()[0];
    int nc = conv_filter[i]->dims()[1];
    int nh = conv_filter[i]->dims()[2];
    int nw = conv_filter[i]->dims()[3];
    xft_conv_weights_.emplace_back(
        const_cast<int16_t*>(
            reinterpret_cast<const int16_t*>(conv_filter[i]->data<int16_t>())),
        const_cast<float*>(conv_filter_max[i]->data<float>()),
        xft::xftTensor<int16_t, 4>::dim_t{channel, xn, nh, nw});
    kernel_dims_2d.emplace_back(std::vector<int>{xn, nc, nh, nw});
  }

  // prepare bias
  for (auto* conv_bias : conv_bias) {
    xft_conv_bias_.emplace_back(
        const_cast<float*>(conv_bias->data<float>()),
        xft::xftVec<float>::dim_t{conv_bias->dims()[0]});
  }

  xft::STResBlockParam resblock_param_;

  std::vector<std::vector<int>> strides_2d{IntVec1DTo2D(strides, 2)};
  std::vector<std::vector<int>> paddings_2d{IntVec1DTo2D(paddings, 4)};
  std::vector<std::vector<int>> dilations_2d{IntVec1DTo2D(dilations, 2)};

  // achieve params from model
  resblock_param_.conv_fix = conv_fix;
  resblock_param_.has_silu_fc_input = has_silu_fc_input;
  resblock_param_.include_silu = include_silu;
  resblock_param_.conv_groups = groups;
  resblock_param_.kernel_dims = kernel_dims_2d;
  resblock_param_.dilations = dilations_2d;
  resblock_param_.paddings = paddings_2d;
  resblock_param_.strides = strides_2d;
  resblock_param_.gn_groups = gn_groups;
  resblock_param_.gn_eps = gn_eps;

  // input
  xft::xftTensor<XPUType, 4> in_tensor(const_cast<XPUType*>(in1),
                                       const_cast<float*>(input_max_[0]),
                                       {batch, channel, nh, nw});
  xft::xftMat<XPUType> in_silu_tensor(
      const_cast<XPUType*>(in2), nullptr, {batch, input2_dim});
  // output
  xft::xftTensor<XPUType, 4> output_tensor(out_data, {batch, channel, nh, nw});
  int r = xft::st_resblock_fusion<XPUType, int16_t, int16_t>(
      ctx.x_context(),
      in_tensor,
      in_silu_tensor,
      xft_gn_weight_,
      xft_gn_bias_,
      xft_fc_weights_,  // has_silu_fc_input
      xft_fc_bias_,     // has_silu_fc_input_
      xft_conv_weights_,
      xft_conv_bias_,
      &output_tensor,
      resblock_param_);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "st_resblock_fusion");
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "spatial_transformer_resblock_xpu is not supported since it's not "
      "compiled with XPU_XFT"));
#endif
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(spatial_transformer_resblock_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::SpatialTransformerResblockXPUKernel,
                   float,
                   phi::dtype::float16) {}
