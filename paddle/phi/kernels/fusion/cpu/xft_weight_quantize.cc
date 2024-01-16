/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "glog/logging.h"
#include "xft/common/my_types.h"
#include "xft/utils/matmul_helper.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedXFTWeightQuantizeKernel(const Context& dev_ctx,
                                  const DenseTensor& weight,
                                  const std::string& algo,
                                  DenseTensor* out,
                                  DenseTensor* scale,
                                  DenseTensor* zero_point) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(scale);
  dev_ctx.template Alloc<T>(zero_point);
  const T* weight_data = weight.data<T>();

  auto weight_dims = weight.dims();
  DenseTensor tmp_tensor(weight.dtype());
  tmp_tensor.Resize(weight_dims);
  dev_ctx.template Alloc<T>(&tmp_tensor);

  // auto scale_dims = make_ddim({1, weight_dims[0]});

  const bool trans = false;
  // const int SPLIT_OFFSET = 0;
  // const int NUM_SPLIT = 1; // same as the number of numa node;
  hpj::Vector<float> scaleWeight;
  auto cpu_place = dev_ctx.GetPlace();
  scaleWeight.data = scale->mutable_data<T>(cpu_place);
  hpj::Vector<float> zeroWeight;
  zeroWeight.data = zero_point->mutable_data<T>(cpu_place);
  if constexpr (std::is_same_v<T, float>) {
    if (algo == "weight_only_int8") {
      hpj::Matrix<int8_t> quantizedWeight;
      hpj::Matrix<int8_t> qkvWeight;
      quantizedWeight.data.buf = tmp_tensor.mutable_data<int8_t>(cpu_place);
      qkvWeight.data.buf = out->mutable_data<int8_t>(cpu_place);
      qkvWeight.rows = weight_dims[0];
      qkvWeight.cols = weight_dims[1];
      qkvWeight.stride = weight_dims[1];
      MMHelper::convertWeight<int8_t>(trans,
                                      weight_dims[0],
                                      weight_dims[1],
                                      weight_data,
                                      quantizedWeight,
                                      scaleWeight,
                                      zeroWeight);
      MMHelper::packWeight<int8_t>(trans, quantizedWeight, qkvWeight);
    } else if (algo == "weight_only_int4") {
      hpj::Matrix<uint4x2_t> quantizedWeight;
      hpj::Matrix<uint4x2_t> qkvWeight;

      qkvWeight.Resize(weight_dims[0], weight_dims[1]);
      qkvWeight.data.buf =
          reinterpret_cast<uint4x2_t*>(out->mutable_data<int8_t>(cpu_place));
      MMHelper::convertWeight<uint4x2_t>(trans,
                                         weight_dims[1],
                                         weight_dims[0],
                                         weight_data,
                                         quantizedWeight,
                                         scaleWeight,
                                         zeroWeight);
      MMHelper::packWeight(trans, quantizedWeight, qkvWeight);
    } else if (algo == "weight_only_nf4") {
      hpj::Matrix<nf4x2_t> quantizedWeight;
      hpj::Matrix<nf4x2_t> qkvWeight;
      qkvWeight.Resize(weight_dims[0], weight_dims[1]);
      qkvWeight.data.buf =
          reinterpret_cast<nf4x2_t*>(out->mutable_data<int8_t>(cpu_place));
      MMHelper::convertWeight<nf4x2_t>(trans,
                                       weight_dims[1],
                                       weight_dims[0],
                                       weight_data,
                                       quantizedWeight,
                                       scaleWeight,
                                       zeroWeight);
      MMHelper::packWeight(trans, quantizedWeight, qkvWeight);
    } else {
      throw std::runtime_error("algo not support");
    }
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(xft_weight_quantize,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedXFTWeightQuantizeKernel,
                   float) {}
