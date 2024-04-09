/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
  dev_ctx.template Alloc<T>(scale);
  dev_ctx.template Alloc<T>(zero_point);
  const T* weight_data = weight.data<T>();

  auto weight_dims = weight.dims();

  // auto scale_dims = make_ddim({1, weight_dims[0]});

  const bool trans = false;
  // const int SPLIT_OFFSET = 0;
  // const int NUM_SPLIT = 1; // same as the number of numa node;
  hpj::Vector<float> scaleWeight;
  // auto cpu_place = dev_ctx.GetPlace();
  scaleWeight.data = scale->data<T>();
  hpj::Vector<float> zeroWeight;
  zeroWeight.data = zero_point->data<T>();
  if constexpr (std::is_same_v<T, float>) {
    if (algo == "weight_only_int8") {
      dev_ctx.template Alloc<int8_t>(out);
      DenseTensor tmp_tensor(weight.dtype());
      tmp_tensor.Resize(weight_dims);
      dev_ctx.template Alloc<int8_t>(&tmp_tensor);
      hpj::Matrix<int8_t> quantizedWeight;
      hpj::Matrix<int8_t> qkvWeight;
      quantizedWeight.data.buf = tmp_tensor.data<int8_t>();
      qkvWeight.data.buf = out->data<int8_t>();
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
