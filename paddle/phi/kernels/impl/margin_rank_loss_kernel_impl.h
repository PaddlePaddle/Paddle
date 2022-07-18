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

#include "paddle/phi/kernels/margin_rank_loss_kernel.h"

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
//#include "paddle/fluid/framework/eigen.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
struct ReLU {
  HOSTDEVICE T operator()(const T& val) const {
    return val > 0 ? val : static_cast<T>(0);
  }
};

template <typename T>
struct Heaviside {
  HOSTDEVICE T operator()(const T& val) const {
    return static_cast<T>(val > 0 ? 1 : 0);
  }
};

template <typename T, typename Context>
void MarginRankLossKernel(const Context& dev_ctx,
                          const DenseTensor& label,
                          const DenseTensor& left,
                          const DenseTensor& right,
                          float margin,
                          DenseTensor* out,
                          DenseTensor* activated){
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(activated);

  auto out_ = phi::EigenVector<T>::Flatten(*out);
  auto act_ = phi::EigenVector<T>::Flatten(*activated);

  auto label_ = phi::EigenVector<T>::Flatten(label);
  auto left_ = phi::EigenVector<T>::Flatten(left);
  auto right_ = phi::EigenVector<T>::Flatten(right);

  auto& dev = *dev_ctx.eigen_device();
  out_.device(dev) = (-label_ * (left_ - right_) + margin).unaryExpr(ReLU<T>());
  act_.device(dev) = out_.unaryExpr(Heaviside<T>());
}
}  // namespace phi
