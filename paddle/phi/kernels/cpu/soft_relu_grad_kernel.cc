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

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <type_traits>

#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
struct SoftReluGradFunctor {
  float threshold;
  void SetAttrs(float threshold_) { threshold = threshold_; }

  template <typename Device,
            typename X,
            typename Out,
            typename dOut,
            typename dX>
  void operator()(Device d, X x UNUSED, Out out, dOut dout, dX dx) {
    auto tmp = static_cast<T>(threshold);
    auto temp = ((out > -tmp) * (out < tmp)).template cast<T>();
    dx.device(d) = dout * (static_cast<T>(1) - (-out).exp()) * temp;
  }
};

template <typename T, typename Context>
void SoftmaxGradKernel(const Context& dev_ctx,
                       const DenseTensor& x_in,
                       const DenseTensor& out_in,
                       const DenseTensor& out_grad,
                       float threshold,
                       DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  auto dout = phi::EigenVector<T>::Flatten(out_grad);
  auto out = phi::EigenVector<T>::Flatten(out_in);
  auto dx = phi::EigenVector<T>::Flatten(*x_grad);
  auto x = phi::EigenVector<T>::Flatten(x_in);
  auto* eigen_dev = dev_ctx.eigen_device();
  SoftReluGradFunctor<T> functor;
  functor.SetAttrs(threshold);
  // use 32bit index to speed up computation
  bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  if (use_32bit_index && is_gpu_place) {
    functor(*eigen_dev,
            To32BitIndex(x),
            To32BitIndex(out),
            To32BitIndex(dout),
            To32BitIndex(dx));
  } else {
    functor(*eigen_dev, x, out, dout, dx);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    soft_relu_grad, CPU, ALL_LAYOUT, phi::SoftmaxGradKernel, float, double) {}
