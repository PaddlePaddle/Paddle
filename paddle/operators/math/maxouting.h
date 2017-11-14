/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
#include "paddle/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {


#define FLT_MAX \
    __FLT_MAX__

/*
 * \brief Extracting simple operations from maxout.
 *        need "initial", "compute"
 * operation.
 */
template <class T>
class MaxOut {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(T& y, const T& x) { y = y > x ? y : x; }
};

template <class T>
class MaxOutGrad {
 public:
  DEVICE inline void compute(const T& x, const T& y, const T& dy, T& dx,
                             T scale) {
    dx += dy * (x == y);
  }
};


/*
 * \brief Getting pooling results, and calculating gradient.
 *
 * In pool2d, all tensors are in NCHW format. Where N is batch size, C is the
 * number of channels, H and W is the height and width of feature.
 * In pool3d, all tensors are in NCDHW format. Where N is batch size, C is the
 * number of channels, D, H and W is the depth, height and width of feature.
 *
 * In max pooling, it is possible that the pooling region has multiple maximum
 * elements. In this case, we should compute the gradient of the first maximum
 * element.
 * This is different from average pooling. So we rewrite the max_pool_grad:
 * MaxPool2dGradFunctor, MaxPool3dGradFunctor.
 */
template <typename Place, typename MaxOutProcess, typename T>

class MaxOutFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor * output,
                  int groups, MaxOutProcess maxout_compute);
};


template <typename Place, class T>
class MaxOutGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, int groups);
};







}  // namespace math
}  // namespace operators
}  // namespace paddle
