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
//////////////////////
#define FLT_MAX __FLT_MAX__  //

template <class T>
class MaxPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(T& y, const T& x) { y = y > x ? y : x; }
  DEVICE inline void finalize(T& y, const T& poo_size) {}
};

template <class T>
class AvgPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(0); }
  DEVICE inline void compute(T& y, const T& x) { y += x; }
  DEVICE inline void finalize(T& y, const T& poo_size) { y /= poo_size; }
};
template <class T>
class MaxPoolGrad {
 public:
  DEVICE inline void compute(const T& x, const T& y, const T& dy, T& dx,
                             T scale) {
    dx += dy * (x == y);
  }
};

template <class T>
class AvgPoolGrad {
 public:
  DEVICE inline void compute(const T& x, const T& y, const T& dy, T& dx,
                             T scale) {
    dx += (scale * dy);
  }
};

template <typename Place, typename PoolProcess, typename T>
class Pool2dFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_compute);
};

template <typename Place, typename PoolProcess, typename T>
class Pool2dGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_compute);
};

template <typename Place, class T>
class MaxPool2dGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename PoolProcess, typename T>
class Pool3dFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  std::vector<int>& ksize, std::vector<int>& strides,
                  std::vector<int>& paddings, PoolProcess pool_compute);
};

template <typename Place, typename PoolProcess, typename T>
class Pool3dGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings,
                  PoolProcess pool_compute);
};

template <typename Place, class T>
class MaxPool3dGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& input_grad,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
