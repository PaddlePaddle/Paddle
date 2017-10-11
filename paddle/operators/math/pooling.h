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
  __FLT_MAX__  // It might need to be placed in another file, but I'm still
               // wondering where to put it.

/*
 * \brief Extracting simple operations from pooling.
 *        Both MaxPool and AvgPool need "initial", "compute" and "finalize"
 * operation.
 *        MaxPool initializes temp variable to the negative maximum to find the
 * maximum value in the pooling field.
 *        AvgPool initializes temp variable to the zero to accumulate all values
 * in pool pooling, and finally takes the average.
 *        MaxPoolGrad and AvgPoolGrad are gradient operations respectively.
 */
template <class T>
class MaxPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(T& y, const T& x) { y = y > x ? y : x; }
  DEVICE inline void finalize(T& y, const T& pool_field) {}
};

template <class T>
class AvgPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(0); }
  DEVICE inline void compute(T& y, const T& x) { y += x; }
  DEVICE inline void finalize(T& y, const T& pool_field) { y /= pool_field; }
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

/*
 * \brief Getting max pooling results and corresponding max index, and
 * calculating gradient.
 * In up-sampling-pooling, it is necessary to know max element index.
 * In pool2d, all tensors are in NCHW format. In pool3d, all tensors are in
 * NCDHW format.
 */
template <typename Place, typename T>
class MaxPool2dWithIndexFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool2dWithIndexGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool3dWithIndexFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  const framework::Tensor& input, framework::Tensor& output,
                  framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

template <typename Place, typename T>
class MaxPool3dWithIndexGradFunctor {
 public:
  void operator()(const platform::DeviceContext& context,
                  framework::Tensor& input_grad,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, std::vector<int>& ksize,
                  std::vector<int>& strides, std::vector<int>& paddings);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
