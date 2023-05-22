/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/macros.h"  // import FLT_MAX

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/backends/gpu/gpu_decls.h"
#endif

namespace phi {
namespace funcs {

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
  HOSTDEVICE inline void compute(const T& x, T* y) { *y = *y > x ? *y : x; }
  DEVICE inline void finalize(const T& pool_field UNUSED, T* y UNUSED) {}
};

template <class T>
class AvgPool {
  using MT = typename dtype::MPTypeTrait<T>::Type;
  MT intermediate_res;

 public:
  DEVICE inline T initial() {
    intermediate_res = static_cast<MT>(0.0f);
    return static_cast<T>(0);
  }

  DEVICE inline void compute(const T& x, T* y UNUSED) {
    intermediate_res += static_cast<MT>(x);
  }

  DEVICE inline void finalize(const T& pool_field, T* y) {
    *y = static_cast<T>(intermediate_res / (static_cast<MT>(pool_field)));
  }
};

template <class T>
class MaxPoolGrad {
 public:
  static constexpr bool use_x = true;
  HOSTDEVICE inline void compute(
      const T& x, const T& y, const T& dy, T scale UNUSED, T* dx) {
    *dx += dy * static_cast<T>(x == y);
  }
};

template <class T>
class AvgPoolGrad {
 public:
  static constexpr bool use_x = false;
  HOSTDEVICE inline void compute(
      const T& x UNUSED, const T& y UNUSED, const T& dy, T scale, T* dx) {
    *dx += (scale * dy);
  }
};

/* used for adaptive pool to calculate start and end index of each divided grid
 */
HOSTDEVICE inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<float>(ph * input_size) / output_size));
}

HOSTDEVICE inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<float>((ph + 1) * input_size) / output_size));
}

/*
 * \brief Getting pooling results, and calculating gradient.
 *
 * In pool2d, all Tensors are in NCHW or NHWC format. Where N is batch size, C
 * is the number of channels, H and W is the height and width of feature.
 * In pool3d, all Tensors are in NCDHW or NDHWC format. Where N is batch size, C
 * is the number of channels, D, H and W is the depth, height and width of
 * feature.
 *
 * In max pooling, it is possible that the pooling region has multiple maximum
 * elements. In this case, we should compute the gradient of the first maximum
 * element.
 * This is different from average pooling. So we rewrite the max_pool_grad:
 * MaxPool2dGradFunctor, MaxPool3dGradFunctor.
 */
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename PoolProcess, typename T>
class Pool2dDirectCUDAFunctor {
 public:
  void operator()(const T* input,
                  const std::vector<int>& input_shape,
                  const std::vector<int>& output_shape,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  T* output,
                  gpuStream_t stream,
                  PoolProcess pool_compute);
};
#endif

template <typename Context, typename PoolProcess, typename T>
class Pool2dFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_compute);

  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_compute);
};

template <typename Context, typename PoolProcess, typename T>
class Pool2dGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_compute);
};

template <typename Context, class T>
class MaxPool2dGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  DenseTensor* input_grad);
  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad);
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename PoolProcess, typename T>
class Pool3dDirectCUDAFunctor {
 public:
  void operator()(const T* input,
                  const std::vector<int>& input_shape,
                  const std::vector<int>& output_shape,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  T* output,
                  gpuStream_t stream,
                  PoolProcess pool_compute);
};
#endif

template <typename Context, typename PoolProcess, typename T>
class Pool3dFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_compute);
};

template <typename Context, typename PoolProcess, typename T>
class Pool3dGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_compute);
};

template <typename Context, class T>
class MaxPool3dGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  DenseTensor* input_grad);
  // overload operator() to support argument data_format
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad);
};

/*
 * \brief Getting max pooling results and corresponding max index, and
 * calculating gradient.
 * In up-sampling-pooling, it is necessary to know max element index.
 * In pool2d, all tensors are in NCHW format. In pool3d, all tensors are in
 * NCDHW format.
 */
template <typename Context, typename T1, typename T2>
class MaxPool2dWithIndexFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask);
};

template <typename Context, typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* input_grad);
};

template <typename Context, typename T1, typename T2>
class MaxPool3dWithIndexFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask);
};

template <typename Context, typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* input_grad);
};

inline int PoolOutputSize(int input_size,
                          int filter_size,
                          int padding_1,
                          int padding_2,
                          int stride,
                          bool ceil_mode) {
  PADDLE_ENFORCE_NE(
      stride,
      0,
      phi::errors::InvalidArgument(
          "The stride of PoolOutputSize shall not be 0, but received %d.",
          stride));

  int output_size;
  if (!ceil_mode) {
    output_size =
        (input_size - filter_size + padding_1 + padding_2) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + padding_1 + padding_2 + stride - 1) /
            stride +
        1;
  }
  PADDLE_ENFORCE_GT(
      output_size,
      0,
      errors::InvalidArgument(
          "the output size must be greater than 0. But received: "
          "output_size = %d due to the settings of input_size(%d), "
          "padding(%d,%d), "
          "k_size(%d) and stride(%d). Please check again!",
          output_size,
          input_size,
          padding_1,
          padding_2,
          filter_size,
          stride));
  return output_size;
}

inline int MaxPoolOutputSize(int input_size,
                             int filter_size,
                             int padding,
                             int stride) {
  PADDLE_ENFORCE_NE(
      stride,
      0,
      phi::errors::InvalidArgument(
          "The stride of MaxPool shall not be 0, but received %d.", stride));
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2,
                      paddings->size(),
                      errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(),
                          data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T = int>
inline void UpdateKernelSize(std::vector<T>* kernel_size,
                             const DDim data_dims) {
  kernel_size->resize(static_cast<size_t>(data_dims.size()));
  for (size_t i = 0; i < kernel_size->size(); ++i) {
    *(kernel_size->begin() + i) = static_cast<T>(data_dims[i]);
  }
}

}  // namespace funcs
}  // namespace phi
