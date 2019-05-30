/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/sum.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
__global__ void Sum2CUDAKernel(const T *in_0, const T *in_1, T *out,
                               int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    out[id] = in_0[id] + in_1[id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T, bool AddTo>
__global__ void SumArrayCUDAKernel(const T **ins, T *out, int64_t N,
                                   size_t in_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    T total(0);
    for (int i = 0; i < in_size; ++i) {
      const T *tmp = ins[i];
      total += tmp[id];
    }
    if (AddTo) {
      out[id] += total;
    } else {
      out[id] = total;
    }
    id += blockDim.x * gridDim.x;
  }
}

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

/*
 * All inputs' dimension should be the same and the values of
 * each dimension must be the same.
 */
template <typename T>
class SumLoDTensorFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext &context,
                  const std::vector<framework::Tensor *> &inputs,
                  framework::Tensor *output) {
    size_t in_num = inputs.size();
    PADDLE_ENFORCE_LE(in_num, 2UL,
                      "The number of inputs should be not less than 2.");

    bool in_place = (inputs[0] == output) ? true : false;

    if (in_num == 2) {
      if (inputs[0]->numel() > 0 && inputs[1]->numel() > 0) {
        PADDLE_ENFORCE_EQ(inputs[0]->numel(), inputs[1]->numel());
        auto &place = *context.eigen_device();

        auto result = framework::EigenVector<T>::Flatten(*output);
        auto in_0_e = framework::EigenVector<T>::Flatten(*inputs[0]);
        auto in_1_e = framework::EigenVector<T>::Flatten(*inputs[1]);
        result.device(place) = in_0_e + in_1_e;
      } else if (inputs[0]->numel() == 0) {
        // Copy inputs[1] -> output
        framework::TensorCopy(*inputs[1], context.GetPlace(), context, output);
      } else if (inputs[1]->numel() == 0) {
        // Copy inputs[0] -> output
        framework::TensorCopy(*inputs[0], context.GetPlace(), context, output);
      }
    } else {
      size_t start = in_place ? 1 : 0;
      std::vector<const T *> in_data;
      int64_t length = 0;
      for (size_t i = start; i < in_num; ++i) {
        if (inputs[i]->numel() > 0) {
          in_data.emplace_back(inputs[i]->data<T>());
          if (length == 0) {
            length = inputs[i]->numel();
          } else {
            PADDLE_ENFORCE_EQ(length, inputs[i]->numel());
          }
        }
      }

      if (in_data.size() > 0) {
        auto tmp_in_array = platform::DeviceTemporaryAllocator::Instance()
                                .Get(context)
                                .Allocate(in_data.size() * sizeof(T *));
        memory::Copy(boost::get<platform::CUDAPlace>(context.GetPlace()),
                     tmp_in_array->ptr(), platform::CPUPlace(),
                     reinterpret_cast<void *>(in_data.data()),
                     in_data.size() * sizeof(T *), context.stream());
        const T **in_array_data =
            reinterpret_cast<const T **>(tmp_in_array->ptr());

        constexpr size_t theory_sm_threads = 1024;
        auto max_threads = context.GetMaxPhysicalThreadCount();
        auto sm_count = max_threads / theory_sm_threads;
        size_t tile_size = 0;
        if (length >= max_threads)
          tile_size = 1024;
        else if (length < max_threads && length > sm_count * 128)
          tile_size = 512;
        else if (length <= sm_count * 128)
          tile_size = 256;
        dim3 grids = dim3(CEIL_DIV(length, tile_size), 1, 1);
        dim3 blocks = dim3(tile_size, 1, 1);

        if (in_place) {
          SumArrayCUDAKernel<T, true><<<grids, blocks, 0, context.stream()>>>(
              in_array_data, output->data<T>(), length, in_data.size());
        } else {
          SumArrayCUDAKernel<T, false><<<grids, blocks, 0, context.stream()>>>(
              in_array_data, output->data<T>(), length, in_data.size());
        }
      }
    }
  }
};

template class SumLoDTensorFunctor<platform::CUDADeviceContext, float>;
template class SumLoDTensorFunctor<platform::CUDADeviceContext, double>;
template class SumLoDTensorFunctor<platform::CUDADeviceContext, int>;
template class SumLoDTensorFunctor<platform::CUDADeviceContext, int64_t>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
