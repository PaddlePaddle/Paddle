/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>
#include <type_traits>
#include "paddle/fluid/operators/sum_op.h"
#include "paddle/fluid/platform/cuda_helper.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
using platform::Times;
using platform::PosixInNsec;
using platform::in_var_length;
using platform::single_var_length;
using platform::in_var_length_total;

template <typename T>
void inline HOSTDEVICE SumImpl(const size_t numel, T *a, T *b) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (; idx < numel; idx += gridDim.x * blockDim.x) {
    atomicAdd(&a[idx], b[idx]);
  }
}

template <typename T, typename... ARGS>
void inline HOSTDEVICE SumImpl(const size_t numel, T *a, T *b, ARGS... args) {
  SumImpl(numel, a, b);
  if (sizeof...(args) == 0) {
    return;
  } else {
    SumImpl(numel, a, args...);
  }
}

template <typename T, typename... ARGS>
__global__ void sum_kernel(const size_t numel, ARGS... args) {
  SumImpl(numel, args...);
}

template <typename T>
struct MySumFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext &context,
                  const std::vector<const Variable *> &in_vars,
                  Variable *out_var) {
    std::list<uint64_t> local_times;
    auto *ctx = &context;
    platform::RecordEvent record_event("Sum LoDTensorArray", ctx);
    auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
    bool in_place = out_var == in_vars[0];
    in_var_length = std::max(in_var_length, in_vars.size());
    uint64_t batch_in_length = 0;

    for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
      PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                     "Only support all inputs are TensorArray");
      auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

      batch_in_length += in_array.size();
      single_var_length = std::max(single_var_length, in_array.size());
      for (size_t i = 0; i < in_array.size(); ++i) {
        if (in_array[i].numel() != 0) {
          if (i >= out_array.size()) {
            out_array.resize(i + 1);
          }
          if (out_array[i].numel() == 0) {
            platform::RecordEvent record_event("Sum TensorCopy", ctx);
            framework::TensorCopy(in_array[i], in_array[i].place(), context,
                                  &out_array[i]);
            out_array[i].set_lod(in_array[i].lod());
          } else {
            PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
            platform::RecordEvent record_event("Sum TensorCompute", ctx);
            auto in = EigenVector<T>::Flatten(in_array[i]);
            auto result = EigenVector<T>::Flatten(out_array[i]);
            // result.device(*context.template device_context<DeviceContext>()
            //                    .eigen_device()) = result + in;
            local_times.push_back(PosixInNsec());
            result.device(*context.eigen_device()) = result + in;
            local_times.push_back(PosixInNsec());
            uint64_t local_cost =
                local_times.back() - *(std::prev(std::prev(local_times.end())));
          }
        }
      }
    }
    in_var_length_total = std::max(in_var_length_total, batch_in_length);
    Times.push_back(local_times);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sum, ops::SumKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SumKernel<paddle::platform::CUDADeviceContext, int64_t>);
