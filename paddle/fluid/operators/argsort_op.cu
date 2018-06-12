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

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/argsort_op.h"
#include "paddle/fluid/platform/assert.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void PermuteInData(const T* in, const int64_t* trg_idx, int64_t n,
                              T* med_out) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    med_out[trg_idx[index]] = in[index];
  }
}

template <typename T>
__global__ void Sort(int64_t axis_dim, int64_t groups, T* med_out,
                     int64_t* med_ids) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < groups) {
    thrust::sort_by_key(thrust::device, med_out + index * axis_dim,
                        med_out + axis_dim * (1 + index),
                        med_ids + index * axis_dim);
  }
}

template <typename T>
__global__ void PermuteMediateData(const T* med_out, const int64_t* med_ids,
                                   const int64_t* trg_idx, int64_t n, T* out,
                                   int64_t* indices) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    out[index] = med_out[trg_idx[index]];
    indices[index] = med_ids[trg_idx[index]];
  }
}

template <typename T>
class ArgsortOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");
    int axis = ctx.Attr<int>("axis");

    auto in_dims = input->dims();
    axis = (axis == -1) ? (in_dims.size() - 1) : axis;

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* ids_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    int64_t numel = input->numel();
    int64_t groups = numel / in_dims[axis];

    // Mediate tensor for sorting
    Tensor mediate_output;
    T* med_out_data =
        mediate_output.mutable_data<T>(input->dims(), ctx.GetPlace());

    // The target index of each elemement in mediate tensor
    std::vector<int64_t> target_idx(numel, 0);
    // To record the index along the given axis for the data in mediate tensor
    std::vector<int64_t> mediate_indices(numel, 0);
    std::vector<int64_t> in_dims_out_axis = vectorize(in_dims);
    in_dims_out_axis.erase(in_dims_out_axis.begin() + axis);
    for (int64_t index = 0; index < numel; ++index) {
      int64_t tmp = index;
      int64_t pos_in_axis = 0;
      std::vector<int64_t> shape;
      for (int64_t j = in_dims.size() - 1; j >= 0; --j) {
        if (j != axis) {
          shape.push_back(tmp % in_dims[j]);
        } else {
          pos_in_axis = tmp % in_dims[j];
        }
        tmp /= in_dims[j];
      }
      std::reverse(shape.begin(), shape.end());
      int64_t group = (shape.size() > 0) ? shape[0] : 0;
      for (size_t j = 0; j < shape.size() - 1; ++j) {
        group = group * in_dims_out_axis[j + 1] + shape[j + 1];
      }

      target_idx[index] = group * in_dims[axis] + pos_in_axis;
      mediate_indices[target_idx[index]] = pos_in_axis;
    }

    thrust::device_vector<int64_t> med_ids_dev(mediate_indices.begin(),
                                               mediate_indices.end());
    int64_t* med_ids_data = thrust::raw_pointer_cast(med_ids_dev.data());
    thrust::device_vector<int64_t> trg_idx_dev(target_idx.begin(),
                                               target_idx.end());
    int64_t* trg_idx = thrust::raw_pointer_cast(trg_idx_dev.data());

    auto stream = reinterpret_cast<const platform::CUDADeviceContext&>(
                      ctx.device_context())
                      .stream();
    auto num_threads = PADDLE_CUDA_NUM_THREADS;

    PermuteInData<<<(numel - 1) / num_threads + 1, num_threads, 0, stream>>>(
        in_data, trg_idx, numel, med_out_data);

    Sort<<<(groups - 1) / num_threads + 1, num_threads, 0, stream>>>(
        in_dims[axis], groups, med_out_data, med_ids_data);

    PermuteMediateData<<<(numel - 1) / num_threads + 1, num_threads, 0,
                         stream>>>(med_out_data, med_ids_data, trg_idx, numel,
                                   out_data, ids_data);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(argsort, paddle::operators::ArgsortOpCUDAKernel<float>,
                        paddle::operators::ArgsortOpCUDAKernel<double>);
