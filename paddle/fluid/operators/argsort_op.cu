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
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;

const int kMaxRank = 9;  // The max rank of a tensor allowed in Fluid

__global__ void ComputeTargetIdx(const int64_t* in_dims, int dims_size,
                                 int axis, int64_t n, int64_t* trg_idx,
                                 int64_t* med_ids) {
  int64_t index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < n) {
    int64_t shape_out_axis[kMaxRank - 1] = {0};
    int64_t dims_out_axis[kMaxRank - 1] = {0};
    int64_t tmp = index;
    int64_t pos_in_axis = 0;
    int64_t i = dims_size - 2;
    int64_t dim_axis = 0;
    for (int64_t j = dims_size - 1; j >= 0; --j) {
      int64_t dim = in_dims[j];
      if (j != axis) {
        shape_out_axis[i] = tmp % dim;
        dims_out_axis[i] = dim;
        i--;
      } else {
        dim_axis = dim;
        pos_in_axis = tmp % dim_axis;
      }
      tmp /= dim;
    }
    int64_t group = (dims_size > 1) ? shape_out_axis[0] : 0;
    for (int64_t j = 0; j < dims_size - 2; ++j) {
      group = group * dims_out_axis[j + 1] + shape_out_axis[j + 1];
    }

    int64_t traget_idx = group * dim_axis + pos_in_axis;
    trg_idx[index] = traget_idx;
    med_ids[traget_idx] = pos_in_axis;
  }
}

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
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    const T* in_data = input->data<T>();
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* ids_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    int64_t numel = input->numel();
    int64_t groups = numel / in_dims[axis];

    std::vector<int64_t> in_dims_vec = vectorize(in_dims);
    thrust::device_vector<int64_t> in_dims_dev(in_dims_vec.begin(),
                                               in_dims_vec.end());
    int64_t* in_dims_data = thrust::raw_pointer_cast(in_dims_dev.data());
    // Mediate tensor for sorting data and indices
    Tensor mediate_output, mediate_indices;
    T* med_out_data =
        mediate_output.mutable_data<T>(input->dims(), ctx.GetPlace());
    int64_t* med_ids_data =
        mediate_indices.mutable_data<int64_t>(in_dims, ctx.GetPlace());
    // Target index of each element along the given axis in the mediate tensors
    Tensor trg_idx_t;
    int64_t* trg_idx = trg_idx_t.mutable_data<int64_t>(in_dims, ctx.GetPlace());

    auto stream = ctx.cuda_device_context().stream();
    const int num_threads = PADDLE_CUDA_NUM_THREADS;

    ComputeTargetIdx<<<(numel - 1) / num_threads + 1, num_threads, 0, stream>>>(
        in_dims_data, in_dims.size(), axis, numel, trg_idx, med_ids_data);

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
