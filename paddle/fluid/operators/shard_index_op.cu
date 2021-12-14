//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/shard_index_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void ShardIndexInner(const T* in_data, T* out_data,
                                const int64_t numel, const int index_num,
                                const int nshards, const int shard_id,
                                const int ignore_value) {
  int shard_size = (index_num + nshards - 1) / nshards;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    assert(in_data[idx] >= 0 && in_data[idx] < index_num);
    if (in_data[idx] / shard_size == shard_id) {
      out_data[idx] = in_data[idx] % shard_size;
    } else {
      out_data[idx] = ignore_value;
    }
  }
}

using LoDTensor = framework::LoDTensor;

template <typename T>
class ShardIndexCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int index_num = context.Attr<int>("index_num");
    int nshards = context.Attr<int>("nshards");
    int shard_id = context.Attr<int>("shard_id");
    int ignore_value = context.Attr<int>("ignore_value");
    PADDLE_ENFORCE_GT(
        index_num, 0,
        platform::errors::InvalidArgument(
            "The value 'index_num' for Op(shard_index) must be greater than 0, "
            "but the value given is %d.",
            index_num));
    PADDLE_ENFORCE_GT(nshards, 0,
                      platform::errors::InvalidArgument(
                          "The value 'nshard' for Op(shard_index) must be "
                          "greater than 0, but the value given is %d.",
                          nshards));
    PADDLE_ENFORCE_GE(
        shard_id, 0,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be greater or "
            "equal to 0, but the value given is %d.",
            shard_id));
    PADDLE_ENFORCE_LT(
        shard_id, nshards,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be less than "
            "nshards (%d), but the value given is %d.",
            nshards, shard_id));

    out->Resize(in->dims());
    out->set_lod(in->lod());
    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();
    ShardIndexInner<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                          PADDLE_CUDA_NUM_THREADS,
                      PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        in_data, out_data, numel, index_num, nshards, shard_id, ignore_value);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(shard_index, ops::ShardIndexCUDAKernel<int>,
                        ops::ShardIndexCUDAKernel<int64_t>);
