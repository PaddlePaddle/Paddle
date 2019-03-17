//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "paddle/fluid/operators/sequence_ops/sequence_enumerate_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {
using platform::PADDLE_CUDA_NUM_THREADS;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void CalcOutPut(const T* in_data, const size_t* in_lod,
                           const size_t lod_len, const int64_t win_size,
                           const int64_t pad_value, T* out_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < in_lod[lod_len - 1]) {
    int end_idx = 0;
    // Get LoD interval of index
    for (int i = 1; i < lod_len; ++i) {
      if (index < in_lod[i]) {
        end_idx = in_lod[i];
        break;
      }
    }
    for (size_t i = 0; i < win_size; ++i) {
      int word_pos = index + i;
      out_data[index * win_size + i] =
          word_pos < end_idx ? in_data[word_pos] : pad_value;
    }
  }
}

template <typename T>
__global__ void CalcOutPutNoPad(const T* in_data, const size_t* in_lod,
                           const size_t lod_len, const int64_t win_size,
                           const size_t* new_lod, T* out_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < in_lod[lod_len - 1]) {
    int end_idx = 0;
    int new_index  = 0;
    // Get LoD interval of index
    for (int i = 1; i < lod_len; ++i) {
      if (index < in_lod[i]) {
        end_idx = in_lod[i] - win_size;
        new_index = index - in_lod[i-1] + new_lod[i]
        break;
      }
    }
    if (index < end_idx) {
      new_index = index - in_lod[i-1]
      for (size_t i = 0; i < win_size; ++i) {
        int word_pos = index + i;
        out_data[new_index * win_size + i] = in_data[word_pos];
      }
    }
  }
}


template <typename T>
class SequenceEnumerateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int win_size = context.Attr<int>("win_size");
    int pad_value = context.Attr<int>("pad_value");
    bool need_pad = context.Attr<bool>("need_pad");

    auto in_dims = in->dims();
    auto in_lod = in->lod();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    /* Generate enumerate sequence set */
    auto stream = context.cuda_device_context().stream();
    auto lod0 = in_lod[0];
    auto in_len = in->numel();
    auto in_data = in->data<T>();
    out->Resize({in_dims[0], win_size});
    auto out_data = out->mutable_data<T>(context.GetPlace());
    
    int enumerate_shift = 0;
    if (need_pad) {
      // Copy LoD to GPU
      const size_t* dev_in_lod_ptr = lod0.CUDAData(context.GetPlace());
      // Calc output tensor
      CalcOutPut<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                 PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      in_data, dev_in_lod_ptr, lod0.size(), win_size, pad_value, out_data);
    } else {
      framework::LoD new_lod;
      enumerate_shift = win_size;
      new_lod.emplace_back(1, 0);  // size = 1, value = 0;
      auto new_lod0 = new_lod[0];
      new_lod0.push_back(0);
      for (size_t i = 1; i < lod0.size() - 1; ++i) {
          if (lod0[i] - lod0[i - 1] - enumerate_shift > 0) {
              offset = offset + lod0[i] - lod0[i - 1] - enumerate_shift;
          }
          new_lod0.push_back(offset);
      }
      new_lod.push_back(new_lod0);
      out->set_lod(new_lod);
      // Copy LoD to GPU
      const size_t* dev_in_lod_ptr = lod0.CUDAData(context.GetPlace());
      const size_t* dev_new_lod_ptr = new_lod0.CUDAData(context.GetPlace());
      // Calc output tensor
      CalcOutPutNoPad<<<(in_len - 1) / PADDLE_CUDA_NUM_THREADS + 1,
                 PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
      in_data, dev_in_lod_ptr, lod0.size(), win_size, dev_new_lod_ptr, out_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    sequence_enumerate,
    paddle::operators::SequenceEnumerateOpCUDAKernel<int32_t>,
    paddle::operators::SequenceEnumerateOpCUDAKernel<int64_t>);
