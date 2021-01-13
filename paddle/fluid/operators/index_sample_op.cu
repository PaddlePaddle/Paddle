// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/index_sample_op.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;

template <typename T, typename IndexT = int>
__global__ void IndexSampleCUDAInner(const T* input, int64_t input_width,
        const IndexT* index, int64_t index_width, int64_t numel,
        T* output) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numel) {
        // ignore the idle thread
        return ;
    }

    int64_t input_idx = (int64_t)(idx / index_width) * input_width + index[idx];

    output[idx] = input[input_idx];
}

template <typename T>
class IndexSampleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *index_var = ctx.InputVar("Index");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &index_tensor = index_var->Get<LoDTensor>();

    auto *out_var = ctx.OutputVar("Out");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

    const auto &index_type = index_tensor.type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto input_dims = input_tensor.dims();
    int64_t input_width = input_dims[1];

    auto index_dims = index_tensor.dims();
    int64_t index_width = index_dims[1];

    T* output = static_cast<T*>(out_tensor->mutable_data<T>(ctx.GetPlace()));

    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleCUDAInner<T, int><<<1, index_tensor.numel()>>>(input_tensor.data<T>(), input_width,
              index_tensor.data<int>(), index_width, index_tensor.numel(),
              output);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSampleCUDAInner<T, int64_t><<<1, index_tensor.numel()>>>(input_tensor.data<T>(), input_width,
              index_tensor.data<int64_t>(), index_width, index_tensor.numel(),
              output);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_sample,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_sample_grad,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
