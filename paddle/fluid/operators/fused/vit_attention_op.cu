// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <paddle/fluid/platform/device_context.h>
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void set_ptr_kernel(const void **array_a, const void **array_b,
                               void **array_c, T *device_a, T *device_b,
                               T *device_c, int batch_a, int batch_b,
                               int batch_c, int head_a, int head_b,
                               int head_c) {
  int batch_id = blockIdx.x;
  int head_id = threadIdx.x;
  array_a[batch_id * blockDim.x + head_id] =
      device_a + batch_id * batch_a + head_id * head_a;
  array_b[batch_id * blockDim.x + head_id] =
      device_b + batch_id * batch_b + head_id * head_b;
  array_c[batch_id * blockDim.x + head_id] =
      device_c + batch_id * batch_c + head_id * head_c;
}

template <typename DeviceContext, typename T>
class VitAttentionKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    using Tensor = framework::Tensor;
    // input
    /*
    demo:
    input => {batch, seq_len, hidden_size*3} => [Q,K,V]
    Q=[aaa,bbb,ccc,...yyy,zzz]=>head_number=24,head_size=3,hidden_size=72
    */
    auto *input = context.Input<framework::Tensor>("Input");
    auto *input_d = input->data<T>();
    auto input_dims = input->dims();
    int batch = input_dims[0];
    int seq_len = input_dims[1];
    int hidden_three = input_dims[2];
    int hidden_size = hidden_three / 3;
    // prepare attr and context
    int head_number = context.Attr<int>("head_number");
    int head_size = hidden_size / head_number;
    float scale = context.Attr<float>("scale");
    auto &device_ctx = context.template device_context<DeviceContext>();
    auto stream = device_ctx.stream();
    // out
    auto *out = context.Output<framework::Tensor>("Out");
    out->Resize({batch, seq_len, head_number * head_size});
    auto *output_d = out->mutable_data<T>(context.GetPlace());
    // prepare tmp tensor(softmax_d)
    Tensor temp_tensor;
    temp_tensor.Resize({batch, head_number, seq_len, seq_len});
    auto *temp_softmax_d = temp_tensor.mutable_data<T>(context.GetPlace());
    // qkv ptr
    auto *input_q_d = const_cast<T *>(input_d + hidden_size * 0);
    auto *input_k_d = const_cast<T *>(input_d + hidden_size * 1);
    auto *input_v_d = const_cast<T *>(input_d + hidden_size * 2);

    // prepare q * k
    int batch_count = batch * head_number;

    const void **d_a_array, **d_b_array, **array;
    void **d_c_array;
    cudaMalloc(&array, 3 * batch_count * sizeof(T *));
    d_a_array = array;
    d_b_array = &array[batch_count];
    d_c_array = const_cast<void **>(&array[2 * batch_count]);
    // first:set_ptr_kernel
    dim3 grid_ptr(batch);
    dim3 block_ptr(head_number);
    set_ptr_kernel<<<grid_ptr, block_ptr, 0, stream>>>(
        d_a_array, d_b_array, d_c_array, input_q_d, input_k_d, temp_softmax_d,
        seq_len * hidden_three, seq_len * hidden_three,
        seq_len * seq_len * head_number, head_size, head_size, seq_len);
    // second:compute Q*K
    auto alpha = (T)scale;
    auto beta = (T)0.0f;
    int lda = hidden_three;
    int ldb = hidden_three;
    int ldc = seq_len * head_number;

    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(device_ctx);
    blas.BatchedGemmArray(CblasTrans, CblasNoTrans, seq_len, seq_len, head_size,
                          alpha, d_b_array, ldb, d_a_array, lda, beta,
                          d_c_array, ldc, batch_count);
    // softmax
    phi::SoftmaxForwardCUDAKernelDriver<T>(device_ctx, temp_tensor, -1,
                                           &temp_tensor);
    // softmax * v
    set_ptr_kernel<<<grid_ptr, block_ptr, 0, stream>>>(
        d_a_array, d_b_array, d_c_array, temp_softmax_d, input_v_d, output_d,
        seq_len * seq_len * head_number, seq_len * hidden_three,
        seq_len * hidden_size, seq_len, head_size, head_size);

    alpha = (T)1.0f;
    lda = seq_len * head_number;
    ldb = hidden_three;
    ldc = hidden_size;

    blas.BatchedGemmArray(CblasNoTrans, CblasNoTrans, head_size, seq_len,
                          seq_len, alpha, d_b_array, ldb, d_a_array, lda, beta,
                          d_c_array, ldc, batch_count);
    cudaFree(array);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    vit_attention,
    ops::VitAttentionKernel<paddle::platform::CUDADeviceContext, float>,
    ops::VitAttentionKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::float16>);
