/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/batch_fc_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void add_bias_kernel(
    T* data, int slot_pairs_num, int ins_num, int out_dim, const T* bias) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * ins_num * out_dim) {
    int block_len = ins_num * out_dim;
    int slot_index = idx / block_len;
    int out_dim_index = (idx % block_len) % out_dim;
    T temp = data[idx] + bias[slot_index * out_dim + out_dim_index];
    data[idx] = temp;
  }
}

template <typename T>
void add_bias(gpuStream_t stream,
              T* data,
              int slot_pairs_num,
              int ins_num,
              int out_dim,
              const T* bias) {
  add_bias_kernel<<<GET_BLOCKS(slot_pairs_num * ins_num * out_dim),
                    CUDA_NUM_THREADS,
                    0,
                    stream>>>(data, slot_pairs_num, ins_num, out_dim, bias);
}

template <typename T>
__global__ void add_bias_grad_kernel(const T* dout_data,
                                     int slot_pairs_num,
                                     int ins_num,
                                     int out_dim,
                                     T* db_data) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * out_dim) {
    int row = idx / out_dim;
    int col = idx % out_dim;
    T temp = static_cast<T>(0);
    for (int i = 0; i < ins_num; ++i) {
      int select_indx = ((row + 1) * i + 1) * col;
      temp += dout_data[select_indx];
    }
    db_data[idx] += temp;
  }
}

template <typename T>
void add_bias_grad(gpuStream_t stream,
                   const T* dout_data,
                   int slot_pairs_num,
                   int ins_num,
                   int out_dim,
                   T* db_data) {
  add_bias_grad_kernel<<<GET_BLOCKS(slot_pairs_num * out_dim),
                         CUDA_NUM_THREADS,
                         0,
                         stream>>>(
      dout_data, slot_pairs_num, ins_num, out_dim, db_data);
}

template <typename DeviceContext, typename T>
class BatchFCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // X.dim = slot_pairs_num * ins_num * in_dim
    // W.dim = slot_pairs_num * in_dim * out_dim
    // b.dim = slot_pairs_num * out_dim
    // output.dim = slot_pairs_num * ins_num * out_dim
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* w = ctx.Input<phi::DenseTensor>("W");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto slot_pairs_num = input_dims[0];
    auto ins_num = input_dims[1];
    auto in_dim = input_dims[2];
    auto out_dim = w_dims[2];

    // get data ptr
    const T* in_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();

    output->Resize({slot_pairs_num, ins_num, out_dim});
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    // initialize
    auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto& place =
        *ctx.template device_context<phi::GPUContext>().eigen_device();
    out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    T alpha = 1;
    T beta = 0;
    int64_t strideA = ins_num * in_dim;
    int64_t strideB = in_dim * out_dim;

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    blas.BatchedGEMM(transA,
                     transB,
                     ins_num,
                     out_dim,
                     in_dim,
                     alpha,
                     in_data,
                     w_data,
                     beta,
                     out_data,
                     slot_pairs_num,
                     strideA,
                     strideB);
    add_bias<T>(ctx.cuda_device_context().stream(),
                out_data,
                slot_pairs_num,
                ins_num,
                out_dim,
                bias_data);
  }
};

template <typename DeviceContext, typename T>
class BatchFCGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("Input");
    auto* w = ctx.Input<phi::DenseTensor>("W");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("Input"));
    auto* dw = ctx.Output<phi::DenseTensor>(framework::GradVarName("W"));
    auto* db = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto slot_pairs_num = input_dims[0];
    auto ins_num = input_dims[1];
    auto in_dim = input_dims[2];
    auto out_dim = w_dims[2];

    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto& place =
        *ctx.template device_context<phi::GPUContext>().eigen_device();
    // initialize
    dx->mutable_data<T>(ctx.GetPlace());
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));

    dw->mutable_data<T>(ctx.GetPlace());
    auto dw_eigen = framework::EigenVector<T>::Flatten(*dw);
    dw_eigen.device(place) = dw_eigen.constant(static_cast<T>(0));

    // get data ptr
    const T* x_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* dout_data = dout->data<T>();
    T* dx_data = dx->data<T>();
    T* dw_data = dw->data<T>();

    db->mutable_data<T>(ctx.GetPlace());
    auto db_eigen = framework::EigenVector<T>::Flatten(*db);
    db_eigen.device(place) = db_eigen.constant(static_cast<T>(0));
    T* db_data = db->data<T>();
    add_bias_grad<T>(ctx.cuda_device_context().stream(),
                     dout_data,
                     slot_pairs_num,
                     ins_num,
                     out_dim,
                     db_data);

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    T alpha = 1;
    T beta = 0;

    // dx = dout_data * y^T
    blas.BatchedGEMM(CblasNoTrans,
                     CblasTrans,
                     ins_num,
                     in_dim,
                     out_dim,
                     alpha,
                     dout_data,
                     w_data,
                     beta,
                     dx_data,
                     slot_pairs_num,
                     ins_num * out_dim,
                     out_dim * in_dim);
    // dy = x^T * dout_data
    blas.BatchedGEMM(CblasTrans,
                     CblasNoTrans,
                     in_dim,
                     out_dim,
                     ins_num,
                     alpha,
                     x_data,
                     dout_data,
                     beta,
                     dw_data,
                     slot_pairs_num,
                     in_dim * ins_num,
                     ins_num * out_dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = phi::GPUContext;
REGISTER_OP_CUDA_KERNEL(batch_fc,
                        ops::BatchFCCUDAKernel<GPUCtx, float>,
                        ops::BatchFCCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(batch_fc_grad,
                        ops::BatchFCGradOpCUDAKernel<GPUCtx, float>,
                        ops::BatchFCGradOpCUDAKernel<GPUCtx, double>);
