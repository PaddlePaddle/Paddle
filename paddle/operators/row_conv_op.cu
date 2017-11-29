/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/math/math_function.h"
#include "paddle/operators/row_conv_op.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using framework::Tensor;

namespace {

inline int DivUp(int x, int y) { return (x + y - 1) / y; }

template <typename T>
__global__ void RowConvForward(const T *in, const T *wt, int num_sequence,
                               int input_dim, int context_length,
                               size_t *batch_indices, T *out) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int bly = blockDim.y;
  int thy = threadIdx.y;

  if (d >= input_dim) return;

  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;
    for (int k = thy; k < current_timesteps; k += bly) {
      T sum = 0;
      for (int w = 0; (w < context_length) && ((k + w) < current_timesteps);
           w++) {
        // sum += wt[w, d] * in[start + k + w, d];
        sum += wt[w * input_dim + d] * in[(start + k + w) * input_dim + d];
      }
      // out[start + k, d] = sum;
      out[(start + k) * input_dim + d] = sum;
    }
  }
}

// Compute input gradient
template <typename T>
__global__ void RowConvGradInput(const T *dout, const T *wt, int num_sequence,
                                 int input_dim, int context_length,
                                 size_t *batch_indices, T *din) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int bly = blockDim.y;
  int thy = threadIdx.y;
  if (d >= input_dim) return;

  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;
    for (int k = thy; k < current_timesteps; k += bly) {
      for (int w = 0; (w < context_length) && ((k + w) < current_timesteps);
           w++) {
        // cur_dip(start + k + w, d) += wt(w, d) * dout(start + k, d);
        din[(start + k + w) * input_dim + d] +=
            wt[w * input_dim + d] * dout[(start + k) * input_dim + d];
      }
    }
  }
}

// Compute weight gradient
template <typename T>
__global__ void RowConvGradFilter(const T *in, const T *dout, int num_sequence,
                                  int input_dim, int context_length,
                                  size_t *batch_indices, T *dfilter) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;  // index along input_dim
  int w = blockIdx.y * blockDim.y + threadIdx.y;  // index along context_length

  if (d >= input_dim || w >= context_length) return;
  for (size_t i = 0; i < num_sequence; i++) {  // For different sequences
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = end - start;
    for (int k = 0; k < current_timesteps; k++) {
      if ((k + w) > current_timesteps) return;
      dfilter[w * input_dim + d] += in[(start + k + w) * input_dim + d] *
                                    dout[(start + k) * input_dim + d];
    }
  }
}
}  // namespace

template <typename T>
class RowConvKernel<platform::GPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<LoDTensor>("X");
    auto *Filter = context.Input<Tensor>("Filter");
    auto *Out = context.Output<LoDTensor>("Out");

    const T *in = X->data<T>();
    const T *weight = Filter->data<T>();
    T *out = Out->mutable_data<T>(context.GetPlace());

    auto batch_indices = X->lod()[0];
    int input_dim = X->dims()[1];
    int num_sequence = batch_indices.size() - 1;
    int context_length = Filter->dims()[0];
    size_t *idx = batch_indices.data();

    auto stream = context.cuda_device_context().stream();
    dim3 block_dim = dim3(32, 32);
    dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);

    RowConvForward<T><<<grid_dim, block_dim, 0, stream>>>(
        in, weight, num_sequence, input_dim, context_length, idx, out);
  }
};

template <typename T>
class RowConvGradKernel<platform::GPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<LoDTensor>("X");
    auto *Filter = context.Input<Tensor>("Filter");
    auto *dOut = context.Input<LoDTensor>(framework::GradVarName("Out"));
    const T *in = X->data<T>();
    const T *weights = Filter->data<T>();
    const T *dout = dOut->data<T>();

    Tensor *dX = context.Output<LoDTensor>(framework::GradVarName("X"));
    Tensor *dFilter = context.Output<Tensor>(framework::GradVarName("Filter"));

    auto batch_indices = X->lod()[0];
    int input_dim = X->dims()[1];
    int num_sequence = batch_indices.size() - 1;
    int context_length = Filter->dims()[0];
    size_t *idx = batch_indices.data();

    auto &device_ctx = context.cuda_device_context();
    math::SetConstant<platform::GPUPlace, T> zero;

    if (dFilter) {
      T *dfilter = dFilter->mutable_data<T>(context.GetPlace());
      zero(device_ctx, dFilter, static_cast<T>(0.0));  // May not need, CHECK ME

      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x),
                           DivUp(context_length, block_dim.y));

      RowConvGradFilter<T><<<grid_dim, block_dim, 0, device_ctx.stream()>>>(
          in, dout, num_sequence, input_dim, context_length, idx, dfilter);
    }

    if (dX) {
      T *din = dX->mutable_data<T>(context.GetPlace());
      zero(device_ctx, dX, static_cast<T>(0.0));
      dim3 block_dim = dim3(32, 32);
      dim3 grid_dim = dim3(DivUp(input_dim, block_dim.x), 1);
      RowConvGradInput<T><<<grid_dim, block_dim, 0, device_ctx.stream()>>>(
          dout, weights, num_sequence, input_dim, context_length, idx, din);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(row_conv,
                       ops::RowConvKernel<paddle::platform::GPUPlace, float>);
REGISTER_OP_GPU_KERNEL(
    row_conv_grad, ops::RowConvGradKernel<paddle::platform::GPUPlace, float>);
