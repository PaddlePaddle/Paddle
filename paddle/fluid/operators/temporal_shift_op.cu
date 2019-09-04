/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/temporal_shift_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void KeTemporalShiftFw(const T* input, T* output, const int ntchw,
                                  const int tchw, const int chw, const int hw,
                                  const int w, const int t, const int c,
                                  const float shift_ratio) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;
  for (; tid < ntchw; tid += stride) {
    int in = tid / tchw;
    int it = (tid % tchw) / chw;
    int ic = (tid % chw) / hw;
    int ih = (tid % hw) / w;
    int iw = tid % w;

    const int c1 = static_cast<T>(c * shift_ratio);
    const int c2 = static_cast<T>(c * 2 * shift_ratio);

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it < 0 || src_it >= t) {
      output[tid] = 0;
    } else {
      int src_idx = GetEntryIndex(in, src_it, ic, ih, iw, tchw, chw, hw, w);
      output[tid] = input[src_idx];
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftBw(const T* output_grad, T* input_grad,
                                  const int ntchw, const int tchw,
                                  const int chw, const int hw, const int w,
                                  const int t, const int c,
                                  const float shift_ratio) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;
  for (; tid < ntchw; tid += stride) {
    int in = tid / tchw;
    int it = (tid % tchw) / chw;
    int ic = (tid % chw) / hw;
    int ih = (tid % hw) / w;
    int iw = tid % w;

    const int c1 = static_cast<T>(c * shift_ratio);
    const int c2 = static_cast<T>(c * 2 * shift_ratio);

    if (ic < c1) {
      src_it = it - 1;
    } else if (ic < c2) {
      src_it = it + 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      int src_idx = GetEntryIndex(in, src_it, ic, ih, iw, tchw, chw, hw, w);
      input_grad[src_idx] = output_grad[tid];
    }
  }
}

template <typename T>
class TemporalShiftOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");

    const int nt = input->dims()[0];
    const int c = input->dims()[1];
    const int h = input->dims()[2];
    const int w = input->dims()[3];

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>({nt, c, h, w}, ctx.GetPlace());

    int pixelNum = nt * chw;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeTemporalShiftFw<
        T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        input_data, output_data, ntchw, tchw, chw, hw, w, t, c, shift_ratio);
  }
};

template <typename T>
class TemporalShiftGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");

    const int nt = output_grad->dims()[0];
    const int c = output_grad->dims()[1];
    const int h = output_grad->dims()[2];
    const int w = output_grad->dims()[3];

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>({nt, c, h, w}, ctx.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, T>()(
        ctx.template device_context<platform::CUDADeviceContext>(), input_grad,
        static_cast<T>(0));

    int pixelNum = nt * chw;
    int grid_dim = (pixelNum + 512 - 1) / 512;
    grid_dim = grid_dim > 8 ? 8 : grid_dim;

    KeTemporalShiftBw<
        T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
        output_grad_data, input_grad_data, ntchw, tchw, chw, hw, w, t, c,
        shift_ratio);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(temporal_shift, ops::TemporalShiftOpCUDAKernel<float>,
                        ops::TemporalShiftOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(temporal_shift_grad,
                        ops::TemporalShiftGradOpCUDAKernel<float>,
                        ops::TemporalShiftGradOpCUDAKernel<double>);
