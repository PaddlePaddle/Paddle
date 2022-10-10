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
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeTemporalShiftFwNCHW(const T* input,
                                      T* output,
                                      const int ntchw,
                                      const int tchw,
                                      const int chw,
                                      const int hw,
                                      const int t,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < ntchw; tid += stride) {
    int it = (tid % tchw) / chw;
    int ic = (tid % chw) / hw;

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
      output[tid] = input[tid + (src_it - it) * chw];
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftFwNHWC(const T* input,
                                      T* output,
                                      const int nthwc,
                                      const int thwc,
                                      const int hwc,
                                      const int t,
                                      const int c,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < nthwc; tid += stride) {
    int it = (tid % thwc) / hwc;
    int ic = tid % c;

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
      output[tid] = input[tid + (src_it - it) * hwc];
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftBwNCHW(const T* output_grad,
                                      T* input_grad,
                                      const int ntchw,
                                      const int tchw,
                                      const int chw,
                                      const int hw,
                                      const int t,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < ntchw; tid += stride) {
    int it = (tid % tchw) / chw;
    int ic = (tid % chw) / hw;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[tid] = output_grad[tid + (src_it - it) * chw];
    } else {
      input_grad[tid] = 0;
    }
  }
}

template <typename T>
__global__ void KeTemporalShiftBwNHWC(const T* output_grad,
                                      T* input_grad,
                                      const int nthwc,
                                      const int thwc,
                                      const int hwc,
                                      const int t,
                                      const int c,
                                      const int c1,
                                      const int c2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int src_it = 0;

  for (; tid < nthwc; tid += stride) {
    int it = (tid % thwc) / hwc;
    int ic = tid % c;

    if (ic < c1) {
      src_it = it + 1;
    } else if (ic < c2) {
      src_it = it - 1;
    } else {
      src_it = it;
    }

    if (src_it >= 0 && src_it < t) {
      input_grad[tid] = output_grad[tid + (src_it - it) * hwc];
    } else {
      input_grad[tid] = 0;
    }
  }
}

template <typename T>
class TemporalShiftOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()),
                      true,
                      platform::errors::InvalidArgument(
                          "This kernel only runs on GPU device."));
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");
    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_format_str);

    const int nt = input->dims()[0];
    const int c = (data_layout == DataLayout::kNCHW ? input->dims()[1]
                                                    : input->dims()[3]);
    const int h = (data_layout == DataLayout::kNCHW ? input->dims()[2]
                                                    : input->dims()[1]);
    const int w = (data_layout == DataLayout::kNCHW ? input->dims()[3]
                                                    : input->dims()[2]);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    framework::DDim out_dims =
        (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                          : phi::make_ddim({nt, h, w, c}));
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(out_dims, ctx.GetPlace());

    int pixelNum = nt * chw;
    int threads = 1024;
    int grid = (pixelNum + threads - 1) / threads;
    const auto& dev_ctx = ctx.cuda_device_context();
    int blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() / threads;
    grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

    if (data_layout == DataLayout::kNCHW) {
      KeTemporalShiftFwNCHW<T>
          <<<grid, threads, 0, ctx.cuda_device_context().stream()>>>(
              input_data, output_data, ntchw, tchw, chw, hw, t, c1, c2);
    } else {
      KeTemporalShiftFwNHWC<T>
          <<<grid, threads, 0, ctx.cuda_device_context().stream()>>>(
              input_data, output_data, ntchw, tchw, chw, t, c, c1, c2);
    }
  }
};

template <typename T>
class TemporalShiftGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    int t = ctx.Attr<int>("seg_num");
    float shift_ratio = ctx.Attr<float>("shift_ratio");
    const std::string data_format_str = ctx.Attr<std::string>("data_format");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_format_str);

    const int nt = output_grad->dims()[0];
    const int c = (data_layout == DataLayout::kNCHW ? output_grad->dims()[1]
                                                    : output_grad->dims()[3]);
    const int h = (data_layout == DataLayout::kNCHW ? output_grad->dims()[2]
                                                    : output_grad->dims()[1]);
    const int w = (data_layout == DataLayout::kNCHW ? output_grad->dims()[3]
                                                    : output_grad->dims()[2]);

    const int hw = h * w;
    const int chw = c * hw;
    const int tchw = t * chw;
    const int ntchw = nt * chw;

    const int c1 = static_cast<int>(c * shift_ratio);
    const int c2 = static_cast<int>(c * 2 * shift_ratio);

    framework::DDim in_grad_dims =
        (data_layout == DataLayout::kNCHW ? phi::make_ddim({nt, c, h, w})
                                          : phi::make_ddim({nt, h, w, c}));
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>(in_grad_dims, ctx.GetPlace());

    int pixelNum = nt * chw;
    int threads = 1024;
    int grid = (pixelNum + threads - 1) / threads;
    const auto& dev_ctx = ctx.cuda_device_context();
    int blocks_per_sm = dev_ctx.GetMaxPhysicalThreadCount() / threads;
    grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

    if (data_layout == DataLayout::kNCHW) {
      KeTemporalShiftBwNCHW<
          T><<<grid, threads, 0, ctx.cuda_device_context().stream()>>>(
          output_grad_data, input_grad_data, ntchw, tchw, chw, hw, t, c1, c2);
    } else {
      KeTemporalShiftBwNHWC<
          T><<<grid, threads, 0, ctx.cuda_device_context().stream()>>>(
          output_grad_data, input_grad_data, ntchw, tchw, chw, t, c, c1, c2);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    temporal_shift,
    ops::TemporalShiftOpCUDAKernel<float>,
    ops::TemporalShiftOpCUDAKernel<double>,
    ops::TemporalShiftOpCUDAKernel<paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    temporal_shift_grad,
    ops::TemporalShiftGradOpCUDAKernel<float>,
    ops::TemporalShiftGradOpCUDAKernel<double>,
    ops::TemporalShiftGradOpCUDAKernel<paddle::platform::float16>);
