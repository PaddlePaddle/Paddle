/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <algorithm>
#include <string>

#include "paddle/fluid/operators/bilateral_slice_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using DataLayout = framework::DataLayout;

template <typename T>
__device__ T DiffAbs(T x) {
  T eps = 1e-8;
  return sqrt(x * x + eps);
}

template <typename T>
__device__ T DdiffAbs(T x) {
  T eps = 1e-8;
  return x / sqrt(x * x + eps);
}

template <typename T>
__device__ T WeightZ(T x) {
  T abx = DiffAbs(x);
  return max(1.0f - abx, 0.0f);
}

template <typename T>
__device__ T DweightZ(T x) {
  T abx = DiffAbs(x);
  if (abx > 1.0f) {
    return 0.0f;
  } else {
    return DdiffAbs(x);
  }
}

template <typename T>
__global__ void BilateralSliceCudaForwardKernel(T* output,
                                                const T* bilateral_grid,
                                                const T* guide,
                                                const T* input,
                                                GridSizes gsz,
                                                bool has_offset,
                                                int total_count,
                                                int output_chans) {
  int h = gsz.h;
  int w = gsz.w;
  int gd = gsz.gd;
  int gh = gsz.gh;
  int gw = gsz.gw;
  int input_chans = gsz.input_chans;
  int coeff_stride = input_chans;
  int grid_chans = input_chans * output_chans;

  if (has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_count;
       idx += blockDim.x * gridDim.x) {
    int x = idx % w;
    int y = (idx / w) % h;
    int out_c = (idx / (h * w)) % output_chans;
    int b = (idx / (output_chans * w * h));

    T gx = (x + 0.5f) * gw / (1.0f * w);
    T gy = (y + 0.5f) * gh / (1.0f * h);
    T gz = guide[x + w * (y + h * b)] * gd;

    int fx = static_cast<int>(floor(gx - 0.5f));
    int fy = static_cast<int>(floor(gy - 0.5f));
    int fz = static_cast<int>(floor(gz - 0.5f));

    int sy = gw;
    int sz = gw * gh;
    int sc = gd * gw * gh;
    int sb = grid_chans * gd * gw * gh;

    T value = 0.0f;
    for (int in_c = 0; in_c < coeff_stride; ++in_c) {
      T coeff_sample = 0.0f;

      for (int xx = fx; xx < fx + 2; ++xx) {
        int x_ = max(min(xx, gw - 1), 0);
        T wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

        for (int yy = fy; yy < fy + 2; ++yy) {
          int y_ = max(min(yy, gh - 1), 0);
          T wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

          for (int zz = fz; zz < fz + 2; ++zz) {
            int z_ = max(min(zz, gd - 1), 0);
            T wz = WeightZ(zz + 0.5 - gz);
            int c_ = coeff_stride * out_c + in_c;
            int grid_idx = x_ + sy * y_ + sz * z_ + sc * c_ + sb * b;

            coeff_sample += bilateral_grid[grid_idx] * wx * wy * wz;
          }
        }
      }
      if (in_c < input_chans) {
        int input_idx = x + w * (y + h * (in_c + input_chans * b));
        value += coeff_sample * input[input_idx];
      } else {
        value += coeff_sample;
      }
    }

    output[idx] = value;
  }
}

template <typename T>
class BilateralSliceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* grid = ctx.Input<phi::DenseTensor>("Grid");
    auto* guide = ctx.Input<phi::DenseTensor>("Guide");
    auto* output = ctx.Output<phi::DenseTensor>("Out");

    auto* output_data = output->mutable_data<T>(ctx.GetPlace());
    auto* grid_data = grid->data<T>();
    auto* guide_data = guide->data<T>();
    auto* input_data = input->data<T>();

    bool has_offset = ctx.Attr<bool>("has_offset");
    auto input_dims = input->dims();
    auto output_dims = output->dims();
    auto grid_dims = grid->dims();

    int batch_size = input_dims[0];
    int h = input_dims[2];
    int w = input_dims[3];
    int input_chans = input_dims[1];
    int coeff_stride = input_chans;
    int grid_chans = input_chans * output_dims[1];

    int64_t coeffs_chans = grid_dims[1];
    int64_t gd = grid_dims[2];
    int64_t gh = grid_dims[3];
    int64_t gw = grid_dims[4];

    GridSizes grid_sizes;
    grid_sizes.h = h;
    grid_sizes.w = w;
    grid_sizes.bs = batch_size;
    grid_sizes.coeffs_chans = coeffs_chans;
    grid_sizes.gd = gd;
    grid_sizes.gh = gh;
    grid_sizes.gw = gw;
    grid_sizes.input_chans = input_chans;

    int total_count = batch_size * h * w * output_dims[1];

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), total_count);

    BilateralSliceCudaForwardKernel<T>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           ctx.cuda_device_context().stream()>>>(output_data,
                                                 grid_data,
                                                 guide_data,
                                                 input_data,
                                                 grid_sizes,
                                                 has_offset,
                                                 total_count,
                                                 output_dims[1]);
  }
};

template <typename T>
__global__ void BilateralSliceCudaGridGradKernel(T* out_grid_grad,
                                                 const T* upstream_grad,
                                                 const T* guide,
                                                 const T* input,
                                                 GridSizes gsz,
                                                 bool has_offset,
                                                 int grid_count,
                                                 int output_chans) {
  int h = gsz.h;
  int w = gsz.w;
  int gd = gsz.gd;
  int gh = gsz.gh;
  int gw = gsz.gw;
  int input_chans = gsz.input_chans;
  int grid_chans = input_chans * output_chans;
  int coeff_stride = input_chans;

  if (has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < grid_count;
       idx += blockDim.x * gridDim.x) {
    int gx = idx % gw;
    int gy = (idx / gw) % gh;
    int gz = (idx / (gh * gw)) % gd;
    int c = (idx / (gd * gh * gw)) % grid_chans;
    int b = (idx / (grid_chans * gd * gw * gh));

    T scale_w = w * 1.0 / gw;
    T scale_h = h * 1.0 / gh;

    int left_x = static_cast<int>(floor(scale_w * (gx + 0.5 - 1)));
    int right_x = static_cast<int>(ceil(scale_w * (gx + 0.5 + 1)));
    int left_y = static_cast<int>(floor(scale_h * (gy + 0.5 - 1)));
    int right_y = static_cast<int>(ceil(scale_h * (gy + 0.5 + 1)));

    int sy = w;
    int sc = w * h;
    int sb = output_chans * w * h;

    int isy = w;
    int isc = h * w;
    int isb = input_chans * h * w;

    int out_c = c / coeff_stride;
    int in_c = c % coeff_stride;

    T value = 0.0f;
    for (int x = left_x; x < right_x; ++x) {
      int x_ = x;

      if (x_ < 0) {
        x_ = -x_ - 1;
      }
      if (x_ >= w) {
        x_ = 2 * w - 1 - x_;
      }

      T gx2 = (x + 0.5f) / scale_w;
      T wx = max(1.0f - abs(gx + 0.5 - gx2), 0.0f);

      for (int y = left_y; y < right_y; ++y) {
        int y_ = y;

        if (y_ < 0) {
          y_ = -y_ - 1;
        }
        if (y_ >= h) {
          y_ = 2 * h - 1 - y_;
        }

        T gy2 = (y + 0.5f) / scale_h;
        T wy = max(1.0f - abs(gy + 0.5 - gy2), 0.0f);

        int guide_idx = x_ + w * y_ + h * w * b;
        T gz2 = guide[guide_idx] * gd;
        T wz = WeightZ(gz + 0.5f - gz2);
        if (((gz == 0) && (gz2 < 0.5f)) ||
            ((gz == (gd - 1)) && (gz2 > (gd - 0.5f)))) {
          wz = 1.0f;
        }

        int back_idx = x_ + sy * y_ + sc * out_c + sb * b;
        if (in_c < input_chans) {
          int input_idx = x_ + isy * y_ + isc * in_c + isb * b;
          value += wz * wx * wy * upstream_grad[back_idx] * input[input_idx];
        } else {
          value += wz * wx * wy * upstream_grad[back_idx];
        }
      }
    }
    out_grid_grad[idx] = value;
  }
}

template <typename T>
__global__ void BilateralSliceCudaGuideGradKernel(T* out_guide_grad,
                                                  const T* upstream_grad,
                                                  const T* bilateral_grid,
                                                  const T* guide,
                                                  const T* input,
                                                  GridSizes gsz,
                                                  bool has_offset,
                                                  int guide_count,
                                                  int output_chans) {
  int h = gsz.h;
  int w = gsz.w;
  int gd = gsz.gd;
  int gh = gsz.gh;
  int gw = gsz.gw;
  int input_chans = gsz.input_chans;
  int grid_chans = input_chans * output_chans;
  int coeff_stride = input_chans;

  if (has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < guide_count;
       idx += blockDim.x * gridDim.x) {
    int x = idx % w;
    int y = (idx / w) % h;
    int b = (idx / (w * h));

    T gx = (x + 0.5f) * gw / (1.0f * w);
    T gy = (y + 0.5f) * gh / (1.0f * h);
    T gz = guide[x + w * (y + h * b)] * gd;

    int fx = static_cast<int>(floor(gx - 0.5f));
    int fy = static_cast<int>(floor(gy - 0.5f));
    int fz = static_cast<int>(floor(gz - 0.5f));

    int sy = gw;
    int sz = gh * gw;
    int sc = gd * gh * gw;
    int sb = grid_chans * gd * gw * gh;

    T out_sum = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {
      T in_sum = 0.0f;
      for (int in_c = 0; in_c < coeff_stride; ++in_c) {
        T grid_sum = 0.0f;
        for (int xx = fx; xx < fx + 2; ++xx) {
          int x_ = max(min(xx, gw - 1), 0);
          T wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

          for (int yy = fy; yy < fy + 2; ++yy) {
            int y_ = max(min(yy, gh - 1), 0);
            T wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

            for (int zz = fz; zz < fz + 2; ++zz) {
              int z_ = max(min(zz, gd - 1), 0);
              T dwz = gd * DweightZ(zz + 0.5 - gz);

              int c_ = coeff_stride * out_c + in_c;
              int grid_idx = x_ + sy * y_ + sz * z_ + sc * c_ + sb * b;
              grid_sum += bilateral_grid[grid_idx] * wx * wy * dwz;
            }
          }
        }

        if (in_c < input_chans) {
          in_sum +=
              grid_sum * input[x + w * (y + h * (in_c + input_chans * b))];
        } else {
          in_sum += grid_sum;
        }
      }

      out_sum +=
          in_sum * upstream_grad[x + w * (y + h * (out_c + output_chans * b))];
    }

    out_guide_grad[idx] = out_sum;
  }
}

template <typename T>
__global__ void BilateralSliceCudaInputGradKernel(T* out_input_grad,
                                                  const T* upstream_grad,
                                                  const T* bilateral_grid,
                                                  const T* guide,
                                                  GridSizes gsz,
                                                  bool has_offset,
                                                  int input_count,
                                                  int output_chans) {
  int h = gsz.h;
  int w = gsz.w;
  int gd = gsz.gd;
  int gh = gsz.gh;
  int gw = gsz.gw;
  int input_chans = gsz.input_chans;
  int grid_chans = input_chans * output_chans;
  int coeff_stride = input_chans;

  if (has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < input_count;
       idx += blockDim.x * gridDim.x) {
    int x = idx % w;
    int y = (idx / w) % h;
    int in_c = (idx / (h * w)) % input_chans;
    int b = (idx / (input_chans * w * h));

    T gx = (x + 0.5f) * gw / (1.0f * w);
    T gy = (y + 0.5f) * gh / (1.0f * h);
    T gz = guide[x + w * (y + h * b)] * gd;

    int fx = static_cast<int>(floor(gx - 0.5f));
    int fy = static_cast<int>(floor(gy - 0.5f));
    int fz = static_cast<int>(floor(gz - 0.5f));

    int sy = gw;
    int sz = gh * gw;
    int sc = gd * gh * gw;
    int sb = grid_chans * gd * gh * gw;

    T value = 0.0f;
    for (int out_c = 0; out_c < output_chans; ++out_c) {
      T chan_val = 0.0f;

      for (int xx = fx; xx < fx + 2; ++xx) {
        int x_ = max(min(xx, gw - 1), 0);
        T wx = max(1.0f - abs(xx + 0.5 - gx), 0.0f);

        for (int yy = fy; yy < fy + 2; ++yy) {
          int y_ = max(min(yy, gh - 1), 0);
          T wy = max(1.0f - abs(yy + 0.5 - gy), 0.0f);

          for (int zz = fz; zz < fz + 2; ++zz) {
            int z_ = max(min(zz, gd - 1), 0);

            T wz = WeightZ(zz + 0.5 - gz);

            int c_ = coeff_stride * out_c + in_c;
            int grid_idx = x_ + sy * y_ + sz * z_ + sc * c_ + sb * b;
            chan_val += bilateral_grid[grid_idx] * wx * wy * wz;
          }
        }
      }

      value += chan_val *
               upstream_grad[x + w * (y + h * (out_c + output_chans * b))];
    }
    out_input_grad[idx] = value;
  }
}

template <typename T>
class BilateralSliceGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* guide = ctx.Input<phi::DenseTensor>("Guide");
    auto* grid = ctx.Input<phi::DenseTensor>("Grid");
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* grid_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Grid"));
    auto* guide_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Guide"));
    auto* output_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    const T* input_data = input->data<T>();
    const T* guide_data = guide->data<T>();
    const T* grid_data = grid->data<T>();
    const T* output_grad_data = output_grad->data<T>();

    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());
    T* guide_grad_data = guide_grad->mutable_data<T>(ctx.GetPlace());
    T* grid_grad_data = grid_grad->mutable_data<T>(ctx.GetPlace());

    bool has_offset = ctx.Attr<bool>("has_offset");

    auto input_grad_dims = input_grad->dims();
    auto grid_dims = grid_grad->dims();

    int batch_size = input_grad_dims[0];
    int h = input_grad_dims[2];
    int w = input_grad_dims[3];
    int input_chans = input_grad_dims[1];

    int64_t coeffs_chans = grid_dims[1];
    int64_t gd = grid_dims[2];
    int64_t gh = grid_dims[3];
    int64_t gw = grid_dims[4];

    int output_chans = 0;
    if (has_offset) {
      output_chans = coeffs_chans / (input_chans + 1);
    } else {
      output_chans = coeffs_chans / input_chans;
    }
    int grid_count = batch_size * gh * gw * gd * coeffs_chans;
    int guide_count = batch_size * h * w;
    int input_count = batch_size * h * w * input_chans;

    GridSizes grid_sizes;
    grid_sizes.h = h;
    grid_sizes.w = w;
    grid_sizes.bs = batch_size;
    grid_sizes.coeffs_chans = coeffs_chans;
    grid_sizes.gd = gd;
    grid_sizes.gh = gh;
    grid_sizes.gw = gw;
    grid_sizes.input_chans = input_chans;

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), grid_count);

    BilateralSliceCudaGridGradKernel<T>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           ctx.cuda_device_context().stream()>>>(grid_grad_data,
                                                 output_grad_data,
                                                 guide_data,
                                                 input_data,
                                                 grid_sizes,
                                                 has_offset,
                                                 grid_count,
                                                 output_chans);

    config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), guide_count);

    BilateralSliceCudaGuideGradKernel<T>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           ctx.cuda_device_context().stream()>>>(guide_grad_data,
                                                 output_grad_data,
                                                 grid_data,
                                                 guide_data,
                                                 input_data,
                                                 grid_sizes,
                                                 has_offset,
                                                 guide_count,
                                                 output_chans);

    config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), input_count);

    BilateralSliceCudaInputGradKernel<T>
        <<<config.block_per_grid,
           config.thread_per_block,
           0,
           ctx.cuda_device_context().stream()>>>(input_grad_data,
                                                 output_grad_data,
                                                 grid_data,
                                                 guide_data,
                                                 grid_sizes,
                                                 has_offset,
                                                 input_count,
                                                 output_chans);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bilateral_slice,
                        ops::BilateralSliceOpCUDAKernel<float>,
                        ops::BilateralSliceOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(bilateral_slice_grad,
                        ops::BilateralSliceGradOpCUDAKernel<float>,
                        ops::BilateralSliceGradOpCUDAKernel<double>);
