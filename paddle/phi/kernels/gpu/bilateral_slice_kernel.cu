// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/bilateral_slice_kernel_impl.cu.h"

namespace phi {

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

template <typename T, typename Context>
void BilateralSliceOpCUDAKernel(const Context& dev_ctx,
                                const DenseTensor& x_in,
                                const DenseTensor& grid_in,
                                const DenseTensor& guide_in,
                                bool has_offset,
                                DenseTensor* out) {
  auto* input = &x_in;
  auto* grid = &grid_in;
  auto* guide = &guide_in;
  auto* output = out;

  auto* output_data = dev_ctx.template Alloc<T>(output);
  auto* grid_data = grid->data<T>();
  auto* guide_data = guide->data<T>();
  auto* input_data = input->data<T>();

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

  phi::backends::gpu::GpuLaunchConfig config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total_count);

  BilateralSliceCudaForwardKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          output_data,
          grid_data,
          guide_data,
          input_data,
          grid_sizes,
          has_offset,
          total_count,
          output_dims[1]);
}
}  // namespace phi
PD_REGISTER_KERNEL(bilateral_slice,
                   GPU,
                   ALL_LAYOUT,
                   phi::BilateralSliceOpCUDAKernel,
                   float,
                   double) {}
