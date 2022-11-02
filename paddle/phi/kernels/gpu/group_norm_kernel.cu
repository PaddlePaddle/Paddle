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

#include "paddle/phi/kernels/group_norm_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

namespace phi {

template <typename T, typename AccT = T>
struct GroupNormFunctor {
  inline HOSTDEVICE T operator()(const T& x,
                                 const AccT& scale,
                                 const AccT& bias) const {
    return static_cast<T>(scale * static_cast<AccT>(x) + bias);
  }
};

template <typename T>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int N,
                                              int C,
                                              int W,
                                              int imsize,
                                              int groups,
                                              int group_size,
                                              T* mean,
                                              T* var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = 0, x_var = 0;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val;
    int hid = imid / W;
    int wid = imid % W;
    val = x[(bid * H + hid) * W * C + wid * C + ccid];

    x_mean += val;
    x_var += val * val;
  }
  x_mean /= number * imsize;
  x_var /= number * imsize;
  CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
  CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
}

template <typename T, int flags>
__global__ void GroupNormForward(const T* x,
                                 const T* mean,
                                 const T* var,
                                 const T* scale,
                                 const T* bias,
                                 int N,
                                 int C,
                                 int W,
                                 int imsize,
                                 int groups,
                                 int group_size,
                                 T epsilon,
                                 T* y,
                                 T* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  T x_mean = mean[ng];
  T x_var = var[ng];
  x_var = x_var - x_mean * x_mean;
  T var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = x_var;
  }
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val;
    int hid, wid;
    int index = (bid * C + ccid) * imsize + imid;
    if (data_layout == DataLayout::kNCHW) {
      val = x[index];
    } else {
      hid = imid / W;
      wid = imid % W;
      val = x[(bid * H + hid) * W * C + wid * C + ccid];
    }
    val = (val - x_mean) * var_inv;
    if (flags & kHasScale) {
      val *= scale[ccid];
    }
    if (flags & kHasBias) {
      val += bias[ccid];
    }
    if (data_layout == DataLayout::kNCHW) {
      y[index] = val;
    } else {
      y[(bid * H + hid) * W * C + wid * C + ccid] = val;
    }
  }
}

template <typename T>
__global__ void UpdateParams(const T* mean,
                             const T* var,
                             const T* scale,
                             const T* bias,
                             const int N,
                             const int C,
                             const int G,
                             const int group_size,
                             float eps,
                             T* real_var,
                             T* eq_scale,
                             T* eq_bias) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N * C) {
    const int ng = tid / group_size;
    const int c = tid % C;
    T x_mean = mean[ng];
    T x_var = var[ng];
    x_var = x_var - x_mean * x_mean;
    T var_inv = rsqrt(x_var + eps);
    T scale_val = (scale == nullptr) ? var_inv : scale[c] * var_inv;
    T bias_val = -scale_val * x_mean + ((bias == nullptr) ? 0 : bias[c]);
    eq_scale[tid] = scale_val;
    eq_bias[tid] = bias_val;
    if (tid - ng * group_size == 0) {
      real_var[ng] = x_var;
    }
  }
}

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  const auto x_dims = x.dims();
  const int N = x_dims[0];
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1]
                                                  : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;

  const int W = (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                                  : x_dims[x_dims.size() - 2]);

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(var);
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  DenseTensor temp_var;
  DenseTensor eq_scale;
  DenseTensor eq_bias;
  temp_var.Resize(var->dims());
  DDim param_dims = (data_layout == DataLayout::kNCHW)
                        ? make_ddim({N, C, 1, 1})
                        : make_ddim({N, 1, 1, C});
  eq_scale.Resize(param_dims);
  eq_bias.Resize(param_dims);
  dev_ctx.template Alloc<T>(&temp_var);
  dev_ctx.template Alloc<T>(&eq_scale);
  dev_ctx.template Alloc<T>(&eq_bias);
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<T>();
  auto* var_data = var->data<T>();
  auto* temp_var_data = temp_var.data<T>();
  auto* eq_scale_data = eq_scale.data<T>();
  auto* eq_bias_data = eq_bias.data<T>();

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();

  int imsize = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < x_dims.size(); ++i) {
      imsize *= x_dims[i];
    }
  } else {
    for (int i = 1; i < x_dims.size() - 1; ++i) {
      imsize *= x_dims[i];
    }
  }

#ifdef __HIPCC__
  int block_size = std::max(std::min(256, imsize), 64);
#else
  int block_size = std::min(1024, imsize);
#endif

  dim3 grid(group_size, groups, N);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
    using AccT = typename kps::details::MPTypeTrait<T>::Type;
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    int size = group_size * imsize;
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }
    block_size_nchw = std::max(block_size_nchw, kps::details::kWarpSize);
    dim3 grids(N * groups);
    dim3 blocks(block_size_nchw);
    if (size < vec_size * block_size_nchw) {
      ScalarGetMeanAndVarNCHW<T><<<grids, blocks, 0, dev_ctx.stream()>>>(
          x_data, mean_data, temp_var_data, size);
    } else {
      VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, dev_ctx.stream()>>>(
              x_data, mean_data, temp_var_data, size);
    }
  } else {
    set_zero(dev_ctx, mean, static_cast<T>(0));
    set_zero(dev_ctx, &temp_var, static_cast<T>(0));
    GroupNormForwardGetMeanAndVar<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, N, C, W, imsize, groups, group_size, mean_data, temp_var_data);
  }

  const int64_t block_num = (N * C + 256 - 1) / 256;
  UpdateParams<T><<<block_num, 256, 0, dev_ctx.stream()>>>(mean_data,
                                                           temp_var_data,
                                                           scale_data,
                                                           bias_data,
                                                           N,
                                                           C,
                                                           groups,
                                                           group_size,
                                                           epsilon,
                                                           var_data,
                                                           eq_scale_data,
                                                           eq_bias_data);
  std::vector<const DenseTensor*> ins{&x, &eq_scale, &eq_bias};
  std::vector<DenseTensor*> outs{y};
  funcs::BroadcastKernel<ElementwiseType::kTernary, T, T>(
      dev_ctx, ins, &outs, -1, GroupNormFunctor<T>());
}

template <typename T>
void GroupNormDirectCUDAFunctor<T>::operator()(gpuStream_t stream,
                                               const T* input,
                                               std::vector<int> input_shape,
                                               const T* bias,
                                               const T* scale,
                                               T* temp_mean,
                                               T* temp_variance,
                                               int groups,
                                               float eps,
                                               T* output,
                                               T* mean,
                                               T* variance,
                                               const DataLayout data_layout) {
  const auto input_ddim = phi::make_ddim(input_shape);
  const int C =
      (data_layout == DataLayout::kNCHW ? input_ddim[1]
                                        : input_ddim[input_ddim.size() - 1]);
  const int group_size = C / groups;
  const int W =
      (data_layout == DataLayout::kNCHW ? input_ddim[input_ddim.size() - 1]
                                        : input_ddim[input_ddim.size() - 2]);

  int image_size = 1;
  if (data_layout == DataLayout::kNCHW) {
    for (int i = 2; i < input_ddim.size(); ++i) {
      image_size *= input_ddim[i];
    }
  } else {
    for (int i = 1; i < input_ddim.size() - 1; ++i) {
      image_size *= input_ddim[i];
    }
  }
#ifdef __HIPCC__
  int block_size = std::max(std::min(256, image_size), 64);
#else
  int block_size = std::min(1024, image_size);
#endif
  dim3 grid(group_size, groups, input_ddim[0]);
  dim3 threads(block_size, 1, 1);
  if (data_layout == DataLayout::kNCHW) {
    using AccT = typename phi::kps::details::MPTypeTrait<float>::Type;
    constexpr int vec_size = sizeof(float4) / sizeof(float);
    int size = group_size * image_size;  // group element size
    const int max_num_threads = 1024;
    int max_block_size = std::min(size / vec_size, max_num_threads);
    int block_size_nchw = 1;
    while (block_size_nchw < max_block_size) {
      block_size_nchw *= 2;
    }

    block_size_nchw = std::max(block_size_nchw, phi::kps::details::kWarpSize);
    dim3 grids(input_ddim[0] * groups);
    dim3 blocks(block_size_nchw);

    if (size < vec_size * block_size_nchw) {
      phi::ScalarGetMeanAndVarNCHW<T>
          <<<grids, blocks, 0, stream>>>(input, temp_mean, temp_variance, size);
    } else {
      phi::VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, stream>>>(input, temp_mean, temp_variance, size);
    }
  } else {
    phi::GroupNormForwardGetMeanAndVar<T>
        <<<grid, threads, 0, stream>>>(input,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       temp_mean,
                                       temp_variance);
  }
  GroupNormForward<T, 3><<<grid, threads, 0, stream>>>(
      input,
      temp_mean,
      temp_variance,
      scale,
      bias,
      input_ddim[0],
      C,
      W,
      image_size,
      groups,
      group_size,
      eps,
      output,
      variance,
      data_layout);  // for now, we only support nchw for group norm
}
template class GroupNormDirectCUDAFunctor<float>;
}  // namespace phi

PD_REGISTER_KERNEL(
    group_norm, GPU, ALL_LAYOUT, phi::GroupNormKernel, float, double) {}
