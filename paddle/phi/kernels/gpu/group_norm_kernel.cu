
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

template <typename T, typename AccT>
__global__ void GroupNormForwardGetMeanAndVar(const T* x,
                                              int N,
                                              int C,
                                              int W,
                                              int imsize,
                                              int groups,
                                              int group_size,
                                              AccT* mean,
                                              AccT* var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  AccT x_mean = static_cast<AccT>(0);
  AccT x_var = static_cast<AccT>(0);
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val;
    int hid = imid / W;
    int wid = imid % W;
    val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);

    x_mean += val;
    x_var += val * val;
  }
  x_mean /= number * imsize;
  x_var /= number * imsize;
  CudaAtomicAddWithWarp(&mean[bid * groups + gid], x_mean);
  CudaAtomicAddWithWarp(&var[bid * groups + gid], x_var);
}

template <typename T, typename AccT, int flags>
__global__ void GroupNormForward(const T* x,
                                 const AccT* mean,
                                 const AccT* var,
                                 const T* scale,
                                 const T* bias,
                                 int N,
                                 int C,
                                 int W,
                                 int imsize,
                                 int groups,
                                 int group_size,
                                 float epsilon,
                                 T* y,
                                 T* real_mean,
                                 T* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  AccT x_mean = static_cast<AccT>(mean[ng]);
  AccT x_var = static_cast<AccT>(var[ng]);
  x_var = x_var - x_mean * x_mean;
  AccT var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = static_cast<T>(x_var);
    real_mean[ng] = static_cast<T>(x_mean);
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
    val = static_cast<T>((static_cast<AccT>(val) - x_mean) * var_inv);
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

template <typename T, typename AccT>
__global__ void UpdateParams(const AccT* mean,
                             const AccT* var,
                             const T* scale,
                             const T* bias,
                             const int N,
                             const int C,
                             const int G,
                             const int group_size,
                             float eps,
                             T* real_mean,
                             T* real_var,
                             T* eq_scale,
                             T* eq_bias) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N * C) {
    const int ng = tid / group_size;
    const int c = tid % C;
    AccT x_mean = static_cast<AccT>(mean[ng]);
    AccT x_var = static_cast<AccT>(var[ng]);
    x_var = x_var - x_mean * x_mean;
    AccT var_inv = rsqrt(x_var + eps);
    AccT scale_val = (scale == nullptr) ? var_inv : static_cast<AccT>(scale[c]) * var_inv;
    AccT bias_val = -scale_val * x_mean + ((bias == nullptr) ? static_cast<AccT>(0) : static_cast<AccT>(bias[c]));
    eq_scale[tid] = static_cast<T>(scale_val);
    eq_bias[tid] = static_cast<T>(bias_val);
    if (tid - ng * group_size == 0) {
      real_var[ng] = static_cast<T>(x_var);
      real_mean[ng] = static_cast<T>(x_mean);
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
  using AccT = typename kps::details::MPTypeTrait<T>::Type;

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
  VLOG(1)<<"@@@ gn pin 1";
  dev_ctx.template Alloc<T>(y);
  VLOG(1)<<"@@@ gn pin 2";
  dev_ctx.template Alloc<T>(mean);
  VLOG(1)<<"@@@ gn pin 3";
  dev_ctx.template Alloc<T>(var);
  phi::funcs::SetConstant<GPUContext, AccT> set_zero;
  
  DenseTensor temp_mean;
  DenseTensor temp_var;
  DenseTensor eq_scale;
  DenseTensor eq_bias;
  temp_var.Resize(var->dims());
  temp_mean.Resize(mean->dims());
  auto param_dims = make_ddim(std::vector<int>(x_dims.size(), 1));
  param_dims[0] = N;
  if (data_layout == DataLayout::kNCHW) {
    param_dims[1] = C;
  } else {
    param_dims[param_dims.size() - 1] = C;
  }
  eq_scale.Resize(param_dims);
  eq_bias.Resize(param_dims);
  dev_ctx.template Alloc<AccT>(&temp_var);
  dev_ctx.template Alloc<AccT>(&temp_mean);
  dev_ctx.template Alloc<T>(&eq_scale);
  dev_ctx.template Alloc<T>(&eq_bias);
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<T>();
  auto* var_data = var->data<T>();
  auto* temp_mean_data = temp_mean.data<AccT>();
  auto* temp_var_data = temp_var.data<AccT>();
  auto* eq_scale_data = eq_scale.data<T>();
  auto* eq_bias_data = eq_bias.data<T>();
  VLOG(1)<<"@@@ gn pin 4";

  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
    VLOG(1)<<"@@@ gn pin 5";
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();
  VLOG(1)<<"@@@ gn pin 6";
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
      ScalarGetMeanAndVarNCHW<T, AccT><<<grids, blocks, 0, dev_ctx.stream()>>>(
          x_data, temp_mean_data, temp_var_data, size);
    } else {
      VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, dev_ctx.stream()>>>(
              x_data, temp_mean_data, temp_var_data, size);
    }
  } else {
    set_zero(dev_ctx, &temp_mean, static_cast<AccT>(0));
    set_zero(dev_ctx, &temp_var, static_cast<AccT>(0));
    GroupNormForwardGetMeanAndVar<T, AccT><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, N, C, W, imsize, groups, group_size, temp_mean_data, temp_var_data);
  }

  const int64_t block_num = (N * C + 256 - 1) / 256;
  UpdateParams<T, AccT><<<block_num, 256, 0, dev_ctx.stream()>>>(
                                                           temp_mean_data,
                                                           temp_var_data,
                                                           scale_data,
                                                           bias_data,
                                                           N,
                                                           C,
                                                           groups,
                                                           group_size,
                                                           epsilon,
                                                           mean_data,
                                                           var_data,
                                                           eq_scale_data,
                                                           eq_bias_data);
  std::vector<const DenseTensor*> ins{&x, &eq_scale, &eq_bias};
  std::vector<DenseTensor*> outs{y};
  VLOG(1)<<"@@@ gn pin 7";
  funcs::BroadcastKernel<ElementwiseType::kTernary, T, T>(
      dev_ctx, ins, &outs, -1, GroupNormFunctor<T, AccT>());
  VLOG(1)<<"@@@ gn pin8";
  VLOG(1)<<"@@@ gn output:"<<*y;
}

template <typename T, typename AccT>
void GroupNormDirectCUDAFunctor<T, AccT>::operator()(gpuStream_t stream,
                                               const T* input,
                                               std::vector<int> input_shape,
                                               const T* bias,
                                               const T* scale,
                                               AccT* temp_mean,
                                               AccT* temp_variance,
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
      phi::ScalarGetMeanAndVarNCHW<T, AccT>
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
  GroupNormForward<T, T, 3><<<grid, threads, 0, stream>>>(
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
      mean,
      variance,
      data_layout);  // for now, we only support nchw for group norm
}
template class GroupNormDirectCUDAFunctor<float>;
}  // namespace phi

PD_REGISTER_KERNEL(
    group_norm, GPU, ALL_LAYOUT, phi::GroupNormKernel, float, double, phi::dtype::float16) {}
