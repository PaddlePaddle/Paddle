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
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/group_norm_utils.h"

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

static inline int32_t divUp(int32_t m, int32_t n) { return (m + n - 1) / n; }

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sumSq;
};

template <typename T>
struct GroupNormNHWCParams {
  // The output buffer. Layout NHWC.
  T* dst;
  // The output buffer. Layout NHWC.
  // The input buffer. Layout NHWC.
  T const* srcX;
  // The input buffer. Layout NHWC.
  // The gamma scaling factor.
  void const* gamma;
  // The beta term to add in GN.
  void const* beta;
  // The temporary buffer to do the global parallel reduction. Size:
  // BLOCKS_PER_BATCH x C x 2.
  float* redBuffer;

  // The number of instances in the batch.
  int32_t n;
  // The height and width of each activation map.
  int32_t h, w;
  // The number of channels.
  int32_t c;
  // The number of groups.
  int32_t groups;
  // Do we apply the Silu activation function?
  bool withSilu;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (h * w) and the number of
  // activations per block.
  int32_t hw, hwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock, cPerGroup;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hwc in floats (to compute mean/var).
  float invHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;
  // epsilon, Constant for numerical stability
  float eps;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const &a,
                                         GroupSums const &b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

static int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void UpdateSum(const T * srcX, float &sum, float &sumSq) {
  printf("###############  UpdateSum error ####################\n");
  float src_data = *reinterpret_cast<float const *>(srcX);
  sum += src_data;
  sumSq += src_data * src_data;
}

template <>
inline __device__ void UpdateSum<phi::dtype::float16, 2>(const phi::dtype::float16 * srcX, float &sum, float &sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const *>(srcX);
  float2 f2 = __half22float2(h2);
  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}


template <>
inline __device__ void UpdateSum<phi::dtype::bfloat16, 2>(const phi::dtype::bfloat16 * srcX, float &sum, float &sumSq) {
  __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const *>(srcX);
  float2 f2 = __bfloat1622float2(h2);
  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <typename T, int THREADS_PER_BLOCK, int THREADS_PER_CHANNEL>
__global__ void groupNormNHWCSumKernel(const GroupNormNHWCParams<T> params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, THREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * THREADS_PER_CHANNEL;
  if (ci >= params.c) {
    return;
  }
  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The offset.
    int64_t offset = static_cast<int64_t>(ni) * params.hwc +
                    static_cast<int64_t>(hwi) * params.c + ci;
    float src_data = *reinterpret_cast<float const *>(&params.srcX[offset]);
    UpdateSum<T, THREADS_PER_CHANNEL>(&params.srcX[offset], sum, sumSq);
  }
    

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = threadIdx.x * THREADS_PER_CHANNEL / params.cPerGroup;
  int32_t cj = threadIdx.x * THREADS_PER_CHANNEL - params.cPerGroup * gi;


  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  
  if (cj == params.cPerGroup - THREADS_PER_CHANNEL || cj == params.cPerBlock - 1) { 
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

  int32_t blockThrs = params.groupsPerBlock;

  if (params.groupsPerBlock == 0) {
    blockThrs = 1;
    gj = ci / params.cPerGroup;
  }

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= blockThrs || gj >= params.groups) {
    return;
  }

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];
  // printf("    $$$$$$$$ sumx, sum y %f  %f  $$$$$$$$$\n", sums.x, sums.y);
  atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
  atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

template <typename T>
void groupNormNHWCSum(const GroupNormNHWCParams<T> &params, cudaStream_t stream) {
  dim3 grid;
  grid.x = divUp(params.c, params.cPerBlock);
  grid.y = divUp(params.hw, params.hwPerBlock);
  grid.z = params.n;
  int sharedMemSize = 0;
  printf("########  params.cPerBlock  %d   %d  \n", params.cPerBlock, params.cPerGroup);  
  if (params.cPerGroup % 2 == 0) {
    switch (params.cPerBlock) {
      case 512:
      case 480:
        groupNormNHWCSumKernel<T, 256, 2><<<grid, 256, 0, stream>>>(params);
        break;
      case 320:
        groupNormNHWCSumKernel<T, 160, 2><<<grid, 160, 0, stream>>>(params);
        break;
      case 256:
        groupNormNHWCSumKernel<T, 128, 2><<<grid, 128, 0, stream>>>(params);
        break;
      case 128:
        groupNormNHWCSumKernel<T, 64, 2><<<grid, 64, 0, stream>>>(params);
        break;
      default:
        sharedMemSize = sizeof(float2) * params.cPerBlock;
        printf("########  share_men  %d  \n", sharedMemSize);
        // sharedMemSize = sizeof(float2) * params.cPerBlock / 2;
        // groupNormNHWCSumKernel<T, 2><<<grid, params.cPerBlock / 2, sharedMemSize, stream>>>(params);
    }
  } else {
    switch (params.cPerBlock) {
      case 512:
        groupNormNHWCSumKernel<T, 512, 1><<<grid, 512, 0, stream>>>(params);
        break;
      case 480:
        groupNormNHWCSumKernel<T, 480, 1><<<grid, 480, 0, stream>>>(params);
        break;
      case 320:
        groupNormNHWCSumKernel<T, 320, 1><<<grid, 320, 0, stream>>>(params);
        break;
      case 256:
        groupNormNHWCSumKernel<T, 256, 1><<<grid, 256, 0, stream>>>(params);
        break;
      case 128:
        groupNormNHWCSumKernel<T, 128, 1><<<grid, 128, 0, stream>>>(params);
        break;
      default:
        sharedMemSize = sizeof(float2) * params.cPerBlock;
        printf("########  share_men  %d  \n", sharedMemSize);
        // groupNormNHWCSumKernel<T, 1><<<grid, params.cPerBlock, sharedMemSize, stream>>>(params);
    }
  }
}


template <typename T, int THREADS_PER_CHANNEL>
inline __device__ void GroupNormCompute(
    int32_t hwBegin, int32_t hwEnd, int32_t ci, const GroupNormNHWCParams<T> &params, float mean, float invStdDev) {
  float gamma, beta;
  gamma = *reinterpret_cast<float const *>(
        reinterpret_cast<T const *>(params.gamma) + ci);
  beta = *reinterpret_cast<float const *>(
        reinterpret_cast<T const *>(params.beta) + ci);

  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.hwc + hwi * params.c + ci;
    float src_data = *reinterpret_cast<float const *>(&params.srcX[offset]);
  
    // Normalize the channels.
    src_data = (src_data - mean) * invStdDev;
    // Scale by gamma and add beta.
    src_data = gamma * src_data + beta;

    // Apply Silu if needed.
    if (params.withSilu) {
      src_data = src_data * sigmoid(src_data);
    }

    // Store the scaled values.
    *reinterpret_cast<T *>(&params.dst[offset]) = src_data;
  }
}

template <>
inline __device__ void GroupNormCompute<phi::dtype::float16, 2>( 
    int32_t hwBegin, int32_t hwEnd, int32_t ci, const GroupNormNHWCParams<phi::dtype::float16> &params, float mean, float invStdDev) {

  float2 gammaF2, betaF2;
  gammaF2 = __half22float2(*reinterpret_cast<__half2 const *>(
        reinterpret_cast<half const *>(params.gamma) + ci));
  betaF2 = __half22float2(*reinterpret_cast<__half2 const *>(
        reinterpret_cast<half const *>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.hwc + hwi * params.c + ci;

    // Fetch two channels per thread.
    __half2 h2 = *reinterpret_cast<__half2 const *>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Silu if needed.
    if (params.withSilu) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }
    // Store the scaled values.
    *reinterpret_cast<__half2 *>(&params.dst[offset]) = __float22half2_rn(f2);
  }
}


template <>
inline __device__ void GroupNormCompute<phi::dtype::bfloat16, 2> ( 
    int32_t hwBegin, int32_t hwEnd, int32_t ci, const GroupNormNHWCParams<phi::dtype::bfloat16> &params, float mean, float invStdDev){
  float2 gammaF2, betaF2;
  gammaF2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(
        reinterpret_cast<__nv_bfloat16 const *>(params.gamma) + ci));
  betaF2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(
        reinterpret_cast<__nv_bfloat16 const *>(params.beta) + ci));

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)blockIdx.z * params.hwc + hwi * params.c + ci;

    // Fetch two channels per thread.
    __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162 const *>(&params.srcX[offset]);

    // Extract the two half values.
    float2 f2 = __bfloat1622float2(h2);

    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    f2.x = gammaF2.x * f2.x + betaF2.x;
    f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Silu if needed.
    if (params.withSilu) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }
    // Store the scaled values.
    *reinterpret_cast<__nv_bfloat162 *>(&params.dst[offset]) = __float22bfloat162_rn(f2);
  }
}


template <typename T, int THREADS_PER_CHANNEL>
__global__ void groupNormNHWCScaleKernel(const GroupNormNHWCParams<T> params) {
  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * THREADS_PER_CHANNEL;

  if (ci >= params.c) {
    return;
  }
  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < params.groups) {
    sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
  }

  // Compute the mean.
  float mean = sum * params.invHWC;
  // Compute the variance.
  float var = sumSq * params.invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = rsqrtf(var + params.eps);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);
  GroupNormCompute<T, THREADS_PER_CHANNEL>(hwBegin, hwEnd, ci, params, mean, invStdDev);
}

template <typename T>
void groupNormNHWCScale(const GroupNormNHWCParams<T> &params,
                        cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = divUp(params.c, params.cPerBlock);
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;
  if (params.cPerGroup % 2 == 0) {
    switch (params.cPerBlock) {
      case 512:
      case 480:
        groupNormNHWCScaleKernel<T, 2><<<grid, 256, 0, stream>>>(params);
        break;
      case 320:
        groupNormNHWCScaleKernel<T, 2><<<grid, 160, 0, stream>>>(params);
        break;
      case 256:
        groupNormNHWCScaleKernel<T, 2><<<grid, 128, 0, stream>>>(params);
        break;
      case 128:
        groupNormNHWCScaleKernel<T, 2><<<grid, 64, 0, stream>>>(params);
        break;
      default:
        groupNormNHWCScaleKernel<T, 2><<<grid, params.cPerBlock / 2, 0, stream>>>(params);
    }
  } else {
    switch (params.cPerBlock) {
      case 512:
        groupNormNHWCScaleKernel<T, 1><<<grid, 512, 0, stream>>>(params);
        break;
      case 480:
        groupNormNHWCScaleKernel<T, 1><<<grid, 480, 0, stream>>>(params);
        break;
      case 320:
        groupNormNHWCScaleKernel<T, 1><<<grid, 320, 0, stream>>>(params);
        break;
      case 256:
        groupNormNHWCScaleKernel<T, 1><<<grid, 256, 0, stream>>>(params);
        break;
      case 128:
        groupNormNHWCScaleKernel<T, 1><<<grid, 128, 0, stream>>>(params);
        break;
      default:
        groupNormNHWCScaleKernel<T, 1><<<grid, params.cPerBlock, 0, stream>>>(params);
    }
  }
}

template <typename Context, typename T>
void GroupNormNHWCKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& scale,
    const paddle::optional<DenseTensor>& bias,
    float epsilon,
    int groups,
    const std::string& data_layout_str,
    DenseTensor* y) {
  GroupNormNHWCParams<T> params_;
  params_.withSilu = false;

  const auto x_dims = x.dims();
  dev_ctx.template Alloc<T>(y);
  const T* x_data = x.data<T>();
  T* y_data = y->data<T>();
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const T* scale_data = nullptr;
  if (scale_ptr) scale_data = scale_ptr->data<T>();
  const T* bias_data = nullptr;
  if (bias_ptr) bias_data = bias_ptr->data<T>();
  params_.n = x_dims[0];
  params_.c = x_dims[3];
  params_.h = x_dims[1];
  params_.w = x_dims[2];
    
  int32_t cPerBlock = 320;
  int32_t maxBlocksPerHW = 1024;
  switch (params_.c) {
      case 2048:
      case 1024:
          cPerBlock = 512;
          break;
      case 960:
      case 1920:
          cPerBlock = 480;
          break;
      case 512:
      case 256:
          cPerBlock = 256;
          break;
      case 128:
          cPerBlock = 128;
          break;
      default:
          cPerBlock = 320;
  }
  params_.groups = groups;
  params_.cPerGroup = params_.c / params_.groups;
  if (cPerBlock % params_.cPerGroup != 0) {
    cPerBlock = params_.cPerGroup;
  }
  params_.srcX = (T const *)(x_data);
  params_.dst = (T *)(y_data);
    
  params_.gamma = scale_data;
  params_.beta = bias_data;
  params_.hw = params_.h * params_.w;
  const int32_t blocksPerHW = findMaxDivisor(params_.hw, maxBlocksPerHW);
  params_.hwPerBlock = divUp(params_.hw, blocksPerHW);
  params_.cPerBlock = cPerBlock;
  params_.hwc = params_.hw * params_.c;
  params_.invHWC = 1.F / static_cast<float>(params_.hw * params_.cPerGroup);
  params_.groupsPerBlock = cPerBlock / params_.cPerGroup;
  params_.eps = epsilon;
  auto stream = dev_ctx.stream();
  int Bytes = 2 * sizeof(float) * params_.n * groups;
  cudaMalloc((void **)&params_.redBuffer, Bytes);
  cudaMemsetAsync(params_.redBuffer, 0, Bytes, stream);
  groupNormNHWCSum<T>(params_, stream);
  groupNormNHWCScale<T>(params_, stream);
  cudaFree(params_.redBuffer);
}


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
                                 AccT epsilon,
                                 T* y,
                                 AccT* real_var,
                                 const DataLayout data_layout) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int H = imsize / W;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  auto ng = bid * groups + gid;
  AccT x_mean = mean[ng];
  AccT x_var = var[ng];
  x_var = x_var - x_mean * x_mean;

  AccT var_inv = rsqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) {
    real_var[ng] = x_var;
  }
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    AccT val;
    int hid, wid;
    int index = (bid * C + ccid) * imsize + imid;
    if (data_layout == DataLayout::kNCHW) {
      val = static_cast<AccT>(x[index]);
    } else {
      hid = imid / W;
      wid = imid % W;
      val = static_cast<AccT>(x[(bid * H + hid) * W * C + wid * C + ccid]);
    }
    val = (val - x_mean) * var_inv;
    if (flags & kHasScale) {
      val *= static_cast<AccT>(scale[ccid]);
    }
    if (flags & kHasBias) {
      val += static_cast<AccT>(bias[ccid]);
    }
    if (data_layout == DataLayout::kNCHW) {
      y[index] = static_cast<T>(val);
    } else {
      y[(bid * H + hid) * W * C + wid * C + ccid] = static_cast<T>(val);
    }
  }
}

template <typename T, typename AccT>
void GroupNormDirectCUDAFunctor<T, AccT>::operator()(
    gpuStream_t stream,
    const T* input,
    std::vector<int> input_shape,
    const T* bias,
    const T* scale,
    AccT* temp_variance,
    int groups,
    float eps,
    T* output,
    AccT* mean,
    AccT* variance,
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
    constexpr int vec_size = sizeof(float4) / sizeof(T);
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
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    } else {
      phi::VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, stream>>>(input, mean, temp_variance, size);
    }
  } else {
#ifdef PADDLE_WITH_HIP
    hipMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    hipMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#else
    cudaMemset(mean, 0, sizeof(AccT) * input_ddim[0] * groups);
    cudaMemset(temp_variance, 0, sizeof(AccT) * input_ddim[0] * groups);
#endif

    phi::GroupNormForwardGetMeanAndVar<T, AccT>
        <<<grid, threads, 0, stream>>>(input,
                                       input_ddim[0],
                                       C,
                                       W,
                                       image_size,
                                       groups,
                                       group_size,
                                       mean,
                                       temp_variance);
  }
  GroupNormForward<T, AccT, 3>
      <<<grid, threads, 0, stream>>>(input,
                                     mean,
                                     temp_variance,
                                     scale,
                                     bias,
                                     input_ddim[0],
                                     C,
                                     W,
                                     image_size,
                                     groups,
                                     group_size,
                                     static_cast<AccT>(eps),
                                     output,
                                     variance,
                                     data_layout);
}
template class GroupNormDirectCUDAFunctor<float, float>;
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
template class GroupNormDirectCUDAFunctor<half, float>;
#endif

template <typename Context, typename T>
void GroupNormGeneralCaseKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& scale,
                     const paddle::optional<DenseTensor>& bias,
                     float epsilon,
                     int groups,
                     const std::string& data_layout_str,
                     DenseTensor* y,
                     DenseTensor* mean,
                     DenseTensor* var) {
  using AccT = typename phi::dtype::MPTypeTrait<T>::Type;
  const DataLayout data_layout = phi::StringToDataLayout(data_layout_str);
  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();
  const auto x_dims = x.dims();
  const int C = (data_layout == DataLayout::kNCHW ? x_dims[1] : x_dims[x_dims.size() - 1]);
  const int group_size = C / groups;
  const int W = (data_layout == DataLayout::kNCHW ? x_dims[x_dims.size() - 1]
                                                      : x_dims[x_dims.size() - 2]);

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<AccT>(mean);
  dev_ctx.template Alloc<AccT>(var);
  // temp_var is used to calculate the mean^2
  DenseTensor temp_var;
  temp_var.Resize(var->dims());
  dev_ctx.template Alloc<AccT>(&temp_var);
  phi::funcs::SetConstant<GPUContext, T> set_zero;
  phi::funcs::SetConstant<GPUContext, AccT> set_zero_AccT;
  auto* x_data = x.data<T>();
  auto* y_data = y->data<T>();
  auto* mean_data = mean->data<AccT>();
  auto* var_data = var->data<AccT>();
  auto* temp_var_data = temp_var.data<AccT>();

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

  dim3 grid(group_size, groups, x_dims[0]);
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
    dim3 grids(x_dims[0] * groups);
    dim3 blocks(block_size_nchw);
    if (size < vec_size * block_size_nchw) {
      ScalarGetMeanAndVarNCHW<T, AccT><<<grids, blocks, 0, dev_ctx.stream()>>>(
          x_data, mean_data, temp_var_data, size);
    } else {
      VectorizedGetMeanAndVarNCHW<T, AccT, vec_size>
          <<<grids, blocks, 0, dev_ctx.stream()>>>(
              x_data, mean_data, temp_var_data, size);
    }
  } else {
    set_zero_AccT(dev_ctx, mean, static_cast<AccT>(0));
    set_zero_AccT(dev_ctx, &temp_var, static_cast<AccT>(0));
    GroupNormForwardGetMeanAndVar<T, AccT><<<grid, threads, 0, dev_ctx.stream()>>>(x_data,
                                                    x_dims[0],
                                                    C,
                                                    W,
                                                    imsize,
                                                    groups,
                                                    group_size,
                                                    mean_data,
                                                    temp_var_data);
    }
    int flags =
        (scale_data != nullptr) * kHasScale + (bias_data != nullptr) * kHasBias;
    UNROLL_ALL_CASES(flags,
                    GroupNormForward,
                    x_data,
                    mean_data,
                    temp_var_data,
                    scale_data,
                    bias_data,
                    x_dims[0],
                    C,
                    W,
                    imsize,
                    groups,
                    group_size,
                    static_cast<AccT>(epsilon),
                    y_data,
                    var_data,
                    data_layout);
}

template <typename Context, typename T>
class GroupNormCustomKernel {
  public:
    GroupNormCustomKernel() = default;
    void operator()(
        const Context& dev_ctx,
        const DenseTensor& x,
        const paddle::optional<DenseTensor>& scale,
        const paddle::optional<DenseTensor>& bias,
        float epsilon,
        int groups,
        const std::string& data_layout_str,
        DenseTensor* y,
        DenseTensor* mean,
        DenseTensor* var) {
      printf(" ################################## 222222222 \n");
      GroupNormGeneralCaseKernel<Context, T>(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y, mean, var);
    }
};

template <typename Context>
class GroupNormCustomKernel<Context, phi::dtype::bfloat16> {
  public:
    GroupNormCustomKernel() = default;
    void operator()(
        const Context& dev_ctx,
        const DenseTensor& x,
        const paddle::optional<DenseTensor>& scale,
        const paddle::optional<DenseTensor>& bias,
        float epsilon,
        int groups,
        const std::string& data_layout_str,
        DenseTensor* y,
        DenseTensor* mean,
        DenseTensor* var) {
      using T = phi::dtype::bfloat16;
      printf(" ################################## 3333333333 \n");
      std::cout << data_layout_str << "\n";
      const auto x_dims = x.dims();
      if (data_layout_str == "NHWC") {
        GroupNormNHWCKernel<Context, T>(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y);
      } else {
        GroupNormGeneralCaseKernel<Context, T>(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y, mean, var);
      }
    }
};

template <typename Context>
class GroupNormCustomKernel<Context, phi::dtype::float16> {
  public:
    GroupNormCustomKernel() = default;
    void operator()(
        const Context& dev_ctx,
        const DenseTensor& x,
        const paddle::optional<DenseTensor>& scale,
        const paddle::optional<DenseTensor>& bias,
        float epsilon,
        int groups,
        const std::string& data_layout_str,
        DenseTensor* y,
        DenseTensor* mean,
        DenseTensor* var) {
      using T = phi::dtype::float16;
      const auto x_dims = x.dims();
      if (data_layout_str == "NHWC") {
        GroupNormNHWCKernel<Context, T>(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y);
      } else {
        GroupNormGeneralCaseKernel<Context, T>(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y, mean, var);
      }
    }
};


template <typename T, typename Context>
void GroupNormKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& scale,
    const paddle::optional<DenseTensor>& bias,
    float epsilon,
    int groups,
    const std::string& data_layout_str,
    DenseTensor* y,
    DenseTensor* mean,
    DenseTensor* var) {
  GroupNormCustomKernel<Context, T>()(dev_ctx, x, scale, bias, epsilon, groups, data_layout_str, y, mean, var);
}

}  // namespace phi

PD_REGISTER_KERNEL(group_norm,
                   GPU,
                   ALL_LAYOUT,
                   phi::GroupNormKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::BFLOAT16 ||
      kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}