// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "glog/logging.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/load_store_util.h"
#include "paddle/phi/kernels/gpu/gelu_funcs.h"

PHI_DECLARE_bool(use_fast_math);

namespace phi {
namespace fusion {

using phi::funcs::CudaSwishFunctor;
using phi::funcs::DequantLoad;
using phi::funcs::Load;
using phi::funcs::QuantStore;
using phi::funcs::Store;

template <typename T>
using CudnnDataType = phi::backends::gpu::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

// TODO(lzc): transfer to phi::funcs
template <typename T>
struct GeluFunctor {
  inline __host__ __device__ T operator()(const T x) const {
    using U = LayerNormParamType<T>;
    const U casted_x = static_cast<U>(x);
    const U temp = erf(casted_x * static_cast<U>(M_SQRT1_2));
    const U out = (casted_x * static_cast<U>(0.5) * (static_cast<U>(1) + temp));
    return static_cast<T>(out);
  }
};

template <typename T>
struct FastGeluFunctor {
  inline __device__ T operator()(const T x) const {
    return phi::GeluFwd<T, true>(x);
  }
};

inline cudaError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMultiProcessors(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return cudaSuccess;
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void ActFFNGlu(const T *bias,
                          Functor act_functor,
                          const int token_num,
                          const int hid_dim,
                          const int elem_num,
                          LoadFunc load_func,
                          StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec1;
  LoadT src_vec2;
  LoadT bias_vec1;
  LoadT bias_vec2;
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int bi = i / hid_dim;
    int idx = i % hid_dim;

    load_func.template load<VecSize>(&src_vec1, bi * hid_dim * 2 + idx);
    load_func.template load<VecSize>(&src_vec2,
                                     bi * hid_dim * 2 + idx + hid_dim);

    if (bias) {
      phi::Load<T, VecSize>(&bias[idx], &bias_vec1);
      phi::Load<T, VecSize>(&bias[idx + hid_dim], &bias_vec2);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec1[j] += bias_vec1[j];
        src_vec2[j] += bias_vec2[j];
      }
      src_vec1[j] = act_functor(src_vec1[j]);
      src_vec1[j] *= src_vec2[j];
    }
    // phi::Store<T, VecSize>(src_vec1, &output_this_thread[idx]);
    store_func.template store<VecSize>(src_vec1, bi * hid_dim + idx);
  }
}

template <typename T,
          typename Context,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchActFFNGlu(const Context &dev_ctx,
                     const T *bias,
                     const int token_num,
                     const int hid_dim,
                     LoadFunc load_func,
                     StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      ActFFNGlu<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      ActFFNGlu<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void BiasAct(const T *bias,
                        Functor act_functor,
                        const int rows,
                        const int cols,
                        const int elem_num,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

// Zero Initialize BiasVec.
#pragma unroll
  for (int unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
    bias_vec[unroll_idx] = 0;
  }

  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int row_idx = i / cols;
    int col_idx = i % cols;
    int linear_idx = row_idx * cols + col_idx;
    load_func.template load<VecSize>(&src_vec, linear_idx);
    if (bias) {
      phi::Load<T, VecSize>(&bias[col_idx], &bias_vec);
    }
#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if (bias) {
        src_vec[j] += bias_vec[j];
      }
      src_vec[j] = act_functor(src_vec[j]);
    }
    store_func.template store<VecSize>(src_vec, linear_idx);
  }
}

template <typename T,
          typename Context,
          typename Functor,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void LaunchBiasAct(const Context &dev_ctx,
                   const T *bias,
                   const int token_num,
                   const int hid_dim,
                   LoadFunc load_func,
                   StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int elem_cnt = token_num * hid_dim;
  const int blocksize = 128;
  int grid_size = 1;
  Functor functor;
  switch (hid_dim % PackSize) {
    case 0:
      GetNumBlocks(elem_cnt / PackSize, &grid_size);
      BiasAct<T, Functor, PackSize>
          <<<grid_size, blocksize, 0, dev_ctx.stream()>>>(bias,
                                                          functor,
                                                          token_num,
                                                          hid_dim,
                                                          elem_cnt,
                                                          load_func,
                                                          store_func);
      break;
    default:
      GetNumBlocks(elem_cnt, &grid_size);
      BiasAct<T, Functor, 1><<<grid_size, blocksize, 0, dev_ctx.stream()>>>(
          bias, functor, token_num, hid_dim, elem_cnt, load_func, store_func);
      break;
  }
}

template <typename T,
          typename Context,
          typename LoadFunc,
          typename StoreFunc,
          typename LoadT = T>
void ComputeImpl(const Context &dev_ctx,
                 const T *bias_data,
                 const std::string &act_method,
                 int rows,
                 int cols,
                 LoadFunc load_func,
                 StoreFunc store_func) {
  if (act_method == "geglu") {
    // Note(Zhengzekang): For GLU structure, we need divide the cols by 2.
    VLOG(8) << "Doing geglu";
    LaunchActFFNGlu<T, Context, GeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
        dev_ctx, bias_data, rows, cols / 2, load_func, store_func);
  } else if (act_method == "swiglu") {
    VLOG(8) << "Doing swiglu";
    LaunchActFFNGlu<T,
                    Context,
                    CudaSwishFunctor<T>,
                    LoadFunc,
                    StoreFunc,
                    LoadT>(
        dev_ctx, bias_data, rows, cols / 2, load_func, store_func);
  } else if (act_method == "gelu") {
    if (FLAGS_use_fast_math) {
      VLOG(8) << "Doing Fast GELU";
      LaunchBiasAct<T, Context, FastGeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
          dev_ctx, bias_data, rows, cols, load_func, store_func);
    } else {
      VLOG(8) << "Doing GELU";
      LaunchBiasAct<T, Context, GeluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
          dev_ctx, bias_data, rows, cols, load_func, store_func);
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Currently Only Support GeGLU, SwiGLU, GeLU"));
  }
}

template <typename T, typename Context>
void DispatchComputeImpl(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor *bias,
                         const DenseTensor *dequant_scales,
                         const std::string &act_method,
                         int rows,
                         int cols,
                         const float quant_scale,
                         const int quant_round_type,
                         const float quant_max_bound,
                         const float quant_min_bound,
                         DenseTensor *out) {
  const T *bias_data = bias == nullptr ? nullptr : bias->data<T>();
  if (dequant_scales != nullptr && quant_scale > 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    QuantStore<T> store_func(out->data<int8_t>(),
                             quant_round_type,
                             quant_scale,
                             quant_max_bound,
                             quant_min_bound);
    ComputeImpl<T, Context, DequantLoad<T>, QuantStore<T>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales == nullptr && quant_scale > 0) {
    Load<T> load_func(x.data<T>());
    QuantStore<T> store_func(out->data<int8_t>(),
                             quant_round_type,
                             quant_scale,
                             quant_max_bound,
                             quant_min_bound);
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales != nullptr && quant_scale <= 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    Store<T> store_func(out->data<T>());
    ComputeImpl<T, Context, DequantLoad<T>, Store<T>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else {
    Load<T> load_func(x.data<T>());
    Store<T> store_func(out->data<T>());
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  }
}

template <typename T, typename Context>
void DispatchComputeImpl(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor *bias,
                         const DenseTensor *dequant_scales,
                         const DenseTensor *shift,
                         const DenseTensor *smooth,
                         const std::string &act_method,
                         int rows,
                         int cols,
                         const float quant_scale,
                         const int quant_round_type,
                         const float quant_max_bound,
                         const float quant_min_bound,
                         DenseTensor *out) {
  bool use_glu = (act_method == "geglu" || act_method == "swiglu");
  const T *bias_data = bias == nullptr ? nullptr : bias->data<T>();
  if (dequant_scales != nullptr && quant_scale > 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    QuantStore<T, true> store_func(out->data<int8_t>(),
                                   shift->data<T>(),
                                   smooth->data<T>(),
                                   use_glu ? cols / 2 : cols,
                                   quant_round_type,
                                   quant_scale,
                                   quant_max_bound,
                                   quant_min_bound);
    ComputeImpl<T, Context, DequantLoad<T>, QuantStore<T, true>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales == nullptr && quant_scale > 0) {
    Load<T> load_func(x.data<T>());
    QuantStore<T, true> store_func(out->data<int8_t>(),
                                   shift->data<T>(),
                                   smooth->data<T>(),
                                   use_glu ? cols / 2 : cols,
                                   quant_round_type,
                                   quant_scale,
                                   quant_max_bound,
                                   quant_min_bound);
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales != nullptr && quant_scale <= 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    Store<T, true> store_func(out->data<T>(),
                              shift->data<T>(),
                              smooth->data<T>(),
                              use_glu ? cols / 2 : cols);
    ComputeImpl<T, Context, DequantLoad<T>, Store<T, true>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else {
    Load<T> load_func(x.data<T>());
    Store<T, true> store_func(out->data<T>(),
                              shift->data<T>(),
                              smooth->data<T>(),
                              use_glu ? cols / 2 : cols);
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  }
}

struct NormalVersion {};
struct UnusedVersion {};

template <typename T>
struct DispatchDtypeTrait {
  using FuncVersion = NormalVersion;
};

template <>
struct DispatchDtypeTrait<int32_t> {
  using FuncVersion = UnusedVersion;
};

template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &dequant_scales,
                       const paddle::optional<DenseTensor> &shift,
                       const paddle::optional<DenseTensor> &smooth,
                       const std::string &act_method,
                       int rows,
                       int cols,
                       float quant_scale,
                       int quant_round_type,
                       float quant_max_bound,
                       float quant_min_bound,
                       DenseTensor *out,
                       NormalVersion) {
  auto *bias_p = bias.get_ptr();
  auto *dequant_scales_p = dequant_scales.get_ptr();
  auto *shift_p = shift.get_ptr();
  auto *smooth_p = smooth.get_ptr();
  if (dequant_scales_p != nullptr) {
    if (shift_p != nullptr) {
      DispatchComputeImpl<T>(dev_ctx,
                             x,
                             bias_p,
                             dequant_scales_p,
                             shift_p,
                             smooth_p,
                             act_method,
                             rows,
                             cols,
                             quant_scale,
                             quant_round_type,
                             quant_max_bound,
                             quant_min_bound,
                             out);
    } else {
      DispatchComputeImpl<T>(dev_ctx,
                             x,
                             bias_p,
                             dequant_scales_p,
                             act_method,
                             rows,
                             cols,
                             quant_scale,
                             quant_round_type,
                             quant_max_bound,
                             quant_min_bound,
                             out);
    }
  } else {
    const T *bias_data = bias_p == nullptr ? nullptr : bias_p->data<T>();
    Load<T> load_func(x.data<T>());
    Store<T> store_func(out->data<T>());
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  }
}

// (not use) only for registering int32_t
template <typename T, typename Context>
void DispatchWithDtype(const Context &dev_ctx,
                       const DenseTensor &x,
                       const paddle::optional<DenseTensor> &bias,
                       const paddle::optional<DenseTensor> &dequant_scales,
                       const paddle::optional<DenseTensor> &shift,
                       const paddle::optional<DenseTensor> &smooth,
                       const std::string &act_method,
                       int rows,
                       int cols,
                       float quant_scale,
                       int quant_round_type,
                       float quant_max_bound,
                       float quant_min_bound,
                       DenseTensor *out,
                       UnusedVersion) {}

template <typename T, typename Context>
void FusedBiasActKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const paddle::optional<DenseTensor> &bias,
                        const paddle::optional<DenseTensor> &dequant_scales,
                        const paddle::optional<DenseTensor> &shift,
                        const paddle::optional<DenseTensor> &smooth,
                        const std::string &act_method,
                        const std::string &compute_dtype,
                        int rows,
                        int cols,
                        float quant_scale,
                        int quant_round_type,
                        float quant_max_bound,
                        float quant_min_bound,
                        DenseTensor *out) {
  if (x.dtype() == phi::DataType::INT32) {
    if (compute_dtype == "bf16") {
      DispatchWithDtype<phi::dtype::bfloat16, Context>(
          dev_ctx,
          x,
          bias,
          dequant_scales,
          shift,
          smooth,
          act_method,
          rows,
          cols,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound,
          out,
          typename DispatchDtypeTrait<phi::dtype::bfloat16>::FuncVersion{});

    } else if (compute_dtype == "fp16") {
      DispatchWithDtype<phi::dtype::float16, Context>(
          dev_ctx,
          x,
          bias,
          dequant_scales,
          shift,
          smooth,
          act_method,
          rows,
          cols,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound,
          out,
          typename DispatchDtypeTrait<phi::dtype::float16>::FuncVersion{});
    } else if (compute_dtype == "fp32") {
      DispatchWithDtype<float, Context>(
          dev_ctx,
          x,
          bias,
          dequant_scales,
          shift,
          smooth,
          act_method,
          rows,
          cols,
          quant_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound,
          out,
          typename DispatchDtypeTrait<phi::dtype::float32>::FuncVersion{});

    } else {
      PADDLE_THROW("Only bf16, fp16 and fp32 are supported. ");
    }
  } else {
    DispatchWithDtype<T, Context>(
        dev_ctx,
        x,
        bias,
        dequant_scales,
        shift,
        smooth,
        act_method,
        rows,
        cols,
        quant_scale,
        quant_round_type,
        quant_max_bound,
        quant_min_bound,
        out,
        typename DispatchDtypeTrait<T>::FuncVersion{});
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_act,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasActKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int32_t) {}
