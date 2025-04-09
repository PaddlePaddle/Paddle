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

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/phi/kernels/fusion/gpu/fused_bias_act_utils.h"

COMMON_DECLARE_bool(use_fast_math);

namespace phi {
namespace fusion {

template <typename T,
          typename Functor,
          int VecSize,
          typename LoadFunc,
          typename StoreFunc>
__global__ void ActFFNGlu(const T *bias,
                          Functor act_functor,
                          const int64_t token_num,
                          const int64_t hid_dim,
                          const int64_t elem_num,
                          LoadFunc load_func,
                          StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec1;
  LoadT src_vec2;
  LoadT bias_vec1;
  LoadT bias_vec2;
  const int64_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int64_t bi = i / hid_dim;
    int64_t idx = i % hid_dim;

    int64_t index = bi * hid_dim * 2 + idx;
    load_func.template load<VecSize>(&src_vec1, index);
    load_func.template load<VecSize>(&src_vec2, index + hid_dim);

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
                     const int64_t token_num,
                     const int64_t hid_dim,
                     LoadFunc load_func,
                     StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int64_t elem_cnt = token_num * hid_dim;
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
                        const int64_t rows,
                        const int64_t cols,
                        const int64_t elem_num,
                        LoadFunc load_func,
                        StoreFunc store_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;

// Zero Initialize BiasVec.
#pragma unroll
  for (int64_t unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
    bias_vec[unroll_idx] = 0;
  }

  const int64_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t i = global_tid * VecSize; i < elem_num;
       i += gridDim.x * blockDim.x * VecSize) {
    int64_t row_idx = i / cols;
    int64_t col_idx = i % cols;
    int64_t linear_idx = row_idx * cols + col_idx;
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
                   const int64_t token_num,
                   const int64_t hid_dim,
                   LoadFunc load_func,
                   StoreFunc store_func) {
  constexpr int VecSize = 16;
  constexpr int PackSize = VecSize / sizeof(LoadT);
  const int64_t elem_cnt = token_num * hid_dim;
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
                 int64_t rows,
                 int64_t cols,
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
  } else if (act_method == "relu") {
    VLOG(8) << "Doing RELU";
    // for opt model
    LaunchBiasAct<T, Context, ReluFunctor<T>, LoadFunc, StoreFunc, LoadT>(
        dev_ctx, bias_data, rows, cols, load_func, store_func);
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently Only Support GeGLU, SwiGLU, GeLU"));
  }
}

template <typename T, typename OutT, typename Context>
void DispatchComputeImpl(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor *bias,
                         const std::string &act_method,
                         int rows,
                         int cols,
                         const float quant_scale,
                         const int quant_round_type,
                         const float quant_max_bound,
                         const float quant_min_bound,
                         DenseTensor *out) {
  const T *bias_data = bias == nullptr ? nullptr : bias->data<T>();
  Load<T> load_func(x.data<T>());
  QuantStore<T, OutT> store_func(dev_ctx.template Alloc<OutT>(out),
                                 quant_round_type,
                                 quant_scale,
                                 quant_max_bound,
                                 quant_min_bound);
  ComputeImpl<T>(
      dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
}

template <typename T, typename Context>
void DispatchComputeImpl(const Context &dev_ctx,
                         const DenseTensor &x,
                         const DenseTensor *bias,
                         const DenseTensor *dequant_scales,
                         const std::string &act_method,
                         int64_t rows,
                         int64_t cols,
                         const float quant_scale,
                         const int quant_round_type,
                         const float quant_max_bound,
                         const float quant_min_bound,
                         DenseTensor *out) {
  const T *bias_data = bias == nullptr ? nullptr : bias->data<T>();
  if (dequant_scales != nullptr && quant_scale > 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    QuantStore<T, int8_t> store_func(dev_ctx.template Alloc<int8_t>(out),
                                     quant_round_type,
                                     quant_scale,
                                     quant_max_bound,
                                     quant_min_bound);
    ComputeImpl<T, Context, DequantLoad<T>, QuantStore<T, int8_t>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales == nullptr && quant_scale > 0) {
    Load<T> load_func(x.data<T>());
    QuantStore<T, int8_t> store_func(dev_ctx.template Alloc<int8_t>(out),
                                     quant_round_type,
                                     quant_scale,
                                     quant_max_bound,
                                     quant_min_bound);
    ComputeImpl<T>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales != nullptr && quant_scale <= 0) {
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    Store<T> store_func(dev_ctx.template Alloc<T>(out));
    ComputeImpl<T, Context, DequantLoad<T>, Store<T>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else {
    Load<T> load_func(x.data<T>());
    Store<T> store_func(dev_ctx.template Alloc<T>(out));
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
                         int64_t rows,
                         int64_t cols,
                         const float quant_scale,
                         const int quant_round_type,
                         const float quant_max_bound,
                         const float quant_min_bound,
                         DenseTensor *out) {
  bool use_glu = (act_method == "geglu" || act_method == "swiglu");
  const T *bias_data = bias == nullptr ? nullptr : bias->data<T>();

  if (dequant_scales != nullptr && quant_scale > 0) {
    int8_t *out_data = dev_ctx.template Alloc<int8_t>(out);
    DequantLoad<T> load_func(
        x.data<int32_t>(), dequant_scales->data<float>(), cols);
    QuantStore<T, int8_t, true> store_func(dev_ctx.template Alloc<int8_t>(out),
                                           shift->data<T>(),
                                           smooth->data<T>(),
                                           use_glu ? cols / 2 : cols,
                                           quant_round_type,
                                           quant_scale,
                                           quant_max_bound,
                                           quant_min_bound);
    ComputeImpl<T,
                Context,
                DequantLoad<T>,
                QuantStore<T, int8_t, true>,
                int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else if (dequant_scales == nullptr && quant_scale > 0) {
    Load<T> load_func(x.data<T>());
    QuantStore<T, int8_t, true> store_func(dev_ctx.template Alloc<int8_t>(out),
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
    Store<T, true> store_func(dev_ctx.template Alloc<T>(out),
                              shift->data<T>(),
                              smooth->data<T>(),
                              use_glu ? cols / 2 : cols);
    ComputeImpl<T, Context, DequantLoad<T>, Store<T, true>, int32_t>(
        dev_ctx, bias_data, act_method, rows, cols, load_func, store_func);
  } else {
    Load<T> load_func(x.data<T>());
    Store<T, true> store_func(dev_ctx.template Alloc<T>(out),
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
                       int64_t rows,
                       int64_t cols,
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
    if (out->dtype() == phi::DataType::FLOAT8_E4M3FN) {
      DispatchComputeImpl<T, phi::dtype::float8_e4m3fn>(dev_ctx,
                                                        x,
                                                        bias_p,
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
                       int64_t rows,
                       int64_t cols,
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
                        float quant_scale,
                        int quant_round_type,
                        float quant_max_bound,
                        float quant_min_bound,
                        DenseTensor *out) {
  int64_t cols = x.dims()[x.dims().size() - 1];
  int64_t rows = x.numel() / cols;
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
          typename DispatchDtypeTrait<float>::FuncVersion{});
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "In the case of quantization enabled with Input(x) INT32, "
          "Attr(compute_dtype) must be set in (bf16, fp16, fp32), "
          "but get compute_dtype (%s)",
          compute_dtype));
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
