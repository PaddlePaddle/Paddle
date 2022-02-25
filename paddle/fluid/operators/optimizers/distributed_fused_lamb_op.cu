// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include "paddle/fluid/memory/buffer.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/cast_with_ptr.h"
#include "paddle/fluid/operators/optimizers/distributed_fused_lamb_op.h"
#include "paddle/fluid/operators/optimizers/multi_tensor_apply.h"
#include "paddle/fluid/operators/tensor_to_string.h"
#include "paddle/fluid/platform/aligned_vector.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/string/string_helper.h"
#include "paddle/phi/core/utils/data_type.h"

#ifdef __NVCC__
#include "cub/cub.cuh"
#include "math.h"  // NOLINT
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#include "math.h"  // NOLINT
namespace cub = hipcub;
#endif

namespace paddle {
namespace operators {

template <typename T>
using MasterT = typename details::MPTypeTrait<T>::Type;

template <typename T>
static void FillZeroWithPtr(T *x, size_t n, gpuStream_t stream) {
  static_assert(!std::is_same<T, void>::value, "T cannot be void.");
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemsetAsync(x, 0, n * sizeof(T), stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(x, 0, n * sizeof(T), stream));
#endif
}

template <typename T, int BlockDim, int VecSize>
struct L2NormFunctor {
  DEVICE void operator()(int tensor_id, int chunk_id, int offset, int size,
                         const T *x, MasterT<T> *y, int max_chunk_num) const {
    using MT = MasterT<T>;
    const T *ptr = x + offset;

    using BlockReduce = cub::BlockReduce<MT, BlockDim>;
    __shared__ typename BlockReduce::TempStorage storage;

    MT square_sum = static_cast<MT>(0);
    int i;
    for (i = threadIdx.x * VecSize; i + VecSize <= size;
         i += (BlockDim * VecSize)) {
      platform::AlignedVector<T, VecSize> tmp_vec;
      platform::Load(ptr + i, &tmp_vec);
#pragma unroll
      for (int j = 0; j < VecSize; ++j) {
        auto tmp = static_cast<MT>(tmp_vec[j]);
        square_sum += (tmp * tmp);
      }
    }

    for (; i < size; ++i) {
      auto tmp = static_cast<MT>(ptr[i]);
      square_sum += (tmp * tmp);
    }

    square_sum = BlockReduce(storage).Reduce(square_sum, cub::Sum());
    if (threadIdx.x == 0) {
      y[tensor_id * max_chunk_num + chunk_id] = square_sum;
    }
  }
};

template <typename InT, typename OutT, int BlockDim, bool NeedSqrt>
static __global__ void MultiTensorL2NormReduceAgainCUDAKernel(
    const InT *x, OutT *y, int max_chunk_num) {
  int tensor_id = blockIdx.x;
  x += (tensor_id * max_chunk_num);
  using BlockReduce = cub::BlockReduce<InT, BlockDim>;
  __shared__ typename BlockReduce::TempStorage storage;
  InT sum = static_cast<InT>(0);
  for (int i = threadIdx.x; i < max_chunk_num; i += BlockDim) {
    sum += x[i];
  }
  sum = BlockReduce(storage).Reduce(sum, cub::Sum());
  if (threadIdx.x == 0) {
    if (NeedSqrt) {
      y[blockIdx.x] = static_cast<OutT>(sqrtf(sum));
    } else {
      y[blockIdx.x] = static_cast<OutT>(sum);
    }
  }
}

template <typename T>
static int GetChunkedVecSize(const T *ptr, int chunk_size) {
  static_assert(!std::is_same<T, void>::value, "T cannot be void.");

  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  auto address = reinterpret_cast<uintptr_t>(ptr);
  constexpr int vec8 = alignof(platform::AlignedVector<T, 8>);
  constexpr int vec4 = alignof(platform::AlignedVector<T, 4>);
  constexpr int vec2 = alignof(platform::AlignedVector<T, 2>);
  if (address % vec8 == 0 && chunk_size % vec8 == 0) {
    return std::min(8, valid_vec_size);
  } else if (address % vec4 == 0 && chunk_size % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0 && chunk_size % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

#define PD_VEC_MULTI_TENSOR_APPLY_CASE(__vec_size, ...) \
  case __vec_size: {                                    \
    constexpr int kVecSize = __vec_size;                \
    __VA_ARGS__;                                        \
    break;                                              \
  }

#define PD_VEC_MULTI_TENSOR_APPLY(__vec_size, ...)    \
  do {                                                \
    switch (__vec_size) {                             \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(8, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(4, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(2, __VA_ARGS__); \
      PD_VEC_MULTI_TENSOR_APPLY_CASE(1, __VA_ARGS__); \
    }                                                 \
  } while (0)

// TODO(zengjinle): which chunk_size is better?
template <typename InT, typename OutT, bool NeedSqrt = false,
          int MaxTensorNumPerLaunch = 50, int MaxChunkNumPerLaunch = 680,
          int BlockDim = 512>
static void MultiTensorL2Norm(const platform::CUDAPlace &place,
                              gpuStream_t stream, const InT *x,
                              const int *offsets, int n, OutT *y,
                              int chunk_size = 65536) {
  if (n <= 0) return;

  constexpr int kNumTensor = MaxTensorNumPerLaunch;
  constexpr int kNumChunk = MaxChunkNumPerLaunch;
  constexpr int kBlockDim = BlockDim;

  int max_chunk_num = -1;
  int vec_size = 8;
  int total_chunk_num = 0;
  for (int i = 0; i < n; ++i) {
    vec_size = std::min(
        vec_size, GetChunkedVecSize(x + offsets[i] - offsets[0], chunk_size));
    int length = offsets[i + 1] - offsets[i];
    auto tmp_chunk_num = (length + chunk_size - 1) / chunk_size;
    max_chunk_num = std::max(max_chunk_num, tmp_chunk_num);
    total_chunk_num += tmp_chunk_num;
  }

  VLOG(1) << "MultiTensorL2Norm max_chunk_num = " << max_chunk_num
          << " , total_chunk_num = " << total_chunk_num
          << " , tensor_num = " << n;

  using MT = MasterT<InT>;
  memory::Buffer tmp_out(place);
  auto *tmp_out_ptr = tmp_out.Alloc<MT>(n * max_chunk_num);
  FillZeroWithPtr(tmp_out_ptr, n * max_chunk_num, stream);

#define PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL                         \
  do {                                                              \
    using FunctorT = L2NormFunctor<InT, kBlockDim, kVecSize>;       \
    VLOG(10) << __func__ << " " << typeid(InT).name()               \
             << " VecSize = " << kVecSize;                          \
    MultiTensorApply<FunctorT, kBlockDim, kNumTensor, kNumChunk>(   \
        FunctorT(), stream, offsets, n, chunk_size, x, tmp_out_ptr, \
        max_chunk_num);                                             \
  } while (0)

  PD_VEC_MULTI_TENSOR_APPLY(vec_size, PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL);
#undef PD_LAUNCH_MULTI_TENSOR_APPLY_KERNEL

  MultiTensorL2NormReduceAgainCUDAKernel<MT, OutT, kBlockDim,
                                         NeedSqrt><<<n, kBlockDim, 0, stream>>>(
      tmp_out_ptr, y, max_chunk_num);
}

template <int LogLevel>
static void LogParamAndTrustRatioDivSquareNorm(
    const framework::ExecutionContext &ctx, const float *param_square_norm,
    const float *trust_ratio_div_square_norm) {
  if (!VLOG_IS_ON(LogLevel)) return;

  auto tensors = ctx.MultiInput<framework::Tensor>("Param");
  if (tensors.empty()) return;

  size_t n = tensors.size();
  auto place = tensors[0]->place();

  auto pn_vec = ToVector(param_square_norm, n, place);
  auto tn_vec = ToVector(trust_ratio_div_square_norm, n, place);

  std::vector<size_t> fp32_indices, fp16_indices;
  fp32_indices.reserve(n);
  fp16_indices.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const auto *t = tensors[i];
    if (t->dtype() == phi::DataType::FLOAT32) {
      fp32_indices.push_back(i);
    } else if (t->dtype() == phi::DataType::FLOAT16) {
      fp16_indices.push_back(i);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported data type %s.", t->dtype()));
    }
  }

  for (auto idx : fp16_indices) {
    fp32_indices.push_back(idx);
  }

  const auto &names = ctx.GetOp().Inputs("Param");
  for (size_t i = 0; i < fp32_indices.size(); ++i) {
    auto idx = fp32_indices[i];
    VLOG(LogLevel) << "Param " << tensors[idx]->dtype() << " " << names[idx]
                   << " pn = " << pn_vec[i] << " , tn = " << tn_vec[i];
  }
}

static bool IsFinite(const platform::CUDADeviceContext &dev_ctx,
                     const float *ptr) {
  auto stream = dev_ctx.stream();
  float cpu_value;
#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(&cpu_value, ptr, sizeof(float),
                                            hipMemcpyDeviceToHost, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&cpu_value, ptr, sizeof(float),
                                             cudaMemcpyDeviceToHost, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
  LOG(INFO) << "NAN_INF indicator value: " << cpu_value;
  return isfinite(cpu_value);
}

template <typename T>
static const T *GetInputTensorPtr(const framework::ExecutionContext &ctx,
                                  const char *in_name,
                                  int64_t *numel = nullptr) {
  const auto *in_tensor = ctx.Input<framework::Tensor>(in_name);
  PADDLE_ENFORCE_NOT_NULL(in_tensor, platform::errors::InvalidArgument(
                                         "Input(%s) cannot be NULL.", in_name));
  if (in_tensor->IsInitialized()) {
    if (numel) *numel = in_tensor->numel();
    return in_tensor->data<T>();
  } else {
    if (numel) *numel = 0;
    return nullptr;
  }
}

template <typename T, bool AllowNotExist = false>
static T *GetSameInOutTensorPtr(const framework::ExecutionContext &ctx,
                                const platform::Place &place,
                                const char *in_name, const char *out_name,
                                int64_t *numel = nullptr) {
  const auto *in_tensor = ctx.Input<framework::Tensor>(in_name);
  if (in_tensor == nullptr || !in_tensor->IsInitialized()) {
    PADDLE_ENFORCE_EQ(AllowNotExist, true,
                      platform::errors::InvalidArgument(
                          "Input(%s) cannot be NULL.", in_name));
    if (numel) *numel = 0;
    return nullptr;
  }

  auto *out_tensor = ctx.Output<framework::Tensor>(out_name);
  PADDLE_ENFORCE_NOT_NULL(in_tensor, platform::errors::InvalidArgument(
                                         "Input(%s) cannot be NULL.", in_name));
  PADDLE_ENFORCE_NOT_NULL(out_tensor,
                          platform::errors::InvalidArgument(
                              "Output(%s) cannot be NULL.", out_name));
  const T *in_data = in_tensor->data<T>();
  T *out_data = out_tensor->mutable_data<T>(place);
  PADDLE_ENFORCE_EQ(in_data, out_data,
                    platform::errors::InvalidArgument(
                        "Input(%s) and Output(%s) must be the same Tensor.",
                        in_name, out_name));
  if (numel) *numel = out_tensor->numel();
  return out_data;
}

template <typename T>
struct SquareFunctor {
  HOSTDEVICE MasterT<T> operator()(T x) const {
    auto y = static_cast<MasterT<T>>(x);
    return y * y;
  }
};

template <typename T>
struct IsNanInfFunctor {
  HOSTDEVICE bool operator()(T x) const { return !isfinite(x); }
};

struct OrFunctor {
  HOSTDEVICE bool operator()(bool x, bool y) const { return x || y; }
};

struct AndFunctor {
  HOSTDEVICE bool operator()(bool x, bool y) const { return x && y; }
};

template <typename T1, typename T2>
static __global__ void ScaleCUDAKernel(const T1 *__restrict__ x,
                                       const T2 *__restrict__ scale,
                                       T1 *__restrict__ y, int num) {
  static_assert(sizeof(T1) <= sizeof(T2),
                "sizeof(T1) must be not greater than sizeof(T2).");
  T2 s = scale[0];
  CUDA_KERNEL_LOOP(i, num) {
    y[i] = static_cast<T1>(static_cast<T2>(x[i]) * s);
  }
}

template <typename T>
static __global__ void AddToCUDAKernel(const T *__restrict__ x,
                                       T *__restrict__ y) {
  y[0] += x[0];
}

// If clip before allreduce,
// coeff = global_scale * max_global_grad_norm / (1e-6 + sqrt(square_grad_norm)
// * rescale_grad)
// if coeff >= 1 or coeff is Nan/Inf, scale = 1.0
// else scale = coeff
template <typename T1, typename T2>
static __global__ void CalcGradNormClipBeforeAllReduceScale(
    const T1 *__restrict__ global_scale, T1 max_global_grad_norm,
    const T1 *__restrict__ square_grad_norm, T1 *__restrict__ out1,
    T2 *__restrict__ out2, T1 clip_rescale_grad) {
  T1 grad_norm = static_cast<T1>(sqrt(*square_grad_norm)) * clip_rescale_grad;
  T1 scale = global_scale[0] * max_global_grad_norm / (1e-6 + grad_norm);
  bool found_nan_inf = !isfinite(scale);
  if (scale >= 1 || found_nan_inf) {
    scale = static_cast<T1>(1.0);
  }

  if (out1) {
    *out1 = scale;
  }
  if (out2) {
    *out2 = static_cast<T2>(scale);
  }
}

static __global__ void SetNanInfValueCUDAKernelOneFlag(const bool *in_flag_p,
                                                       float *out_p) {
  *out_p = (*in_flag_p) ? __int_as_float(0x7fffffffU) : 0.0f;
}

static __global__ void SetNanInfValueCUDAKernelTwoFlag(const bool *in_flag_p_1,
                                                       const bool *in_flag_p_2,
                                                       float *out_p) {
  *out_p =
      ((*in_flag_p_1) || (*in_flag_p_2)) ? __int_as_float(0x7fffffffU) : 0.0f;
}

// TODO(zengjinle): Vectorize this function
// NOTE: this method does not update Beta1Pow and Beta2Pow!
template <typename T, typename GradT, typename IndexT>
static __global__ void UpdateLambMoment(
    const T *__restrict__ param_p, const GradT *__restrict__ grad_p,
    const T *__restrict__ square_grad_norm_p,
    const T *__restrict__ global_scale, const IndexT *__restrict__ indices,
    const T *__restrict__ weight_decay_p, const T *__restrict__ beta1pow_p,
    const T *__restrict__ beta2pow_p, T *__restrict__ mom1_p,
    T *__restrict__ mom2_p, T *__restrict__ trust_ratio_div_p, T beta1, T beta2,
    T epsilon, T max_global_grad_norm, int num, T rescale_grad) {
  T square_grad_norm = *square_grad_norm_p;
  if (!isfinite(square_grad_norm)) return;

  T scale = rescale_grad / global_scale[0];
  if (max_global_grad_norm > 0) {
    T clip_scale =
        max_global_grad_norm / (sqrtf(square_grad_norm) * scale + 1e-6);
    if (clip_scale < static_cast<T>(1)) {
      scale *= clip_scale;
    }
  }

  T one_minus_beta1pow = 1 - beta1pow_p[0];
  T one_minus_beta2pow = 1 - beta2pow_p[0];

  CUDA_KERNEL_LOOP(i, num) {
    T p = param_p[i];
    T g = static_cast<T>(grad_p[i]) * scale;
    T weight_decay = weight_decay_p[i];
    T mom1 = mom1_p[i];
    T mom2 = mom2_p[i];

    mom1 = beta1 * mom1 + (1 - beta1) * g;
    mom2 = beta2 * mom2 + (1 - beta2) * g * g;

    T mom1_unbiased = mom1 / one_minus_beta1pow;
    T mom2_unbiased = mom2 / one_minus_beta2pow;
    T trust_ratio_div =
        mom1_unbiased / (sqrtf(mom2_unbiased) + epsilon) + weight_decay * p;

    mom1_p[i] = mom1;
    mom2_p[i] = mom2;
    trust_ratio_div_p[i] = trust_ratio_div;
  }
}

template <typename T, bool NeedUpdate /*=true*/>
struct LambBetaPowUpdateOnceHelper {
  LambBetaPowUpdateOnceHelper(T *beta1pow, T *beta2pow, T beta1, T beta2) {
    PADDLE_ENFORCE_NOT_NULL(beta1pow,
                            platform::errors::InvalidArgument(
                                "The beta1pow should not be nullptr."));
    PADDLE_ENFORCE_NOT_NULL(beta2pow,
                            platform::errors::InvalidArgument(
                                "The beta2pow should not be nullptr."));
    beta1pow_ = beta1pow;
    beta2pow_ = beta2pow;
    beta1_ = beta1;
    beta2_ = beta2;
  }

  HOSTDEVICE void UpdateBetaPows() const {
    beta1pow_[0] *= beta1_;
    beta2pow_[0] *= beta2_;
  }

 private:
  T *__restrict__ beta1pow_;
  T *__restrict__ beta2pow_;
  T beta1_;
  T beta2_;
};

template <typename T>
struct LambBetaPowUpdateOnceHelper<T, false> {
  LambBetaPowUpdateOnceHelper(T *beta1pow, T *beta2pow, T beta1, T beta2) {
    PADDLE_ENFORCE_EQ(
        beta1pow, nullptr,
        platform::errors::InvalidArgument("The beta1pow should be nullptr."));
    PADDLE_ENFORCE_EQ(
        beta2pow, nullptr,
        platform::errors::InvalidArgument("The beta2pow should be nullptr."));
  }

  HOSTDEVICE void UpdateBetaPows() const {}
};

template <bool HasFoundInf /*=true*/>
struct LambFoundInfHelper {
 public:
  explicit LambFoundInfHelper(bool *found_inf) : found_inf_(found_inf) {
    PADDLE_ENFORCE_NOT_NULL(found_inf,
                            platform::errors::InvalidArgument(
                                "The found_inf should not be nullptr."));
  }

  HOSTDEVICE void UpdateFoundInf(bool value) { *found_inf_ = value; }

 private:
  bool *__restrict__ found_inf_;
};

template <>
struct LambFoundInfHelper<false> {
 public:
  explicit LambFoundInfHelper(bool *found_inf) {
    PADDLE_ENFORCE_EQ(
        found_inf, nullptr,
        platform::errors::InvalidArgument("The found_inf should be nullptr."));
  }

  HOSTDEVICE void UpdateFoundInf(bool) {}
};

template <typename T, bool HasMasterParam /*=true*/>
struct LambParamHelper {
  LambParamHelper(T *param, MasterT<T> *master_param) {
    constexpr bool kIsSameType = std::is_same<T, MasterT<T>>::value;
    PADDLE_ENFORCE_EQ(kIsSameType, false,
                      platform::errors::InvalidArgument(
                          "T must not be the same with MasterT<T>."));
    PADDLE_ENFORCE_NOT_NULL(master_param,
                            platform::errors::InvalidArgument(
                                "Master parameter must be provided."));
    param_ = param;
    master_param_ = master_param;
  }

  HOSTDEVICE void SetParam(int i, MasterT<T> updated_p) {
    param_[i] = static_cast<T>(updated_p);
    master_param_[i] = updated_p;
  }

  HOSTDEVICE MasterT<T> GetParam(int i) { return master_param_[i]; }

 private:
  T *__restrict__ param_;
  MasterT<T> *__restrict__ master_param_;
};

template <typename T>
struct LambParamHelper<T, false> {
  LambParamHelper(T *param, MasterT<T> *master_param) {
    constexpr bool kIsSameType = std::is_same<T, MasterT<T>>::value;
    PADDLE_ENFORCE_EQ(kIsSameType, true,
                      platform::errors::InvalidArgument(
                          "T must be the same with MasterT<T>."));
    if (master_param != nullptr) {
      PADDLE_ENFORCE_EQ(static_cast<void *>(param),
                        static_cast<void *>(master_param),
                        platform::errors::InvalidArgument(
                            "Master parameter must be nullptr or the same as "
                            "non-master parameter."));
    }
    param_ = param;
  }

  HOSTDEVICE void SetParam(int i, MasterT<T> updated_p) {
    param_[i] = static_cast<T>(updated_p);
  }

  HOSTDEVICE MasterT<T> GetParam(int i) {
    return static_cast<MasterT<T>>(param_[i]);
  }

 private:
  T *__restrict__ param_;
};

template <typename ParamT, typename IndexT, bool HasMasterParam,
          bool NeedUpdateBetaPow, bool HasFoundInf>
struct LambParamAndBetaPowsUpdateHelper
    : public LambParamHelper<ParamT, HasMasterParam>,
      public LambBetaPowUpdateOnceHelper<MasterT<ParamT>, NeedUpdateBetaPow>,
      public LambFoundInfHelper<HasFoundInf> {
  LambParamAndBetaPowsUpdateHelper(
      ParamT *param, MasterT<ParamT> *master_param, MasterT<ParamT> *beta1pow,
      MasterT<ParamT> *beta2pow, MasterT<ParamT> beta1, MasterT<ParamT> beta2,
      bool *found_inf, const MasterT<ParamT> *trust_ratio_div,
      const MasterT<ParamT> *lr, const IndexT *index,
      const MasterT<ParamT> *param_square_norm,
      const MasterT<ParamT> *trust_ratio_div_square_norm,
      const MasterT<ParamT> *update_flag)
      : LambParamHelper<ParamT, HasMasterParam>(param, master_param),
        LambBetaPowUpdateOnceHelper<MasterT<ParamT>, NeedUpdateBetaPow>(
            beta1pow, beta2pow, beta1, beta2),
        LambFoundInfHelper<HasFoundInf>(found_inf),
        trust_ratio_div(trust_ratio_div),
        lr(lr),
        index(index),
        param_square_norm(param_square_norm),
        trust_ratio_div_square_norm(trust_ratio_div_square_norm),
        update_flag(update_flag) {}

  const MasterT<ParamT> *__restrict__ trust_ratio_div;
  const MasterT<ParamT> *__restrict__ lr;
  const IndexT *__restrict__ index;
  const MasterT<ParamT> *__restrict__ param_square_norm;
  const MasterT<ParamT> *__restrict__ trust_ratio_div_square_norm;
  const MasterT<ParamT> *__restrict__ update_flag;
};

template <typename ParamT, typename IndexT, bool HasMasterParam,
          bool NeedUpdateBetaPow, bool HasFoundInf>
static __global__ void LambUpdateParamAndBetaPowsCUDAKernel(
    LambParamAndBetaPowsUpdateHelper<ParamT, IndexT, HasMasterParam,
                                     NeedUpdateBetaPow, HasFoundInf>
        args,
    int num) {
  auto should_update = *args.update_flag;
  if (!isfinite(should_update)) {
    if (HasFoundInf && threadIdx.x == 0 && blockIdx.x == 0) {
      args.UpdateFoundInf(true);
    }
    return;
  } else if (HasFoundInf && threadIdx.x == 0 && blockIdx.x == 0) {
    args.UpdateFoundInf(false);
  }

  if (NeedUpdateBetaPow && threadIdx.x == 0 && blockIdx.x == 0) {
    args.UpdateBetaPows();
  }

  using MT = MasterT<ParamT>;

  MT lr_value = *args.lr;
  CUDA_KERNEL_LOOP(i, num) {
    MT p = args.GetParam(i);
    MT t = args.trust_ratio_div[i];
    auto norm_idx = args.index[i];
    MT p_square_norm = args.param_square_norm[norm_idx];
    MT t_square_norm = args.trust_ratio_div_square_norm[norm_idx];

    MT p_norm = static_cast<MT>(sqrtf(p_square_norm));
    MT t_norm = static_cast<MT>(sqrtf(t_square_norm));

    auto update = (p_norm != static_cast<MT>(0) && t_norm != static_cast<MT>(0))
                      ? p_norm / t_norm
                      : static_cast<MT>(1);

    MT updated_p = p - lr_value * update * t;
    args.SetParam(i, updated_p);
  }
}

template <typename ParamT, typename IndexT>
static void LambUpdateParamAndBetaPows(
    const platform::CUDADeviceContext &dev_ctx,
    const MasterT<ParamT> *trust_ratio_div, const MasterT<ParamT> *lr,
    const IndexT *index, const MasterT<ParamT> *param_square_norm,
    const MasterT<ParamT> *trust_ratio_div_square_norm,
    const MasterT<ParamT> *update_flag, MasterT<ParamT> **beta1pow,
    MasterT<ParamT> **beta2pow, bool **found_inf, MasterT<ParamT> beta1,
    MasterT<ParamT> beta2, int num, ParamT *param,
    MasterT<ParamT> *master_param, gpuStream_t stream) {
  if (num == 0) return;

  bool has_master_param = !(std::is_same<ParamT, MasterT<ParamT>>::value);
  auto has_beta_pow = (*beta1pow) != nullptr && (*beta2pow) != nullptr;
  auto has_found_inf = (*found_inf) != nullptr;

#define PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(                              \
    __has_master_param, __has_beta_pow, __has_found_inf)                     \
  do {                                                                       \
    LambParamAndBetaPowsUpdateHelper<ParamT, IndexT, __has_master_param,     \
                                     __has_beta_pow, __has_found_inf>        \
        helper(param, master_param, *beta1pow, *beta2pow, beta1, beta2,      \
               *found_inf, trust_ratio_div, lr, index, param_square_norm,    \
               trust_ratio_div_square_norm, update_flag);                    \
    auto config = platform::GetGpuLaunchConfig1D(dev_ctx, num);              \
    LambUpdateParamAndBetaPowsCUDAKernel<<<                                  \
        config.block_per_grid, config.thread_per_block, 0, stream>>>(helper, \
                                                                     num);   \
  } while (0)

  if (has_master_param) {
    if (has_beta_pow) {
      if (has_found_inf) {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(true, true, true);
      } else {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(true, true, false);
      }
    } else {
      if (has_found_inf) {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(true, false, true);
      } else {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(true, false, false);
      }
    }
  } else {
    if (has_beta_pow) {
      if (has_found_inf) {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(false, true, true);
      } else {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(false, true, false);
      }
    } else {
      if (has_found_inf) {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(false, false, true);
      } else {
        PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL(false, false, false);
      }
    }
  }

  *beta1pow = nullptr;
  *beta2pow = nullptr;
  *found_inf = nullptr;
#undef PADDLE_LAUNCH_LAMB_UPDATE_PARAM_KERNEL
}

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
static bool CreatePreMulScaleOpIfSupported(ncclDataType_t dtype,
                                           ncclComm_t comm, const void *scale,
                                           ncclRedOp_t *op) {
#if NCCL_VERSION_CODE >= 21100
  int ver;
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGetVersion(&ver));
  if (ver >= 21100) {
    VLOG(10) << "ncclRedOpCreatePreMulSum is supported.";
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRedOpCreatePreMulSum(
        op, const_cast<void *>(scale), dtype, ncclScalarDevice, comm));
    return true;
  }
#endif
  VLOG(10) << "ncclRedOpCreatePreMulSum is not supported.";
  return false;
}

template <typename T>
static void NCCLReduceScatterWithScale(
    const T *sendbuff, T *recvbuff, size_t recvcount, size_t nranks,
    ncclComm_t comm, gpuStream_t stream,
    const platform::CUDADeviceContext &dev_ctx, const T *scale = nullptr) {
  static_assert(std::is_same<T, float>::value ||
                    std::is_same<T, platform::float16>::value,
                "T must be either float32 or float16.");
  if (recvcount == 0) return;

  if (comm == nullptr) {
    if (scale != nullptr) {
      PADDLE_ENFORCE_EQ(nranks, 1,
                        platform::errors::InvalidArgument(
                            "nranks must be 1 when scale != nullptr."));
      auto numel = recvcount * nranks;
      auto config = platform::GetGpuLaunchConfig1D(dev_ctx, numel);
      ScaleCUDAKernel<<<config.block_per_grid, config.thread_per_block, 0,
                        stream>>>(sendbuff, scale, recvbuff, numel);
    }
    return;
  }

  ncclRedOp_t op = ncclSum;
  ncclDataType_t dtype =
      std::is_same<T, float>::value ? ncclFloat32 : ncclFloat16;
  bool should_destroy_op =
      scale && CreatePreMulScaleOpIfSupported(dtype, comm, scale, &op);
  memory::Buffer buffer(dev_ctx.GetPlace());
  if (scale && !should_destroy_op) {
    size_t numel = recvcount * nranks;
    T *new_sendbuff = buffer.Alloc<T>(numel);
    auto config = platform::GetGpuLaunchConfig1D(dev_ctx, numel);
    ScaleCUDAKernel<<<config.block_per_grid, config.thread_per_block, 0,
                      stream>>>(sendbuff, scale, new_sendbuff, numel);
    sendbuff = new_sendbuff;
  }

  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduceScatter(
      sendbuff, recvbuff, recvcount, dtype, op, comm, stream));

#if NCCL_VERSION_CODE >= 21100
  if (should_destroy_op) {
    VLOG(10) << "ncclRedOpDestroy starts";
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRedOpDestroy(op, comm));
    VLOG(10) << "ncclRedOpDestroy ends";
  }
#endif
}
#endif

template <typename InputIteratorT, typename OutputIteratorT, typename ReduceOpT,
          typename T>
static void CubDeviceReduce(InputIteratorT d_in, OutputIteratorT d_out,
                            int num_items, ReduceOpT reduction_op, T init,
                            gpuStream_t stream, memory::Buffer *buffer) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                num_items, reduction_op, init, stream));
  d_temp_storage = buffer->Alloc<void>(temp_storage_bytes);
  VLOG(10) << "cub::DeviceReduce::Reduce needs " << temp_storage_bytes
           << " byte(s), ptr = " << d_temp_storage;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                num_items, reduction_op, init, stream));
}

template <typename T>
static void GetSquareGradNormImpl(const T *grad, int n, float *square_norm,
                                  gpuStream_t stream,
                                  memory::Buffer *cub_tmp_buffer) {
  using Iterator =
      cub::TransformInputIterator<float, SquareFunctor<T>, const T *>;
  Iterator iter(grad, SquareFunctor<T>());
  CubDeviceReduce(iter, square_norm, n, cub::Sum(), static_cast<float>(0),
                  stream, cub_tmp_buffer);
}

// square_norm is of length 2 at least
static void GetSquareGradNorm(const float *fp32_grad, int fp32_numel,
                              const platform::float16 *fp16_grad,
                              int fp16_numel, float *square_norm,
                              gpuStream_t stream,
                              memory::Buffer *cub_tmp_buffer) {
  VLOG(10) << "GetSquareGradNorm starts, fp32_numel = " << fp32_numel
           << " , fp16_numel = " << fp16_numel;
  if (fp32_numel > 0) {
    GetSquareGradNormImpl(fp32_grad, fp32_numel, square_norm, stream,
                          cub_tmp_buffer);
    VLOG(10) << "FP32 square L2-Norm: "
             << FlattenToString(square_norm, 1, cub_tmp_buffer->GetPlace());
  }

  if (fp16_numel > 0) {
    float *fp16_square_norm = fp32_numel > 0 ? square_norm + 1 : square_norm;
    GetSquareGradNormImpl(fp16_grad, fp16_numel, fp16_square_norm, stream,
                          cub_tmp_buffer);
    VLOG(10) << "FP16 square L2-Norm: "
             << FlattenToString(fp16_square_norm, 1,
                                cub_tmp_buffer->GetPlace());
    if (fp32_numel > 0) {
      AddToCUDAKernel<<<1, 1, 0, stream>>>(fp16_square_norm, square_norm);
      VLOG(10) << "FP32+FP16 square L2-Norm: "
               << FlattenToString(square_norm, 1, cub_tmp_buffer->GetPlace());
    }
  }
  VLOG(10) << "GetSquareGradNorm ends, fp32_numel = " << fp32_numel
           << " , fp16_numel = " << fp16_numel;
}

template <typename T>
std::string NumToString(T x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

template <typename T>
static std::string GetMinMaxStr(const T *x, size_t n,
                                const platform::Place &place) {
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::InvalidArgument("Only support CUDAPlace currently."));

  auto *dev_ctx = static_cast<platform::CUDADeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();

  memory::Buffer ret_buffer(place);
  T *ret = ret_buffer.Alloc<T>(2);

  if (n > 0) {
    memory::Buffer cub_buffer(place);
    CubDeviceReduce(x, ret, n, cub::Min(), std::numeric_limits<T>::max(),
                    stream, &cub_buffer);
    CubDeviceReduce(x, ret + 1, n, cub::Max(), std::numeric_limits<T>::lowest(),
                    stream, &cub_buffer);
    T ret_cpu[2];
#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpyAsync(&ret_cpu[0], ret, 2 * sizeof(T),
                                              hipMemcpyDeviceToHost, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(hipStreamSynchronize(stream));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&ret_cpu[0], ret, 2 * sizeof(T),
                                               cudaMemcpyDeviceToHost, stream));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#endif
    return std::string("{\"min\": ") + NumToString(ret_cpu[0]) +
           " , \"max\": " + NumToString(ret_cpu[1]) + "}";
  } else {
    return "{\"min\": null, \"max\": null}";
  }
}

struct VisitDTypeFunctor {
  VisitDTypeFunctor(const framework::Tensor *x, std::string *s)
      : x_(x), s_(s) {}

  template <typename T>
  void apply() const {
    *s_ = GetMinMaxStr<T>(x_->template data<T>(), x_->numel(), x_->place());
  }

 private:
  const framework::Tensor *x_;
  std::string *s_;
};

static std::string GetMinMaxStr(const framework::Tensor *x) {
  if (x == nullptr) return "null";
  if (!x->IsInitialized()) return "not_inited";
  if (!platform::is_gpu_place(x->place())) return "CPUTensor";
  std::string str;
  VisitDTypeFunctor functor(x, &str);
  phi::VisitDataType(x->dtype(), functor);
  return str;
}

static void PrintAllMinMaxRange(const framework::ExecutionContext &ctx,
                                bool only_inputs) {
  if (!VLOG_IS_ON(1)) return;
  for (const auto &pair : ctx.GetOp().Inputs()) {
    const auto &key = pair.first;
    const auto tensors = ctx.MultiInput<framework::Tensor>(key);
    size_t n = tensors.size();
    for (size_t i = 0; i < n; ++i) {
      VLOG(1) << "Input(" << key + ")[" << i << "] = " << pair.second[i]
              << " , " << GetMinMaxStr(tensors[i]);
    }
  }

  if (only_inputs) return;
  for (const auto &pair : ctx.GetOp().Outputs()) {
    const auto &key = pair.first;
    const auto tensors = ctx.MultiOutput<framework::Tensor>(key);
    size_t n = tensors.size();
    for (size_t i = 0; i < n; ++i) {
      VLOG(1) << "Output(" << key + ")[" << i << "] = " << pair.second[i]
              << " , " << GetMinMaxStr(tensors[i]);
    }
  }
}

static void CheckHasNanInfGrad(const float *fp32_grad, int fp32_numel,
                               const platform::float16 *fp16_grad,
                               int fp16_numel, float *nan_inf_flag,
                               gpuStream_t stream,
                               memory::Buffer *cub_tmp_buffer) {
  bool *fp32_has_nan_inf = nullptr;
  bool *fp16_has_nan_inf = nullptr;
  if (fp32_numel > 0) {
    fp32_has_nan_inf = reinterpret_cast<bool *>(nan_inf_flag + 1);
    cub::TransformInputIterator<bool, IsNanInfFunctor<float>, const float *>
    iter(fp32_grad, IsNanInfFunctor<float>());
    CubDeviceReduce(iter, fp32_has_nan_inf, fp32_numel, OrFunctor(), false,
                    stream, cub_tmp_buffer);
  }

  if (fp16_numel > 0) {
    fp16_has_nan_inf = reinterpret_cast<bool *>(nan_inf_flag + 1) + 1;
    cub::TransformInputIterator<bool, IsNanInfFunctor<platform::float16>,
                                const platform::float16 *>
        iter(fp16_grad, IsNanInfFunctor<platform::float16>());
    CubDeviceReduce(iter, fp16_has_nan_inf, fp16_numel, OrFunctor(), false,
                    stream, cub_tmp_buffer);
  }

  if (fp32_has_nan_inf && fp16_has_nan_inf) {
    SetNanInfValueCUDAKernelTwoFlag<<<1, 1, 0, stream>>>(
        fp32_has_nan_inf, fp16_has_nan_inf, nan_inf_flag);
  } else if (fp32_has_nan_inf) {
    SetNanInfValueCUDAKernelOneFlag<<<1, 1, 0, stream>>>(fp32_has_nan_inf,
                                                         nan_inf_flag);
  } else {
    SetNanInfValueCUDAKernelOneFlag<<<1, 1, 0, stream>>>(fp16_has_nan_inf,
                                                         nan_inf_flag);
  }
}

template <typename T>
class DistributedFusedLambOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    auto place = dev_ctx.GetPlace();

    // Step 1: Get fp16 param and grad tensors
    int64_t fp16_numel;
    auto *fp16_param = GetSameInOutTensorPtr<platform::float16, true>(
        ctx, place, "FP16FusedParam", "FP16FusedParamOut", &fp16_numel);
    bool has_fp16_param = (fp16_numel > 0);
    const platform::float16 *fp16_grad = nullptr;
    if (has_fp16_param) {
      fp16_grad = GetInputTensorPtr<platform::float16>(ctx, "FP16FusedGrad");
    } else {
      fp16_param = nullptr;
    }

    // Step 2: Get fp32 param and grad tensors
    int64_t fp32_numel = 0;
    auto *fp32_param = GetSameInOutTensorPtr<float, true>(
        ctx, place, "FP32FusedParam", "FP32FusedParamOut", &fp32_numel);
    PADDLE_ENFORCE_GE(fp32_numel, fp16_numel,
                      platform::errors::InvalidArgument(
                          "The element number in FP32FusedParam should be not "
                          "less than FP16FusedParam."));

    fp32_numel -= fp16_numel;  // the FP32FusedParam contains fp32 param and
                               // fp16 master weight
    bool has_fp32_param = (fp32_numel > 0);
    const float *fp32_grad = nullptr;
    if (has_fp32_param) {
      fp32_grad = GetInputTensorPtr<float>(ctx, "FP32FusedGrad");
    } else {
      PADDLE_ENFORCE_EQ(
          has_fp16_param, true,
          platform::errors::InvalidArgument(
              "Either FP32FusedGrad or FP16FusedGrad cannot be NULL."));
    }

    auto numel = fp32_numel + fp16_numel;
    VLOG(1) << "numel = " << numel << " , fp32_numel = " << fp32_numel
            << " , fp16_numel = " << fp16_numel;

    // The NVIDIA cub library does not support number > INT32_MAX
    PADDLE_ENFORCE_LE(numel, std::numeric_limits<int>::max(),
                      platform::errors::Unimplemented(
                          "Too many parameter number. Only <= %d is supported.",
                          std::numeric_limits<int>::max()));

    // Step 3: Get FusedIndices, ParamInfo
    const auto *indices = GetInputTensorPtr<int>(ctx, "FusedIndices");
    const auto *param_info_tensor = GetInputTensorPtr<int>(ctx, "ParamInfo");
    auto fp32_local_start_idx = param_info_tensor[0];
    auto fp32_local_param_num = param_info_tensor[1];
    auto fp32_global_param_num = param_info_tensor[2];
    auto fp16_local_start_idx = param_info_tensor[3];
    auto fp16_local_param_num = param_info_tensor[4];
    auto fp16_global_param_num = param_info_tensor[5];

    auto local_param_num = fp32_local_param_num + fp16_local_param_num;
    auto param_num = fp32_global_param_num + fp16_global_param_num;
    PADDLE_ENFORCE_LE(local_param_num, param_num,
                      platform::errors::InvalidArgument(
                          "The local parameter number should not exceed the "
                          "global parameter number."));
    VLOG(1) << "local_param_num = " << local_param_num
            << " , global_param_num = " << param_num
            << " , fp32_local_start_idx = " << fp32_local_start_idx
            << " , fp32_local_param_num = " << fp32_local_param_num
            << " , fp32_global_param_num = " << fp32_global_param_num
            << " , fp16_local_start_idx = " << fp16_local_start_idx
            << " , fp16_local_param_num = " << fp16_local_param_num
            << " , fp16_global_param_num = " << fp16_global_param_num;

    // Step 4: Get LearningRate, Moment1, Moment2, Beta1Pow, Beta2Pow,
    // WeightDecay, GlobalScale, FoundInf
    const auto *global_scale = GetInputTensorPtr<float>(ctx, "GlobalScale");
    const auto *lr = GetInputTensorPtr<float>(ctx, "LearningRate");
    int64_t partial_numel = 0;
    auto *moment1 = GetSameInOutTensorPtr<float>(ctx, place, "Moment1",
                                                 "Moment1Out", &partial_numel);

    PADDLE_ENFORCE_EQ(numel % partial_numel, 0,
                      platform::errors::InvalidArgument(
                          "The total parameter number %d should be divided "
                          "exactly by the element number %d of Moment1.",
                          numel, partial_numel));

    int64_t num_devices = numel / partial_numel;
    VLOG(1) << "num_devices = " << num_devices
            << " , partial_numel = " << partial_numel;

    PADDLE_ENFORCE_EQ(fp32_numel % num_devices, 0,
                      platform::errors::InvalidArgument(
                          "The fp32 parameter number %d should be divided "
                          "exactly by the device number %d.",
                          fp32_numel, num_devices));
    PADDLE_ENFORCE_EQ(fp16_numel % num_devices, 0,
                      platform::errors::InvalidArgument(
                          "The fp16 parameter number %d should be divided "
                          "exactly by the device number %d.",
                          fp16_numel, num_devices));

    auto *moment2 =
        GetSameInOutTensorPtr<float>(ctx, place, "Moment2", "Moment2Out");
    auto *beta1pow =
        GetSameInOutTensorPtr<float>(ctx, place, "Beta1Pow", "Beta1PowOut");
    auto *beta2pow =
        GetSameInOutTensorPtr<float>(ctx, place, "Beta2Pow", "Beta2PowOut");
    const float *weight_decay = GetInputTensorPtr<float>(ctx, "WeightDecay");

    auto *found_inf_t = ctx.Output<framework::Tensor>("FoundInf");
    found_inf_t->Resize({1});
    auto *found_inf = found_inf_t->mutable_data<bool>(place);

    // Step 5: Get attributes beta1, beta2, epsilon, max_grad_norm, ring_id,
    // use_master_param_norm, is_grad_scaled_by_nranks
    auto beta1 = ctx.Attr<float>("beta1");
    auto beta2 = ctx.Attr<float>("beta2");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto max_global_grad_norm = ctx.Attr<float>("max_global_grad_norm");
    auto clip_after_allreduce = ctx.Attr<bool>("clip_after_allreduce");
    auto ring_id = ctx.Attr<int>("ring_id");
    auto use_master_param_norm = ctx.Attr<bool>("use_master_param_norm");
    auto is_grad_scaled_by_nranks = ctx.Attr<bool>("is_grad_scaled_by_nranks");
    VLOG(10) << "max_global_grad_norm = " << max_global_grad_norm
             << " , clip_after_allreduce = " << clip_after_allreduce
             << " , use_master_param_norm = " << use_master_param_norm
             << " , is_grad_scaled_by_nranks = " << is_grad_scaled_by_nranks;

    // Step 6: allreduce + global norm gradient clip
    int rank = 0;
    ncclComm_t comm = nullptr;
    if (num_devices > 1) {
      auto *nccl_comm_handle =
          platform::NCCLCommContext::Instance().Get(ring_id, place);
      comm = nccl_comm_handle->comm();
      rank = nccl_comm_handle->rank();
    }

    memory::Buffer grad_norm_square_buffer(place);
    auto *fp32_square_grad_norm = grad_norm_square_buffer.Alloc<float>(2);
    memory::Buffer cub_tmp_buffer(place);

    memory::Buffer sum_grad_buffer(place);
    float *fp32_sum_grad;
    platform::float16 *fp16_sum_grad;
    auto fp32_numel_each_device = fp32_numel / num_devices;
    auto fp16_numel_each_device = fp16_numel / num_devices;
    if (num_devices > 1) {
      auto ptr = sum_grad_buffer.Alloc<uint8_t>(
          fp32_numel_each_device * sizeof(float) +
          fp16_numel_each_device * sizeof(platform::float16));
      fp32_sum_grad = has_fp32_param ? reinterpret_cast<float *>(ptr) : nullptr;
      fp16_sum_grad = has_fp16_param
                          ? reinterpret_cast<platform::float16 *>(
                                ptr + fp32_numel_each_device * sizeof(float))
                          : nullptr;
    } else {
      // NOTE: The const_cast here is not important. The fp32_sum_grad and
      // fp16_sum_grad would not be changed when num_devices == 1
      // But if I do not perform const_cast here, there would be more
      // if-else codes (num_devices > 1) when I write the following code.
      // So I prefer to use const_cast to unify the following code to reduce
      // the if-else codes.
      fp32_sum_grad = const_cast<float *>(fp32_grad);
      fp16_sum_grad = const_cast<platform::float16 *>(fp16_grad);
    }

    float rescale_grad = 1.0f;
    if (!is_grad_scaled_by_nranks) {
      rescale_grad /= num_devices;
    }

    if (max_global_grad_norm > 0) {
      if (clip_after_allreduce) {
        // (1) ReduceScater first
        NCCLReduceScatterWithScale(fp32_grad, fp32_sum_grad,
                                   fp32_numel_each_device, num_devices, comm,
                                   stream, dev_ctx);
        NCCLReduceScatterWithScale(fp16_grad, fp16_sum_grad,
                                   fp16_numel_each_device, num_devices, comm,
                                   stream, dev_ctx);
        // (2) Calculate the global grad norm
        GetSquareGradNorm(fp32_sum_grad, fp32_numel_each_device, fp16_sum_grad,
                          fp16_numel_each_device, fp32_square_grad_norm, stream,
                          &cub_tmp_buffer);
        VLOG(1) << "Grad square norm before all reduce: "
                << FlattenToString(fp32_square_grad_norm, 1, place);
        if (num_devices > 1) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
              fp32_square_grad_norm, fp32_square_grad_norm, 1, ncclFloat32,
              ncclSum, comm, stream));
        }
        VLOG(1) << "Grad square norm after all reduce: "
                << FlattenToString(fp32_square_grad_norm, 1, place);
      } else {
        // (1) Calculate the local grad norm
        GetSquareGradNorm(fp32_grad, fp32_numel, fp16_grad, fp16_numel,
                          fp32_square_grad_norm, stream, &cub_tmp_buffer);
        VLOG(1) << "Grad square norm before all reduce: "
                << FlattenToString(fp32_square_grad_norm, 1, place);
        // (2) Calculate the gradient clip scale
        float *fp32_scale = nullptr;
        platform::float16 *fp16_scale = nullptr;
        if (has_fp32_param && has_fp16_param) {
          auto *ptr = cub_tmp_buffer.Alloc<uint8_t>(sizeof(float) +
                                                    sizeof(platform::float16));
          fp32_scale = reinterpret_cast<float *>(ptr);
          fp16_scale =
              reinterpret_cast<platform::float16 *>(ptr + sizeof(float));
        } else if (has_fp32_param) {
          fp32_scale = cub_tmp_buffer.Alloc<float>(1);
        } else {
          fp16_scale = cub_tmp_buffer.Alloc<platform::float16>(1);
        }

        float clip_scale = 1.0f;
        if (is_grad_scaled_by_nranks) {
          clip_scale *= num_devices;
        }
        CalcGradNormClipBeforeAllReduceScale<
            float, platform::float16><<<1, 1, 0, stream>>>(
            global_scale, max_global_grad_norm, fp32_square_grad_norm,
            fp32_scale, fp16_scale, clip_scale);
        VLOG(1) << "Grad scale: " << FlattenToString(fp32_scale, 1, place);
        if (num_devices > 1) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
              fp32_square_grad_norm, fp32_square_grad_norm, 1, ncclFloat32,
              ncclSum, comm, stream));
        }
        // (3) Do ReduceScatter with scale
        NCCLReduceScatterWithScale(fp32_grad, fp32_sum_grad,
                                   fp32_numel_each_device, num_devices, comm,
                                   stream, dev_ctx, fp32_scale);
        NCCLReduceScatterWithScale(fp16_grad, fp16_sum_grad,
                                   fp16_numel_each_device, num_devices, comm,
                                   stream, dev_ctx, fp16_scale);
        // (4) mark max_global_grad_norm as 0, meaning that clip has been
        // already performed
        max_global_grad_norm = 0;
      }
    } else {
      NCCLReduceScatterWithScale(fp32_grad, fp32_sum_grad,
                                 fp32_numel_each_device, num_devices, comm,
                                 stream, dev_ctx);
      NCCLReduceScatterWithScale(fp16_grad, fp16_sum_grad,
                                 fp16_numel_each_device, num_devices, comm,
                                 stream, dev_ctx);
      CheckHasNanInfGrad(fp32_sum_grad, fp32_numel_each_device, fp16_sum_grad,
                         fp16_numel_each_device, fp32_square_grad_norm, stream,
                         &cub_tmp_buffer);
      if (num_devices > 1) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
            fp32_square_grad_norm, fp32_square_grad_norm, 1, ncclFloat32,
            ncclSum, comm, stream));
      }
      max_global_grad_norm = 0;
    }
    VLOG(10) << "ReduceScatter done";

    // Step 7: update the moment1, moment2. Calcuate the trust_ratio_div
    memory::Buffer trust_ratio_div_buffer(place);
    auto *trust_ratio_div = trust_ratio_div_buffer.Alloc<float>(partial_numel);
    auto fp32_offset = rank * fp32_numel_each_device;
    auto fp16_offset = rank * fp16_numel_each_device;
    if (has_fp32_param) {
      auto config =
          platform::GetGpuLaunchConfig1D(dev_ctx, fp32_numel_each_device);
      VLOG(10) << "Update FP32 Moment and TrustRatioDiv starts";
      UpdateLambMoment<<<config.block_per_grid, config.thread_per_block, 0,
                         stream>>>(
          fp32_param + fp32_offset, fp32_sum_grad, fp32_square_grad_norm,
          global_scale, indices + fp32_offset, weight_decay, beta1pow, beta2pow,
          moment1, moment2, trust_ratio_div, beta1, beta2, epsilon,
          max_global_grad_norm, fp32_numel_each_device, rescale_grad);
      VLOG(10) << "Update FP32 Moment and TrustRatioDiv done";
    }
    float *master_param = nullptr;
    if (has_fp16_param) {
      master_param = fp32_param + fp32_numel;
      auto config =
          platform::GetGpuLaunchConfig1D(dev_ctx, fp16_numel_each_device);
      VLOG(10) << "Update FP16 Moment and TrustRatioDiv starts";
      UpdateLambMoment<<<config.block_per_grid, config.thread_per_block, 0,
                         stream>>>(
          master_param + fp16_offset, fp16_sum_grad, fp32_square_grad_norm,
          global_scale, indices + fp32_numel + fp16_offset, weight_decay,
          beta1pow, beta2pow, moment1 + fp32_numel_each_device,
          moment2 + fp32_numel_each_device,
          trust_ratio_div + fp32_numel_each_device, beta1, beta2, epsilon,
          max_global_grad_norm, fp16_numel_each_device, rescale_grad);
      VLOG(10) << "Update FP16 Moment and TrustRatioDiv done";
    }

    VLOG(10) << "Update Moment and TrustRatioDiv done hehahaha";

    // Step 8: calculate L2-Norm square of parameter and trust_ratio_div
    memory::Buffer square_norm_buffer(place);
    auto *param_square_norm = square_norm_buffer.Alloc<float>(2 * param_num);
    auto *trust_ratio_div_square_norm = param_square_norm + param_num;

    auto *fused_offsets_t = ctx.Input<framework::Tensor>("FusedParamOffsets");
    auto *fused_offsets = fused_offsets_t->data<int>();
    auto *fp32_partial_fused_offsets_t =
        ctx.Input<framework::Tensor>("FP32ShardFusedParamOffsets");
    const auto *fp32_partial_fused_offsets =
        fp32_partial_fused_offsets_t->data<int>();
    auto *fp16_partial_fused_offsets_t =
        ctx.Input<framework::Tensor>("FP16ShardFusedParamOffsets");
    const auto *fp16_partial_fused_offsets =
        fp16_partial_fused_offsets_t->data<int>();

    VLOG(1) << "FusedParamOffsets: "
            << FlattenToString(fused_offsets, fused_offsets_t->numel(),
                               fused_offsets_t->place());
    VLOG(1) << "FP32ShardFusedParamOffsets: "
            << FlattenToString(fp32_partial_fused_offsets,
                               fp32_partial_fused_offsets_t->numel(),
                               fp32_partial_fused_offsets_t->place());
    VLOG(1) << "FP16ShardFusedParamOffsets: "
            << FlattenToString(fp16_partial_fused_offsets,
                               fp16_partial_fused_offsets_t->numel(),
                               fp16_partial_fused_offsets_t->place());

    if (num_devices > 1) {
      if (use_master_param_norm) {
        FillZeroWithPtr(param_square_norm + fp32_global_param_num,
                        2 * param_num - fp32_global_param_num, stream);
      } else {
        FillZeroWithPtr(trust_ratio_div_square_norm, param_num, stream);
      }
    }
    MultiTensorL2Norm(place, stream, fp32_param, fused_offsets,
                      fp32_global_param_num, param_square_norm);
    if (use_master_param_norm) {
      MultiTensorL2Norm(place, stream, master_param + fp16_offset,
                        fp16_partial_fused_offsets, fp16_local_param_num,
                        param_square_norm + fp16_local_start_idx);
    } else {
      // NOTE: extra computation is performed. We can improve this performance
      // if needed in the future.
      MultiTensorL2Norm(
          place, stream, fp16_param, fused_offsets + fp32_global_param_num,
          fp16_global_param_num, param_square_norm + fp32_global_param_num);
    }

    MultiTensorL2Norm(place, stream, trust_ratio_div,
                      fp32_partial_fused_offsets, fp32_local_param_num,
                      trust_ratio_div_square_norm + fp32_local_start_idx);
    MultiTensorL2Norm(place, stream, trust_ratio_div + fp32_numel_each_device,
                      fp16_partial_fused_offsets, fp16_local_param_num,
                      trust_ratio_div_square_norm + fp16_local_start_idx);

    VLOG(1) << "TrustRatioDiv L2-Norm before allreduce: "
            << FlattenToString(trust_ratio_div_square_norm, param_num, place);
    if (num_devices > 1) {
      if (use_master_param_norm) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
            param_square_norm + fp32_global_param_num,
            param_square_norm + fp32_global_param_num,
            2 * param_num - fp32_global_param_num, ncclFloat32, ncclSum, comm,
            stream));
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
            trust_ratio_div_square_norm, trust_ratio_div_square_norm, param_num,
            ncclFloat32, ncclSum, comm, stream));
      }
      VLOG(10) << "ncclAllReduce done";
    }

    LogParamAndTrustRatioDivSquareNorm<1>(ctx, param_square_norm,
                                          trust_ratio_div_square_norm);
    VLOG(10) << "Calculate L2-Norm of Param and TrustRatioDiv done";

    // Step 9: update parameter, beta1pow, beta2pow. All gather parameters.
    if (has_fp32_param) {
      LambUpdateParamAndBetaPows<float>(
          dev_ctx, trust_ratio_div, lr, indices + fp32_offset,
          param_square_norm, trust_ratio_div_square_norm, fp32_square_grad_norm,
          &beta1pow, &beta2pow, &found_inf, beta1, beta2,
          fp32_numel_each_device, fp32_param + fp32_offset, nullptr, stream);
      if (num_devices > 1) {
        // ncclAllGather
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
            fp32_param + fp32_offset, fp32_param, fp32_numel_each_device,
            ncclFloat32, comm, stream));
      }
    }
    if (has_fp16_param) {
      LambUpdateParamAndBetaPows<platform::float16>(
          dev_ctx, trust_ratio_div + fp32_numel_each_device, lr,
          indices + fp32_numel + fp16_offset, param_square_norm,
          trust_ratio_div_square_norm, fp32_square_grad_norm, &beta1pow,
          &beta2pow, &found_inf, beta1, beta2, fp16_numel_each_device,
          fp16_param + fp16_offset, master_param + fp16_offset, stream);

      if (num_devices > 1) {
        // ncclAllGather
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
            fp16_param + fp16_offset, fp16_param, fp16_numel_each_device,
            ncclFloat16, comm, stream));
      }
    }
    VLOG(10) << "Update Param done";

    VLOG(1) << "IsFinite: " << IsFinite(dev_ctx, fp32_square_grad_norm);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "distributed_fused_lamb op should be used with NCCL/RCCL."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    distributed_fused_lamb,
    ops::DistributedFusedLambOpKernel<plat::CUDADeviceContext, float>);
