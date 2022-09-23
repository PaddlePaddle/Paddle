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

#pragma once
#include <math.h>  // for sqrt in CPU and CUDA

#include <Eigen/Dense>

#include "paddle/phi/kernels/funcs/algorithm.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/memcpy.h"
#endif

namespace phi {
namespace funcs {
using float16 = dtype::float16;

#ifdef PADDLE_WITH_XPU

template <typename Context, typename T1, typename T2>
static int ConvertDataByType(
    const T1* x, T2** y, int len, bool allocateFlag, const Context& dev_ctx) {
  if (nullptr == x || nullptr == y || len <= 0)
    return xpu::Error_t::INVALID_PARAM;
  int r = 0;
  if (allocateFlag) {
    r = xpu_malloc(reinterpret_cast<void**>(y), sizeof(T2) * len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
  }

  T1* cpu_data = reinterpret_cast<T1*>(malloc(sizeof(T1) * len));

  paddle::memory::Copy(
      CPUPlace(), cpu_data, dev_ctx.GetPlace(), x, len * sizeof(T1));

  T2* cpu_real_data = reinterpret_cast<T2*>(malloc(sizeof(T2) * len));
  for (int i = 0; i < len; i++) cpu_real_data[i] = static_cast<T2>(cpu_data[i]);

  paddle::memory::Copy(
      dev_ctx.GetPlace(), *y, CPUPlace(), cpu_real_data, len * sizeof(T2));

  free(cpu_data);
  free(cpu_real_data);

  return xpu::Error_t::SUCCESS;
}

template <typename Context, typename T>
static void GetDataPointer(const phi::DenseTensor& tensorData,
                           T** result,
                           const Context& dev_ctx) {
  if (tensorData.dtype() == DataType::FLOAT16) {
    const float16* real_data = tensorData.template data<float16>();
    int len = tensorData.numel();

    int r = ConvertDataByType<Context, float16, T>(
        real_data, result, len, true, dev_ctx);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
  }
}

template <typename Context, typename T>
static void GetOutDataPointer(DenseTensor* tensorData,
                              DenseTensor* out,
                              T** result,
                              const Context& dev_ctx) {
  if (tensorData->dtype() == DataType::FLOAT16) {
    *result = dev_ctx.template Alloc<T>(out);
  } else {
    *result = dev_ctx.template Alloc<T>(tensorData);
  }
}

template <typename Context, typename T>
static void CopyOutData(const DenseTensor& srcTensor,
                        phi::DenseTensor* dstTensor,
                        const Context& dev_ctx) {
  if (dstTensor->dtype() == DataType::FLOAT16) {
    const T* xpu_out_data = srcTensor.template data<T>();
    float16* out_data = dev_ctx.template Alloc<float16>(dstTensor);
    int len = srcTensor.numel();

    int r = ConvertDataByType<Context, T, float16>(
        xpu_out_data, &out_data, len, false, dev_ctx);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
  }
}

template <typename T>
static void FreeData(const phi::DenseTensor& tensorData, T* dataPtr) {
  if (tensorData.dtype() == DataType::FLOAT16) xpu_free(dataPtr);
}

template <typename Context, typename T>
static void SetBetaData(const phi::DenseTensor& beta_pow,
                        phi::DenseTensor* beta_pow_out,
                        const T& beta,
                        const Context& dev_ctx) {
  if (beta_pow.dtype() == DataType::FLOAT16) {
    const float16* beta_pow_p = beta_pow.template data<float16>();
    dev_ctx.template HostAlloc<float16>(beta_pow_out)[0] =
        static_cast<float16>(beta) * beta_pow_p[0];
  } else {
    const T* beta_pow_p = beta_pow.template data<T>();
    dev_ctx.template HostAlloc<T>(beta_pow_out)[0] = beta * beta_pow_p[0];
  }
}

template <typename Context, typename T>
static void Scale(phi::DenseTensor* beta_pow_out,
                  const phi::DenseTensor& beta_pow,
                  T* beta_pow_ptr,
                  const T& beta,
                  const Context& dev_ctx) {
  float16* beta_pow_out_p2 = dev_ctx.template Alloc<float16>(beta_pow_out);

  DenseTensor xpu_beta_pow_out;
  const phi::DenseTensorMeta meta_beta_pow_out(DataType::FLOAT32,
                                               beta_pow_out->dims());
  xpu_beta_pow_out.set_meta(meta_beta_pow_out);

  T* beta_pow_out_ptr = dev_ctx.template Alloc<T>(&xpu_beta_pow_out);

  int r = xpu::scale(dev_ctx.x_context(),
                     beta_pow_ptr,
                     beta_pow_out_ptr,
                     beta_pow.numel(),
                     false,
                     beta,
                     0.0f);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");

  const float* xpu_beta_pow_out_data =
      dev_ctx.template Alloc<T>(&xpu_beta_pow_out);
  int len = xpu_beta_pow_out.numel();

  r = ConvertDataByType<Context, T, float16>(
      xpu_beta_pow_out_data, &beta_pow_out_p2, len, false, dev_ctx);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "adam");
}
#endif

struct GPUAdam;
struct CPUAdam;

template <typename T, typename Flavour>
class AdamFunctor;

template <typename T>
class AdamFunctor<T, GPUAdam> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

 public:
  AdamFunctor(T beta1,
              T beta2,
              T epsilon,
              const T* beta1_pow,
              const T* beta2_pow,
              const T* mom1,
              T* mom1_out,
              const T* mom2,
              T* mom2_out,
              const T* lr,
              const T* grad,
              const T* param,
              T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // Merge all memory access together.
    T g = grad_[i];
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_ * sqrt(1 - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }
};

template <typename T>
class AdamFunctor<T, CPUAdam> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

 public:
  AdamFunctor(T beta1,
              T beta2,
              T epsilon,
              const T* beta1_pow,
              const T* beta2_pow,
              const T* mom1,
              T* mom1_out,
              const T* mom2,
              T* mom2_out,
              const T* lr,
              const T* grad,
              const T* param,
              T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  void operator()(size_t numel) const {
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> g{
        grad_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom1{
        moment1_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom2{
        moment2_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param_out{
        param_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment1_out{
        moment1_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment2_out{
        moment2_out_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    moment1_out = beta1_ * mom1 + (1 - beta1_) * g;
    moment2_out = beta2_ * mom2 + (1 - beta2_) * g * g;
    param_out = param - lr * (moment1_out / (moment2_out.sqrt() +
                                             epsilon_ * sqrt(1 - beta2_pow)));
  }
};

template <typename T, typename Flavour, typename MT = T>
class SparseAdamFunctor;

template <typename T, typename MT>
class SparseAdamFunctor<T, GPUAdam, MT> {
 private:
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  const MT* beta1_pow_;
  const MT* beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const MT* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;
  const MT* master_param_;
  MT* master_param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;
  bool lazy_mode_;

 public:
  SparseAdamFunctor(MT beta1,
                    MT beta2,
                    MT epsilon,
                    const MT* beta1_pow,
                    const MT* beta2_pow,
                    const MT* mom1,
                    MT* mom1_out,
                    const MT* mom2,
                    MT* mom2_out,
                    const MT* lr,
                    const T* grad,
                    const T* param,
                    T* param_out,
                    const MT* master_param,
                    MT* master_param_out,
                    const int64_t* rows,
                    int64_t row_numel,
                    int64_t row_count,
                    bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        master_param_(master_param),
        master_param_out_(master_param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count),
        lazy_mode_(lazy_mode) {}

  inline HOSTDEVICE void adam_update(size_t i, MT g) const {
    // The following code is the same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT lr = *lr_;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    // Calculation
    lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
          (static_cast<MT>(1.0) - beta1_pow);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1.0) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1.0) - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) +
                       epsilon_ * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = static_cast<T>(p);
    if (master_param_out_) {
      master_param_out_[i] = p;
    }
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        phi::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    if (lazy_mode_ && row_idx < 0) {
      return;
    } else {
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
                 : static_cast<MT>(0);
      adam_update(i, g);
    }
  }
};

template <typename T>
class SparseAdamFunctor<T, CPUAdam, T> {
 private:
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

 public:
  SparseAdamFunctor(T beta1,
                    T beta2,
                    T epsilon,
                    const T* beta1_pow,
                    const T* beta2_pow,
                    const T* mom1,
                    T* mom1_out,
                    const T* mom2,
                    T* mom2_out,
                    const T* lr,
                    const T* grad,
                    const T* param,
                    T* param_out,
                    const int64_t* rows,
                    int64_t row_numel,
                    int64_t row_count,
                    bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void adam_update(size_t i, T g) const {
    // The following code is the same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_ * sqrt(1 - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }

  inline void operator()(size_t numel) const {
    // lr could be reuse
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);
    int64_t row_count = static_cast<int64_t>(numel / row_numel_);

    for (int64_t i = 0, j = 0; i != row_count; ++i) {
      if (i == *(rows_ + j)) {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T g = grad_[j * row_numel_ + k];
          adam_update(i * row_numel_ + k, g);
        }
        ++j;
      } else {
        for (int64_t k = 0; k != row_numel_; ++k) {
          T mom1 = moment1_[i * row_numel_ + k];
          T mom2 = moment2_[i * row_numel_ + k];
          T p = param_[i * row_numel_ + k];

          mom1 = beta1_ * mom1;
          mom2 = beta2_ * mom2;

          p -= lr * (mom1 / (sqrt(mom2) + epsilon_));
          // Write back to global memory
          moment1_out_[i * row_numel_ + k] = mom1;
          moment2_out_[i * row_numel_ + k] = mom2;
          param_out_[i * row_numel_ + k] = p;
        }
      }
    }
  }
};

struct GPUAdamW;
struct CPUAdamW;

template <typename T, typename Flavour>
class AdamWFunctor;

template <typename T>
class AdamWFunctor<T, CPUAdamW> {
 private:
  const T coeff_;
  const T lr_ratio_;
  const T* lr_;
  T* param_;

 public:
  AdamWFunctor(const T coeff, const T lr_ratio, const T* lr, T* param)
      : coeff_(coeff), lr_ratio_(lr_ratio), lr_(lr), param_(param) {}

  inline HOSTDEVICE void operator()(size_t numel) const {
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;

    // Calculation
    param -= lr * lr_ratio_ * coeff_ * param;
  }
};

template <typename T, typename Flavour, typename MT = T>
class SparseAdamWFunctor;

template <typename T, typename MT>
class SparseAdamWFunctor<T, GPUAdamW, MT> {
 private:
  MT beta1_;
  MT beta2_;
  MT epsilon_;
  MT coeff_;
  MT lr_ratio_;

  const MT* beta1_pow_;
  const MT* beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const MT* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;
  const MT* master_param_;
  MT* master_param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;
  bool lazy_mode_;

 public:
  SparseAdamWFunctor(MT beta1,
                     MT beta2,
                     MT epsilon,
                     MT coeff,
                     MT lr_ratio,
                     const MT* beta1_pow,
                     const MT* beta2_pow,
                     const MT* mom1,
                     MT* mom1_out,
                     const MT* mom2,
                     MT* mom2_out,
                     const MT* lr,
                     const T* grad,
                     const T* param,
                     T* param_out,
                     const MT* master_param,
                     MT* master_param_out,
                     const int64_t* rows,
                     int64_t row_numel,
                     int64_t row_count,
                     bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        coeff_(coeff),
        lr_ratio_(lr_ratio),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out),
        master_param_(master_param),
        master_param_out_(master_param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count),
        lazy_mode_(lazy_mode) {}

  inline HOSTDEVICE void adamw_update(size_t i, MT g) const {
    // The following code is the same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT lr = *lr_ * lr_ratio_;
    MT lr_orig = lr;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    // Calculation
    lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
          (static_cast<MT>(1.0) - beta1_pow);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1.0) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1.0) - beta2_) * g * g;
    p -= lr_orig * coeff_ * p;
    p -= lr * (mom1 / (sqrt(mom2) +
                       epsilon_ * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = static_cast<T>(p);
    if (master_param_out_) {
      master_param_out_[i] = p;
    }
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        phi::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    if (lazy_mode_ && row_idx < 0) {
      return;
    } else {
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
                 : static_cast<MT>(0);
      adamw_update(i, g);
    }
  }
};

}  // namespace funcs
}  // namespace phi
