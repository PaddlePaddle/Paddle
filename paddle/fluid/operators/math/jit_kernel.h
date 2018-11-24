/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <functional>
#include <memory>  // for shared_ptr
#include <string>
#include <unordered_map>
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/macros.h"

// Note: Only support on CPU yet.
namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

// TODO(TJ): move these to some proper place
#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0
#define XMM_FLOAT_BLOCK 4
#define YMM_FLOAT_BLOCK 8
#define ZMM_FLOAT_BLOCK 16

typedef enum { kLT8, kEQ8, kGT8LT16, kEQ16, kGT16 } jit_block;

class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
  // TODO(TJ): below members should be deprecated.
  int num_{0};
  int end_{0};
  int rest_{0};
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

class KernelPool {
 public:
  static KernelPool &Instance();

  template <typename Ker, typename... ARGS>
  std::shared_ptr<const Ker> Get(ARGS... args);

  std::shared_ptr<const Kernel> Get(const std::string &key) const;

 private:
  KernelPool() = default;
  std::unordered_map<std::string, std::shared_ptr<const Kernel>> kers_;

  DISABLE_COPY_AND_ASSIGN(KernelPool);
};

template <typename T>
class VMulKernel : public Kernel {
 public:
  void (*Compute)(const T *, const T *, T *, int);
};

template <typename T>
class VAddKernel : public Kernel {
 public:
  void (*Compute)(const T *, const T *, T *, int);
};

template <typename T>
class VAddReluKernel : public Kernel {
 public:
  void (*Compute)(const T *, const T *, T *, int);
};

template <typename T>
class VScalKernel : public Kernel {
 public:
  // y = a.*x
  void (*Compute)(const T *, const T *, T *, int);
};

template <typename T>
class VAddBiasKernel : public Kernel {
 public:
  // y = a.+x
  void (*Compute)(const T *, const T *, T *, int);
};

#ifdef PADDLE_WITH_MKLDNN
template <typename T>
class EltwiseMulnChw16cNCKernel : public Kernel {
 public:
  // nChw16c = nChw16c .* NC
  void (*Compute)(const float *, const float *, float *, int, int);
};
#endif

template <typename T>
class VActKernel : public Kernel {
 public:
  void (*Compute)(const T *, T *, int);
};

template <typename T>
class VReluKernel : public VActKernel<T> {};

template <typename T>
class VIdentityKernel : public VActKernel<T> {};

template <typename T>
class VExpKernel : public VActKernel<T> {};

template <typename T>
class VSigmoidKernel : public VActKernel<T> {};

template <typename T>
class VTanhKernel : public VActKernel<T> {};

template <typename T>
class LSTMKernel : public Kernel {
 public:
  virtual void ComputeCtHt(T *gates, const T *ct_1, T *ct, T *ht,
                           /* below only used in peephole*/
                           const T *wp_data = nullptr,
                           T *checked = nullptr) const = 0;

  // compute c1 and h1 without c0 or h0
  virtual void ComputeC1H1(T *gates, T *ct, T *ht,
                           /* below only used in peephole*/
                           const T *wp_data = nullptr) const = 0;
};

template <typename T>
class GRUKernel : public Kernel {
 public:
  // compute h1 without h0
  virtual void ComputeH1(T *gates, T *ht) const = 0;
  virtual void ComputeHtPart1(T *gates, const T *ht_1, T *ht) const = 0;
  virtual void ComputeHtPart2(T *gates, const T *ht_1, T *ht) const = 0;
};

template <typename T>
class CRFDecodeKernel : public Kernel {
 public:
  virtual void Compute(const int seq_len, const T *x, const T *w, T *alpha,
                       int *track) const = 0;
};

template <typename T>
class LayerNormKernel : public Kernel {
 public:
  virtual void Compute(T *x, T *out, T *mean, T *var, const T *scale,
                       const T *bias, int height,
                       const float epsilon) const = 0;
};

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
