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

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0

#define AVX_FLOAT_BLOCK 8
#define AVX_DOUBLE_BLOCK 4
#define AVX2_FLOAT_BLOCK 8
#define AVX2_DOUBLE_BLOCK 4
#define AVX512_FLOAT_BLOCK 16
#define AVX512_DOUBLE_BLOCK 8

typedef enum { kLT8, kEQ8, kGT8LT16, kEQ16, kGT16 } jit_block;

class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;

 private:
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
  virtual void Compute(const int n, const T *x, const T *y, T *z) const = 0;
};

template <typename T>
class VAddKernel : public Kernel {
 public:
  virtual void Compute(const int n, const T *x, const T *y, T *z) const = 0;
};

template <typename T>
class VScalKernel : public Kernel {
 public:
  virtual void Compute(const int n, const T a, const T *x, T *y) const = 0;
  virtual void Compute(const int n, const T a, T *x) const = 0;
};

template <typename T>
class VExpKernel : public Kernel {
 public:
  virtual void Compute(const int n, const T *x, T *y) const = 0;
};

template <typename T>
class VSigmoidKernel : public Kernel {
 public:
  virtual void Compute(const int n, const T *x, T *y) const = 0;
};

template <typename T>
class VTanhKernel : public Kernel {
 public:
  virtual void Compute(const int n, const T *x, T *y) const = 0;
};

template <typename T>
class LSTMKernel : public Kernel {
 public:
  explicit LSTMKernel(int d, const std::string &act_gate,
                      const std::string &act_cand, const std::string &act_cell);

  void (*jit_ker)(T *, const T *, T *, T *);
  std::function<void(T *, const T *, T *, T *)> ComputeCtHt, ComputeCtHt_NoC0H0;

 private:
  int d_, d2_, d3_;
  std::function<void(const int, const T *, T *)> act_gate_, act_cell_,
      act_cand_;
};

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
