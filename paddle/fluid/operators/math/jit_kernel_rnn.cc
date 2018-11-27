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

#include "paddle/fluid/operators/math/jit_kernel.h"
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"
#include "paddle/fluid/operators/math/jit_kernel_refer.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

#ifdef PADDLE_WITH_XBYAK
#include "paddle/fluid/operators/math/jit_code.h"
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

/* LSTM JitKernel */
template <typename T>
class LSTMKernelImpl : public LSTMKernel<T> {
 public:
  static inline std::string name(const lstm_attr_t& attr) {
    PADDLE_THROW("DType should be either float or double");
  }
  static inline bool useJIT(int d) { return false; }
  static inline bool useMKL(int d) { return false; }
  explicit LSTMKernelImpl(const lstm_attr_t& attr) : LSTMKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(attr.d)) {
      size_t sz = 96 + attr.d / YMM_FLOAT_BLOCK * 90 * 4 * 8;
      jitcode0_.reset(new gen::LSTMJitCode(false, attr, sz > 4096 ? sz : 4096));
      this->ComputeCtHt =
          jitcode0_->getCode<void (*)(lstm_t*, const lstm_attr_t*)>();

      jitcode1_.reset(new gen::LSTMJitCode(true, attr, sz > 4096 ? sz : 4096));
      this->ComputeC1H1 =
          jitcode1_->getCode<void (*)(lstm_t*, const lstm_attr_t*)>();
      return;
    }
#endif

    this->ComputeCtHt = refer::LSTMCtHt<T>;
    this->ComputeC1H1 = refer::LSTMC1H1<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::LSTMJitCode> jitcode0_{nullptr}, jitcode1_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool LSTMKernelImpl<float>::useJIT(int d) {
  return gen::LSTMJitCode::init(d);
}
#endif

/* Peephole JitKernel */
template <typename T>
class PeepholeKernelImpl : public LSTMKernel<T> {
 public:
  static inline std::string name(const lstm_attr_t& attr) {
    PADDLE_THROW("DType should be either float or double");
  }
  static inline bool useJIT(int d) { return false; }
  static inline bool useMKL(int d) { return false; }
  explicit PeepholeKernelImpl(const lstm_attr_t& attr) : LSTMKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(attr.d)) {
      size_t sz = 96 + attr.d / YMM_FLOAT_BLOCK * 96 * 4 * 8;
      jitcode0_.reset(new gen::LSTMJitCode(false, attr, sz > 4096 ? sz : 4096));
      this->ComputeCtHt =
          jitcode0_->getCode<void (*)(lstm_t*, const lstm_attr_t*)>();

      jitcode1_.reset(new gen::LSTMJitCode(true, attr, sz > 4096 ? sz : 4096));
      this->ComputeC1H1 =
          jitcode1_->getCode<void (*)(lstm_t*, const lstm_attr_t*)>();
      return;
    }
#endif

    this->ComputeCtHt = refer::LSTMCtHt<T>;
    this->ComputeC1H1 = refer::LSTMC1H1<T>;
  }

#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::LSTMJitCode> jitcode0_{nullptr}, jitcode1_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool PeepholeKernelImpl<float>::useJIT(int d) {
  return gen::LSTMJitCode::init(d);
}
#endif

#define JITKERNEL_DEFINE_NAME_LSTM(ker_key, ker_class)                 \
  template <>                                                          \
  std::string ker_class##Impl<float>::name(const lstm_attr_t& attr) {  \
    std::string key(#ker_key "f");                                     \
    key += (attr.act_gate + attr.act_cand + attr.act_cell +            \
            (attr.use_peephole ? "p" : "n"));                          \
    if (useJIT(attr.d)) {                                              \
      /* only jit code need record d*/                                 \
      return key + "jit" + std::to_string(attr.d);                     \
    } else if (useMKL(attr.d)) {                                       \
      return key + "mkl";                                              \
    } else {                                                           \
      return key + "any";                                              \
    }                                                                  \
  }                                                                    \
  template <>                                                          \
  std::string ker_class##Impl<double>::name(const lstm_attr_t& attr) { \
    std::string key(#ker_key "d");                                     \
    /* jit code do not support double yet*/                            \
    if (useMKL(attr.d)) {                                              \
      return key + "mkl";                                              \
    } else {                                                           \
      return key + "any";                                              \
    }                                                                  \
  }

#define JITKERNEL_DECLARE_LSTM(ker_class, ker_dtype)          \
  template <>                                                 \
  std::shared_ptr<const LSTMKernel<ker_dtype>>                \
  KernelPool::Get<LSTMKernel<ker_dtype>, const lstm_attr_t&>( \
      const lstm_attr_t& attr)

#define JITKERNEL_FIND_KEY_LSTM(ker_class, ker_dtype) \
  std::string key = ker_class##Impl<ker_dtype>::name(attr)

#define JITKERNEL_LSTM_IMPL(ker, dtype)                     \
  if (attr.use_peephole) {                                  \
    p = std::dynamic_pointer_cast<ker<dtype>>(              \
        std::make_shared<PeepholeKernelImpl<dtype>>(attr)); \
  } else {                                                  \
    p = std::dynamic_pointer_cast<ker<dtype>>(              \
        std::make_shared<ker##Impl<dtype>>(attr));          \
  }

REGISTER_JITKERNEL_ARGS(lstm, LSTMKernel, JITKERNEL_DEFINE_NAME_LSTM,
                        JITKERNEL_DECLARE_LSTM, JITKERNEL_FIND_KEY_LSTM,
                        JITKERNEL_LSTM_IMPL);

#undef JITKERNEL_LSTM_IMPL
#undef JITKERNEL_FIND_KEY_LSTM
#undef JITKERNEL_DECLARE_LSTM
#undef JITKERNEL_DEFINE_NAME_LSTM

/* GRU JitKernel */
template <typename T>
class GRUKernelImpl : public GRUKernel<T> {
 public:
  static inline std::string name(const gru_attr_t& attr) {
    PADDLE_THROW("DType should be either float or double");
  }
  static inline bool useJIT(int d) { return false; }
  static inline bool useMKL(int d) { return false; }
  explicit GRUKernelImpl(const gru_attr_t& attr) : GRUKernel<T>() {
#ifdef PADDLE_WITH_XBYAK
    if (useJIT(attr.d)) {
      size_t sz = 96 + attr.d / YMM_FLOAT_BLOCK * 96 * 2 * 8;
      jitcode0_.reset(new gen::GRUJitCode(0, attr, sz > 4096 ? sz : 4096));
      this->ComputeH1 =
          jitcode0_->getCode<void (*)(gru_t*, const gru_attr_t*)>();

      jitcode1_.reset(new gen::GRUJitCode(1, attr, sz > 4096 ? sz : 4096));
      this->ComputeHtPart1 =
          jitcode1_->getCode<void (*)(gru_t*, const gru_attr_t*)>();

      jitcode2_.reset(new gen::GRUJitCode(2, attr, sz > 4096 ? sz : 4096));
      this->ComputeHtPart2 =
          jitcode2_->getCode<void (*)(gru_t*, const gru_attr_t*)>();
      return;
    }
#endif
    this->ComputeH1 = refer::GRUH1<T>;
    this->ComputeHtPart1 = refer::GRUHtPart1<T>;
    this->ComputeHtPart2 = refer::GRUHtPart2<T>;
  }
#ifdef PADDLE_WITH_XBYAK

 private:
  std::unique_ptr<gen::GRUJitCode> jitcode0_{nullptr}, jitcode1_{nullptr},
      jitcode2_{nullptr};
#endif
};

#ifdef PADDLE_WITH_XBYAK
template <>
bool GRUKernelImpl<float>::useJIT(int d) {
  return gen::GRUJitCode::init(d);
}
#endif

#define JITKERNEL_DEFINE_NAME_GRU(ker_key, ker_class)                 \
  template <>                                                         \
  std::string ker_class##Impl<float>::name(const gru_attr_t& attr) {  \
    std::string key(#ker_key "f");                                    \
    key += (attr.act_gate + attr.act_cand);                           \
    if (useJIT(attr.d)) {                                             \
      /* only jit code need record d*/                                \
      return key + "jit" + std::to_string(attr.d);                    \
    } else if (useMKL(attr.d)) {                                      \
      return key + "mkl";                                             \
    } else {                                                          \
      return key + "any";                                             \
    }                                                                 \
  }                                                                   \
  template <>                                                         \
  std::string ker_class##Impl<double>::name(const gru_attr_t& attr) { \
    std::string key(#ker_key "d");                                    \
    /* jit code do not support double yet*/                           \
    if (useMKL(attr.d)) {                                             \
      return key + "mkl";                                             \
    } else {                                                          \
      return key + "any";                                             \
    }                                                                 \
  }

#define JITKERNEL_DECLARE_GRU(ker_class, ker_dtype)         \
  template <>                                               \
  std::shared_ptr<const ker_class<ker_dtype>>               \
  KernelPool::Get<ker_class<ker_dtype>, const gru_attr_t&>( \
      const gru_attr_t& attr)

#define JITKERNEL_FIND_KEY_GRU(ker_class, ker_dtype) \
  std::string key = ker_class##Impl<ker_dtype>::name(attr)

#define JITKERNEL_GRU_IMPL(ker, dtype)       \
  p = std::dynamic_pointer_cast<ker<dtype>>( \
      std::make_shared<ker##Impl<dtype>>(attr));

REGISTER_JITKERNEL_ARGS(gru, GRUKernel, JITKERNEL_DEFINE_NAME_GRU,
                        JITKERNEL_DECLARE_GRU, JITKERNEL_FIND_KEY_GRU,
                        JITKERNEL_GRU_IMPL);

#undef JITKERNEL_GRU_IMPL
#undef JITKERNEL_FIND_KEY_GRU
#undef JITKERNEL_DECLARE_GRU
#undef JITKERNEL_DEFINE_NAME_GRU
}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
