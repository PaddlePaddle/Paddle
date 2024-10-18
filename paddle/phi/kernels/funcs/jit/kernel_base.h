/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once
#include <cstdint>

#include "paddle/common/macros.h"
#include "paddle/phi/kernels/funcs/jit/macro.h"

namespace phi {
namespace jit {

typedef enum {
  kNone = 0,
  // sort by alphabet
  kAdam = 1,
  kAdamW,
  kCRFDecoding,
  kEmbSeqPool,
  kGRUH1,
  kGRUHtPart1,
  kGRUHtPart2,
  kLSTMCtHt,
  kLSTMC1H1,
  kLayerNorm,
  kMatMul,
  kSeqPool,
  kVAdd,
  kVAddBias,
  kVAddRelu,
  kVBroadcast,
  kVCopy,
  kVExp,
  kVIdentity,
  kVMul,
  kVRelu,
  kVScal,
  kSgd,
  kVSigmoid,
  kVSquare,
  kVSub,
  kVTanh,
} KernelType;

typedef enum {
  kNonePoolType = 0,
  kSum = 1,
  kAvg,
  kSqrt,
} SeqPoolType;

// x, y, z, n
template <typename T>
struct XYZNTuple {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int);
};

// a, x, y, n
template <typename T>
struct AXYNTuple : public XYZNTuple<T> {};

// a, x, y, n, stride
template <typename T>
struct AXYNSTuple {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int, int);
};

// x, y, n
template <typename T>
struct XYNTuple {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, T*, int);
};

// x, returned value, n, stride
template <typename T>
struct XRNSTuple {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, T*, int, int);
};

#define DECLARE_KERNELTUPLE(kernel_tuple, type)        \
  template <typename T>                                \
  struct type##Tuple : public kernel_tuple<T> {        \
    static constexpr KernelType kernel_type = k##type; \
  }

// Tuple should be corresponding to the KernelType
DECLARE_KERNELTUPLE(XYZNTuple, VMul);
DECLARE_KERNELTUPLE(XYZNTuple, VAdd);
DECLARE_KERNELTUPLE(XYZNTuple, VAddRelu);
DECLARE_KERNELTUPLE(XYZNTuple, VSub);

DECLARE_KERNELTUPLE(AXYNTuple, VScal);
DECLARE_KERNELTUPLE(AXYNTuple, VAddBias);

DECLARE_KERNELTUPLE(XYNTuple, VRelu);
DECLARE_KERNELTUPLE(XYNTuple, VIdentity);
DECLARE_KERNELTUPLE(XYNTuple, VSquare);
DECLARE_KERNELTUPLE(XYNTuple, VExp);
DECLARE_KERNELTUPLE(XYNTuple, VSigmoid);
DECLARE_KERNELTUPLE(XYNTuple, VTanh);
DECLARE_KERNELTUPLE(XYNTuple, VCopy);

typedef struct lstm_t {
  void* gates;  // gates: x_ch, x_ih, x_fh, x_oh
  const void* ct_1;
  void* ct;
  void* ht;
  /* weight_peephole and checked data are only used in peephole*/
  const void* wp{nullptr};  //  W_ic, W_fc, W_oc
  void* checked{nullptr};   // size: 2 * d
} lstm_t;

typedef struct {
  void* gates;  // gates: {x_update, x_reset; x_state}
  const void* ht_1;
  void* ht;
} gru_t;

struct rnn_attr_s {
  int d;
  KernelType act_gate, act_cand;
  rnn_attr_s() = default;
  explicit rnn_attr_s(int _d, KernelType _act_gate, KernelType _act_cand)
      : d(_d), act_gate(_act_gate), act_cand(_act_cand) {}
};

struct lstm_attr_s : public rnn_attr_s {
  bool use_peephole;
  KernelType act_cell;
  lstm_attr_s() = default;
  explicit lstm_attr_s(int _d,
                       KernelType _act_gate,
                       KernelType _act_cand,
                       KernelType _act_cell,
                       bool _use_peephole = false)
      : rnn_attr_s(_d, _act_gate, _act_cand),
        use_peephole(_use_peephole),
        act_cell(_act_cell) {}
};

typedef struct rnn_attr_s gru_attr_t;
typedef struct lstm_attr_s lstm_attr_t;

template <typename T>
struct LSTMTuple {
  typedef T data_type;
  typedef lstm_attr_t attr_type;
  typedef void (*func_type)(lstm_t*, const lstm_attr_t*);
};

template <typename T>
struct GRUTuple {
  typedef T data_type;
  typedef gru_attr_t attr_type;
  typedef void (*func_type)(gru_t*, const gru_attr_t*);
};

DECLARE_KERNELTUPLE(LSTMTuple, LSTMCtHt);
DECLARE_KERNELTUPLE(LSTMTuple, LSTMC1H1);

DECLARE_KERNELTUPLE(GRUTuple, GRUH1);
DECLARE_KERNELTUPLE(GRUTuple, GRUHtPart1);
DECLARE_KERNELTUPLE(GRUTuple, GRUHtPart2);

#undef DECLARE_KERNELTUPLE

template <typename T>
struct VBroadcastTuple {
  static constexpr KernelType kernel_type = kVBroadcast;
  typedef T data_type;
  typedef int64_t attr_type;
  typedef void (*func_type)(const T*, T*, int64_t, int64_t);
};

typedef struct seq_pool_attr_s {
  int h, w;  // h should always be the first one
  SeqPoolType type;
  seq_pool_attr_s() = default;
  explicit seq_pool_attr_s(int width, SeqPoolType pool_type, int height = 1)
      : h(height), w(width), type(pool_type) {}
} seq_pool_attr_t;

template <typename T>
struct SeqPoolTuple {
  static constexpr KernelType kernel_type = kSeqPool;
  typedef T data_type;
  typedef seq_pool_attr_t attr_type;
  typedef void (*func_type)(const T*, T*, const seq_pool_attr_t*);
};

typedef struct emb_seq_pool_attr_s {
  int64_t table_height, table_width;
  int64_t index_height, index_width;
  int64_t out_width;
  SeqPoolType pool_type;
  emb_seq_pool_attr_s() = default;
  explicit emb_seq_pool_attr_s(int64_t tbl_height,
                               int64_t tbl_width,
                               int64_t idx_height,
                               int64_t idx_width,
                               int64_t output_width,
                               SeqPoolType seqpool_type = SeqPoolType::kSum)
      : table_height(tbl_height),
        table_width(tbl_width),
        index_height(idx_height),
        index_width(idx_width),
        out_width(output_width),
        pool_type(seqpool_type) {}
} emb_seq_pool_attr_t;

template <typename T>
struct EmbSeqPoolTuple {
  static constexpr KernelType kernel_type = kEmbSeqPool;
  typedef T data_type;
  typedef emb_seq_pool_attr_t attr_type;
  typedef void (*func_type)(const T*,
                            const int64_t*,
                            T*,
                            const emb_seq_pool_attr_t*);
};

typedef struct sgd_attr_s {
  int64_t param_height, param_width;
  int64_t grad_height, grad_width;
  int64_t selected_rows_size;
  sgd_attr_s() = default;
  explicit sgd_attr_s(int64_t param_h,
                      int64_t param_w,
                      int64_t grad_h,
                      int64_t grad_w,
                      int64_t selected_rows_sz)
      : param_height(param_h),
        param_width(param_w),
        grad_height(grad_h),
        grad_width(grad_w),
        selected_rows_size(selected_rows_sz) {}
} sgd_attr_t;

template <typename T>
struct SgdTuple {
  static constexpr KernelType kernel_type = kSgd;
  typedef T data_type;
  typedef sgd_attr_t attr_type;
  typedef void (*func_type)(
      const T*, const T*, const T*, const int64_t*, T*, const sgd_attr_t*);
};

typedef struct adam_attr_s {
  float beta1, beta2;
  bool amsgrad;
  adam_attr_s() = default;
  explicit adam_attr_s(float beta1, float beta2, bool amsgrad)
      : beta1(beta1), beta2(beta2), amsgrad(amsgrad) {}
} adam_attr_t;

template <typename T>
struct AdamTuple {
  static constexpr KernelType kernel_type = kAdam;
  typedef T data_type;
  typedef adam_attr_t attr_type;
  typedef void (*func_type)(T,
                            T,
                            T,
                            T,
                            int64_t,
                            const T*,
                            const T*,
                            const T*,
                            const T*,
                            const T*,
                            T*,
                            T*,
                            T*,
                            T*,
                            bool);
};

typedef struct adamw_attr_s {
  float beta1, beta2, coeff;
  bool amsgrad;
  adamw_attr_s() = default;
  explicit adamw_attr_s(float beta1, float beta2, float coeff, bool amsgrad)
      : beta1(beta1), beta2(beta2), coeff(coeff), amsgrad(amsgrad) {}
} adamw_attr_t;

template <typename T>
struct AdamWTuple {
  static constexpr KernelType kernel_type = kAdamW;
  typedef T data_type;
  typedef adamw_attr_t attr_type;
  typedef void (*func_type)(T,
                            T,
                            T,
                            T,
                            T,
                            T,
                            T,
                            int64_t,
                            const T*,
                            const T*,
                            const T*,
                            const T*,
                            const T*,
                            T*,
                            T*,
                            T*,
                            T*,
                            bool);
};

typedef struct matmul_attr_s {
  int m, n, k;
  void* packed_weight{nullptr};
  matmul_attr_s() = default;
  explicit matmul_attr_s(int m_, int n_, int k_, void* packed_weight_ = nullptr)
      : m(m_), n(n_), k(k_), packed_weight(packed_weight_) {}
} matmul_attr_t;

template <typename T>
struct MatMulTuple {
  static constexpr KernelType kernel_type = kMatMul;
  typedef T data_type;
  typedef matmul_attr_t attr_type;
  typedef void (*func_type)(const T*, const T*, T*, const matmul_attr_t*);
};

template <typename T>
struct CRFDecodingTuple {
  static constexpr KernelType kernel_type = kCRFDecoding;
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const int, const T*, const T*, T*, int*, int);
};

template <typename T>
struct LayerNormTuple {
  static constexpr KernelType kernel_type = kLayerNorm;
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(
      T*, T*, T*, T*, const T*, const T*, int, const float, int);
};

// Just for adding to kernel pool without template
class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
  virtual const char* ImplType() const = 0;
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

template <typename KernelTuple>
class KernelMore : public Kernel {
 public:
  using T = typename KernelTuple::data_type;
  using Func = typename KernelTuple::func_type;
  using Attr = typename KernelTuple::attr_type;
  virtual Func GetFunc() const { return func; }
  // specify this kernel can be used, means it should not fail if use it.
  virtual bool CanBeUsed(const Attr& attr) const = 0;

 protected:
  Func func{nullptr};
};

template <typename KernelTuple>
class ReferKernel : public KernelMore<KernelTuple> {
 public:
  // Refer code can always be used
  bool CanBeUsed(
      const typename KernelTuple::attr_type& attr UNUSED) const override {
    return true;
  }
  const char* ImplType() const override { return "Refer"; }
};

}  // namespace jit
}  // namespace phi
