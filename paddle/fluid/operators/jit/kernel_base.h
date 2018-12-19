/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/jit/macro.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace jit {

typedef enum {
  non_kernel = 0,
  vmul = 1,
  vadd = 2,
  vaddrelu,
  vsub,
  vscal,
  vaddbias,
  vrelu,
  videntity,
  vexp,
  vsigmoid,
  vtanh,
  lstmctht,
  lstmc1h1,
  gruh1,
  gruhtpart1,
  gruhtpart2,
  crfdecoding,
  layernorm,
  nchw16cmulnc,
} KernelType;

template <typename T>
struct XYZNTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int);
};

template <typename T>
struct AXYNTuples : public XYZNTuples<T> {};

template <typename T>
struct XYNTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, T*, int);
};

typedef struct {
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
  explicit lstm_attr_s(int _d, KernelType _act_gate, KernelType _act_cand,
                       KernelType _act_cell, bool _use_peephole = false)
      : rnn_attr_s(_d, _act_gate, _act_cand),
        use_peephole(_use_peephole),
        act_cell(_act_cell) {}
};

typedef struct rnn_attr_s gru_attr_t;
typedef struct lstm_attr_s lstm_attr_t;

template <typename T>
struct LSTMTuples {
  typedef T data_type;
  typedef lstm_attr_t attr_type;
  typedef void (*func_type)(lstm_t*, const lstm_attr_t*);
};

template <typename T>
struct GRUTuples {
  typedef T data_type;
  typedef gru_attr_t attr_type;
  typedef void (*func_type)(gru_t*, const gru_attr_t*);
};

template <typename T>
struct CRFDecodingTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const int, const T*, const T*, T*, int*, int);
};

template <typename T>
struct LayerNormTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(T*, T*, T*, T*, const T*, const T*, int,
                            const float, int);
};

// nChw16c = nChw16c .* NC
template <typename T>
struct NCHW16CMulNCTuples {
  typedef T data_type;
  typedef int attr_type;
  typedef void (*func_type)(const T*, const T*, T*, int, int);
};

// Just for adding to kernel pool without template
class Kernel {
 public:
  Kernel() = default;
  virtual ~Kernel() = default;
  DISABLE_COPY_AND_ASSIGN(Kernel);
};

template <typename KernelTuples>
class KernelImpl : public Kernel {
  // TODO(TJ): rename KernelImpl to KernelMore which seems only used in more
  // and add name interface for more implements easy for debug
 public:
  using T = typename KernelTuples::data_type;
  using Func = typename KernelTuples::func_type;
  using Attr = typename KernelTuples::attr_type;
  virtual Func GetFunc() const { return func; }
  // TODO(TJ): const &attr
  virtual bool UseMe(Attr attr) const = 0;

 protected:
  Func func{nullptr};
};

template <typename KernelTuples>
class ReferKernel : public KernelImpl<KernelTuples> {
 public:
  // Refer code can always be used
  bool UseMe(typename KernelTuples::attr_type attr) const override {
    return true;
  }
};

}  // namespace jit
}  // namespace operators
}  // namespace paddle
