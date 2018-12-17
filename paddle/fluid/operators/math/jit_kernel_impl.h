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
#include <string>
#include <type_traits>

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0
#define XMM_FLOAT_BLOCK 4
#define YMM_FLOAT_BLOCK 8
#define ZMM_FLOAT_BLOCK 16

typedef struct {
  void* gates;  // gates: W_ch, W_ih, W_fh, W_oh
  const void* ct_1;
  void* ct;
  void* ht;
  /* weight_peephole and checked data are only used in peephole*/
  const void* wp{nullptr};
  void* checked{nullptr};
} lstm_t;

typedef struct {
  void* gates;  // gates: {W_update, W_reset; W_state}
  const void* ht_1;
  void* ht;
} gru_t;

struct rnn_attr_s {
  int d;
  std::string act_gate, act_cand;
  rnn_attr_s() = default;
  rnn_attr_s(int _d, const std::string& _act_gate, const std::string& _act_cand)
      : d(_d), act_gate(_act_gate), act_cand(_act_cand) {}
};

struct lstm_attr_s : public rnn_attr_s {
  bool use_peephole;
  std::string act_cell;
  lstm_attr_s() = default;
  lstm_attr_s(int _d, const std::string& _act_gate,
              const std::string& _act_cand, const std::string& _act_cell,
              bool _use_peephole = false)
      : rnn_attr_s(_d, _act_gate, _act_cand),
        use_peephole(_use_peephole),
        act_cell(_act_cell) {}
};

typedef struct rnn_attr_s gru_attr_t;
typedef struct lstm_attr_s lstm_attr_t;

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
