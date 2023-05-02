/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <type_traits>

#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/lstm_compute.h"

#if defined(_WIN32)
#if defined(__AVX2__) || defined(__AVX__)
inline __m256 operator+=(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
#endif
#endif

namespace phi {
namespace funcs {
namespace detail {

using Array1 = Eigen::DSizes<int64_t, 1>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = phi::EigenVector<T, MajorType, IndexType>;

#if !defined(__NVCC__) && !defined(__HIPCC___)  // @{ Group LSTM CPU

template <class T, class Op>
void naive_lstm_forward_one_sequence(Op op,
                                     phi::funcs::LstmMetaValue<T> value,
                                     int frame_size,
                                     T cell_clip,
                                     ActivationType active_node,
                                     ActivationType active_gate,
                                     ActivationType active_state,
                                     bool old_api_version) {
  T r_value_in;
  T r_value_ig;
  T r_value_fg;
  T r_value_og;
  T r_checkI;
  T r_checkF;
  T r_checkO;
  T r_state;
  T r_prev_state = 0;
  T r_state_atv;
  T r_out;

  T *value_ig = value.gate_value;
  T *value_fg = value.gate_value + frame_size;
  T *value_in = value.gate_value + frame_size * 2;
  T *value_og = value.gate_value + frame_size * 3;
  if (old_api_version) {
    value_in = value.gate_value;
    value_ig = value.gate_value + frame_size;
    value_fg = value.gate_value + frame_size * 2;
  }

  for (int i = 0; i < frame_size; i++) {
    r_value_in = value_in[i];
    r_value_ig = value_ig[i];
    r_value_fg = value_fg[i];
    r_value_og = value_og[i];
    r_checkI = value.check_ig ? value.check_ig[i] : 0;
    r_checkF = value.check_fg ? value.check_fg[i] : 0;
    r_checkO = value.check_og ? value.check_og[i] : 0;

    if (value.prev_state_value) {
      r_prev_state = value.prev_state_value[i];
    }

    op(&r_value_in,
       &r_value_ig,
       &r_value_fg,
       &r_value_og,
       &r_prev_state,
       &r_state,
       &r_state_atv,
       &r_out,
       &r_checkI,
       &r_checkF,
       &r_checkO,
       &cell_clip,
       active_node,
       active_gate,
       active_state);

    value_in[i] = r_value_in;
    value_ig[i] = r_value_ig;
    value_fg[i] = r_value_fg;
    value_og[i] = r_value_og;
    value.state_value[i] = r_state;
    value.state_active_value[i] = r_state_atv;
    value.output_value[i] = r_out;
  }
}

template <class T, class Op>
void naive_lstm_backward_one_sequence(Op op,
                                      phi::funcs::LstmMetaValue<T> value,
                                      phi::funcs::LstmMetaGrad<T> grad,
                                      int frame_size,
                                      T cell_clip,
                                      ActivationType active_node,
                                      ActivationType active_gate,
                                      ActivationType active_state,
                                      bool old_api_version) {
  T r_value_in;
  T r_value_ig;
  T r_value_fg;
  T r_value_og;
  T r_grad_in;
  T r_grad_ig;
  T r_grad_fg;
  T r_grad_og;
  T r_prev_state = 0;
  T r_prev_state_grad;
  T r_state;
  T r_state_grad;
  T r_state_atv;
  T r_output_grad;
  T r_checkI;
  T r_checkF;
  T r_checkO;
  T r_checkIGrad;
  T r_checkFGrad;
  T r_checkOGrad;

  T *value_ig = value.gate_value;
  T *value_fg = value.gate_value + frame_size;
  T *value_in = value.gate_value + frame_size * 2;
  T *value_og = value.gate_value + frame_size * 3;
  if (old_api_version) {
    value_in = value.gate_value;
    value_ig = value.gate_value + frame_size;
    value_fg = value.gate_value + frame_size * 2;
  }

  T *grad_ig = grad.gate_grad;
  T *grad_fg = grad.gate_grad + frame_size;
  T *grad_in = grad.gate_grad + frame_size * 2;
  T *grad_og = grad.gate_grad + frame_size * 3;
  if (old_api_version) {
    grad_in = grad.gate_grad;
    grad_ig = grad.gate_grad + frame_size;
    grad_fg = grad.gate_grad + frame_size * 2;
  }

  for (int i = 0; i < frame_size; i++) {
    r_value_in = value_in[i];
    r_value_ig = value_ig[i];
    r_value_fg = value_fg[i];
    r_value_og = value_og[i];
    r_checkI = value.check_ig ? value.check_ig[i] : 0;
    r_checkF = value.check_fg ? value.check_fg[i] : 0;
    r_checkO = value.check_og ? value.check_og[i] : 0;
    r_state = value.state_value[i];
    r_state_atv = value.state_active_value[i];
    r_output_grad = grad.output_grad[i];
    r_state_grad = grad.state_grad[i];
    if (value.prev_state_value) {
      r_prev_state = value.prev_state_value[i];
    }

    op(&r_value_in,
       &r_value_ig,
       &r_value_fg,
       &r_value_og,
       &r_grad_in,
       &r_grad_ig,
       &r_grad_fg,
       &r_grad_og,
       &r_prev_state,
       &r_prev_state_grad,
       &r_state,
       &r_state_grad,
       &r_state_atv,
       &r_output_grad,
       &r_checkI,
       &r_checkF,
       &r_checkO,
       &r_checkIGrad,
       &r_checkFGrad,
       &r_checkOGrad,
       &cell_clip,
       active_node,
       active_gate,
       active_state);

    grad_in[i] = r_grad_in;
    grad_ig[i] = r_grad_ig;
    grad_fg[i] = r_grad_fg;
    grad_og[i] = r_grad_og;
    grad.state_grad[i] = r_state_grad;

    if (grad.prev_state_grad) grad.prev_state_grad[i] = r_prev_state_grad;
    if (value.prev_state_value) {
      if (grad.check_ig_grad) grad.check_ig_grad[i] += r_checkIGrad;
      if (grad.check_fg_grad) grad.check_fg_grad[i] += r_checkFGrad;
    }
    if (grad.check_og_grad) grad.check_og_grad[i] += r_checkOGrad;
  }
}

template <class T, class Op>
void avx_lstm_forward_one_sequence(Op op,
                                   phi::funcs::LstmMetaValue<T> value,
                                   int frame_size,
                                   T cell_clip,
                                   ActivationType active_node,
                                   ActivationType active_gate,
                                   ActivationType active_state,
                                   bool old_api_version) {
#ifdef __AVX__
  __m256 r_value_in;
  __m256 r_value_ig;
  __m256 r_value_fg;
  __m256 r_value_og;
  __m256 r_checkI = _mm256_set1_ps(0.0f);
  __m256 r_checkF = _mm256_set1_ps(0.0f);
  __m256 r_checkO = _mm256_set1_ps(0.0f);
  __m256 r_state;
  __m256 r_prev_state = _mm256_set1_ps(0.0f);
  __m256 r_state_atv;
  __m256 r_out;

  __m256 *value_ig = reinterpret_cast<__m256 *>(value.gate_value);
  __m256 *value_fg = reinterpret_cast<__m256 *>(value.gate_value + frame_size);
  __m256 *value_in =
      reinterpret_cast<__m256 *>(value.gate_value + frame_size * 2);
  __m256 *value_og =
      reinterpret_cast<__m256 *>(value.gate_value + frame_size * 3);
  if (old_api_version) {
    value_in = reinterpret_cast<__m256 *>(value.gate_value);
    value_ig = reinterpret_cast<__m256 *>(value.gate_value + frame_size);
    value_fg = reinterpret_cast<__m256 *>(value.gate_value + frame_size * 2);
  }

  for (int i = 0; i < frame_size / 8; i++) {
    r_value_in = value_in[i];
    r_value_ig = value_ig[i];
    r_value_fg = value_fg[i];
    r_value_og = value_og[i];
    if (value.check_ig) {
      r_checkI = (reinterpret_cast<__m256 *>(value.check_ig))[i];
      r_checkF = (reinterpret_cast<__m256 *>(value.check_fg))[i];
      r_checkO = (reinterpret_cast<__m256 *>(value.check_og))[i];
    }

    if (value.prev_state_value) {
      r_prev_state =
          (reinterpret_cast<__m256 const *>(value.prev_state_value))[i];
    }

    op(&r_value_in,
       &r_value_ig,
       &r_value_fg,
       &r_value_og,
       &r_prev_state,
       &r_state,
       &r_state_atv,
       &r_out,
       &r_checkI,
       &r_checkF,
       &r_checkO,
       &cell_clip,
       active_node,
       active_gate,
       active_state);

    value_in[i] = r_value_in;
    value_ig[i] = r_value_ig;
    value_fg[i] = r_value_fg;
    value_og[i] = r_value_og;
    (reinterpret_cast<__m256 *>(value.state_value))[i] = r_state;
    (reinterpret_cast<__m256 *>(value.state_active_value))[i] = r_state_atv;
    (reinterpret_cast<__m256 *>(value.output_value))[i] = r_out;
  }
#endif
}

template <class T, class Op>
void avx_lstm_backward_one_sequence(Op op,
                                    phi::funcs::LstmMetaValue<T> value,
                                    phi::funcs::LstmMetaGrad<T> grad,
                                    int frame_size,
                                    T cell_clip,
                                    ActivationType active_node,
                                    ActivationType active_gate,
                                    ActivationType active_state,
                                    bool old_api_version) {
#ifdef __AVX__
  __m256 r_value_in;
  __m256 r_value_ig;
  __m256 r_value_fg;
  __m256 r_value_og;
  __m256 r_grad_in;
  __m256 r_grad_ig;
  __m256 r_grad_fg;
  __m256 r_grad_og;
  __m256 r_prev_state = _mm256_set1_ps(0.0f);
  __m256 r_prev_state_grad;
  __m256 r_state_grad;
  __m256 r_state;
  __m256 r_state_atv;
  __m256 r_output_grad;
  __m256 r_checkI = _mm256_set1_ps(0.0f);
  __m256 r_checkF = _mm256_set1_ps(0.0f);
  __m256 r_checkO = _mm256_set1_ps(0.0f);
  __m256 r_checkIGrad;
  __m256 r_checkFGrad;
  __m256 r_checkOGrad;

  __m256 *value_ig = reinterpret_cast<__m256 *>(value.gate_value);
  __m256 *value_fg = reinterpret_cast<__m256 *>(value.gate_value + frame_size);
  __m256 *value_in =
      reinterpret_cast<__m256 *>(value.gate_value + frame_size * 2);
  __m256 *value_og =
      reinterpret_cast<__m256 *>(value.gate_value + frame_size * 3);
  if (old_api_version) {
    value_in = reinterpret_cast<__m256 *>(value.gate_value);
    value_ig = reinterpret_cast<__m256 *>(value.gate_value + frame_size);
    value_fg = reinterpret_cast<__m256 *>(value.gate_value + frame_size * 2);
  }

  __m256 *grad_ig = reinterpret_cast<__m256 *>(grad.gate_grad);
  __m256 *grad_fg = reinterpret_cast<__m256 *>(grad.gate_grad + frame_size);
  __m256 *grad_in = reinterpret_cast<__m256 *>(grad.gate_grad + frame_size * 2);
  __m256 *grad_og = reinterpret_cast<__m256 *>(grad.gate_grad + frame_size * 3);
  if (old_api_version) {
    grad_in = reinterpret_cast<__m256 *>(grad.gate_grad);
    grad_ig = reinterpret_cast<__m256 *>(grad.gate_grad + frame_size);
    grad_fg = reinterpret_cast<__m256 *>(grad.gate_grad + frame_size * 2);
  }

  for (int i = 0; i < frame_size / 8; i++) {
    r_value_in = value_in[i];
    r_value_ig = value_ig[i];
    r_value_fg = value_fg[i];
    r_value_og = value_og[i];
    if (value.check_ig) {
      r_checkI = (reinterpret_cast<__m256 *>(value.check_ig))[i];
      r_checkF = (reinterpret_cast<__m256 *>(value.check_fg))[i];
      r_checkO = (reinterpret_cast<__m256 *>(value.check_og))[i];
    }
    r_state = (reinterpret_cast<__m256 *>(value.state_value))[i];
    r_state_atv = (reinterpret_cast<__m256 *>(value.state_active_value))[i];
    r_output_grad = (reinterpret_cast<__m256 *>(grad.output_grad))[i];
    r_state_grad = (reinterpret_cast<__m256 *>(grad.state_grad))[i];
    if (value.prev_state_value) {
      r_prev_state =
          (reinterpret_cast<__m256 const *>(value.prev_state_value))[i];
    }

    op(&r_value_in,
       &r_value_ig,
       &r_value_fg,
       &r_value_og,
       &r_grad_in,
       &r_grad_ig,
       &r_grad_fg,
       &r_grad_og,
       &r_prev_state,
       &r_prev_state_grad,
       &r_state,
       &r_state_grad,
       &r_state_atv,
       &r_output_grad,
       &r_checkI,
       &r_checkF,
       &r_checkO,
       &r_checkIGrad,
       &r_checkFGrad,
       &r_checkOGrad,
       &cell_clip,
       active_node,
       active_gate,
       active_state);

    grad_in[i] = r_grad_in;
    grad_ig[i] = r_grad_ig;
    grad_fg[i] = r_grad_fg;
    grad_og[i] = r_grad_og;
    (reinterpret_cast<__m256 *>(grad.state_grad))[i] = r_state_grad;

    if (grad.prev_state_grad)
      (reinterpret_cast<__m256 *>(grad.prev_state_grad))[i] = r_prev_state_grad;
    if (value.prev_state_value) {
      if (grad.check_ig_grad)
        (reinterpret_cast<__m256 *>(grad.check_ig_grad))[i] += r_checkIGrad;
      if (grad.check_fg_grad)
        (reinterpret_cast<__m256 *>(grad.check_fg_grad))[i] += r_checkFGrad;
    }
    if (grad.check_og_grad)
      (reinterpret_cast<__m256 *>(grad.check_og_grad))[i] += r_checkOGrad;
  }
#endif
}

template <class T, class Context>
void eigen_lstm_forward_one_sequence(const Context &context,
                                     phi::funcs::LstmMetaValue<T> value,
                                     int frame_size) {
  auto eigen_value_ig =
      typename EigenVector<T>::Type(value.gate_value, Array1(frame_size));
  auto eigen_value_fg = typename EigenVector<T>::Type(
      value.gate_value + frame_size, Array1(frame_size));
  auto eigen_value_in = typename EigenVector<T>::Type(
      value.gate_value + frame_size * 2, Array1(frame_size));
  auto eigen_value_og = typename EigenVector<T>::Type(
      value.gate_value + frame_size * 3, Array1(frame_size));
  auto eigen_state =
      typename EigenVector<T>::Type(value.state_value, Array1(frame_size));
  auto eigen_state_act = typename EigenVector<T>::Type(value.state_active_value,
                                                       Array1(frame_size));
  auto eigen_output =
      typename EigenVector<T>::Type(value.output_value, Array1(frame_size));

  auto &place = *context.eigen_device();
  TanhFunctor<T>()(place, eigen_value_in, eigen_value_in);
  SigmoidFunctor<T>()(place, eigen_value_ig, eigen_value_ig);
  SigmoidFunctor<T>()(place, eigen_value_fg, eigen_value_fg);
  SigmoidFunctor<T>()(place, eigen_value_og, eigen_value_og);

  eigen_state.device(place) = eigen_value_in * eigen_value_ig;
  if (value.prev_state_value) {
    auto eigen_prev_state = typename EigenVector<T>::ConstType(
        value.prev_state_value, Array1(frame_size));
    eigen_state.device(place) = eigen_state + eigen_prev_state * eigen_value_fg;
  }

  TanhFunctor<T>()(place, eigen_state, eigen_state_act);
  eigen_output.device(place) = eigen_value_og * eigen_state_act;
}

template <class T, class Context>
void eigen_lstm_backward_one_sequence(const Context &context,
                                      phi::funcs::LstmMetaValue<T> value,
                                      phi::funcs::LstmMetaGrad<T> grad,
                                      int frame_size) {
  auto eigen_value_ig =
      typename EigenVector<T>::Type(value.gate_value, Array1(frame_size));
  auto eigen_value_fg = typename EigenVector<T>::Type(
      value.gate_value + frame_size, Array1(frame_size));
  auto eigen_value_in = typename EigenVector<T>::Type(
      value.gate_value + frame_size * 2, Array1(frame_size));
  auto eigen_value_og = typename EigenVector<T>::Type(
      value.gate_value + frame_size * 3, Array1(frame_size));
  auto eigen_state_act = typename EigenVector<T>::Type(value.state_active_value,
                                                       Array1(frame_size));

  auto eigen_grad_ig =
      typename EigenVector<T>::Type(grad.gate_grad, Array1(frame_size));
  auto eigen_grad_fg = typename EigenVector<T>::Type(
      grad.gate_grad + frame_size, Array1(frame_size));
  auto eigen_grad_in = typename EigenVector<T>::Type(
      grad.gate_grad + frame_size * 2, Array1(frame_size));
  auto eigen_grad_og = typename EigenVector<T>::Type(
      grad.gate_grad + frame_size * 3, Array1(frame_size));
  auto eigen_grad_output =
      typename EigenVector<T>::Type(grad.output_grad, Array1(frame_size));
  auto eigen_grad_state =
      typename EigenVector<T>::Type(grad.state_grad, Array1(frame_size));

  auto &place = *context.eigen_device();
  SigmoidGradFunctor<T>()(place,
                          1 /*useless*/,
                          eigen_value_og,
                          eigen_grad_output * eigen_state_act,
                          eigen_grad_og);
  eigen_grad_state.device(place) =
      eigen_grad_state +
      eigen_grad_output * eigen_value_og *
          (static_cast<T>(1) - eigen_state_act * eigen_state_act);
  TanhGradFunctor<T>()(place,
                       1,
                       eigen_value_in,
                       eigen_grad_state * eigen_value_ig,
                       eigen_grad_in);
  SigmoidGradFunctor<T>()(place,
                          1,
                          eigen_value_ig,
                          eigen_grad_state * eigen_value_in,
                          eigen_grad_ig);
  if (value.prev_state_value) {
    auto eigen_prev_state = typename EigenVector<T>::ConstType(
        value.prev_state_value, Array1(frame_size));
    SigmoidGradFunctor<T>()(place,
                            1,
                            eigen_value_fg,
                            eigen_grad_state * eigen_prev_state,
                            eigen_grad_fg);
  } else {
    SigmoidGradFunctor<T>()(place, 1, eigen_value_fg, 0, eigen_grad_fg);
  }
  if (grad.prev_state_grad) {
    auto eigen_grad_pre_state =
        typename EigenVector<T>::Type(grad.prev_state_grad, Array1(frame_size));
    eigen_grad_pre_state.device(place) = eigen_grad_state * eigen_value_fg;
  }
}

template <class T, class Op, class Context>
void cpu_lstm_forward(const Context &context,
                      Op op,
                      phi::funcs::LstmMetaValue<T> value,
                      int frame_size,
                      T cell_clip,
                      ActivationType active_node,
                      ActivationType active_gate,
                      ActivationType active_state,
                      bool old_api_version) {
  if (!old_api_version) {
    eigen_lstm_forward_one_sequence<T>(context, value, frame_size);
  } else {
    if (Op::avx && !(frame_size & (8 - 1)) && (std::is_same<T, float>::value)) {
      avx_lstm_forward_one_sequence<T>(op,
                                       value,
                                       frame_size,
                                       cell_clip,
                                       active_node,
                                       active_gate,
                                       active_state,
                                       old_api_version);
    } else {
      naive_lstm_forward_one_sequence<T>(op,
                                         value,
                                         frame_size,
                                         cell_clip,
                                         active_node,
                                         active_gate,
                                         active_state,
                                         old_api_version);
    }
  }
}

template <class T, class Op, class Context>
void cpu_lstm_backward(const Context &context,
                       Op op,
                       phi::funcs::LstmMetaValue<T> value,
                       phi::funcs::LstmMetaGrad<T> grad,
                       int frame_size,
                       T cell_clip,
                       ActivationType active_node,
                       ActivationType active_gate,
                       ActivationType active_state,
                       bool old_api_version) {
  if (!old_api_version) {
    eigen_lstm_backward_one_sequence<T>(context, value, grad, frame_size);
  } else {
    if (Op::avx && !(frame_size & (8 - 1)) && (std::is_same<T, float>::value)) {
      avx_lstm_backward_one_sequence<T>(op,
                                        value,
                                        grad,
                                        frame_size,
                                        cell_clip,
                                        active_node,
                                        active_gate,
                                        active_state,
                                        old_api_version);
    } else {
      naive_lstm_backward_one_sequence<T>(op,
                                          value,
                                          grad,
                                          frame_size,
                                          cell_clip,
                                          active_node,
                                          active_gate,
                                          active_state,
                                          old_api_version);
    }
  }
}

#endif  // @{ End Group LSTM CPU

}  // namespace detail
}  // namespace funcs
}  // namespace phi
