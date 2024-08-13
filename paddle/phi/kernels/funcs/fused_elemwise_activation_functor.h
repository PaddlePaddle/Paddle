// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/phi/kernels/funcs/compound_functors.h"
#include "paddle/phi/kernels/funcs/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"

namespace phi {
namespace funcs {

static inline bool IsBcastY(const phi::DDim &x_dim, const phi::DDim &y_dim) {
  bool bcast_y = x_dim.size() >= y_dim.size();
  if (x_dim.size() == y_dim.size()) {
    for (int i = 0; i < x_dim.size(); ++i) {
      if (x_dim[i] < y_dim[i]) {
        bcast_y = false;
        break;
      }
    }
  }
  return bcast_y;
}

/**
 * Whether the compound function is Unary(Binary(X, Y)).
 * For Unary(Binary(X, Y)), the intermediate_out's shape is the same the final
 * out.
 */
static inline bool IsUnaryCompound(
    const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(),
      2,
      common::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(),
          2));
  static std::unordered_set<std::string> binary_fun = {"elementwise_add",
                                                       "elementwise_mul",
                                                       "elementwise_add_grad",
                                                       "elementwise_mul_grad"};
  return binary_fun.count(functor_list[1]) != 0;
}

/**
 *  For the in-place unary functor, the inputs of op_desc only have Out and
 *  Out@Grad.
 */
static inline bool HasInPlaceUnary(
    const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(),
      2,
      common::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(),
          2));
  static std::unordered_set<std::string> InplaceOpSet = {"relu", "relu_grad"};
  bool is_in_place = false;
  for (auto &func_name : functor_list) {
    is_in_place |= (InplaceOpSet.count(func_name) == 1);
  }
  return is_in_place;
}

/**
 * Whether the Input(X) could be absent.
 */
static inline bool InputXCanBeAbsent(
    const std::vector<std::string> &functor_list) {
  PADDLE_ENFORCE_EQ(
      functor_list.size(),
      2,
      common::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functor_list.size(),
          2));
  static std::unordered_set<std::string> binary_fun = {"elementwise_add_grad"};
  return binary_fun.count(functor_list[0]) != 0 ||
         binary_fun.count(functor_list[1]) != 0;
}

/*
 * Whether the compound function is supported.
 * For Unary(Binary(X, Y)), the intermediate_out's shape is the same the final
 * out.
 */
static bool IsSupportedCompound(const std::vector<std::string> &functors) {
  PADDLE_ENFORCE_EQ(
      functors.size(),
      2UL,
      common::errors::InvalidArgument(
          "Invalid functor list size %d, which should be equal to %d.",
          functors.size(),
          2));

  static std::unordered_set<std::string> unary_fun = {
      "scale", "relu", "tanh", "sigmoid", "gelu"};
  static std::unordered_set<std::string> binary_fun = {"elementwise_add",
                                                       "elementwise_mul"};

  std::string unary_fun_str;
  if (binary_fun.count(functors[0])) {
    unary_fun_str = functors[1];
  } else if (binary_fun.count(functors[1])) {
    unary_fun_str = functors[0];
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "%s and %s are not included in fused_list.", functors[0], functors[1]));
  }
  PADDLE_ENFORCE_EQ(unary_fun.count(unary_fun_str),
                    1,
                    common::errors::InvalidArgument(
                        "%s is not included in fused_list.", unary_fun_str));
  return true;
}

template <typename DeviceContext,
          typename T,
          typename BinaryFunctor,
          typename UnaryFunctor>
void RunBinaryCompoundFunctor(const DeviceContext &dev_ctx,
                              const BinaryFunctor &binary_functor,
                              const UnaryFunctor &unary_functor,
                              const phi::DenseTensor &in_x,
                              const phi::DenseTensor &in_y,
                              std::vector<phi::DenseTensor *> *outputs,
                              int axis,
                              bool save_intermediate_out) {
  // Z = Binary(X, Unary(Y))
  // intermediate_out = Unary(Y)
  // out = Binary(X, Unary(Y))
  // In this case, the shape of intermediate_out and out are different.
  phi::funcs::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>
      compound_func(binary_functor, unary_functor);
  if (save_intermediate_out) {
    phi::funcs::FusedElemwiseAndActComputeEx<
        DeviceContext,
        T,
        phi::funcs::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        true /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    phi::funcs::FusedElemwiseAndActComputeEx<
        DeviceContext,
        T,
        phi::funcs::BinaryCompoundFunctor<T, BinaryFunctor, UnaryFunctor>,
        false /*KeepIntermediateValue*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext,
          typename T,
          typename UnaryFunctor,
          typename BinaryFunctor>
void RunUnaryCompoundFunctors(const DeviceContext &dev_ctx,
                              const UnaryFunctor &unary_functor,
                              const BinaryFunctor &binary_functor,
                              const phi::DenseTensor &in_x,
                              const phi::DenseTensor &in_y,
                              std::vector<phi::DenseTensor *> *outputs,
                              int axis,
                              bool save_intermediate_out) {
  // Z = Unary(Binary(X, Y))
  // intermediate_out = Binary(X, Y)
  // out = Unary(Binary(X, Y))
  // In this case, the shape of intermediate_out and out are the same.

  phi::funcs::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>
      compound_func(unary_functor, binary_functor);

  if (save_intermediate_out) {
    phi::funcs::FusedElemwiseAndActComputeEx<
        DeviceContext,
        T,
        phi::funcs::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        true /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  } else {
    phi::funcs::FusedElemwiseAndActComputeEx<
        DeviceContext,
        T,
        phi::funcs::UnaryCompoundFunctor<T, UnaryFunctor, BinaryFunctor>,
        false /*KeepIntermediateValue*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx, in_x, in_y, axis, compound_func, (*outputs)[0], (*outputs)[1]);
  }
}

template <typename DeviceContext,
          typename T,
          typename BinaryGradFunctor,
          typename UnaryFunctor,
          typename UnaryGradFunctor,
          bool InPlace>
void RunBinaryCompoundGradFunctors(const DeviceContext &dev_ctx,
                                   const BinaryGradFunctor &binary_grad_functor,
                                   const UnaryFunctor &unary_functor,
                                   const UnaryGradFunctor &unary_grad_functor,
                                   const phi::DenseTensor *in_x,
                                   const phi::DenseTensor *in_y,
                                   const phi::DenseTensor *in_out,
                                   const phi::DenseTensor *in_intermediate_out,
                                   const phi::DenseTensor *in_out_grad,
                                   phi::DenseTensor *x_grad,
                                   phi::DenseTensor *y_grad,
                                   phi::DenseTensor *d_intermediate_out,
                                   int axis) {
  // Z = Binary(X, Unary(Y))

  using BinaryCompoundDxFunctor = phi::funcs::
      BinaryCompoundGradDxFunctor<T, BinaryGradFunctor, UnaryFunctor>;
  using BinaryCompoundDyFunctor =
      phi::funcs::BinaryCompoundGradDyFunctor<T,
                                              BinaryGradFunctor,
                                              UnaryFunctor,
                                              UnaryGradFunctor,
                                              InPlace>;
  using BinaryCompoundDIntermediateOutFunctor =
      phi::funcs::BinaryCompoundGradDIntermediateOutFunctor<T,
                                                            BinaryGradFunctor,
                                                            UnaryFunctor>;

  if (in_intermediate_out) {
    phi::funcs::FusedElemwiseAndActGradComputeEx<
        DeviceContext,
        T,
        BinaryCompoundDxFunctor,
        BinaryCompoundDyFunctor,
        BinaryCompoundDIntermediateOutFunctor,
        true /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        axis,
        x_grad,
        y_grad,
        d_intermediate_out,
        BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(
            binary_grad_functor, unary_functor, unary_grad_functor),
        BinaryCompoundDIntermediateOutFunctor(binary_grad_functor,
                                              unary_functor));
  } else {
    phi::funcs::FusedElemwiseAndActGradComputeEx<
        DeviceContext,
        T,
        BinaryCompoundDxFunctor,
        BinaryCompoundDyFunctor,
        BinaryCompoundDIntermediateOutFunctor,
        false /*UseIntermediateOut*/,
        false /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        axis,
        x_grad,
        y_grad,
        d_intermediate_out,
        BinaryCompoundDxFunctor(binary_grad_functor, unary_functor),
        BinaryCompoundDyFunctor(
            binary_grad_functor, unary_functor, unary_grad_functor),
        BinaryCompoundDIntermediateOutFunctor(binary_grad_functor,
                                              unary_functor));
  }
}

template <typename DeviceContext,
          typename T,
          typename UnaryGradFunctor,
          typename BinaryFunctor,
          typename BinaryGradFunctor,
          bool InPlace>
void RunUnaryCompoundGradFunctors(const DeviceContext &dev_ctx,
                                  const UnaryGradFunctor &unary_grad_functor,
                                  const BinaryFunctor &binary_functor,
                                  const BinaryGradFunctor &binary_grad_functor,
                                  const phi::DenseTensor *in_x,
                                  const phi::DenseTensor *in_y,
                                  const phi::DenseTensor *in_out,
                                  const phi::DenseTensor *in_intermediate_out,
                                  const phi::DenseTensor *in_out_grad,
                                  phi::DenseTensor *x_grad,
                                  phi::DenseTensor *y_grad,
                                  phi::DenseTensor *d_intermediate_out,
                                  int axis) {
  // Z = Unary(Binary(X, Y))

  using UnaryCompoundDxFunctor =
      phi::funcs::UnaryCompoundGradDxFunctor<T,
                                             UnaryGradFunctor,
                                             BinaryFunctor,
                                             BinaryGradFunctor,
                                             InPlace>;
  using UnaryCompoundDyFunctor =
      phi::funcs::UnaryCompoundGradDyFunctor<T,
                                             UnaryGradFunctor,
                                             BinaryFunctor,
                                             BinaryGradFunctor,
                                             InPlace>;
  using UnaryCompoundDIntermediateFunctor =
      phi::funcs::UnaryCompoundGradDIntermediateFunctor<T,
                                                        UnaryGradFunctor,
                                                        BinaryFunctor,
                                                        InPlace>;

  if (in_intermediate_out) {
    phi::funcs::FusedElemwiseAndActGradComputeEx<
        DeviceContext,
        T,
        UnaryCompoundDxFunctor,
        UnaryCompoundDyFunctor,
        UnaryCompoundDIntermediateFunctor,
        true /*UseIntermediateOut*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        axis,
        x_grad,
        y_grad,
        d_intermediate_out,
        UnaryCompoundDxFunctor(
            unary_grad_functor, binary_functor, binary_grad_functor),
        UnaryCompoundDyFunctor(
            unary_grad_functor, binary_functor, binary_grad_functor),
        UnaryCompoundDIntermediateFunctor(unary_grad_functor, binary_functor));
  } else {
    phi::funcs::FusedElemwiseAndActGradComputeEx<
        DeviceContext,
        T,
        UnaryCompoundDxFunctor,
        UnaryCompoundDyFunctor,
        UnaryCompoundDIntermediateFunctor,
        false /*UseIntermediateOut*/,
        true /*SameShapeOfIntermediateOutAndOut*/>(
        dev_ctx,
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        axis,
        x_grad,
        y_grad,
        d_intermediate_out,
        UnaryCompoundDxFunctor(
            unary_grad_functor, binary_functor, binary_grad_functor),
        UnaryCompoundDyFunctor(
            unary_grad_functor, binary_functor, binary_grad_functor),
        UnaryCompoundDIntermediateFunctor(unary_grad_functor, binary_functor));
  }
}

template <typename DeviceContext, typename T>
void RunFunctors(const DeviceContext &dev_ctx,
                 const phi::DenseTensor &in_x,
                 const phi::DenseTensor &in_y,
                 std::vector<phi::DenseTensor *> *outputs,
                 std::vector<std::string> functor_list,
                 float in_scale,
                 int axis,
                 bool save_intermediate_out) {
  auto &functors = functor_list;

  // TODO(zcd): The following code can be refined.
  auto funcs_str = functors[0] + "," + functors[1];
  if (funcs_str == "elementwise_add,scale") {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(in_scale);
    RunBinaryCompoundFunctor<DeviceContext,
                             T,
                             phi::funcs::AddFunctor<T>,
                             phi::funcs::ScaleFunctor<T>>(
        dev_ctx,
        phi::funcs::AddFunctor<T>(),
        phi::funcs::ScaleFunctor<T>(scale),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "scale,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(in_scale);
    RunUnaryCompoundFunctors<DeviceContext,
                             T,
                             phi::funcs::ScaleFunctor<T>,
                             phi::funcs::AddFunctor<T>>(
        dev_ctx,
        phi::funcs::ScaleFunctor<T>(scale),
        phi::funcs::AddFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "elementwise_add,relu") {
    // Z = Binary(X, Unary(Y))
    RunBinaryCompoundFunctor<DeviceContext,
                             T,
                             phi::funcs::AddFunctor<T>,
                             phi::funcs::ReluFunctor<T>>(
        dev_ctx,
        phi::funcs::AddFunctor<T>(),
        phi::funcs::ReluFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "relu,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    RunUnaryCompoundFunctors<DeviceContext,
                             T,
                             phi::funcs::ReluFunctor<T>,
                             phi::funcs::AddFunctor<T>>(
        dev_ctx,
        phi::funcs::ReluFunctor<T>(),
        phi::funcs::AddFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "elementwise_mul,scale") {
    // Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(in_scale);
    RunBinaryCompoundFunctor<DeviceContext,
                             T,
                             phi::funcs::MultiplyFunctor<T>,
                             phi::funcs::ScaleFunctor<T>>(
        dev_ctx,
        phi::funcs::MultiplyFunctor<T>(),
        phi::funcs::ScaleFunctor<T>(scale),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "tanh,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    RunUnaryCompoundFunctors<DeviceContext,
                             T,
                             phi::funcs::TanhFunctor<T>,
                             phi::funcs::AddFunctor<T>>(
        dev_ctx,
        phi::funcs::TanhFunctor<T>(),
        phi::funcs::AddFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "elementwise_mul,tanh") {
    // Z = Binary(X, Unary(Y))
    RunBinaryCompoundFunctor<DeviceContext,
                             T,
                             phi::funcs::MultiplyFunctor<T>,
                             phi::funcs::TanhFunctor<T>>(
        dev_ctx,
        phi::funcs::MultiplyFunctor<T>(),
        phi::funcs::TanhFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "elementwise_mul,sigmoid") {
    // Z = Binary(X, Unary(Y))
    RunBinaryCompoundFunctor<DeviceContext,
                             T,
                             phi::funcs::MultiplyFunctor<T>,
                             phi::funcs::SigmoidFunctor<T>>(
        dev_ctx,
        phi::funcs::MultiplyFunctor<T>(),
        phi::funcs::SigmoidFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else if (funcs_str == "gelu,elementwise_add") {
    // Z = Unary(Binary(X, Y))
    RunUnaryCompoundFunctors<DeviceContext,
                             T,
                             phi::funcs::GeluFunctor<T>,
                             phi::funcs::AddFunctor<T>>(
        dev_ctx,
        phi::funcs::GeluFunctor<T>(),
        phi::funcs::AddFunctor<T>(),
        in_x,
        in_y,
        outputs,
        axis,
        save_intermediate_out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument("%s has not been implemented.",
                                                 funcs_str));
  }
}

template <typename DeviceContext, typename T, bool InPlace>
void RunGradFunctors(const DeviceContext &dev_ctx,
                     const phi::DenseTensor *in_x,
                     const phi::DenseTensor *in_y,
                     const phi::DenseTensor *in_out,
                     const phi::DenseTensor *in_intermediate_out,
                     const phi::DenseTensor *in_out_grad,
                     phi::DenseTensor *x_grad,
                     phi::DenseTensor *y_grad,
                     phi::DenseTensor *d_intermediate_out,
                     std::vector<std::string> functor_list,
                     float in_scale,
                     int axis) {
  auto &functors = functor_list;
  auto funcs_str = functors[0] + "," + functors[1];

  if (funcs_str == "elementwise_add_grad,scale_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(in_scale);
    RunBinaryCompoundGradFunctors<DeviceContext,
                                  T,
                                  phi::funcs::AddGradFunctor<T>,
                                  phi::funcs::ScaleFunctor<T>,
                                  phi::funcs::ScaleGradFunctor<T>,
                                  InPlace>(
        dev_ctx,
        phi::funcs::AddGradFunctor<T>(),
        phi::funcs::ScaleFunctor<T>(scale),
        phi::funcs::ScaleGradFunctor<T>(scale),
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        x_grad,
        y_grad,
        d_intermediate_out,
        axis);
  } else if (funcs_str == "scale_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    T scale = static_cast<T>(in_scale);
    RunUnaryCompoundGradFunctors<DeviceContext,
                                 T,
                                 phi::funcs::ScaleGradFunctor<T>,
                                 phi::funcs::AddFunctor<T>,
                                 phi::funcs::AddGradFunctor<T>,
                                 InPlace>(
        dev_ctx,
        phi::funcs::ScaleGradFunctor<T>(scale),
        phi::funcs::AddFunctor<T>(),
        phi::funcs::AddGradFunctor<T>(),
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        x_grad,
        y_grad,
        d_intermediate_out,
        axis);
  } else if (funcs_str == "elementwise_add_grad,relu_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    RunBinaryCompoundGradFunctors<DeviceContext,
                                  T,
                                  phi::funcs::AddGradFunctor<T>,
                                  phi::funcs::ReluFunctor<T>,
                                  phi::funcs::ReluGradFunctor<T>,
                                  InPlace>(dev_ctx,
                                           phi::funcs::AddGradFunctor<T>(),
                                           phi::funcs::ReluFunctor<T>(),
                                           phi::funcs::ReluGradFunctor<T>(),
                                           in_x,
                                           in_y,
                                           in_out,
                                           in_intermediate_out,
                                           in_out_grad,
                                           x_grad,
                                           y_grad,
                                           d_intermediate_out,
                                           axis);
  } else if (funcs_str == "relu_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    RunUnaryCompoundGradFunctors<DeviceContext,
                                 T,
                                 phi::funcs::ReluGradFunctor<T>,
                                 phi::funcs::AddFunctor<T>,
                                 phi::funcs::AddGradFunctor<T>,
                                 InPlace>(dev_ctx,
                                          phi::funcs::ReluGradFunctor<T>(),
                                          phi::funcs::AddFunctor<T>(),
                                          phi::funcs::AddGradFunctor<T>(),
                                          in_x,
                                          in_y,
                                          in_out,
                                          in_intermediate_out,
                                          in_out_grad,
                                          x_grad,
                                          y_grad,
                                          d_intermediate_out,
                                          axis);
  } else if (funcs_str == "elementwise_mul_grad,scale_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    T scale = static_cast<T>(in_scale);
    RunBinaryCompoundGradFunctors<DeviceContext,
                                  T,
                                  phi::funcs::MulGradFunctor<T>,
                                  phi::funcs::ScaleFunctor<T>,
                                  phi::funcs::ScaleGradFunctor<T>,
                                  InPlace>(
        dev_ctx,
        phi::funcs::MulGradFunctor<T>(),
        phi::funcs::ScaleFunctor<T>(scale),
        phi::funcs::ScaleGradFunctor<T>(scale),
        in_x,
        in_y,
        in_out,
        in_intermediate_out,
        in_out_grad,
        x_grad,
        y_grad,
        d_intermediate_out,
        axis);
  } else if (funcs_str == "tanh_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    RunUnaryCompoundGradFunctors<DeviceContext,
                                 T,
                                 phi::funcs::TanhGradFunctor<T>,
                                 phi::funcs::AddFunctor<T>,
                                 phi::funcs::AddGradFunctor<T>,
                                 InPlace>(dev_ctx,
                                          phi::funcs::TanhGradFunctor<T>(),
                                          phi::funcs::AddFunctor<T>(),
                                          phi::funcs::AddGradFunctor<T>(),
                                          in_x,
                                          in_y,
                                          in_out,
                                          in_intermediate_out,
                                          in_out_grad,
                                          x_grad,
                                          y_grad,
                                          d_intermediate_out,
                                          axis);
  } else if (funcs_str == "elementwise_mul_grad,tanh_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    RunBinaryCompoundGradFunctors<DeviceContext,
                                  T,
                                  phi::funcs::MulGradFunctor<T>,
                                  phi::funcs::TanhFunctor<T>,
                                  phi::funcs::TanhGradFunctor<T>,
                                  InPlace>(dev_ctx,
                                           phi::funcs::MulGradFunctor<T>(),
                                           phi::funcs::TanhFunctor<T>(),
                                           phi::funcs::TanhGradFunctor<T>(),
                                           in_x,
                                           in_y,
                                           in_out,
                                           in_intermediate_out,
                                           in_out_grad,
                                           x_grad,
                                           y_grad,
                                           d_intermediate_out,
                                           axis);
  } else if (funcs_str == "elementwise_mul_grad,sigmoid_grad") {
    // The backward of Z = Binary(X, Unary(Y))
    RunBinaryCompoundGradFunctors<DeviceContext,
                                  T,
                                  phi::funcs::MulGradFunctor<T>,
                                  phi::funcs::SigmoidFunctor<T>,
                                  phi::funcs::SigmoidGradFunctor<T>,
                                  InPlace>(dev_ctx,
                                           phi::funcs::MulGradFunctor<T>(),
                                           phi::funcs::SigmoidFunctor<T>(),
                                           phi::funcs::SigmoidGradFunctor<T>(),
                                           in_x,
                                           in_y,
                                           in_out,
                                           in_intermediate_out,
                                           in_out_grad,
                                           x_grad,
                                           y_grad,
                                           d_intermediate_out,
                                           axis);
  } else if (funcs_str == "gelu_grad,elementwise_add_grad") {
    // The backward of Z = Unary(Binary(X, Y))
    RunUnaryCompoundGradFunctors<DeviceContext,
                                 T,
                                 phi::funcs::GeluGradFunctor<T>,
                                 phi::funcs::AddFunctor<T>,
                                 phi::funcs::AddGradFunctor<T>,
                                 InPlace>(dev_ctx,
                                          phi::funcs::GeluGradFunctor<T>(),
                                          phi::funcs::AddFunctor<T>(),
                                          phi::funcs::AddGradFunctor<T>(),
                                          in_x,
                                          in_y,
                                          in_out,
                                          in_intermediate_out,
                                          in_out_grad,
                                          x_grad,
                                          y_grad,
                                          d_intermediate_out,
                                          axis);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument("%s has not been implemented.",
                                                 funcs_str));
  }
}

}  // namespace funcs
}  // namespace phi
