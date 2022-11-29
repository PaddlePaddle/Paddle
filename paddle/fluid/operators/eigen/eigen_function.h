/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

template <typename EigenDevice, typename T, int Rank>
using EigenBroadcast = phi::funcs::EigenBroadcast<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenBroadcastGrad = phi::funcs::EigenBroadcastGrad<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenConstant = phi::funcs::EigenConstant<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenSign = phi::funcs::EigenSign<EigenDevice, T>;

template <typename EigenDevice, typename T, int Rank>
using EigenReverse = phi::funcs::EigenReverse<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenAdd = phi::funcs::EigenAdd<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenSub = phi::funcs::EigenSub<EigenDevice, T>;

template <typename EigenDevice, typename T, int Rank>
using EigenSlice = phi::funcs::EigenSlice<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenPad = phi::funcs::EigenPad<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenScale = phi::funcs::EigenScale<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenErf = phi::funcs::EigenErf<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenErfGrad = phi::funcs::EigenErfGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenRankLoss = phi::funcs::EigenRankLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenRankLossGrad = phi::funcs::EigenRankLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenLogLoss = phi::funcs::EigenLogLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenLogLossGrad = phi::funcs::EigenLogLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenHingeLoss = phi::funcs::EigenHingeLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenHingeLossGrad = phi::funcs::EigenHingeLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenL1Norm = phi::funcs::EigenL1Norm<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenL1NormGrad = phi::funcs::EigenL1NormGrad<EigenDevice, T>;

}  // namespace operators
}  // namespace paddle
