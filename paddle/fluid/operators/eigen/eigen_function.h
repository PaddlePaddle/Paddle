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
#include "paddle/pten/kernels/funcs/eigen/eigen_function.h"

namespace paddle {
namespace operators {

template <typename EigenDevice, typename T, int Rank>
using EigenBroadcast = pten::funcs::EigenBroadcast<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenBroadcastGrad =
    pten::funcs::EigenBroadcastGrad<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenConstant = pten::funcs::EigenConstant<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenSign = pten::funcs::EigenSign<EigenDevice, T>;

template <typename EigenDevice, typename T, int Rank>
using EigenReverse = pten::funcs::EigenReverse<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenAdd = pten::funcs::EigenAdd<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenSub = pten::funcs::EigenSub<EigenDevice, T>;

template <typename EigenDevice, typename T, int Rank>
using EigenSlice = pten::funcs::EigenSlice<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T, int Rank>
using EigenPad = pten::funcs::EigenPad<EigenDevice, T, Rank>;

template <typename EigenDevice, typename T>
using EigenScale = pten::funcs::EigenScale<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenErf = pten::funcs::EigenErf<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenErfGrad = pten::funcs::EigenErfGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenRankLoss = pten::funcs::EigenRankLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenRankLossGrad = pten::funcs::EigenRankLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenLogLoss = pten::funcs::EigenLogLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenLogLossGrad = pten::funcs::EigenLogLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenHingeLoss = pten::funcs::EigenHingeLoss<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenHingeLossGrad = pten::funcs::EigenHingeLossGrad<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenL1Norm = pten::funcs::EigenL1Norm<EigenDevice, T>;

template <typename EigenDevice, typename T>
using EigenL1NormGrad = pten::funcs::EigenL1NormGrad<EigenDevice, T>;

}  // namespace operators
}  // namespace paddle
