/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "BaseMatrix.h"
#include "paddle/utils/Logging.h"

namespace paddle {

/**
 * \brief Sparse Momentum optimizer.
 */
extern void sparseMomentumApply(BaseMatrix& value,
                                BaseMatrix& grad,
                                BaseMatrix& momU,
                                BaseMatrix& momV,
                                real alpha,
                                real beta,
                                real gamma,
                                real tau,
                                real learningRate);

/**
 * \brief AdaDelta optimizer.
 */
extern void adadeltaApply(BaseMatrix& value,
                          BaseMatrix& grad,
                          BaseMatrix& sum,
                          BaseMatrix& sum1,
                          BaseMatrix& mom,
                          BaseMatrix& lr,
                          real rou,
                          real epsilon,
                          real learningRate,
                          real momentum,
                          real decayRate);

/**
 * \brief AdaGrad optimizer.
 */
extern void adagradApply(BaseMatrix& value,
                         BaseMatrix& grad,
                         BaseMatrix& sum,
                         BaseMatrix& sum1,
                         BaseMatrix& mom,
                         BaseMatrix& lr,
                         real epsilon,
                         real learningRate,
                         real momentum,
                         real decayRate);

/**
 * \brief RMSProp optimizer.
 */
extern void rmspropApply(BaseMatrix& value,
                         BaseMatrix& grad,
                         BaseMatrix& g,
                         BaseMatrix& f,
                         BaseMatrix& mom,
                         BaseMatrix& lr,
                         real accumulatedRou,
                         real rou,
                         real epsilon,
                         real learningRate,
                         real momentum,
                         real decayRate,
                         bool firstTime);

/**
 * \brief Decayed AdaGrad optimizer.
 */
extern void decayedAdagradApply(BaseMatrix& value,
                                BaseMatrix& grad,
                                BaseMatrix& mom,
                                BaseMatrix& accum,
                                BaseMatrix& lr,
                                real accumulatedRou,
                                real rou,
                                real epsilon,
                                real learningRate,
                                real momentum,
                                real decayRate,
                                bool firstTime);

/**
 * \brief Adam optimizer.
 */
extern void adamApply(BaseMatrix& value,
                      BaseMatrix& grad,
                      BaseMatrix& mom,
                      BaseMatrix& v,
                      real beta1,
                      real beta2,
                      real beta1_power,
                      real beta2_power,
                      real epsilon,
                      real learningRate);

/**
 * \brief AdaMax optimizer.
 */
extern void adamaxApply(BaseMatrix& value,
                        BaseMatrix& grad,
                        BaseMatrix& mom,  // firse moment
                        BaseMatrix& u,    // weighted infinity norm
                        real beta1,
                        real beta2,
                        int64_t step,
                        real alpha);
}  // namespace paddle
