/**
 * TrainingAlgorithmOp.h
 *
 * Author: hedaoyuan (hedaoyuan@baidu.com)
 * Created on: 2016-06-29
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */

#pragma once

#include "paddle/utils/Logging.h"
#include "BaseMatrix.h"

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
