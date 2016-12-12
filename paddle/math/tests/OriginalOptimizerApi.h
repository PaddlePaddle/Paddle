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

#include "paddle/math/Vector.h"
#include "paddle/utils/GlobalConstants.h"

using namespace paddle;  // NOLINT

void SparseMomentumParameterOptimizer(const VectorPtr vecs[],
                                      real alpha,
                                      real beta,
                                      real gamma,
                                      real tau,
                                      real learningRate) {
  vecs[PARAMETER_MOMENTUM_UT]->add(*vecs[PARAMETER_GRADIENT],
                                   -alpha * gamma * learningRate);
  vecs[PARAMETER_MOMENTUM_VT]->add(*vecs[PARAMETER_GRADIENT],
                                   tau * alpha * gamma * learningRate);
  vecs[PARAMETER_VALUE]->add(*vecs[PARAMETER_MOMENTUM_UT],
                             tau / beta + 1.0 / alpha,
                             *vecs[PARAMETER_MOMENTUM_VT],
                             1.0 / beta);
}

void AdagradParameterOptimizer(const VectorPtr vecs[],
                               real epsilon,
                               real learningRate,
                               real momentum,
                               real decayRate) {
  vecs[PARAMETER_GRADIENT_SQURESUM1]->addSquare(*vecs[PARAMETER_GRADIENT],
                                                1.0f);
  vecs[PARAMETER_LEARNING_RATE]->add(*vecs[PARAMETER_GRADIENT_SQURESUM],
                                     *vecs[PARAMETER_GRADIENT_SQURESUM1]);
  vecs[PARAMETER_LEARNING_RATE]->add(epsilon);
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(*vecs[PARAMETER_GRADIENT],
                                   *vecs[PARAMETER_MOMENTUM],
                                   *vecs[PARAMETER_LEARNING_RATE],
                                   learningRate,
                                   momentum,
                                   decayRate);
}

void AdaDeltaParameterOptimizer(const VectorPtr vecs[],
                                real rou,
                                real epsilon,
                                real learningRate,
                                real momentum,
                                real decayRate) {
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(
      *vecs[PARAMETER_GRADIENT], rou, 1.0f - rou);

  // learn_rate = sqrt( ( E(dx_{t-1}^2) + epsilon ) / ( E(g_t^2) + epsilon ) )
  vecs[PARAMETER_LEARNING_RATE]->dotDiv(*vecs[PARAMETER_GRADIENT_SQURESUM1],
                                        *vecs[PARAMETER_GRADIENT_SQURESUM],
                                        epsilon,
                                        epsilon);
  vecs[PARAMETER_LEARNING_RATE]->sqrt2();

  // E(dx_t^2) = \rou * E(dx_{t-1}^2) + (1-\rou) * (-g*learn_rate)^2
  vecs[PARAMETER_GRADIENT_SQURESUM1]->decayAddSquareMul(
      *vecs[PARAMETER_GRADIENT],
      *vecs[PARAMETER_LEARNING_RATE],
      rou,
      1.0f - rou);

  vecs[PARAMETER_VALUE]->sgdUpdate(*vecs[PARAMETER_GRADIENT],
                                   *vecs[PARAMETER_MOMENTUM],
                                   *vecs[PARAMETER_LEARNING_RATE],
                                   learningRate,
                                   momentum,
                                   decayRate);
}

void RMSPropParameterOptimizer(const VectorPtr vecs[],
                               real accumulatedRou,
                               real rou,
                               real epsilon,
                               real learningRate,
                               real momentum,
                               real decayRate,
                               bool firstTime) {
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  // For the first time update, make the sum be the current square
  // so that the initial estimation of E(g_t^2) will not be too small.
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(
      *vecs[PARAMETER_GRADIENT], accumulatedRou, firstTime ? 1.0f : 1.0f - rou);

  // E(g_t) = \rou * E(g_{t-1}) + (1-\rou) * g
  vecs[PARAMETER_GRADIENT_SQURESUM1]->add(
      *vecs[PARAMETER_GRADIENT], accumulatedRou, 1.0f - rou);

  // learn_rate = 1/sqrt( ( E(g_t^2) - (E(g_t))^2 + epsilon )
  // Basiclly if the sign of the gradient changes more often,
  // the learning rate will be decreased.
  vecs[PARAMETER_LEARNING_RATE]->assign(*vecs[PARAMETER_GRADIENT_SQURESUM]);
  vecs[PARAMETER_LEARNING_RATE]->addSquare(*vecs[PARAMETER_GRADIENT_SQURESUM1],
                                           -1.0f);
  vecs[PARAMETER_LEARNING_RATE]->add(epsilon);
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(*vecs[PARAMETER_GRADIENT],
                                   *vecs[PARAMETER_MOMENTUM],
                                   *vecs[PARAMETER_LEARNING_RATE],
                                   learningRate,
                                   momentum,
                                   decayRate);
}

void DecayedAdagradParameterOptimizer(const VectorPtr vecs[],
                                      real accumulatedRou,
                                      real rou,
                                      real epsilon,
                                      real learningRate,
                                      real momentum,
                                      real decayRate,
                                      bool firstTime) {
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  // For the first time update, make the sum be the current square
  // so that the initial estimation of E(g_t^2) will not be too small.
  vecs[PARAMETER_GRADIENT_SQURESUM]->decayAddSquare(
      *vecs[PARAMETER_GRADIENT], accumulatedRou, firstTime ? 1.0f : 1.0f - rou);

  // learn_rate = 1/sqrt( ( E(g_t^2) + epsilon )
  // Basiclly if the bigger the magnitude gradient is,
  // the smaller the learning rate will be.
  vecs[PARAMETER_LEARNING_RATE]->assign(epsilon);
  vecs[PARAMETER_LEARNING_RATE]->add(*vecs[PARAMETER_GRADIENT_SQURESUM]);
  vecs[PARAMETER_LEARNING_RATE]->invSqrt(*vecs[PARAMETER_LEARNING_RATE]);

  vecs[PARAMETER_VALUE]->sgdUpdate(*vecs[PARAMETER_GRADIENT],
                                   *vecs[PARAMETER_MOMENTUM],
                                   *vecs[PARAMETER_LEARNING_RATE],
                                   learningRate,
                                   momentum,
                                   decayRate);
}

void AdamParameterOptimizer(const VectorPtr vecs[],
                            real beta1,
                            real beta2,
                            real beta1_power,
                            real beta2_power,
                            real epsilon,
                            real learningRate) {
  Vector* m = vecs[PARAMETER_MOMENTUM].get();
  Vector* g = vecs[PARAMETER_GRADIENT].get();
  Vector* v = vecs[PARAMETER_SECOND_MOMENTUM].get();
  Vector* theta = vecs[PARAMETER_VALUE].get();

  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  m->add(*g, beta1, 1 - beta1);

  // v_t = \beta_2 * v_{t-1} + (1-\beta_2)* g_{t-1}^2
  g->square2();
  v->add(*g, beta2, 1 - beta2);

  // tmp = m_t / ( \sqrt{v_t} + \epsilon )
  // \theta_t = \theta_{t-1} - \alpha * \sqrt(1-\beta_2^t) / (1-\beta_1^t) * tmp
  g->sqrt2(*v);
  g->dotDiv(*m, *g, 0., epsilon);
  real alpha =
      learningRate * std::sqrt((real)1 - beta2_power) / ((real)1 - beta1_power);
  theta->add(*theta, 1.0, *g, -alpha);
}

void AdamaxParameterOptimizer(
    const VectorPtr vecs[], real beta1, real beta2, int64_t step, real alpha) {
  Vector* m = vecs[PARAMETER_MOMENTUM].get();
  Vector* g = vecs[PARAMETER_GRADIENT].get();
  Vector* u = vecs[PARAMETER_WEIGHTED_INFINITY_NORM].get();
  Vector* theta = vecs[PARAMETER_VALUE].get();

  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  m->add(*g, beta1, 1 - beta1);

  // u_t = max(\beta_2*u_{t-1}, abs(g_t))
  u->mulScalar(beta2);
  g->abs2();
  u->max2(*u, *g);

  // \theta_t = \theta_{t-1} - (\alpha/(1-\beta_1^t))*m_t/u_t
  g->dotDiv(*m, *u);
  real learningRate = alpha / (1 - std::pow(beta1, step));
  theta->add(*theta, 1.0, *g, -learningRate);
}
