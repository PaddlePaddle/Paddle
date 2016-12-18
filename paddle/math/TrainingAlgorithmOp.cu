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

#include "paddle/utils/Logging.h"
#include "BaseMatrix.h"
#include "TrainingAlgorithmOp.h"

#if __cplusplus > 199711L

#include "TensorAssign.h"

namespace paddle {

void sparseMomentumApply(BaseMatrix& value,
                         BaseMatrix& grad,
                         BaseMatrix& momU,
                         BaseMatrix& momV,
                         real alpha,
                         real beta,
                         real gamma,
                         real tau,
                         real learningRate) {
  auto expr1 = momU.lazyAssign(momU - (alpha * gamma * learningRate) * grad);
  auto expr2 = momV.lazyAssign(
    momV + (tau * alpha * gamma * learningRate) * grad);
  auto expr3 = value.lazyAssign(
    (tau / beta + (real)1 / alpha) * momU + ((real)1 / beta) * momV);

  AssignEvaluate(expr1, expr2, expr3);
}

void adadeltaApply(BaseMatrix& value,
                   BaseMatrix& grad,
                   BaseMatrix& mom,
                   BaseMatrix& accum,
                   BaseMatrix& accum_update,
                   BaseMatrix& lr,
                   real rou,
                   real epsilon,
                   real learningRate,
                   real momentum,
                   real decayRate) {
  auto expr1 = accum.lazyAssign(rou * accum + ((real)1 - rou) * grad.square());
  auto expr2 = lr.lazyAssign(
    ((accum_update + epsilon) / (accum + epsilon)).sqrt());
  auto expr3 = accum_update.lazyAssign(
    rou * accum_update + ((real)1 - rou) * (grad * lr).square());
  auto expr4 = mom.lazyAssign(
    mom * momentum - learningRate * lr * (grad + value * decayRate));
  auto expr5 = value.lazyAssign(value + mom);

  AssignEvaluate(expr1, expr2, expr3, expr4, expr5);
}

void adagradApply(BaseMatrix& value,
                  BaseMatrix& grad,
                  BaseMatrix& mom,
                  BaseMatrix& accum_buffer,
                  BaseMatrix& accum,
                  BaseMatrix& lr,
                  real epsilon,
                  real learningRate,
                  real momentum,
                  real decayRate) {
  auto expr1 = accum.lazyAssign(accum + grad.square());
  auto expr2 = lr.lazyAssign(
    (accum_buffer + accum + epsilon).sqrt().reciprocal());
  auto expr3 = mom.lazyAssign(
    mom * momentum - learningRate * lr * (grad + value * decayRate));
  auto expr4 = value.lazyAssign(value + mom);

  AssignEvaluate(expr1, expr2, expr3, expr4);
}

void rmspropApply(BaseMatrix& value,
                  BaseMatrix& grad,
                  BaseMatrix& mom,
                  BaseMatrix& g,
                  BaseMatrix& f,
                  BaseMatrix& lr,
                  real accumulatedRou,
                  real rou,
                  real epsilon,
                  real learningRate,
                  real momentum,
                  real decayRate,
                  bool firstTime) {
  auto expr2 = f.lazyAssign(accumulatedRou * f + ((real)1 - rou) * grad);
  auto expr3 = lr.lazyAssign((g - f.square() + epsilon).sqrt().reciprocal());
  auto expr4 = mom.lazyAssign(
    mom * momentum - learningRate * lr * (grad + value * decayRate));
  auto expr5 = value.lazyAssign(value + mom);

  if (firstTime) {
    auto expr1 = g.lazyAssign(accumulatedRou * g + grad.square());

    AssignEvaluate(expr1, expr2, expr3, expr4, expr5);
  } else {
    auto expr1 = g.lazyAssign(
      accumulatedRou * g + ((real)1 - rou) * grad.square());

    AssignEvaluate(expr1, expr2, expr3, expr4, expr5);
  }
}

void decayedAdagradApply(BaseMatrix& value,
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
                         bool firstTime) {
  auto expr2 = lr.lazyAssign((accum + epsilon).sqrt().reciprocal());
  auto expr3 = mom.lazyAssign(
    mom * momentum - learningRate * lr * (grad + value * decayRate));
  auto expr4 = value.lazyAssign(value + mom);

  if (firstTime) {
    auto expr1 = accum.lazyAssign(accumulatedRou * accum + grad.square());

    AssignEvaluate(expr1, expr2, expr3, expr4);
  } else {
    auto expr1 = accum.lazyAssign(
      accumulatedRou * accum + ((real)1 - rou) * grad.square());

    AssignEvaluate(expr1, expr2, expr3, expr4);
  }
}

void adamApply(BaseMatrix& value,
               BaseMatrix& grad,
               BaseMatrix& mom,  // firse moment
               BaseMatrix& v,    // second moment
               real beta1,
               real beta2,
               real beta1_power,
               real beta2_power,
               real epsilon,
               real learningRate) {
  real alpha = learningRate *
      std::sqrt((real)1 - beta2_power) / ((real)1 - beta1_power);

  auto expr1 = mom.lazyAssign(beta1 * mom + ((real)1 - beta1) * grad);
  auto expr2 = v.lazyAssign(beta2 * v + ((real)1 - beta2) * grad.square());
  auto expr3 = value.lazyAssign(
    value - (mom * alpha) / (v.sqrt() + epsilon));

  AssignEvaluate(expr1, expr2, expr3);
}

void adamaxApply(BaseMatrix& value,
                 BaseMatrix& grad,
                 BaseMatrix& mom,  // firse moment
                 BaseMatrix& u,    // weighted infinity norm
                 real beta1,
                 real beta2,
                 int64_t step,
                 real alpha) {
  auto expr1 = mom.lazyAssign(beta1 * mom + ((real)1 - beta1) * grad);
  auto expr2 = u.lazyAssign(
    (beta2 * u > grad.abs()).condition(beta2 * u, grad.abs()));
  auto expr3 = value.lazyAssign(
    value - (alpha / ((real)1 - (real)std::pow(beta1, step))) * (mom / u));

  AssignEvaluate(expr1, expr2, expr3);
}

}  // namespace paddle

#else

namespace paddle {

void sparseMomentumApply(BaseMatrix& value,
                         BaseMatrix& grad,
                         BaseMatrix& momU,
                         BaseMatrix& momV,
                         real alpha,
                         real beta,
                         real gamma,
                         real tau,
                         real learningRate) {
  /**
   * \alpha_t = \alpha_{t-1} / k
   * \beta_t = \beta_{t-1} / (1 + \lambda\gamma_t)
   * u_t = u_{t-1} - \alpha_t \gamma_t g_t
   * v_t = v_{t-1} + \tau_{t-1} \alpha_t \gamma_t g_t
   * \tau_t = \tau_{t-1} + \beta_t / \alpha_t
   */
  momU -= (alpha * gamma * learningRate) * grad;
  momV += (tau * alpha * gamma * learningRate) * grad;
  value = (tau / beta + (real)1 / alpha) * momU + ((real)1 / beta) * momV;
}

void adadeltaApply(BaseMatrix& value,
                   BaseMatrix& grad,
                   BaseMatrix& mom,
                   BaseMatrix& accum,
                   BaseMatrix& accum_update,
                   BaseMatrix& lr,
                   real rou,
                   real epsilon,
                   real learningRate,
                   real momentum,
                   real decayRate) {
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  accum = rou * accum + ((real)1 - rou) * grad.square();

  // learn_rate: sqrt(( E(dx_{t-1}^2) + epsilon ) / ( E(g_t^2) + epsilon ))
  lr = ((accum_update + epsilon) / (accum + epsilon)).sqrt();

  // E(dx_t^2) = \rou * E(dx_{t-1}^2) + (1-\rou) * (-g*learn_rate)^2
  accum_update = rou * accum_update + ((real)1 - rou) * (grad * lr).square();

  mom = mom * momentum - learningRate * lr * (grad + value * decayRate);
  value += mom;
}

void adagradApply(BaseMatrix& value,
                  BaseMatrix& grad,
                  BaseMatrix& mom,
                  BaseMatrix& accum_buffer,
                  BaseMatrix& accum,
                  BaseMatrix& lr,
                  real epsilon,
                  real learningRate,
                  real momentum,
                  real decayRate) {
  accum += grad.square();
  lr = (accum_buffer + accum + epsilon).sqrt().reciprocal();
  mom = mom * momentum - learningRate * lr * (grad + value * decayRate);
  value += mom;
}

void rmspropApply(BaseMatrix& value,
                  BaseMatrix& grad,
                  BaseMatrix& mom,
                  BaseMatrix& g,
                  BaseMatrix& f,
                  BaseMatrix& lr,
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
  if (firstTime) {
    g = accumulatedRou * g + grad.square();
  } else {
    g = accumulatedRou * g + ((real)1 - rou) * grad.square();
  }

  // E(f_t) = \rou * E(f_{t-1}) + (1-\rou) * g
  f = accumulatedRou * f + ((real)1 - rou) * grad;

  // learn_rate = 1/sqrt( ( E(g_t^2) - (E(f_t))^2 + epsilon )
  // Basiclly if the sign of the gradient changes more often,
  // the learning rate will be decreased.
  lr = (g - f.square() + epsilon).sqrt().reciprocal();

  mom = mom * momentum - learningRate * lr * (grad + value * decayRate);
  value += mom;
}

void decayedAdagradApply(BaseMatrix& value,
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
                         bool firstTime) {
  // E(g_t^2) = \rou * E(g_{t-1}^2) + (1-\rou) * g^2
  // For the first time update, make the sum be the current square
  // so that the initial estimation of E(g_t^2) will not be too small.
  if (firstTime) {
    accum = accumulatedRou * accum + grad.square();
  } else {
    accum = accumulatedRou * accum + ((real)1 - rou) * grad.square();
  }

  // learn_rate = 1/sqrt( ( E(g_t^2) + epsilon )
  // Basiclly if the bigger the magnitude gradient is,
  // the smaller the learning rate will be.
  lr = (accum + epsilon).sqrt().reciprocal();

  mom = mom * momentum - learningRate * lr * (grad + value * decayRate);
  value += mom;
}

void adamApply(BaseMatrix& value,
               BaseMatrix& grad,
               BaseMatrix& mom,  // firse moment
               BaseMatrix& v,    // second moment
               real beta1,
               real beta2,
               real beta1_power,
               real beta2_power,
               real epsilon,
               real learningRate) {
  real alpha = learningRate *
      std::sqrt((real)1 - beta2_power) / ((real)1 - beta1_power);

  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  mom = beta1 * mom + ((real)1 - beta1) * grad;

  // v_t = \beta_2 * v_{t-1} + (1-\beta_2)* g_{t-1}^2
  v = beta2 * v + ((real)1 - beta2) * grad.square();

  value -=  (mom * alpha) / (v.sqrt() + epsilon);
}

void adamaxApply(BaseMatrix& value,
                 BaseMatrix& grad,
                 BaseMatrix& mom,  // firse moment
                 BaseMatrix& u,    // weighted infinity norm
                 real beta1,
                 real beta2,
                 int64_t step,
                 real alpha) {
  // m_t = \beta_1 * m_{t-1} + (1-\beta_1)* g_t;
  mom = beta1 * mom + ((real)1 - beta1) * grad;

  // u_t = max(\beta_2*u_{t-1}, abs(g_t))
  u = (beta2 * u > grad.abs()).condition(beta2 * u, grad.abs());

  // \theta_t = \theta_{t-1} - (\alpha/(1-\beta_1^t))*m_t/u_t
  value -= (alpha / ((real)1 - (real)std::pow(beta1, step))) * (mom / u);
}

}  // namespace paddle

#endif
