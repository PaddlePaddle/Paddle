/**
 * TrainingAlgorithmOp.cu
 *
 * Author: hedaoyuan (hedaoyuan@baidu.com)
 * Created on: 2016-06-29
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */

#include "paddle/utils/Logging.h"
#include "BaseMatrix.h"
#include "TrainingAlgorithmOp.h"

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
