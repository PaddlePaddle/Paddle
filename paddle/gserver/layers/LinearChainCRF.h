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

#include "paddle/math/Matrix.h"

namespace paddle {

class LinearChainCRF {
 public:
  /**
   * The size of para must be \f$(numClasses + 2) * numClasses\f$.
   * The first numClasses values of para are for starting weights (\f$a\f$).
   * The next numClasses values of para are for ending weights (\f$b\f$),
   * The remaning values are for transition weights (\f$w\f$).
   *
   * The probability of a state sequence s of length \f$L\f$ is defined as:
   * \f$P(s) = (1/Z) exp(a_{s_1} + b_{s_L}
   *                  + \sum_{l=1}^L x_{s_l}
   *                  + \sum_{l=2}^L w_{s_{l-1},s_l})\f$
   * where \f$Z\f$ is a normalization value so that the sum of \f$P(s)\f$ over
   * all possible
   * sequences is \f$1\f$, and \f$x\f$ is the input feature to the CRF.
   */
  LinearChainCRF(int numClasses, real* para);

  /**
   * Calculate the negative log likelihood of s given x.
   * The size of x must be length * numClasses. Each consecutive numClasses
   * values are the features for one time step.
   */
  real forward(real* x, int* s, int length);

  /**
   * Calculate the gradient with respect to x, a, b, and w.
   * backward() can only be called after a corresponding call to forward() with
   * the same x, s and length.
   * The gradient with respect to a, b, and w will not be calculated if
   * needWGrad is false.
   * @note Please call getWGrad() and getXGrad() to get the gradient with
   * respect to (a, b, w) and x respectively.
   */
  void backward(real* x, int* s, int length, bool needWGrad);

  /**
   * Find the most probable sequence given x. The result will be stored in s.
   */
  void decode(real* x, int* s, int length);

  /*
   * Return the gradient with respect to (a, b, w). It can only be called after
   * a corresponding call to backward().
   */
  MatrixPtr getWGrad() { return matWGrad_; }

  /*
   * Return the gradient with respect to x. It can only be called after a
   * corresponding call to backward().
   */
  MatrixPtr getXGrad() { return matGrad_; }

 protected:
  int numClasses_;
  MatrixPtr a_;
  MatrixPtr b_;
  MatrixPtr w_;
  MatrixPtr matWGrad_;
  MatrixPtr da_;
  MatrixPtr db_;
  MatrixPtr dw_;
  MatrixPtr ones_;

  MatrixPtr expX_;
  MatrixPtr matGrad_;
  MatrixPtr alpha_;
  MatrixPtr beta_;
  MatrixPtr maxX_;
  MatrixPtr expW_;

  // track_(k,i) = j means that the best sequence at time k for class i comes
  // from the sequence at time k-1 for class j
  IVectorPtr track_;
};

}  // namespace paddle
