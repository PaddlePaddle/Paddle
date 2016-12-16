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

#include "paddle/math/Matrix.h"

namespace paddle {

class LinearChainCRF {
public:
  /**
   * The size of para and grad must be \f$(numClasses + 2) * numClasses\f$.
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
  LinearChainCRF(int numClasses, real* para, real* grad);

  /**
   * Calculate the negative log likelihood of s given x.
   * The size of x must be length * numClasses. Each consecutive numClasses
   * values are the features for one time step.
   */
  real forward(real* x, int* s, int length);

  /**
   * Calculate the gradient with respect to x, a, b, and w.
   * The gradient of x will be stored in dx.
   * backward() can only be called after a corresponding call to forward() with
   * the same x, s and length.
   * @note The gradient is added to dx and grad (provided at constructor).
   */
  void backward(real* x, real* dx, int* s, int length);

  /**
   * Find the most probable sequence given x. The result will be stored in s.
   */
  void decode(real* x, int* s, int length);

protected:
  int numClasses_;
  MatrixPtr a_;
  MatrixPtr b_;
  MatrixPtr w_;
  MatrixPtr da_;
  MatrixPtr db_;
  MatrixPtr dw_;
  MatrixPtr ones_;

  MatrixPtr expX_;
  MatrixPtr alpha_;
  MatrixPtr beta_;
  MatrixPtr maxX_;
  MatrixPtr expW_;

  // track_(k,i) = j means that the best sequence at time k for class i comes
  // from the sequence at time k-1 for class j
  IVectorPtr track_;
};

}  // namespace paddle
