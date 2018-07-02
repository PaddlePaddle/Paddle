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

#include "LinearChainCRF.h"
#include <algorithm>

namespace paddle {

LinearChainCRF::LinearChainCRF(int numClasses, real* para)
    : numClasses_(numClasses) {
  a_ = Matrix::create(para, 1, numClasses_);
  b_ = Matrix::create(para + numClasses_, 1, numClasses_);
  w_ = Matrix::create(para + 2 * numClasses_, numClasses_, numClasses_);

  ones_ = Matrix::create(1, numClasses_);
  ones_->one();

  expW_ = Matrix::create(numClasses_, numClasses_);
}

// normalize x so that its sum is 1 and return the original sum;
static real normalizeL1(real* x, int n) {
  real sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i];
  }
  // Right now, we just bet that sum won't be zero. If this really happens,
  // we will figure out what should be done then.
  CHECK_GT(sum, 0);
  real s = 1 / sum;
  for (int i = 0; i < n; ++i) {
    x[i] *= s;
  }
  return sum;
}

real LinearChainCRF::forward(real* x, int* s, int length) {
  Matrix::resizeOrCreate(maxX_, length, 1);
  Matrix::resizeOrCreate(expX_, length, numClasses_);
  Matrix::resizeOrCreate(alpha_, length, numClasses_);
  MatrixPtr matX = Matrix::create(x, length, numClasses_);
  matX->rowMax(*maxX_);
  expX_->assign(*matX);
  // subtract max to avoid overflow or underflow
  expX_->mul(*maxX_, *ones_, (real)-1, (real)1);
  expX_->exp2();

  real* a = a_->getData();
  real* b = b_->getData();
  real* w = w_->getData();
  real* alpha = alpha_->getData();
  real* expX = expX_->getData();
  real* maxX = maxX_->getData();

  expW_->exp2(*w_);
  real* expW = expW_->getData();

  for (int i = 0; i < numClasses_; ++i) {
    alpha[i] = exp(a[i]) * expX[i];
  }
  real ll = -maxX[0] - log(normalizeL1(alpha, numClasses_));

  for (int k = 1; k < length; ++k) {
    for (int i = 0; i < numClasses_; ++i) {
      real sum = 0;
      for (int j = 0; j < numClasses_; ++j) {
        sum += alpha[(k - 1) * numClasses_ + j]  // (*)
               * expW[j * numClasses_ + i];
      }
      alpha[k * numClasses_ + i] = expX[k * numClasses_ + i] * sum;
    }
    // normalizeL1 is to avoid underflow or overflow at (*)
    ll -= maxX[k] + log(normalizeL1(alpha + k * numClasses_, numClasses_));
  }
  real sum = 0;
  for (int i = 0; i < numClasses_; ++i) {
    sum += alpha[(length - 1) * numClasses_ + i] * exp(b[i]);
  }
  ll -= log(sum);
  // Now ll is equal to -log(Z)

  CHECK_LT(*std::max_element(s, s + length), numClasses_);
  // Calculate the nominator part, which depends on s
  ll += a[s[0]] + x[s[0]] + b[s[length - 1]];
  for (int k = 1; k < length; ++k) {
    ll += x[k * numClasses_ + s[k]] + w[s[k - 1] * numClasses_ + s[k]];
  }

  VLOG(1) << "ll=" << ll;
  return -ll;
}

void LinearChainCRF::backward(real* x, int* s, int length, bool needWGrad) {
  Matrix::resizeOrCreate(matGrad_, length, numClasses_);
  Matrix::resizeOrCreate(beta_, length, numClasses_);
  real* b = b_->getData();
  if (needWGrad) {
    Matrix::resizeOrCreate(matWGrad_, numClasses_ + 2, numClasses_);
    matWGrad_->zeroMem();
    da_ = matWGrad_->subRowMatrix(0, 1);
    db_ = matWGrad_->subRowMatrix(1, 2);
    dw_ = matWGrad_->subRowMatrix(2, numClasses_ + 2);
  }

  real* alpha = alpha_->getData();
  real* beta = beta_->getData();
  real* expW = expW_->getData();
  real* expX = expX_->getData();
  real* grad = matGrad_->getData();

  for (int i = 0; i < numClasses_; ++i) {
    beta[(length - 1) * numClasses_ + i] = exp(b[i]);
  }
  normalizeL1(beta + (length - 1) * numClasses_, numClasses_);

  for (int k = length - 2; k >= 0; --k) {
    for (int i = 0; i < numClasses_; ++i) {
      real sum = 0;
      for (int j = 0; j < numClasses_; ++j) {
        sum += expW[i * numClasses_ + j]  // (**)
               * beta[(k + 1) * numClasses_ + j] *
               expX[(k + 1) * numClasses_ + j];
      }
      beta[k * numClasses_ + i] = sum;
    }
    // normalizeL1 is to avoid underflow or overflow at (**)
    normalizeL1(beta + k * numClasses_, numClasses_);
  }

  matGrad_->dotMul(*alpha_, *beta_);
  matGrad_->rowNormalizeL1(*matGrad_);
  for (int k = 0; k < length; ++k) {
    grad[k * numClasses_ + s[k]] -= (real)1;
  }

  if (needWGrad) {
    da_->add(*matGrad_->subMatrix(/* startRow= */ 0, /* numRows= */ 1));
    db_->add(*matGrad_->subMatrix(/* startRow= */ length - 1, 1));

    beta_->dotMul(*beta_, *expX_);
    beta_->rowNormalizeL1(*beta_);

    real* dw = dw_->getData();
    for (int k = 1; k < length; ++k) {
      real sum = 0;
      for (int i = 0; i < numClasses_; ++i) {
        for (int j = 0; j < numClasses_; ++j) {
          sum += expW[i * numClasses_ + j] * alpha[(k - 1) * numClasses_ + i] *
                 beta[k * numClasses_ + j];
        }
      }
      sum = 1 / sum;
      for (int i = 0; i < numClasses_; ++i) {
        for (int j = 0; j < numClasses_; ++j) {
          dw[i * numClasses_ + j] += sum * expW[i * numClasses_ + j] *
                                     alpha[(k - 1) * numClasses_ + i] *
                                     beta[k * numClasses_ + j];
        }
      }
      dw[s[k - 1] * numClasses_ + s[k]] -= (real)1;
    }
  }
}

void LinearChainCRF::decode(real* x, int* s, int length) {
  Matrix::resizeOrCreate(alpha_, length, numClasses_);
  real* a = a_->getData();
  real* b = b_->getData();
  real* w = w_->getData();
  IVector::resizeOrCreate(track_, numClasses_ * length, /* useGpu= */ false);
  int* track = track_->getData();
  real* alpha = alpha_->getData();

  for (int i = 0; i < numClasses_; ++i) {
    alpha[i] = a[i] + x[i];
  }
  for (int k = 1; k < length; ++k) {
    for (int i = 0; i < numClasses_; ++i) {
      real maxScore = -std::numeric_limits<real>::max();
      int maxJ = 0;
      for (int j = 0; j < numClasses_; ++j) {
        real score = alpha[(k - 1) * numClasses_ + j] + w[j * numClasses_ + i];
        if (score > maxScore) {
          maxScore = score;
          maxJ = j;
        }
      }
      alpha[k * numClasses_ + i] = maxScore + x[k * numClasses_ + i];
      track[k * numClasses_ + i] = maxJ;
    }
  }
  real maxScore = -std::numeric_limits<real>::max();
  int maxI = 0;
  for (int i = 0; i < numClasses_; ++i) {
    real score = alpha[(length - 1) * numClasses_ + i] + b[i];
    if (score > maxScore) {
      maxScore = score;
      maxI = i;
    }
  }
  s[length - 1] = maxI;
  for (int k = length - 1; k >= 1; --k) {
    s[k - 1] = maxI = track[k * numClasses_ + maxI];
  }
}

}  // namespace paddle
