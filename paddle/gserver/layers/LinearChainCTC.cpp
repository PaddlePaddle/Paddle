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

#include "LinearChainCTC.h"
#include <math.h>
#include <limits>

namespace paddle {

/* log scale */
const real EXP_MAX = std::numeric_limits<real>::max();
const real EXP_MIN = std::numeric_limits<real>::min();
const real LOG_ZERO = std::log(EXP_MIN);
const real LOG_INFINITY = std::log(EXP_MAX);

static inline real safeExp(real x) {
  if (x <= LOG_ZERO) {
    return 0;
  }
  if (x >= LOG_INFINITY) {
    return EXP_MAX;
  }
  return std::exp(x);
}

static inline real safeLog(real x) {
  if (x <= EXP_MIN) {
    return LOG_ZERO;
  }
  return std::log(x);
}

// x=lna and y=lnb is log scale, ln(a/b)=lna-lnb
static inline real logDiv(real x, real y) {
  if (x - y <= LOG_ZERO) {
    return LOG_ZERO;
  }
  if (x - y >= LOG_INFINITY) {
    return LOG_INFINITY;
  }
  return x - y;
}

// x=lna and y=lnb is log scale, ln(a*b)=lna+lnb
static inline real logMul(real x, real y) {
  if (x + y <= LOG_ZERO) {
    return LOG_ZERO;
  }
  if (x + y >= LOG_INFINITY) {
    return LOG_INFINITY;
  }
  return x + y;
}

// x=lna and y=lnb is log scale, ln(a+b)=lna+ln(1+exp(lnb-lna)), where b > a
static inline real logAdd(real x, real y) {
  if (x < y) {
    real t = y;
    y = x;
    x = t;
  }
  return x + safeLog(1 + safeExp(y - x));
}

static void setLogZero(MatrixPtr mat) {
  size_t size = mat->getElementCnt();
  real* data = mat->getData();
  for (size_t i = 0; i < size; i++) {
    data[i] = LOG_ZERO;
  }
}

LinearChainCTC::LinearChainCTC(int numClasses, bool normByTimes)
    : numClasses_(numClasses), normByTimes_(normByTimes), logProb_(0) {
  // set the class label of blank as "numClasses-1"
  blank_ = numClasses - 1;

  Matrix::resizeOrCreate(gradTerms_, 1, numClasses_);
}

real LinearChainCTC::forward(real* softmaxSeq,
                             int softmaxSeqLen,
                             int* labelSeq,
                             int labelSeqLen) {
  isInvalid_ = false;
  totalTime_ = softmaxSeqLen;
  totalSegments_ = labelSeqLen * 2 + 1;

  int requiredTime = labelSeqLen;
  int oldLabel = -1;

  for (int i = 0; i < labelSeqLen; i++) {
    if (labelSeq[i] == oldLabel) {
      requiredTime++;
    }
    oldLabel = labelSeq[i];
  }

  if (totalTime_ < requiredTime) {
    isInvalid_ = true;
    return 0;
  }

  /* calculate the forward and backward variables,
   * reference Chapter 7.3 of "Alex Grave, Supervised Sequence
   * Labelling with Recurrent Neural Networks" */
  Matrix::resizeOrCreate(logActs_, totalTime_, numClasses_, false, false);
  real* logActsData = logActs_->getData();
  for (int i = 0; i < totalTime_ * numClasses_; i++) {
    logActsData[i] = safeLog(softmaxSeq[i]);
  }

  Matrix::resizeOrCreate(forwardVars_, totalTime_, totalSegments_);
  Matrix::resizeOrCreate(backwardVars_, totalTime_, totalSegments_);

  /* calculate the forward variables */
  setLogZero(forwardVars_);
  real* fwdVars = forwardVars_->getData();

  /* dp initialization at t0 */
  fwdVars[0] = logActs_->getData()[blank_];
  if (totalSegments_ > 1) {
    fwdVars[1] = logActs_->getData()[labelSeq[0]];
  }
  /* dp from t1 */
  for (int i = 1; i < totalTime_; i++) {
    real* dataPerStep = logActsData + i * numClasses_;
    real* oldFvars = fwdVars + (i - 1) * totalSegments_;
    real* fvars = fwdVars + i * totalSegments_;
    int start, end;
    segmentRange(start, end, i);
    for (int j = start; j < end; j++) {
      real fv;
      if (j & 1) {
        int labelIdx = j / 2;
        int labelVal = labelSeq[labelIdx];
        fv = logAdd(oldFvars[j], oldFvars[j - 1]);
        if (j > 1 && (labelVal != labelSeq[labelIdx - 1])) {
          fv = logAdd(fv, oldFvars[j - 2]);
        }
        fv = logMul(fv, dataPerStep[labelVal]);
      } else {
        fv = oldFvars[j];
        if (j) {
          fv = logAdd(fv, oldFvars[j - 1]);
        }
        fv = logMul(fv, dataPerStep[blank_]);
      }
      fvars[j] = fv;
    }
  }

  real* lastFvs = fwdVars + (totalTime_ - 1) * totalSegments_;

  /* sum the last two value as logprob */
  logProb_ = lastFvs[totalSegments_ - 1];
  if (totalSegments_ > 1) {
    logProb_ = logAdd(logProb_, lastFvs[totalSegments_ - 2]);
  }

  /* calculate the backward variables */
  setLogZero(backwardVars_);
  real* bwdVars = backwardVars_->getData();
  real* lastBvs = bwdVars + (totalTime_ - 1) * totalSegments_;

  lastBvs[totalSegments_ - 1] = 0;
  if (totalSegments_ > 1) {
    lastBvs[totalSegments_ - 2] = 0;
  }

  for (int i = totalTime_ - 2; i >= 0; i--) {
    real* oldDataPerStep = logActsData + (i + 1) * numClasses_;
    real* oldBvars = bwdVars + (i + 1) * totalSegments_;
    real* bvars = bwdVars + i * totalSegments_;
    int start, end;
    segmentRange(start, end, i);
    for (int j = start; j < end; j++) {
      real bv;
      if (j & 1) {
        int labelIdx = j / 2;
        int labelVal = labelSeq[labelIdx];

        bv = logAdd(logMul(oldBvars[j], oldDataPerStep[labelVal]),
                    logMul(oldBvars[j + 1], oldDataPerStep[blank_]));
        if (j < (totalSegments_ - 2)) {
          int nextLabelVal = labelSeq[labelIdx + 1];
          if (labelVal != nextLabelVal) {
            bv = logAdd(bv,
                        logMul(oldBvars[j + 2], oldDataPerStep[nextLabelVal]));
          }
        }
      } else {
        bv = logMul(oldBvars[j], oldDataPerStep[blank_]);
        if (j < (totalSegments_ - 1)) {
          bv = logAdd(bv,
                      logMul(oldBvars[j + 1], oldDataPerStep[labelSeq[j / 2]]));
        }
      }
      bvars[j] = bv;
    }
  }

  VLOG(1) << "ctcLoss=" << -logProb_;

  return -logProb_;
}

void LinearChainCTC::backward(real* softmaxSeq,
                              real* grad,
                              int* labelSeq,
                              int labelSeqLen) {
  /* if not meet the conditions of CTC computing, then set the grads to zeros */
  if (isInvalid_) {
    for (int i = 0; i < totalTime_ * numClasses_; i++) {
      grad[i] += 0;
    }
    return;
  }

  real* fwdVars = forwardVars_->getData();
  real* bwdVars = backwardVars_->getData();
  real* logActsData = logActs_->getData();

  for (int i = 0; i < totalTime_; i++) {
    setLogZero(gradTerms_);
    real* gradTermsData = gradTerms_->getData();
    real* fvars = fwdVars + i * totalSegments_;
    real* bvars = bwdVars + i * totalSegments_;
    for (int j = 0; j < totalSegments_; j++) {
      int k = (j & 1) ? labelSeq[j / 2] : blank_;
      gradTermsData[k] = logAdd(gradTermsData[k], logMul(fvars[j], bvars[j]));
    }
    for (int j = 0; j < numClasses_; j++) {
      if (normByTimes_) {
        grad[i * numClasses_ + j] +=
            -safeExp(
                logDiv(gradTermsData[j],
                       logMul(logProb_, logActsData[i * numClasses_ + j]))) /
            totalTime_;
      } else {
        grad[i * numClasses_ + j] += -safeExp(
            logDiv(gradTermsData[j],
                   logMul(logProb_, logActsData[i * numClasses_ + j])));
      }
    }
  }
}

void LinearChainCTC::segmentRange(int& start, int& end, int time) {
  start = std::max(0, totalSegments_ - (2 * (totalTime_ - time)));
  end = std::min(totalSegments_, 2 * (time + 1));
}

}  // namespace paddle
