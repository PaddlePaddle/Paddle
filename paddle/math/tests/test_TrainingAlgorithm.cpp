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

#include <gtest/gtest.h>
#include "OriginalOptimizerApi.h"
#include "PerfUtils.h"
#include "TensorCheck.h"
#include "paddle/math/TrainingAlgorithmOp.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT

#ifndef PADDLE_TYPE_DOUBLE
DEFINE_double(max_diff, 1e-5, "max diff allowed");
#else
DEFINE_double(max_diff, 1e-13, "max diff allowed");
#endif

class SetMaxDiff {
public:
  explicit SetMaxDiff(double max_diff) {
    max_diff_ = FLAGS_max_diff;
    FLAGS_max_diff = max_diff;
  }
  ~SetMaxDiff() { FLAGS_max_diff = max_diff_; }

private:
  double max_diff_;
};

#define COPY_VECTOR_TO_CPU(cpuVec, vector)               \
  do {                                                   \
    if (vector->useGpu()) {                              \
      cpuVec = Vector::create(vector->getSize(), false); \
      cpuVec->copyFrom(*vector);                         \
    } else {                                             \
      cpuVec = vector;                                   \
    }                                                    \
  } while (0)

int VectorCheckErr(const Vector& vector1, const Vector& vector2) {
  CHECK(vector1.getSize() == vector2.getSize());

  const real* data1 = vector1.getData();
  const real* data2 = vector2.getData();
  size_t size = vector1.getSize();
  int count = 0;
  for (size_t i = 0; i < size; i++) {
    real a = data1[i];
    real b = data2[i];
    if (fabs(a - b) > FLAGS_max_diff) {
      if ((fabsf(a - b) / fabsf(a)) > (FLAGS_max_diff / 10.0f)) {
        count++;
      }
    }
  }

  return count;
}

int VectorCheckErr(const VectorPtr& vector1, const VectorPtr& vector2) {
  VectorPtr tmp1;
  VectorPtr tmp2;
  COPY_VECTOR_TO_CPU(tmp1, vector1);
  COPY_VECTOR_TO_CPU(tmp2, vector2);
  return VectorCheckErr(*tmp1, *tmp2);
}

#ifdef PADDLE_DISABLE_TIMER

#define CHECK_VECTORPTR(vector1, vector2) \
  EXPECT_EQ(VectorCheckErr(vector1, vector2), 0)

#else

#define CHECK_VECTORPTR(vector1, vector2)

#endif

typedef std::function<void(size_t size, bool useGpu)> testMatrixFunc;

void testCase(testMatrixFunc matrixFunc) {
#ifndef PADDLE_ONLY_CPU
  for (auto useGpu : {false, true}) {
#else
  for (auto useGpu : {false}) {
#endif
    for (auto size : {1,
                      32,
                      64,
                      128,
                      512,
                      1024,
                      4096,
                      32768,
                      65536,
                      131072,
                      262144,
                      524288,
                      1048576,
                      2097152}) {
      LOG(INFO) << " size=" << size << " useGpu=" << useGpu;
      matrixFunc(size, useGpu);
    }
  }
}

#define INIT_VECTOR(vec1, vec2, type, size, useGpu) \
  vec1[type] = Vector::create(size, useGpu);        \
  vec2[type] = Vector::create(size, useGpu);        \
  vec1[type]->rand();                               \
  vec2[type]->copyFrom(*vec1[type]);

void testAdagrad(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM1, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_LEARNING_RATE, size, useGpu);

  real epsilon = (real)rand() / (real)RAND_MAX;       // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT
  real momentum = (real)rand() / (real)RAND_MAX;      // NOLINT
  real decayRate = (real)rand() / (real)RAND_MAX;     // NOLINT

  EXPRESSION_PERFORMANCE(AdagradParameterOptimizer(
      bufs1, epsilon, learningRate, momentum, decayRate));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& accum_buffer = *bufs2[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& accum = *bufs2[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *bufs2[PARAMETER_LEARNING_RATE];

  EXPRESSION_PERFORMANCE(adagradApply(value,
                                      grad,
                                      mom,
                                      accum_buffer,
                                      accum,
                                      lr,
                                      epsilon,
                                      learningRate,
                                      momentum,
                                      decayRate));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM1],
                  bufs2[PARAMETER_GRADIENT_SQURESUM1]);
  CHECK_VECTORPTR(bufs1[PARAMETER_LEARNING_RATE],
                  bufs2[PARAMETER_LEARNING_RATE]);
}

TEST(Training, Adagrad) { testCase(testAdagrad); }

void testAdaDelta(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM1, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_LEARNING_RATE, size, useGpu);

  real rou = (real)rand() / (real)RAND_MAX;           // NOLINT
  real epsilon = (real)rand() / (real)RAND_MAX;       // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT
  real momentum = (real)rand() / (real)RAND_MAX;      // NOLINT
  real decayRate = (real)rand() / (real)RAND_MAX;     // NOLINT

  EXPRESSION_PERFORMANCE(AdaDeltaParameterOptimizer(
      bufs1, rou, epsilon, learningRate, momentum, decayRate));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& accum = *bufs2[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& accum_update = *bufs2[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *bufs2[PARAMETER_LEARNING_RATE];

  EXPRESSION_PERFORMANCE(adadeltaApply(value,
                                       grad,
                                       mom,
                                       accum,
                                       accum_update,
                                       lr,
                                       rou,
                                       epsilon,
                                       learningRate,
                                       momentum,
                                       decayRate));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM],
                  bufs2[PARAMETER_GRADIENT_SQURESUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM1],
                  bufs2[PARAMETER_GRADIENT_SQURESUM1]);
  CHECK_VECTORPTR(bufs1[PARAMETER_LEARNING_RATE],
                  bufs2[PARAMETER_LEARNING_RATE]);
}

TEST(Training, AdaDelta) { testCase(testAdaDelta); }

template <bool isFirstTime>
void testRMSProp(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM1, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_LEARNING_RATE, size, useGpu);

  /* make sure 'g - f.square()' greater than 0 */
  bufs1[PARAMETER_GRADIENT_SQURESUM]->add(1.0);
  bufs2[PARAMETER_GRADIENT_SQURESUM]->copyFrom(
      *bufs1[PARAMETER_GRADIENT_SQURESUM]);

  real rou = (real)rand() / (real)RAND_MAX;           // NOLINT
  real epsilon = (real)rand() / (real)RAND_MAX;       // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT
  real momentum = (real)rand() / (real)RAND_MAX;      // NOLINT
  real decayRate = (real)rand() / (real)RAND_MAX;     // NOLINT
  real accumulatedRou = rou;

  EXPRESSION_PERFORMANCE(RMSPropParameterOptimizer(bufs1,
                                                   accumulatedRou,
                                                   rou,
                                                   epsilon,
                                                   learningRate,
                                                   momentum,
                                                   decayRate,
                                                   isFirstTime));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& sum = *bufs2[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& sum1 = *bufs2[PARAMETER_GRADIENT_SQURESUM1];
  BaseMatrix& lr = *bufs2[PARAMETER_LEARNING_RATE];

  EXPRESSION_PERFORMANCE(rmspropApply(value,
                                      grad,
                                      mom,
                                      sum,
                                      sum1,
                                      lr,
                                      accumulatedRou,
                                      rou,
                                      epsilon,
                                      learningRate,
                                      momentum,
                                      decayRate,
                                      isFirstTime));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM],
                  bufs2[PARAMETER_GRADIENT_SQURESUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM1],
                  bufs2[PARAMETER_GRADIENT_SQURESUM1]);
  CHECK_VECTORPTR(bufs1[PARAMETER_LEARNING_RATE],
                  bufs2[PARAMETER_LEARNING_RATE]);
}

TEST(Training, RMSProp) {
  testCase(testRMSProp<true>);
  testCase(testRMSProp<false>);
}

template <bool isFirstTime>
void testDecayedAdagrad(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT_SQURESUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_LEARNING_RATE, size, useGpu);

  real rou = (real)rand() / (real)RAND_MAX;           // NOLINT
  real epsilon = (real)rand() / (real)RAND_MAX;       // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT
  real momentum = (real)rand() / (real)RAND_MAX;      // NOLINT
  real decayRate = (real)rand() / (real)RAND_MAX;     // NOLINT
  real accumulatedRou = rou;

  if (isFirstTime) {
    bufs1[PARAMETER_GRADIENT_SQURESUM]->zeroMem();
    bufs2[PARAMETER_GRADIENT_SQURESUM]->zeroMem();
  }

  EXPRESSION_PERFORMANCE(DecayedAdagradParameterOptimizer(bufs1,
                                                          accumulatedRou,
                                                          rou,
                                                          epsilon,
                                                          learningRate,
                                                          momentum,
                                                          decayRate,
                                                          isFirstTime));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& sum = *bufs2[PARAMETER_GRADIENT_SQURESUM];
  BaseMatrix& lr = *bufs2[PARAMETER_LEARNING_RATE];

  EXPRESSION_PERFORMANCE(decayedAdagradApply(value,
                                             grad,
                                             mom,
                                             sum,
                                             lr,
                                             accumulatedRou,
                                             rou,
                                             epsilon,
                                             learningRate,
                                             momentum,
                                             decayRate,
                                             isFirstTime));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_GRADIENT_SQURESUM],
                  bufs2[PARAMETER_GRADIENT_SQURESUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_LEARNING_RATE],
                  bufs2[PARAMETER_LEARNING_RATE]);
}

TEST(Training, DecayedAdagrad) {
  testCase(testDecayedAdagrad<false>);
  testCase(testDecayedAdagrad<true>);
}

void testAdam(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_SECOND_MOMENTUM, size, useGpu);

  real beta1 = (real)rand() / (real)RAND_MAX;         // NOLINT
  real beta2 = (real)rand() / (real)RAND_MAX;         // NOLINT
  real beta1_power = (real)rand() / (real)RAND_MAX;   // NOLINT
  real beta2_power = (real)rand() / (real)RAND_MAX;   // NOLINT
  real epsilon = (real)rand() / (real)RAND_MAX;       // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT

  EXPRESSION_PERFORMANCE(AdamParameterOptimizer(
      bufs1, beta1, beta2, beta1_power, beta2_power, epsilon, learningRate));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& v = *bufs2[PARAMETER_SECOND_MOMENTUM];

  EXPRESSION_PERFORMANCE(adamApply(value,
                                   grad,
                                   mom,
                                   v,
                                   beta1,
                                   beta2,
                                   beta1_power,
                                   beta2_power,
                                   epsilon,
                                   learningRate));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_SECOND_MOMENTUM],
                  bufs2[PARAMETER_SECOND_MOMENTUM]);
}

TEST(Training, Adam) { testCase(testAdam); }

void testAdamax(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_WEIGHTED_INFINITY_NORM, size, useGpu);

  real beta1 = (real)rand() / (real)RAND_MAX;  // NOLINT
  real beta2 = (real)rand() / (real)RAND_MAX;  // NOLINT
  real alpha = (real)rand() / (real)RAND_MAX;  // NOLINT
  int64_t step = 2;

  EXPRESSION_PERFORMANCE(
      AdamaxParameterOptimizer(bufs1, beta1, beta2, step, alpha));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& mom = *bufs2[PARAMETER_MOMENTUM];
  BaseMatrix& u = *bufs2[PARAMETER_WEIGHTED_INFINITY_NORM];

  EXPRESSION_PERFORMANCE(
      adamaxApply(value, grad, mom, u, beta1, beta2, step, alpha));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM], bufs2[PARAMETER_MOMENTUM]);
  CHECK_VECTORPTR(bufs1[PARAMETER_WEIGHTED_INFINITY_NORM],
                  bufs2[PARAMETER_WEIGHTED_INFINITY_NORM]);
}

TEST(Training, Adamax) {
#ifndef PADDLE_TYPE_DOUBLE
  SetMaxDiff diff(1e-4);
#endif
  testCase(testAdamax);
}

void testSparseMomentum(size_t size, bool useGpu) {
  VectorPtr bufs1[NUM_PARAMETER_TYPES];
  VectorPtr bufs2[NUM_PARAMETER_TYPES];
  INIT_VECTOR(bufs1, bufs2, PARAMETER_VALUE, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_GRADIENT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM_UT, size, useGpu);
  INIT_VECTOR(bufs1, bufs2, PARAMETER_MOMENTUM_VT, size, useGpu);

  real alpha = (real)rand() / (real)RAND_MAX;         // NOLINT
  real beta = (real)rand() / (real)RAND_MAX;          // NOLINT
  real gamma = (real)rand() / (real)RAND_MAX;         // NOLINT
  real tau = (real)rand() / (real)RAND_MAX;           // NOLINT
  real learningRate = (real)rand() / (real)RAND_MAX;  // NOLINT

  EXPRESSION_PERFORMANCE(SparseMomentumParameterOptimizer(
      bufs1, alpha, beta, gamma, tau, learningRate));

  BaseMatrix& value = *bufs2[PARAMETER_VALUE];
  BaseMatrix& grad = *bufs2[PARAMETER_GRADIENT];
  BaseMatrix& momU = *bufs2[PARAMETER_MOMENTUM_UT];
  BaseMatrix& momV = *bufs2[PARAMETER_MOMENTUM_VT];

  EXPRESSION_PERFORMANCE(sparseMomentumApply(
      value, grad, momU, momV, alpha, beta, gamma, tau, learningRate));

  CHECK_VECTORPTR(bufs1[PARAMETER_VALUE], bufs2[PARAMETER_VALUE]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM_UT], bufs2[PARAMETER_MOMENTUM_UT]);
  CHECK_VECTORPTR(bufs1[PARAMETER_MOMENTUM_VT], bufs2[PARAMETER_MOMENTUM_VT]);
}

TEST(Training, SparseMomentum) { testCase(testSparseMomentum); }
