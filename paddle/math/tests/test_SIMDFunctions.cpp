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

#include "paddle/math/SIMDFunctions.h"
#include "paddle/utils/Util.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#include <stdlib.h>
#include <time.h>

static constexpr size_t VECTOR_LEN = 3072;
static constexpr size_t BATCH_SIZE = 64;
static constexpr size_t ALIGN = 32;
static_assert(VECTOR_LEN % ALIGN == 0, "VECTOR_LEN % ALIGN == 0");
static_assert(BATCH_SIZE % ALIGN == 0, "BATCH_SIZE % ALIGN == 0");
static constexpr float EPSILON = 1e-5;
static std::mt19937 RandomEngine(time(0));

inline static std::unique_ptr<float[]> NewVector(size_t len = VECTOR_LEN,
                                                 size_t align = ALIGN) {
  float* ptr;
  CHECK_EQ(posix_memalign((void**)&ptr, align, len * sizeof(float)), 0);
  return std::unique_ptr<float[]>(ptr);
}

inline static std::unique_ptr<float[]> NewRandomVector(size_t len = VECTOR_LEN,
                                                       size_t align = ALIGN) {
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  auto generator = std::bind(dist, RandomEngine);
  auto retv = NewVector(len, align);
  std::generate_n(retv.get(), len, generator);
  return retv;
}

TEST(SIMDFunction, addTo) {
  typedef std::function<void(float*, const float*, size_t)> AddToMethodType;

  AddToMethodType naive = paddle::simd::naive::addTo<float>;
  AddToMethodType simd = paddle::simd::addTo<float>;

  auto A = NewRandomVector();
  auto B = NewRandomVector();

  auto ACopy = NewVector();
  memcpy(ACopy.get(), A.get(), VECTOR_LEN * sizeof(float));

  naive(A.get(), B.get(), VECTOR_LEN);
  simd(ACopy.get(), B.get(), VECTOR_LEN);

  for (size_t i = 0; i < VECTOR_LEN; ++i) {
    ASSERT_NEAR(A[i], ACopy[i], EPSILON);
  }
}

TEST(SIMDFunction, batchAddTo) {
  auto A = NewRandomVector();
  auto ACopy = NewVector();
  memcpy(ACopy.get(), A.get(), sizeof(float) * VECTOR_LEN);

  std::vector<std::unique_ptr<float[]>> B;
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    B.emplace_back(NewRandomVector());
  }
  std::unique_ptr<float* []> BRaw(new float*[BATCH_SIZE]);
  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    BRaw[i] = B[i].get();
  }

  typedef std::function<void(float*, const float**, int, size_t)>
      BatchAddToMethodType;

  BatchAddToMethodType naive = paddle::simd::naive::batchAddTo<float>;
  BatchAddToMethodType simd = paddle::simd::batchAddTo<float>;

  naive(A.get(), (const float**)BRaw.get(), BATCH_SIZE, VECTOR_LEN);
  simd(ACopy.get(), (const float**)BRaw.get(), BATCH_SIZE, VECTOR_LEN);

  for (size_t i = 0; i < VECTOR_LEN; ++i) {
    ASSERT_NEAR(A[i], ACopy[i], EPSILON);
  }
}

TEST(SIMDFunction, colMax) {
  auto A = NewRandomVector(VECTOR_LEN * BATCH_SIZE);
  auto naiveResult = NewVector(BATCH_SIZE);
  auto simdResult = NewVector(BATCH_SIZE);

  typedef std::function<void(float*, const float*, int, int)> ColMaxMethodType;
  ColMaxMethodType naive = paddle::simd::naive::colMax<float>;
  ColMaxMethodType simd = paddle::simd::colMax<float>;

  naive(naiveResult.get(), A.get(), BATCH_SIZE, VECTOR_LEN);
  simd(simdResult.get(), A.get(), BATCH_SIZE, VECTOR_LEN);

  for (size_t i = 0; i < BATCH_SIZE; ++i) {
    ASSERT_NEAR(naiveResult[i], simdResult[i], EPSILON);
  }
}

TEST(SIMDFunction, decayL1_WithLR) {
  auto dest = NewRandomVector();
  auto src = NewRandomVector();
  auto lr = NewRandomVector();
  auto lambda = 0.23f;

  auto simd_dest = NewVector();
  memcpy(simd_dest.get(), dest.get(), sizeof(float) * VECTOR_LEN);

  typedef std::function<void(float*, float*, float*, float, size_t)>
      DecayL1MethodType;

  DecayL1MethodType naive = [](
      float* d, float* s, float* lr, float l, size_t len) {
    paddle::simd::naive::decayL1<float>(d, s, lr, l, len);
  };

  DecayL1MethodType simd = [](
      float* d, float* s, float* lr, float l, size_t len) {
    paddle::simd::decayL1<float>(d, s, lr, l, len);
  };

  naive(dest.get(), src.get(), lr.get(), lambda, VECTOR_LEN);
  simd(simd_dest.get(), src.get(), lr.get(), lambda, VECTOR_LEN);

  for (size_t i = 0; i < VECTOR_LEN; ++i) {
    ASSERT_NEAR(dest[i], simd_dest[i], EPSILON);
  }
}

TEST(SIMDFunction, decayL1_WithoutLR) {
  auto dest = NewRandomVector();
  auto src = NewRandomVector();
  auto lambda = 0.23;

  auto simd_dest = NewVector();
  memcpy(simd_dest.get(), dest.get(), sizeof(float) * VECTOR_LEN);

  typedef std::function<void(float*, float*, float, size_t)> DecayL1MethodType;

  DecayL1MethodType naive = [](float* d, float* s, float l, size_t len) {
    paddle::simd::naive::decayL1<float>(d, s, l, len);
  };

  DecayL1MethodType simd = [](float* d, float* s, float l, size_t len) {
    paddle::simd::decayL1<float>(d, s, l, len);
  };

  naive(dest.get(), src.get(), lambda, VECTOR_LEN);
  simd(simd_dest.get(), src.get(), lambda, VECTOR_LEN);

  for (size_t i = 0; i < VECTOR_LEN; ++i) {
    ASSERT_NEAR(dest[i], simd_dest[i], EPSILON);
  }
}
