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

#include <paddle/utils/Util.h>
#include <stdlib.h>

#include <gtest/gtest.h>
#include <paddle/parameter/ParameterUpdateFunctions.h>
#include <paddle/utils/Flags.h>
#include <paddle/utils/Stat.h>
#include <paddle/utils/Thread.h>

using namespace paddle;  // NOLINT

class CommonTest : public ::testing::Test {
protected:
  CommonTest() : testStat_("test") {}
  virtual ~CommonTest() {}
  virtual void SetUp() {
    const size_t buffSize[] = {
        100, 128, 500, 1024, 4096, 10240, 102400, 1000000};
    sizeVec_.resize(8);
    memcpy(&sizeVec_[0], &buffSize[0], 8 * sizeof(size_t));
    valueUint_.resize(4);
    valueUint_[0].first = 0.0;
    valueUint_[0].second = 0.0;
    valueUint_[1].first = 0.0;
    valueUint_[1].second = 1.0;
    valueUint_[2].first = 1.0;
    valueUint_[2].second = 0.0;
    valueUint_[3].first = 1.0;
    valueUint_[3].second = 1.0;
    learningRate_ = 1.0;
  }

  void test_sgdUpadate(real* gradientBuffer,
                       real* valueBuffer,
                       real* momentumBuffer,
                       size_t size);

  virtual void TreaDown() { LOG(INFO) << "All Test Finished."; }

protected:
  std::vector<std::pair<real, real>> valueUint_;
  std::vector<size_t> sizeVec_;
  real learningRate_;
  StatSet testStat_;
};

void CommonTest::test_sgdUpadate(real* gradientBuffer,
                                 real* valueBuffer,
                                 real* momentumBuffer,
                                 size_t size) {
// sgdUpdateAvx has no double version yet
#if defined(__AVX__) && !defined(PADDLE_TYPE_DOUBLE)
  real valueSum1 = 0, valueSum2 = 0, momSum1 = 0, momSum2 = 0;
  real* gradTmp = new real[size];
  real* valueTmp = new real[size];
  real* momentumTmp = new real[size];
  memcpy(gradTmp, gradientBuffer, size * sizeof(real));
  memcpy(valueTmp, valueBuffer, size * sizeof(real));
  memcpy(momentumTmp, momentumBuffer, size * sizeof(real));
  for (auto& arg : valueUint_) {
    {
      {
        struct timeval t;
        REGISTER_TIMER("gettimeofday", 0, testStat_);
        gettimeofday(&t, NULL);
      }
      REGISTER_TIMER("avxTimer", 0);
      sgdUpdateAvx(learningRate_,
                   arg.first,
                   arg.second,
                   size,
                   valueBuffer,
                   gradientBuffer,
                   momentumBuffer);
    }
    for (size_t i = 0; i < size; i++) {
      valueSum1 += valueBuffer[i];
      momSum1 += momentumBuffer[i];
      // std::cout << "["
      //          << valueBuffer[i]
      //          << "," << momentumBuffer[i]
      //          << "," << gradientBuffer[i] << "],";
    }
    {
      REGISTER_TIMER("cpuTimer", 0);
      sgdUpdateCpu(learningRate_,
                   arg.first,
                   arg.second,
                   size,
                   valueTmp,
                   gradTmp,
                   momentumTmp);
    }
    for (size_t i = 0; i < size; i++) {
      valueSum2 += valueTmp[i];
      momSum2 += momentumTmp[i];
      // std::cout << "["
      //          << valueTmp[i]
      //          << "," << momentumTmp[i]
      //          << "," << gradTmp[i] << "],";
    }

    VLOG(3) << "valueSum1 = " << valueSum1 << " ; valueSum2 = " << valueSum2;
    VLOG(3) << "momSum1 = " << momSum1 << " ; momSum2 = " << momSum2;
    ASSERT_EQ(valueSum1, valueSum2);
    ASSERT_EQ(momSum1, momSum2);
  }
  delete[] gradTmp;
  delete[] valueTmp;
  delete[] momentumTmp;
#endif
}

TEST_F(CommonTest, sgdUpdate) {
  const size_t alignHeader[] = {0, 2, 3, 5, 7, 8};
  for (auto& size : sizeVec_) {
    real *gradientBuffer, *valueBuffer, *momentumBuffer;
    CHECK_EQ(posix_memalign((void**)&gradientBuffer, 32, sizeof(real) * size),
             0);
    CHECK_EQ(posix_memalign((void**)&valueBuffer, 32, sizeof(real) * size), 0);
    CHECK_EQ(posix_memalign((void**)&momentumBuffer, 32, sizeof(real) * size),
             0);

    for (size_t i = 0; i < size; i++) {
      gradientBuffer[i] = 1.0;
      valueBuffer[i] = 2.0;
      momentumBuffer[i] = 3.0;
    }
    for (int i = 0; i < 6; i++) {
      LOG(INFO) << "----------------------" << size << ":" << alignHeader[i]
                << "-------------------------";
      test_sgdUpadate(&gradientBuffer[alignHeader[i]],
                      &valueBuffer[alignHeader[i]],
                      &momentumBuffer[alignHeader[i]],
                      size - alignHeader[i]);
    }
    free(gradientBuffer);
    free(valueBuffer);
    free(momentumBuffer);
  }
  globalStat.printAllStatus();
  testStat_.printAllStatus();
}

TEST_F(CommonTest, syncThreadPool) {
  SyncThreadPool pool(10);

  std::vector<int> nums;
  nums.resize(10);

  pool.exec([&](int tid, size_t numThreads) { nums[tid] = tid; });
  for (size_t i = 0; i < nums.size(); ++i) {
    EXPECT_EQ((int)i, nums[i]);
  }

  pool.exec([&](int tid, size_t numThreads) { nums[tid] -= tid; });
  for (size_t i = 0; i < nums.size(); ++i) {
    EXPECT_EQ((int)0, nums[i]);
  }
}

TEST_F(CommonTest, barrierStat) {
  const int threadNum = 10;

  SyncThreadPool pool(threadNum);

#define TEST_BARRIER_RANDOM(statName, numConnThreads, ...)       \
  pool.exec([&](int tid, size_t numThreads) {                    \
    struct timeval time;                                         \
    gettimeofday(&time, nullptr);                                \
    uint64_t usec = timeToMicroSecond(time);                     \
    std::srand(usec);                                            \
    auto value = std::rand() % 100000;                           \
    usleep(value);                                               \
    REGISTER_SLOW_NODES_PROBE(                                   \
        globalStat, statName, numConnThreads, tid, __VA_ARGS__); \
  });

  for (auto i = 0; i < 10; i++) {
    TEST_BARRIER_RANDOM("synThreadBarrier1", threadNum);
    TEST_BARRIER_RANDOM("synThreadBarrier2", threadNum);
  }

  globalStat.printAllStatus();
  globalStat.reset();

  for (auto i = 0; i < 10; i++) {
    TEST_BARRIER_RANDOM("synThreadBarrier3", threadNum, "tag0");
    TEST_BARRIER_RANDOM("synThreadBarrier4", threadNum, "tag1");
  }

  globalStat.printAllStatus();
  globalStat.reset();

// use it to test accurate barrier gap
#define TEST_BARRIER(statName, numConnThreads, ...)              \
  pool.exec([&](int tid, size_t numThreads) {                    \
    usleep(tid * 10000);                                         \
    REGISTER_SLOW_NODES_PROBE(                                   \
        globalStat, statName, numConnThreads, tid, __VA_ARGS__); \
  });

  for (auto i = 0; i < 10; i++) {
    TEST_BARRIER("synThreadBarrier3", threadNum, "tag0");
    TEST_BARRIER("synThreadBarrier4", threadNum, "tag1");
  }

  globalStat.printAllStatus();
  globalStat.reset();
}
