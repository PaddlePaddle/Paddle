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

#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Locks.h"
#include "Logging.h"
#include "ThreadLocal.h"

namespace paddle {

inline uint64_t timeToMicroSecond(struct timeval time) {
  return time.tv_sec * 1000000LU + time.tv_usec;
}

class TimeVectorEnd {
  /*
   * help class for gathering all barrier performance data
   * which shows time point property.
   * freqently used in barrier performance tuning API, such
   * as tuning which is slowest node in sync-sgd mode training.
   */
public:
  explicit TimeVectorEnd(uint16_t size) : size_(size) {
    index_ = 0;
    timeArray_.resize(size);
    trainerIds_.resize(size);
  }
  ~TimeVectorEnd() {}

  uint16_t size() { return size_; }

  bool full() { return index_ == size_; }

  bool empty() { return index_ == 0; }

  void reset() { index_ = 0; }

  void addTimeval(struct timeval time, int32_t trainerId) {
    timeArray_[index_] = time;
    trainerIds_[index_] = trainerId;
    index_++;
  }

  struct timeval getDelta() const {
    struct timeval delta;
    CHECK_GT(size_, 1) << "not support with 1 pserver";
    timersub(&timeArray_[size_ - 1], &timeArray_[0], &delta);
    return delta;
  }

  /* 2, n delta */
  struct timeval get1NDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    struct timeval delta;
    timersub(&timeArray_[size_ - 1], &timeArray_[1], &delta);
    return delta;
  }

  /* n-1, n delta */
  struct timeval getMinus1NDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    struct timeval delta;
    timersub(&timeArray_[size_ - 1], &timeArray_[size_ - 2], &delta);
    return delta;
  }

  /* n/2, n delta */
  struct timeval getMidNDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    struct timeval delta;
    timersub(&timeArray_[size_ - 1], &timeArray_[size_ / 2], &delta);
    return delta;
  }

  int32_t getLastTrainerId() const { return trainerIds_[index_ - 1]; }

private:
  uint16_t size_;
  uint16_t index_;
  std::vector<struct timeval> timeArray_;
  std::vector<int32_t> trainerIds_;
};

class TimeVectorDelta {
  /*
   * help class for gathering performance data which shows time
   * delta property, such as tuning the time distribution of
   * forwardBackward time from all cluster nodes.
   */
public:
  explicit TimeVectorDelta(uint16_t size)
      : size_(size), min_(UINT64_MAX), max_(0) {
    index_ = 0;
    timeArray_.resize(size);
  }
  ~TimeVectorDelta() {}

  uint16_t size() { return size_; }

  bool full() { return index_ == size_; }

  bool empty() { return index_ == 0; }

  void reset() {
    index_ = 0;
    min_ = UINT64_MAX;
    max_ = 0;
  }

  void addTimeval(uint64_t delta, int32_t trainerId) {
    timeArray_[index_] = delta;
    index_++;
    if (delta < min_) {
      min_ = delta;
    }
    if (delta > max_) {
      max_ = delta;
      maxTrainerId_ = trainerId;
    }
  }

  uint64_t getDelta() const {
    CHECK_GT(size_, 1) << "not support with 1 pserver";
    return max_ - min_;
  }

  /* 2, n delta */
  uint64_t get1NDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    LOG(FATAL) << "Not implemented";
  }

  /* n-1, n delta */
  uint64_t getMinus1NDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    LOG(FATAL) << "Not implemented";
  }

  /* n/2, n delta */
  uint64_t getMidNDelta() const {
    CHECK_GT(size_, 2) << "not support with less than 2 pservers";
    LOG(FATAL) << "Not implemented";
  }

  int32_t getMaxTrainerId() const { return maxTrainerId_; }

private:
  uint16_t size_;
  uint16_t index_;
  std::vector<uint64_t> timeArray_;

private:
  uint64_t min_;
  uint64_t max_;
  int32_t maxTrainerId_;
};

// total samples stats, us
struct Abstract {
  // last trainerId for barrier end, maxDelta trainerId for barrier delta
  int32_t trainerId;
  uint64_t minDelta;
  uint64_t maxDelta;
  uint64_t totDelta;
  // first one is probably itself, so discard it.
  uint64_t totSecondDelta;
  // to confirm if last node destroy barrier performance.
  uint64_t totLastTwoDelta;
  // n/2-n delta
  uint64_t totMidDelta;
  uint64_t freq;
};

// barrier performance tunning stats
class BarrierStatBase {
public:
  BarrierStatBase(uint16_t numConnThreads, const std::string &name);

  virtual ~BarrierStatBase() {}

  // if called at pserver end, then trainId means trainer's id.
  // by default trainer does not use trainerId, so set it to -1
  virtual void updateStat(struct timeval &cur, int32_t trainerId = -1) = 0;
  virtual void updateStat(uint64_t delta, int32_t trainerId = -1) = 0;

  const std::string &getName() { return name_; }

  virtual void reset(bool clearRawData = true) {}
  // since the timeVector_ is not stateful, so it's not clear whether the
  // the barrier delta is correct. if one timestamp was lost, the all data
  // from barrier stat becomes rubbish. -_-
  virtual bool checkPassBarrier() {
    LOG(INFO) << "bug implementation found";
    return false;
  }

protected:
  virtual void showAbstract(std::ostream &output) const {}
  friend std::ostream &operator<<(std::ostream &output,
                                  const BarrierStatBase &stat);

protected:
  mutable std::mutex lock_;
  std::mutex abstractLock_;  // see note on updaterStat
  // each freqency for each barrier trainer
  std::vector<struct Abstract> abstract_;
  // it is valuable when do perf-tuining, if lastTrainerId acts uniform
  // distribution
  struct Abstract totAbstract_;
  uint64_t totSamples_;

protected:
  uint16_t numConnThreads_;  // total updates needed
  float rateThreshold_;
  std::string name_;
};

// the end-time of arriving real/forged barrier position
class BarrierEndStat : public BarrierStatBase {
public:
  BarrierEndStat(uint16_t numConnThreads, const std::string &name);
  ~BarrierEndStat() {}

  virtual void updateStat(struct timeval &cur, int32_t trainerId = -1);
  virtual void updateStat(uint64_t delta, int32_t trainerId = -1) {
    LOG(INFO) << "have no delta updateStat in BarrierEndStat";
  }
  virtual void reset(bool clearRawData = true);
  virtual bool checkPassBarrier() { return timeVector_->empty(); }

protected:
  /*
   * LOG:
   * readAllBlocks_denseUpdater
   * trainerId      avgGap         avgSecondGap   avgLastTwoGap  avgMidGap rate
   * 44             86.702         81.022         9.984          50.472 0.144737
   * 46             87.723         82.939         8.737          50.019 0.118421
   * 35             100.923        96.752         14.305         61.979
   * 0.0657895
   * log_barrier_abstract, log_barrier_lowest_nodes, log_barrier_threshold
   * control details.
   */
  virtual void showAbstract(std::ostream &output) const;

private:
  std::unique_ptr<TimeVectorEnd> timeVector_;
};

// the delta-time from different trainers,
// eg, find the degree of imbalance of BP time at pserver end
// the entry value in timerVector_ is BP delta, do evaluation to BP delta.
class BarrierDeltaStat : public BarrierStatBase {
public:
  BarrierDeltaStat(uint16_t numConnThreads, const std::string &name);
  ~BarrierDeltaStat() {}

  virtual void updateStat(uint64_t delta, int32_t trainerId = -1);
  virtual void updateStat(struct timeval &cur, int32_t trainerId = -1) {
    LOG(INFO) << "have no timeval updateStat in BarrierDeltaStat";
  }

  virtual void reset(bool clearRawData = true);

  virtual bool checkPassBarrier() { return timeVector_->empty(); }

protected:
  virtual void showAbstract(std::ostream &outPut) const;

private:
  // store delta time in uint64_t, eg BP time of all trainers
  std::unique_ptr<TimeVectorDelta> timeVector_;
};

// to distinguish different contexts for same parallel threads, and different
// threads with same code-sgement, just use tagName to tag the run-time
// position.
// in Sparse, sendParallel threads can not only run in the stage of push&pull
// with same thread group, but also run in the stage of pull&push with different
// thread group, tag will be used to distinguish different run-time barrier
// position.
// trainerId in REGISTER_BARRIER_TIMER_SERVER is used to retreive lowest trainer
// nodes.

// end barrier
#define __REGISTER_BARRIER_TIMER_SERVER(                            \
    set, statName, numConnThreads, trainerId, ...)                  \
  do {                                                              \
    if (numConnThreads > 2) {                                       \
      std::string internalName =                                    \
          std::string(statName) + std::string(__VA_ARGS__);         \
      BarrierStatPtr __stat =                                       \
          (set).getStat(numConnThreads, internalName, BARRIER_END); \
      struct timeval cur;                                           \
      gettimeofday(&cur, nullptr);                                  \
      __stat->updateStat(cur, trainerId);                           \
    }                                                               \
  } while (0);

// end barrier with user-defined timer
#define __REGISTER_BARRIER_TIMER_SERVER_SET(                        \
    set, statName, numConnThreads, trainerId, cur, ...)             \
  do {                                                              \
    if (numConnThreads > 2) {                                       \
      std::string internalName =                                    \
          std::string(statName) + std::string(__VA_ARGS__);         \
      BarrierStatPtr __stat =                                       \
          (set).getStat(numConnThreads, internalName, BARRIER_END); \
      __stat->updateStat(cur, trainerId);                           \
    }                                                               \
  } while (0);

// delta barrier
#define __REGISTER_BARRIER_DELTA_SERVER_SET(                          \
    set, statName, numConnThreads, trainerId, delta, ...)             \
  do {                                                                \
    if (numConnThreads > 2) {                                         \
      std::string internalName =                                      \
          std::string(statName) + std::string(__VA_ARGS__);           \
      BarrierStatPtr __stat =                                         \
          (set).getStat(numConnThreads, internalName, BARRIER_DELTA); \
      __stat->updateStat(delta, trainerId);                           \
    }                                                                 \
  } while (0);

// check end barrier
#define __CHECK_BARRIER_TIMER(set, statName, numConnThreads, ...)   \
  do {                                                              \
    std::string internalName =                                      \
        std::string(statName) + std::string(__VA_ARGS__);           \
    BarrierStatPtr __stat =                                         \
        (set).getStat(numConnThreads, internalName, BARRIER_END);   \
    PCHECK(__stat->checkPassBarrier()) << internalName              \
                                       << ": invalid barrier data"; \
  } while (0);

/*
 * Note:
 * with sync-sgd algriothm in cluster mode, lots of synchronize action exsit at
 * pserve end. these synchronizaton actions have impact on the efficiency of
 * parameter exchange. the synchronizaton(barrier) GAP is composed of lots of
 * factors, such as the forwardBackward variance, network fluncation. we try
 * to have a quantitative analysis on these factor, so we design lots of barrier
 * time to capture these performance. these barrier also can be placed at
 * implict barrier position.
 *
 * example:
 * in sync-sgd algorithm, each parameter server waits for all gradients from
 * all trainers, thus, an explict barrier point exsit before doing optimization.
 * the barrier timer located before the point can sense the barrier condition.
 *
 */

// try to capture which trainer is slowest node in sync-sgd at pserver.
#define REGISTER_SLOW_NODES_PROBE(                 \
    set, statName, numConnThreads, trainerId, ...) \
  __REGISTER_BARRIER_TIMER_SERVER(                 \
      (set), statName, numConnThreads, trainerId, __VA_ARGS__)
// try to check if all threads or trainers have passed barriers for data
// accuracy.
#define CHECK_BARRIER_TIMER(set, statName, numConnThreads, ...) \
  __CHECK_BARRIER_TIMER((set), statName, numConnThreads, __VA_ARGS__)

#ifdef PADDLE_DISABLE_TIMER

#define REGISTER_BARRIER_TIMER_SERVER( \
    set, statName, numConnThreads, trainerId, ...)
#define REGISTER_BARRIER_TIMER_SERVER_SET( \
    set, statName, numConnThreads, trainerId, cur, ...)
#define REGISTER_BARRIER_DELTA_SERVER_SET( \
    set, statName, numConnThreads, trainerId, cur, ...)

#else

/*
 * sensing barrier time distribution for all parallelization threads.
 * it provides low API for slow node check(REGISTER_SLOW_NODES_PROBE)
 */
#define REGISTER_BARRIER_TIMER_SERVER(             \
    set, statName, numConnThreads, trainerId, ...) \
  __REGISTER_BARRIER_TIMER_SERVER(                 \
      (set), statName, numConnThreads, trainerId, __VA_ARGS__)

/*
 * sensing barrier time distribution for all parallelization threads.
 * but time point for barrier performance is set by user.
 * eg, with this api, you can get implict barrier point such as the beginning
 * time distribution
 * for receiving data.
 */
#define REGISTER_BARRIER_TIMER_SERVER_SET(              \
    set, statName, numConnThreads, trainerId, cur, ...) \
  __REGISTER_BARRIER_TIMER_SERVER_SET(                  \
      (set), statName, numConnThreads, trainerId, cur, __VA_ARGS__)

// try to capture time delta from all trainers, such as forwardBackward time
// which implies
// computation fluctuation
#define REGISTER_BARRIER_DELTA_SERVER_SET(                \
    set, statName, numConnThreads, trainerId, delta, ...) \
  __REGISTER_BARRIER_DELTA_SERVER_SET(                    \
      (set), statName, numConnThreads, trainerId, delta, __VA_ARGS__)

#endif  // DISABLE_TIMER
}  // namespace paddle
