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

#include "BarrierStat.h"
#include "Locks.h"
#include "Logging.h"
#include "ThreadLocal.h"
#include "hl_gpu.h"

namespace paddle {

class Stat;

class StatInfo {
public:
  explicit StatInfo(Stat* stat = nullptr) : stat_(stat) {
    total_ = 0;
    max_ = 0;
    count_ = 0;
    min_ = UINT64_MAX;
  }

  void reset() {
    total_ = 0;
    count_ = 0;
    max_ = 0;
    min_ = UINT64_MAX;
  }

  ~StatInfo();

  Stat* stat_;
  uint64_t total_;
  uint64_t max_;
  uint64_t count_;
  uint64_t min_;
};

class Stat;
typedef std::shared_ptr<Stat> StatPtr;
typedef std::shared_ptr<BarrierStatBase> BarrierStatPtr;

enum BarrierStatType {
  BARRIER_END = 0,
  BARRIER_DELTA = 1,
};

class StatSet {
public:
  explicit StatSet(const std::string& name) : name_(name) {}
  ~StatSet() {}

  // print to LOG(INFO)
  void printSegTimerStatus();
  void printBarrierTimerStatus();
  void printAllStatus();

  void printStatus(const std::string& name);

  StatPtr getStat(const std::string& name) {
    {
      ReadLockGuard guard(lock_);
      auto it = statSet_.find(name);
      if (it != statSet_.end()) {
        return it->second;
      }
    }
    StatPtr stat = std::make_shared<Stat>(name);
    std::lock_guard<RWLock> guard(lock_);
    auto ret = statSet_.insert(std::make_pair(name, stat));
    return ret.first->second;
  }

  BarrierStatPtr getStat(uint16_t numConnThreads,
                         const std::string& name,
                         BarrierStatType bType);

  void deleteStat(const std::string& name);

  // true for showing stats for each thread
  // false for showing stats aggragated over threads
  void setThreadInfo(const std::string& name, bool flag);

  // true for showing stats for each thread
  // false for showing stats aggragated over threads
  void setThreadInfo(bool flag) {
    for (auto& iter : statSet_) {
      setThreadInfo(iter.first, flag);
    }
  }

  // reset the counters for all stats
  // clearRawData means also clearing raw tuning data, because at pserver end,
  // barrier rawData(timeVector_) is stateful, clearing it will cause rubbish
  // data, while rawData should be cleared at the new pass (so complicated
  // pserver code logic, -_- ).
  void reset(bool clearRawData = true);

private:
  std::unordered_map<std::string, StatPtr> statSet_;
  std::unordered_map<std::string, BarrierStatPtr> barrierStatSet_;
  const std::string name_;
  RWLock lock_;
};

extern StatSet globalStat;

/*@brief : a simple stat*/
class Stat {
public:
  explicit Stat(const std::string& statName)
      : destructStat_(nullptr), name_(statName), openThreadInfo_(false) {}
  ~Stat() {}

  typedef std::list<std::pair<StatInfo*, pid_t>> ThreadLocalBuf;

  const std::string& getName() const { return name_; }

  void addSample(uint64_t value);

  // clear all stats
  void reset();

  friend std::ostream& operator<<(std::ostream& outPut, const Stat& stat);

  /*  Set operator << whether to print thread info.
   *  If openThreadInfo_ == true, then print, else print merge thread info.
   */
  void setThreadInfo(bool flag) { openThreadInfo_ = flag; }

  bool getThreadInfo() const { return openThreadInfo_; }

  friend class StatInfo;

private:
  void mergeThreadStat(StatInfo& allThreadStat);

  std::mutex lock_;
  ThreadLocalBuf threadLocalBuf_;
  StatInfo destructStat_;
  ThreadLocal<StatInfo> statInfo_;
  const std::string name_;
  bool openThreadInfo_;
};

extern StatSet globalStat;

inline StatPtr getStat(const std::string& name) {
  return globalStat.getStat(name);
}

inline uint64_t nowInMicroSec() {
  timeval tvTime;
  (void)gettimeofday(&tvTime, NULL);
  return tvTime.tv_sec * 1000000LU + tvTime.tv_usec;
}

/**
 * A simple help class to measure time interval
 */
class Timer {
public:
  explicit Timer(bool autoStart = true) : total_(0), startStamp_(0) {
    if (autoStart) {
      start();
    }
  }
  void start() { startStamp_ = nowInMicroSec(); }
  void setStartStamp(uint64_t startStamp) { startStamp_ = startStamp; }
  uint64_t stop() {
    total_ += nowInMicroSec() - startStamp_;
    return total_;
  }

  uint64_t get() const { return total_; }

  void reset() { total_ = 0; }

protected:
  uint64_t total_;
  uint64_t startStamp_;
};

class TimerOnce {
public:
  TimerOnce(Stat* stat,
            const char* info = "",
            uint64_t threshold = -1,
            bool autoStart = true,
            uint64_t startStamp = 0)
      : stat_(stat), info_(info), timer_(autoStart), threshold_(threshold) {
    if (!autoStart) {
      timer_.setStartStamp(startStamp);
    }
  }
  ~TimerOnce() {
    uint64_t span = timer_.stop();
    if (span >= threshold_) {
      LOG(INFO) << "Stat: [" << stat_->getName() << "] " << info_
                << " [Span:" << span / 1000 << "ms" << span % 1000 << "us"
                << "] ";
    }
    stat_->addSample(span);
  }

private:
  Stat* stat_;
  const char* info_;
  Timer timer_;
  uint64_t threshold_;
};

inline uint64_t registerTimerArg1(uint64_t threshold = -1,
                                  StatSet& statSet = globalStat) {
  return threshold;
}

inline StatSet& registerTimerArg2(uint64_t threshold = -1,
                                  StatSet& statSet = globalStat) {
  return statSet;
}

#ifdef PADDLE_DISABLE_TIMER

#define REGISTER_TIMER(statName, ...)
#define REGISTER_TIMER_SET(statName, start, ...)
#define REGISTER_TIMER_DYNAMIC(statName, ...)
#define REGISTER_TIMER_DYNAMIC_SET(statName, start, ...)
#define REGISTER_TIMER_INFO(statName, info)
#define FOR_TIMING(statement)

#else

#define FOR_TIMING(statement) statement

// The default arguments are shown in the following line:
// REGISTER_TIMER(statName, threshold = -1, statSet = globalStat)
// TODO(yuyang18,wangyanfei01): if UNIQUE_NAME is needed
#define REGISTER_TIMER(statName, ...)                             \
  static ::paddle::StatPtr __stat =                               \
      ::paddle::registerTimerArg2(__VA_ARGS__).getStat(statName); \
  ::paddle::TimerOnce __timerOnce(                                \
      __stat.get(), "", ::paddle::registerTimerArg1(__VA_ARGS__));

#define REGISTER_TIMER_SET(statName, start, ...)                            \
  static ::paddle::StatPtr __stat =                                         \
      ::paddle::registerTimerArg2(__VA_ARGS__).getStat(statName);           \
  ::paddle::TimerOnce __timerOnce(__stat.get(),                             \
                                  "",                                       \
                                  ::paddle::registerTimerArg1(__VA_ARGS__), \
                                  false,                                    \
                                  start);

// dynmaic timer, support to discriminate runtime entity, used in pserver
#define REGISTER_TIMER_DYNAMIC(statName, ...)                     \
  ::paddle::StatPtr __stat =                                      \
      ::paddle::registerTimerArg2(__VA_ARGS__).getStat(statName); \
  ::paddle::TimerOnce __timerOnce(                                \
      __stat.get(), "", ::paddle::registerTimerArg1(__VA_ARGS__));

#define REGISTER_TIMER_DYNAMIC_SET(statName, start, ...)                    \
  ::paddle::StatPtr __stat =                                                \
      ::paddle::registerTimerArg2(__VA_ARGS__).getStat(statName);           \
  ::paddle::TimerOnce __timerOnce(__stat.get(),                             \
                                  "",                                       \
                                  ::paddle::registerTimerArg1(__VA_ARGS__), \
                                  false,                                    \
                                  start);

#define REGISTER_TIMER_INFO(statName, info)                                 \
  static ::paddle::StatPtr __stat = ::paddle::globalStat.getStat(statName); \
  ::paddle::TimerOnce __timerOnce(                                          \
      __stat.get(), info, 10 * 1000000LU /*threshold*/);

#endif  // DISABLE_TIMER

class GpuProfiler final {
public:
  GpuProfiler(std::string statName, std::string info);
  ~GpuProfiler();

private:
  std::lock_guard<std::recursive_mutex> guard_;
};

#ifdef PADDLE_DISABLE_PROFILER

#define REGISTER_GPU_PROFILER(statName, ...)

#else

#define REGISTER_GPU_PROFILER(statName, ...) \
  GpuProfiler __gpuProfiler(statName, #__VA_ARGS__);

#endif  // DISABLE_PROFILER

}  // namespace paddle
