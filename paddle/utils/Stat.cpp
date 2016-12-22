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

#include "Stat.h"
#include <algorithm>
#include <iomanip>
#include "Util.h"

namespace paddle {

StatSet globalStat("GlobalStatInfo");

void Stat::addSample(uint64_t value) {
  StatInfo* statInfo = statInfo_.get(false);
  if (!statInfo) {
    statInfo = new StatInfo(this);
    statInfo_.set(statInfo);
    std::lock_guard<std::mutex> guard(lock_);
    threadLocalBuf_.push_back({statInfo, getTID()});
  }
  if (value > statInfo->max_) {
    statInfo->max_ = value;
  }
  if (value < statInfo->min_) {
    statInfo->min_ = value;
  }
  statInfo->total_ += value;
  statInfo->count_++;
}

void Stat::mergeThreadStat(StatInfo& allThreadStat) {
  allThreadStat = destructStat_;
  for (auto& buf : threadLocalBuf_) {
    if (buf.first->max_ > allThreadStat.max_) {
      allThreadStat.max_ = buf.first->max_;
    }
    if (buf.first->min_ < allThreadStat.min_) {
      allThreadStat.min_ = buf.first->min_;
    }
    allThreadStat.total_ += buf.first->total_;
    allThreadStat.count_ += buf.first->count_;
  }
}

void Stat::reset() {
  std::lock_guard<std::mutex> guard(lock_);
  for (auto& buf : threadLocalBuf_) {
    buf.first->reset();
  }
}

std::ostream& operator<<(std::ostream& outPut, const Stat& stat) {
  std::lock_guard<std::mutex> guard(const_cast<Stat&>(stat).lock_);
  auto showStat = [&](const StatInfo* info, pid_t tid, bool isFirst = true) {
    uint64_t average = 0;
    if (info->count_ > 0) {
      outPut << std::setfill(' ') << std::left;
      if (!isFirst) {
        outPut << std::setw(42) << " ";
      }
      average = info->total_ / info->count_;
      outPut << "Stat=" << std::setw(30) << stat.getName();
      if (tid) {
        outPut << " TID=" << std::setw(6) << tid;
      }
      outPut << " total=" << std::setw(10) << info->total_ * 0.001
             << " avg=" << std::setw(10) << average * 0.001
             << " max=" << std::setw(10) << info->max_ * 0.001
             << " min=" << std::setw(10) << info->min_ * 0.001
             << " count=" << std::setw(10) << info->count_ << std::endl;
    }
  };
  if (!stat.getThreadInfo()) {
    StatInfo infoVarTmp;
    const_cast<Stat&>(stat).mergeThreadStat(infoVarTmp);
    showStat(&infoVarTmp, 0);
  } else {
    bool isFirst = true;
    for (auto& buf : stat.threadLocalBuf_) {
      showStat(buf.first, buf.second, isFirst);
      if (isFirst) isFirst = false;
    }
    showStat(&stat.destructStat_, 0);
  }

  return outPut;
}

BarrierStatPtr StatSet::getStat(uint16_t numConnThreads,
                                const std::string& name,
                                BarrierStatType bType) {
  {
    ReadLockGuard guard(lock_);
    auto it = barrierStatSet_.find(name);
    if (it != barrierStatSet_.end()) {
      return it->second;
    }
  }

  std::lock_guard<RWLock> guard(lock_);
  // test again with lock_guard
  auto it = barrierStatSet_.find(name);
  if (it != barrierStatSet_.end()) {
    return it->second;
  }

  BarrierStatPtr stat;
  if (bType == BARRIER_END) {
    stat = std::make_shared<BarrierEndStat>(numConnThreads, name);
  } else if (bType == BARRIER_DELTA) {
    stat = std::make_shared<BarrierDeltaStat>(numConnThreads, name);
  }
  auto ret = barrierStatSet_.insert(std::make_pair(name, stat));
  return ret.first->second;
}

void StatSet::printSegTimerStatus() {
  ReadLockGuard guard(lock_);
  LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
            << "======= StatSet: [" << name_ << "] status ======" << std::endl;
  for (auto& stat : statSet_) {
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
              << *(stat.second);
  }
}

void StatSet::printBarrierTimerStatus() {
  ReadLockGuard guard(lock_);
  if (barrierStatSet_.empty()) {
    return;
  }
  // control barrierAbstact in runtime, so enable compliation
  LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
            << "======= BarrierStatSet status ======" << std::endl;
  for (auto& stat : barrierStatSet_) {
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
              << *(stat.second);
  }
}

void StatSet::printAllStatus() {
#ifndef PADDLE_DISABLE_TIMER
  printSegTimerStatus();
#endif
  printBarrierTimerStatus();
  LOG(INFO) << std::setiosflags(std::ios::left)
            << "--------------------------------------------------"
            << std::endl;
}

void StatSet::printStatus(const std::string& name) {
  ReadLockGuard guard(lock_);
  auto iter = statSet_.find(name);
  CHECK(iter != statSet_.end()) << name << " is not registed in " << name_;
  LOG(INFO) << *(iter->second);
}

void StatSet::reset(bool clearRawData) {
  ReadLockGuard guard(lock_);
  for (auto& stat : statSet_) {
    stat.second->reset();
  }
  // reset barrierStat
  for (auto& stat : barrierStatSet_) {
    stat.second->reset(clearRawData);
  }
}

void StatSet::setThreadInfo(const std::string& name, bool flag) {
  ReadLockGuard guard(lock_);
  auto iter = statSet_.find(name);
  CHECK(iter != statSet_.end()) << name << " is not registed in " << name_;
  iter->second->setThreadInfo(flag);
}

void StatSet::deleteStat(const std::string& name) {
  std::lock_guard<RWLock> guard(lock_);
  auto iter = statSet_.find(name);
  CHECK(iter != statSet_.end()) << name << " is not registed in " << name_;
  statSet_.erase(iter);
}

StatInfo::~StatInfo() {
  if (stat_) {
    std::lock_guard<std::mutex> guard(stat_->lock_);
    if (stat_->destructStat_.max_ < this->max_) {
      stat_->destructStat_.max_ = this->max_;
    }
    if (stat_->destructStat_.min_ > this->min_) {
      stat_->destructStat_.min_ = this->min_;
    }
    stat_->destructStat_.total_ += this->total_;
    stat_->destructStat_.count_ += this->count_;
    stat_->threadLocalBuf_.remove({this, getTID()});
  }
}

static unsigned g_profileCount = 0;
static std::recursive_mutex g_profileMutex;

GpuProfiler::GpuProfiler(std::string statName, std::string info)
    : guard_(g_profileMutex) {
  if (++g_profileCount == 1) {
    LOG(INFO) << "Enable GPU Profiler Stat: [" << statName << "] " << info;
    hl_profiler_start();
  }
}

GpuProfiler::~GpuProfiler() {
  if (--g_profileCount == 0) {
    hl_profiler_end();
  }
}

}  // namespace paddle
