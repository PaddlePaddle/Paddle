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

#include "paddle/utils/BarrierStat.h"
#include <string.h>
#include <sys/types.h>
#include <algorithm>
#include <iomanip>
#include "paddle/utils/Flags.h"
#include "paddle/utils/Stat.h"

DEFINE_bool(log_barrier_abstract,
            true,
            "if true, show abstract of barrier performance");
DEFINE_int32(log_barrier_lowest_nodes,
             5,
             "how many lowest node will be logged");
DEFINE_bool(log_barrier_show_log,
            false,  // for performance tuning insight
            "if true, always show barrier abstract even with little gap");

namespace paddle {

std::ostream &operator<<(std::ostream &output, const BarrierStatBase &stat) {
  if (FLAGS_log_barrier_abstract) {
    std::lock_guard<std::mutex> guard(stat.lock_);
    stat.showAbstract(output);
  }
  return output;
}

BarrierStatBase::BarrierStatBase(uint16_t numConnThreads,
                                 const std::string &name)
    : totSamples_(0), numConnThreads_(numConnThreads), name_(name) {
  abstract_.resize(numConnThreads_);
  if (FLAGS_log_barrier_show_log) {
    rateThreshold_ = 0.0;
  } else {
    /* probablity of abnormal node
     * p = 1/n + (n/8)/(n+1), n = nodes, n > 1
     * if the freq of lowest trainerId larger than p,
     * output FLAGS_log_barrier_lowest_nodes lastTrainerId.
     * numConnThreads_ indicates nodes
     */
    float n = (float)numConnThreads;
    rateThreshold_ = 1.0 / n + (n / 8.0) / (n + 1.0);
  }
}

BarrierEndStat::BarrierEndStat(uint16_t numConnThreads, const std::string &name)
    : BarrierStatBase(numConnThreads, name) {
  timeVector_.reset(new TimeVectorEnd(numConnThreads_));
  reset(true);
  LOG(INFO) << " create barrierEndStat: " << name
            << " endBarrier warning rate: " << rateThreshold_;
}

/*
 * Note:
 * the design different pserver entity owns different statSet to obey
 * the background that different pserver runs separately.
 */
void BarrierEndStat::updateStat(struct timeval &cur, int32_t trainerId) {
  CHECK_LT(trainerId, numConnThreads_) << "trainerId is invalid in barrier";

  std::lock_guard<std::mutex> guard(lock_);
  timeVector_->addTimeval(cur, trainerId);

  if (timeVector_->full()) {
    std::lock_guard<std::mutex> abstractGuard(abstractLock_);
    auto id = timeVector_->getLastTrainerId();
    auto delta = timeToMicroSecond(timeVector_->getDelta());
    auto secondDelta = timeToMicroSecond(timeVector_->get1NDelta());
    auto lastTwoDelta = timeToMicroSecond(timeVector_->getMinus1NDelta());
    auto midDelta = timeToMicroSecond(timeVector_->getMidNDelta());
    // discard first sample, since first sample probably is abnormal.
    if (totSamples_) {
      abstract_[id].freq++;

      if (delta < abstract_[id].minDelta) {
        abstract_[id].minDelta = delta;
      }
      if (delta > abstract_[id].maxDelta) {
        abstract_[id].maxDelta = delta;
      }
      abstract_[id].totDelta += delta;
      abstract_[id].totSecondDelta += secondDelta;
      abstract_[id].totLastTwoDelta += lastTwoDelta;
      abstract_[id].totMidDelta += midDelta;

      // update totAbstract_
      totAbstract_.freq++;
      if (delta < totAbstract_.minDelta) {
        totAbstract_.minDelta = delta;
      }
      if (delta > totAbstract_.maxDelta) {
        totAbstract_.maxDelta = delta;
      }
      totAbstract_.totDelta += delta;
      totAbstract_.totSecondDelta += secondDelta;
      totAbstract_.totLastTwoDelta += lastTwoDelta;
      totAbstract_.totMidDelta += midDelta;
    }

    totSamples_++;
    timeVector_->reset();
  }
}

void BarrierEndStat::reset(bool clearRawData) {
  int32_t i = 0;

  totSamples_ = 0;

  std::lock_guard<std::mutex> guard(abstractLock_);

  if (clearRawData) {
    timeVector_->reset();
  }

  for (auto &abstract : abstract_) {
    memset((void *)&abstract, 0, sizeof(abstract));
    abstract.minDelta = UINT64_MAX;
    abstract.trainerId = i++;
  }
  memset((void *)&totAbstract_, 0, sizeof(Abstract));
  totAbstract_.minDelta = UINT64_MAX;
}

void BarrierEndStat::showAbstract(std::ostream &output) const {
  // do not support the case "<=2 pserver"
  if (numConnThreads_ <= 2 || !totSamples_) {
    return;
  }

  // duplicate freq info
  std::vector<struct Abstract> outputAbstract = abstract_;
  std::sort(outputAbstract.begin(),
            outputAbstract.end(),
            [](const struct Abstract &a, const struct Abstract &b) {
              return a.freq > b.freq;
            });

  auto rate = (float)outputAbstract[0].freq / (float)totSamples_;
  if (rate < rateThreshold_) {
    return;
  }

  output << std::setw(20) << name_ << std::endl;

  /*
   * Note:
   * avgGap:        the average delta between 1 -- n arriving trainers
   * avgSecondGap:  the average delta between 2 -- n arriving trainers
   * avgLastTwoGap: the average delta between n-1 -- n  arriving trainers
   * avgMidGap:     the average delta between n/2 -- n  arriving trainers
   * rato: samples / totSamples
   *
   * the stat is based on per trainer if trainer_id is set, totAbstract is
   * stat based on all trainers scope.
   */
  output << std::setw(42) << " " << std::setw(15) << "trainerId"
         << std::setw(15) << "avgGap" << std::setw(15) << "avgSecondGap"
         << std::setw(15) << "avgLastTwoGap" << std::setw(15) << "avgMidGap"
         << std::setw(10) << "rate" << std::setw(10) << "samples"
         << std::setw(10) << "totSamples" << std::endl;
  // show totAbstract, it's valuable when lastTrainerId is even-distributed'
  if (!totAbstract_.freq) return;
  output << std::setw(42) << " " << std::setw(15) << "totAbstract"
         << std::setw(15) << (totAbstract_.totDelta / totAbstract_.freq) * 0.001
         << std::setw(15)
         << (totAbstract_.totSecondDelta / totAbstract_.freq) * 0.001
         << std::setw(15)
         << (totAbstract_.totLastTwoDelta / totAbstract_.freq) * 0.001
         << std::setw(15)
         << (totAbstract_.totMidDelta / totAbstract_.freq) * 0.001
         << std::setw(10) << (float)totAbstract_.freq / (float)totSamples_
         << std::setw(10) << (float)totAbstract_.freq << std::setw(10)
         << (float)totSamples_ << std::endl;

  // show lastTrainerId abstract
  int count = 0;
  for (auto &abstract : outputAbstract) {
    if (!abstract.freq || count++ >= FLAGS_log_barrier_lowest_nodes) {
      break;
    }
    // output format control
    output << std::setw(42) << " " << std::setw(15) << abstract.trainerId
           << std::setw(15) << (abstract.totDelta / abstract.freq) * 0.001
           << std::setw(15) << (abstract.totSecondDelta / abstract.freq) * 0.001
           << std::setw(15)
           << (abstract.totLastTwoDelta / abstract.freq) * 0.001
           << std::setw(15) << (abstract.totMidDelta / abstract.freq) * 0.001
           << std::setw(10) << (float)abstract.freq / (float)totSamples_
           << std::setw(10) << (float)abstract.freq << std::setw(10)
           << (float)totSamples_ << std::endl;
  }
}

BarrierDeltaStat::BarrierDeltaStat(uint16_t numConnThreads,
                                   const std::string &name)
    : BarrierStatBase(numConnThreads, name) {
  timeVector_.reset(new TimeVectorDelta(numConnThreads_));
  reset(true);
  LOG(INFO) << " create barrierDeltaStat: " << name
            << " barrierDelta warning rate: " << rateThreshold_;
}

void BarrierDeltaStat::updateStat(uint64_t delta, int32_t trainerId) {
  CHECK_LT(trainerId, numConnThreads_) << "trainerId is invalid in barrier";

  std::lock_guard<std::mutex> guard(lock_);
  timeVector_->addTimeval(delta, trainerId);

  if (timeVector_->full()) {
    std::lock_guard<std::mutex> abstractGuard(abstractLock_);
    auto id = timeVector_->getMaxTrainerId();
    auto delta = timeVector_->getDelta();
    // discard first sample, since first sample probably is abnormal.
    if (totSamples_) {
      abstract_[id].freq++;

      if (delta < abstract_[id].minDelta) {
        abstract_[id].minDelta = delta;
      }
      if (delta > abstract_[id].maxDelta) {
        abstract_[id].maxDelta = delta;
      }
      abstract_[id].totDelta += delta;

      // update totAbstract_
      totAbstract_.freq++;
      if (delta < totAbstract_.minDelta) {
        totAbstract_.minDelta = delta;
      }
      if (delta > totAbstract_.maxDelta) {
        totAbstract_.maxDelta = delta;
      }
      totAbstract_.totDelta += delta;
    }

    totSamples_++;
    timeVector_->reset();
  }
}

void BarrierDeltaStat::reset(bool clearRawData) {
  int32_t i = 0;

  totSamples_ = 0;

  std::lock_guard<std::mutex> guard(abstractLock_);

  if (clearRawData) {
    timeVector_->reset();
  }

  for (auto &abstract : abstract_) {
    memset((void *)&abstract, 0, sizeof(abstract));
    abstract.minDelta = UINT64_MAX;
    abstract.trainerId = i++;
  }
  memset((void *)&totAbstract_, 0, sizeof(Abstract));
  totAbstract_.minDelta = UINT64_MAX;
}

void BarrierDeltaStat::showAbstract(std::ostream &output) const {
  // do not support the case "<=2 pserver"
  if (numConnThreads_ <= 2 || !totSamples_) {
    return;
  }

  // duplicate freq info
  std::vector<struct Abstract> outputAbstract = abstract_;
  std::sort(outputAbstract.begin(),
            outputAbstract.end(),
            [](const struct Abstract &a, const struct Abstract &b) {
              return a.freq > b.freq;
            });

  auto rate = (float)outputAbstract[0].freq / (float)totSamples_;
  if (rate < rateThreshold_) {
    return;
  }

  output << std::setw(20) << name_ << std::endl;

  /* Note:
   * Gap means the delta from all trainers' forwardbackward
   * avgGap: average Gap in log_period batches
   * minGap: min Gap in log_period batches
   * maxGap: max Gap in log_period batches
   * trainerId: the slowest trainer_id
   *
   * the stat is based on per trainer if trainer_id is set, totAbstract is
   * stat based on all trainers scope.
   */
  output << std::setw(42) << " " << std::setw(15) << "trainerId"
         << std::setw(15) << "avgGap" << std::setw(10) << "minGap"
         << std::setw(10) << "maxGap" << std::setw(10) << "rate"
         << std::setw(10) << "samples" << std::setw(10) << "totSamples"
         << std::endl;
  // show totAbstract, it's valuable when lastTrainerId is even-distributed'
  if (!totAbstract_.freq) return;
  output << std::setw(42) << " " << std::setw(15) << "totAbstract"
         << std::setw(15) << (totAbstract_.totDelta / totAbstract_.freq) * 0.001
         << std::setw(10) << totAbstract_.minDelta * 0.001 << std::setw(10)
         << totAbstract_.maxDelta * 0.001 << std::setw(10)
         << (float)totAbstract_.freq / (float)totSamples_ << std::setw(10)
         << (float)totAbstract_.freq << std::setw(10) << (float)totSamples_
         << std::endl;

  // show lastTrainerId abstract
  int count = 0;
  for (auto &abstract : outputAbstract) {
    if (!abstract.freq || count++ >= FLAGS_log_barrier_lowest_nodes) {
      break;
    }
    // output format control
    output << std::setw(42) << " " << std::setw(15) << abstract.trainerId
           << std::setw(15) << (abstract.totDelta / abstract.freq) * 0.001
           << std::setw(10) << abstract.minDelta * 0.001 << std::setw(10)
           << abstract.maxDelta * 0.001 << std::setw(10)
           << (float)abstract.freq / (float)totSamples_ << std::setw(10)
           << (float)abstract.freq << std::setw(10) << (float)totSamples_
           << std::endl;
  }
}
}  // namespace paddle
