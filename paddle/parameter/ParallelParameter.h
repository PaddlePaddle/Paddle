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
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>

#include "hl_gpu.h"
#include "paddle/math/Vector.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/ParameterUpdateFunctions.h"
#include "paddle/utils/Common.h"
#include "paddle/utils/Flags.h"
#include "paddle/utils/Locks.h"

#include "ParameterConfig.pb.h"

namespace paddle {

class ParallelParameter;
class SyncParameter;
class AsyncParameter;

typedef std::shared_ptr<ParallelParameter> ParallelParameterPtr;

const int UPDATE_TYPE_NUM = 32;

/**
 * TrainRole denotes the role of current training, different roles have
 * different jobs.
 *
 * control, major, minor are three kinds of role to support mutiple GPUs
 * parallel SGD training. SM on GPU card has two groups, each group
 * consist of a major and a minor.
 *
 * @param    single  single GPU card single thread training.
 *
 *
 * @param    control current parameter updates via control role,
 *                   not participate in real training. control role is
 *                   responsible for merging all major's gradient and
 *                   update parameter value.
 *
 * @param    major   major role paticipates in real training, when local
 *                   gradient is ready, merge its corresponding minor's
 *                   gradient and notify controller: this group's gradient
 *                   is already ready.
 *
 * @param    minor   minor role participates in real training, when local
 *                   gradient is ready, only notify its corresponding major.
 *                   In order to maximum apportion jobs, after controller
 *                   updates the paramemter value, each group's minior
 *                   reponses to dispatch the latest model into local and
 *                   major.
 */
enum TrainerRole {
  TRAINER_ROLE_SINGLE,
  TRAINER_ROLE_CONTROL,
  TRAINER_ROLE_MAJOR,
  TRAINER_ROLE_MINOR,
  TRAINER_ROLE_MASTER,
  TRAINER_ROLE_SLAVE
};
typedef void (ParallelParameter::*UpdateFunction)(real learnRate);

class ParallelParameter {
public:
  static ParallelParameterPtr create(TrainerRole role,
                                     ParameterPtr localParam,
                                     int asyncCount = 1);

  ParallelParameter(TrainerRole role, ParameterPtr localParam) {
    role_ = role;
    gradSem_.reset(new Semaphore(0));
    valueSem_.reset(new Semaphore(0));
    localParam_ = localParam;
  }

  virtual ~ParallelParameter() {}

  ParameterPtr getLocalParameter() { return localParam_; }
  bool timeWaitGradReady(int sec) {
    struct timespec ts;
    ts.tv_nsec = 0;
    ts.tv_sec = time(NULL) + sec;
    return gradSem_->timeWait(&ts);
  }
  void waitGradReady() { gradSem_->wait(); }
  void postValueReady() { valueSem_->post(); }

  void syncUpdate(TrainerRole role, real learnRate);

  virtual void synchronizeParamter() = 0;

  /**
   * for synchronous
   */
  virtual void singleUpdate(real learnRate) { (void)learnRate; }

  virtual void controlUpdate(const UpdateCallback& callback) { (void)callback; }

  virtual void majorUpdate(real learnRate) { (void)learnRate; }

  virtual void minorUpdate(real learnRate) { (void)learnRate; }

  /**
   * for asynchronous
   */
  virtual void slaveUpdate(real learnRate) { (void)learnRate; }

protected:
  TrainerRole role_;
  ParameterPtr localParam_;
  std::unique_ptr<Semaphore>
      gradSem_;  /// wether the local parameter-gradient is ready
  std::unique_ptr<Semaphore>
      valueSem_;  /// wether the local parameter-value is updated
};

/**
 * this class is designed for multi-threading training.
 *
 * "Synchronous" means multiple GPUs calculate 1/4 mini-Batch,
 * but will get only one gradient
 */
class SyncParameter : public ParallelParameter {
public:
  SyncParameter(TrainerRole role, ParameterPtr localParam)
      : ParallelParameter(role, localParam) {
    controlParam_ = nullptr;
    majorPartners_.clear();
    minorPartners_.clear();
  }
  ~SyncParameter() {
    majorPartners_.clear();
    minorPartners_.clear();
  }
  void attachControlParam(ParallelParameterPtr controler);

  void attachMajorParam(ParallelParameterPtr partner);

  void attachMinorParam(ParallelParameterPtr partner, int deviceId);

  void waitAllMajorGradReady();

  void synchronizeParamter();

  void singleUpdate(real learnRate);

  void controlUpdate(const UpdateCallback& callback);

  void majorUpdate(real learnRate);

  void minorUpdate(real learnRate);

  std::vector<ParallelParameterPtr>& getMajorPartners() {
    return majorPartners_;
  }

  std::vector<ParallelParameterPtr>& getMinorPartners() {
    return minorPartners_;
  }

private:
  // The following variables are used in a multithreaded training situation
  // partnerParam_ is local-parameter's partner
  // controlParam_ is the controller-thread 's parameter
  ParameterPtr partnerParam_;
  std::vector<ParallelParameterPtr> majorPartners_;
  std::vector<ParallelParameterPtr> minorPartners_;
  std::vector<int> minorDeviceIds_;
  ParallelParameterPtr controlParam_;
};

class AsyncParameter : public ParallelParameter {
public:
  AsyncParameter(TrainerRole role, int asyncCount, ParameterPtr localParam);

  void clearCounter() { accumCounter_ = 0; }

  VectorPtr getAccum() { return gradientAccum_; }

  void synchronizeParamter() {
    if (accumCounter_ == asyncCount_) {
      valueSem_->wait();
      clearCounter();
      gradientAccum_->zeroMem();
    }
  }

  /**
   * When asynchronous training, update strategy including slave and master.
   *
   * slave: If in range asyncCount, adopting self-update method.
   *        If beyond asyncCount, waiting for master to update.
   */
  void slaveUpdate(real learnRate);

  /**
   * When asynchronous training, update strategy including slave and master.
   *
   * master: it only polls slaves, do not training data.
   *         If slave's gradient is ready, fetch it.
   *         Update master's parameter, then copy it into
   *         corresponding slave.
   */
  bool masterUpdate(ParallelParameterPtr slaveParam,
                    const UpdateCallback& callback);

private:
  /**
   * When asynchronous training, every aysnc trainer needs to
   * accumulate a number of batch gradient.
   *
   * gradientAccum_ is used to save the sum of gradients.
   */
  VectorPtr gradientAccum_;

  /// Asynchronous count.
  int asyncCount_;
  /// Accumulate counter of current gradients.
  int accumCounter_;
};

typedef std::map<std::string, ParallelParameterPtr> ParallelParameterMap;

}  // namespace paddle
