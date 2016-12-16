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

#include <fstream>
#include "paddle/utils/Logging.h"

#include "ParallelParameter.h"

namespace paddle {

UpdateFunction paramUpdateFunctions[UPDATE_TYPE_NUM] = {
    nullptr,  // &ParallelParameter::singleUpdate,  /* single thread */
    nullptr,  // &ParallelParameter::controlUpdate,    /* controller thread */
    &ParallelParameter::majorUpdate, /* major thread */
    &ParallelParameter::minorUpdate, /* minor thread */

    nullptr,                         /* master */
    &ParallelParameter::slaveUpdate, /* slave */
};
ParallelParameterPtr ParallelParameter::create(TrainerRole role,
                                               ParameterPtr localParam,
                                               int asyncCount) {
  ParallelParameterPtr ptr = nullptr;
  switch (role) {
    case TRAINER_ROLE_CONTROL:
    case TRAINER_ROLE_MAJOR:
    case TRAINER_ROLE_MINOR:
      ptr = std::make_shared<SyncParameter>(role, localParam);
      break;
    case TRAINER_ROLE_MASTER:
    case TRAINER_ROLE_SLAVE:
      ptr = std::make_shared<AsyncParameter>(role, asyncCount, localParam);
      break;
    default:
      LOG(FATAL) << "unknown role " << role << "\n";
  }
  return ptr;
}
void ParallelParameter::syncUpdate(TrainerRole role, real learnRate) {
  if (paramUpdateFunctions[role]) {
    (this->*paramUpdateFunctions[role])(learnRate);
  }
}

void SyncParameter::attachControlParam(ParallelParameterPtr controler) {
  controlParam_ = controler;
}

void SyncParameter::attachMajorParam(ParallelParameterPtr partner) {
  majorPartners_.push_back(partner);
  if (role_ == TRAINER_ROLE_CONTROL) {
    localParam_->setSharedCount(majorPartners_.size());
  }
  // partnerParam_ = partner;
}

void SyncParameter::attachMinorParam(ParallelParameterPtr partner,
                                     int deviceId) {
  minorPartners_.push_back(partner);
  minorDeviceIds_.push_back(deviceId);
  // partnerParam_ = partner;
}

void SyncParameter::waitAllMajorGradReady() {
  for (size_t i = 0; i < majorPartners_.size(); i++) {
    majorPartners_[i]->waitGradReady();
    partnerParam_ = majorPartners_[i]->getLocalParameter();
    VectorPtr localGrad = localParam_->getBuf(PARAMETER_GRADIENT);
    VectorPtr patnrGrad = partnerParam_->getBuf(PARAMETER_GRADIENT);
    if (FLAGS_use_gpu) hl_set_device(minorDeviceIds_[i]);
    localGrad->add(*patnrGrad);
  }
}

void SyncParameter::synchronizeParamter() {
  valueSem_->wait();
  if (role_ == TRAINER_ROLE_MINOR) {
    /* copy the value from controller */
    VectorPtr cntrlVec =
        (controlParam_->getLocalParameter())->getBuf(PARAMETER_VALUE);
    VectorPtr localVec = localParam_->getBuf(PARAMETER_VALUE);
    localVec->copyFrom(*cntrlVec);

    /* dispatch the value to major */
    for (size_t i = 0; i < majorPartners_.size(); i++) {
      VectorPtr majorVec =
          (majorPartners_[i]->getLocalParameter())->getBuf(PARAMETER_VALUE);
      majorVec->copyFrom(*localVec);
      majorPartners_[i]->postValueReady();
    }
  }
}

void SyncParameter::singleUpdate(real learnRate) {
  CHECK(role_ == TRAINER_ROLE_SINGLE);
  localParam_->updateWithGradient(learnRate);
}

void SyncParameter::controlUpdate(const UpdateCallback &callBack) {
  CHECK(role_ == TRAINER_ROLE_CONTROL);
  CHECK(gradSem_ != NULL && valueSem_ != NULL);
  CHECK(majorPartners_.size());

  /* update */
  if (callBack) {
    callBack(localParam_.get());
    localParam_->clearGradient();
  }

  for (size_t i = 0; i < minorPartners_.size(); i++) {
    minorPartners_[i]->postValueReady();
  }
}

void SyncParameter::majorUpdate(real learnRate) {
  (void)learnRate;
  CHECK(role_ == TRAINER_ROLE_MAJOR);
  CHECK(gradSem_ != NULL && valueSem_ != NULL);
  CHECK(minorPartners_.size() && controlParam_);

  /* wait the minor-Gradient is ready */
  for (size_t i = 0; i < minorPartners_.size(); i++) {
    minorPartners_[i]->waitGradReady();
    partnerParam_ = minorPartners_[i]->getLocalParameter();
    VectorPtr localGrad = localParam_->getBuf(PARAMETER_GRADIENT);
    VectorPtr minorGrad = partnerParam_->getBuf(PARAMETER_GRADIENT);
    localGrad->add(*minorGrad);
  }

  /* notice the controller that the gradient is ready */
  gradSem_->post();
}

void SyncParameter::minorUpdate(real learnRate) {
  (void)learnRate;
  CHECK(role_ == TRAINER_ROLE_MINOR);
  CHECK(gradSem_ != NULL && valueSem_ != NULL);

  // notice the major that the gradient is ready
  gradSem_->post();
}

AsyncParameter::AsyncParameter(TrainerRole role,
                               int asyncCount,
                               ParameterPtr localParam)
    : ParallelParameter(role, localParam) {
  asyncCount_ = asyncCount;
  accumCounter_ = 0;
  gradientAccum_ = Vector::create(localParam->getSize(), localParam->useGpu());
  gradientAccum_->zeroMem();
}

void AsyncParameter::slaveUpdate(real learnRate) {
  /* increase the accumCounter_ */
  accumCounter_++;

  /* accumulate the gradient to the buffer */
  VectorPtr grad = localParam_->getBuf(PARAMETER_GRADIENT);
  gradientAccum_->add(*grad);

  /* if need to be synchronized with the master */
  if (accumCounter_ == asyncCount_) {
    gradSem_->post();
    // accumCounter_ = 0; NOTICE: the upper-function need to reset the counter
  } else {  // self update
    localParam_->updateWithGradient(learnRate);
  }
  localParam_->clearGradient();
}

bool AsyncParameter::masterUpdate(ParallelParameterPtr slaveParam,
                                  const UpdateCallback &callback) {
  CHECK(slaveParam && callback);

  /* wait the slave is ready */
  if (!slaveParam->timeWaitGradReady(5)) {
    return false;
  }

  AsyncParameter *asyncParam = dynamic_cast<AsyncParameter *>(slaveParam.get());

  /* get the accum-gradient to update local parameter */
  VectorPtr slaveVec = asyncParam->getAccum();
  localParam_->getBuf(PARAMETER_GRADIENT)->copyFrom(*slaveVec);
  callback(localParam_.get());
  // slaveVec->zeroMem();

  /* copy the newest parameter-value to the slave */
  slaveVec = (slaveParam->getLocalParameter())->getBuf(PARAMETER_VALUE);
  slaveVec->copyFrom(*(localParam_->getBuf(PARAMETER_VALUE)));

  /* release the semphore */
  slaveParam->postValueReady();

  return true;
}

}  // namespace paddle
