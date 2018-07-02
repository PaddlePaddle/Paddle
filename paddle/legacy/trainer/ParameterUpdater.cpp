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

#include "ParameterUpdater.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Thread.h"

namespace paddle {

static const hl_stream_t kDeviceToHostStream = HPPL_STREAM_1;
static const hl_stream_t kHostToDeviceStream = HPPL_STREAM_2;

SgdUpdaterWithCpuAverager::SgdUpdaterWithCpuAverager(
    const OptimizationConfig& optConfig)
    : SgdLocalUpdater(optConfig, false /*with averager*/) {
  CHECK(FLAGS_use_gpu && optConfig.do_average_in_cpu());
  averager_.reset(AverageOptimizer::create(optConfig,
                                           new DummyOptimizer(optConfig),
                                           false /*sparse*/,
                                           true /*apply*/));
  updateWorker_.addJob([]() { hl_set_device(FLAGS_gpu_id); });
}

void SgdUpdaterWithCpuAverager::init(
    const std::vector<ParameterPtr>& parameters) {
  SgdLocalUpdater::init(parameters);
  averager_->init(parameters_.size(), nullptr);
  copyEvents_.resize(parameters_.size());
  for (auto& parameter : parameters) {
    SetDevice device(parameter->getDeviceId());
    cpuParameters_.emplace_back(new Parameter(parameter->getConfig(),
                                              /* useGpu= */ false,
                                              /* doInit= */ false));
    if (parameter->useGpu()) {
      cpuParameters_.back()->enableType(PARAMETER_APPLY);
    } else {
      cpuParameters_.back()->enableSharedType(
          PARAMETER_APPLY, parameter->getBuf(PARAMETER_VALUE));
    }
    for (ParameterType type : averager_->getParameterTypes()) {
      cpuParameters_.back()->enableType(type);
    }

    hl_create_event(&copyEvents_[nonStaticParaIDMap_[parameter->getID()]]);
  }
}

SgdUpdaterWithCpuAverager::~SgdUpdaterWithCpuAverager() {
  for (auto& event : copyEvents_) {
    hl_destroy_event(event);
  }
}

void SgdUpdaterWithCpuAverager::updateImpl(Parameter* para) {
  SgdLocalUpdater::updateImpl(para);

  if (para->useGpu()) {
    size_t pid = nonStaticParaIDMap_[para->getID()];
    Parameter* cpuPara = cpuParameters_[pid].get();
    cpuPara->getBuf(PARAMETER_VALUE)
        ->copyFrom(*para->getBuf(PARAMETER_VALUE), kDeviceToHostStream);
    hl_stream_record_event(kDeviceToHostStream, copyEvents_[pid]);
  }

  updateWorker_.addJob(
      std::bind(&SgdUpdaterWithCpuAverager::updateFunc, this, para));
}

void SgdUpdaterWithCpuAverager::updateFunc(Parameter* para) {
  SetDevice setDevice(para->getDeviceId());
  size_t pid = nonStaticParaIDMap_[para->getID()];
  Parameter* cpuPara = cpuParameters_[pid].get();
  if (para->useGpu()) {
    hl_event_synchronize(copyEvents_[pid]);
  }
  averager_->update(cpuPara->getBufs(), cpuPara->getConfig(), -1LU);
}

void SgdUpdaterWithCpuAverager::finishBatch(real cost) {
  SgdLocalUpdater::finishBatch(cost);

  updateWorker_.wait();
  for (auto para : cpuParameters_) {
    if (auto callback = averager_->needSpecialTraversal(para->getConfig())) {
      callback(para->getBufs(), para->getConfig(), -1LU);
    }
  }
  averager_->finishBatch();
}

void SgdUpdaterWithCpuAverager::apply() {
  // backup gpu value
  for (auto& para : parameters_) {
    SetDevice setDevice(para->getDeviceId());
    para->getBuf(PARAMETER_GRADIENT)
        ->copyFrom(*para->getBuf(PARAMETER_VALUE), kHostToDeviceStream);
  }

  // apply on cpu parameter
  if (auto callback = averager_->apply()) {
    for (auto para : cpuParameters_) {
      callback(para->getBufs(), para->getConfig(), -1LU);
    }
  }

  // copy to gpu value
  for (auto& para : parameters_) {
    SetDevice setDevice(para->getDeviceId());
    size_t pid = nonStaticParaIDMap_[para->getID()];
    Parameter* cpuPara = cpuParameters_[pid].get();
    if (parameters_[pid]->useGpu()) {
      para->getBuf(PARAMETER_VALUE)
          ->copyFrom(*cpuPara->getBuf(PARAMETER_APPLY), kHostToDeviceStream);
    }
  }
  hl_stream_synchronize(kHostToDeviceStream);
  for (auto& para : parameters_) {
    para->setValueUpdated();
  }
}

void SgdUpdaterWithCpuAverager::restore() {
  // restore on cpu parameter
  if (auto callback = averager_->restore()) {
    for (auto para : cpuParameters_) {
      callback(para->getBufs(), para->getConfig(), -1LU);
    }
  }

  // restore gpu value
  for (auto& para : parameters_) {
    SetDevice device(para->getDeviceId());
    para->getBuf(PARAMETER_VALUE)->copyFrom(*para->getBuf(PARAMETER_GRADIENT));
    para->getBuf(PARAMETER_GRADIENT)->zeroMem();
    para->setValueUpdated();
  }
}

}  // namespace paddle
