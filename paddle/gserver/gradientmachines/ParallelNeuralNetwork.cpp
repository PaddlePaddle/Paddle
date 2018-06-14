/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

#include "ParallelNeuralNetwork.h"

#include <pthread.h>
#include <sched.h>

namespace paddle {

void ParallelNeuralNetwork::init(
    const ModelConfig& config,
    ParamInitCallback callback,
    const std::vector<ParameterType>& parameterTypes,
    bool useGpu) {
  NeuralNetwork::init(config, callback, parameterTypes, useGpu);

  if (config.type() == "recurrent_nn") {
    LOG(FATAL)
        << "You can not add `--parallel_nn=true` on the command line, "
        << "parallel_nn training mode does not support the recurrent_nn model.";
  }

  useGpu_ = useGpu;
  numDevices_ = 0;
  if (useGpu_) {
    numDevices_ = hl_get_device_count();
  }

  for (auto& layer : layers_) {
    int deviceId = layer->getDeviceId();
    CHECK_LT(deviceId, numDevices_);
    addComputeThread(deviceId);
  }
}

void ParallelNeuralNetwork::addComputeThread(int deviceId) {
  for (auto& thread : threads_) {
    if (thread->getDeviceId() == deviceId) {
      return;
    }
  }

  threads_.emplace_back(new ParallelThread(
      threads_.size(), deviceId, deviceId >= 0 ? useGpu_ : false));
}

void ParallelNeuralNetwork::waitAllThread() {
  for (auto& thread : threads_) {
    thread->jobEnqueue(NULL, TASK_END_LAYER);
  }

  for (size_t i = 0; i < threads_.size(); i++) {
    threads_[i]->queue_.waitEmpty();
  }
}

void ParallelNeuralNetwork::dispatchByDeviceId(int deviceId,
                                               LayerPtr layer,
                                               TaskType task) {
  for (auto& thread : threads_) {
    if (thread->getDeviceId() == deviceId) {
      thread->jobEnqueue(layer, task);
      return;
    }
  }
  LOG(FATAL) << "No specific device thread ";
}

void ParallelNeuralNetwork::forward(const std::vector<Argument>& inArgs,
                                    std::vector<Argument>* outArgs,
                                    PassType passType) {
  for (auto& thread : threads_) {
    thread->setForwardPassType(passType);
  }
  CHECK_EQ(inArgs.size(), dataLayers_.size());
  outArgs->resize(outputLayers_.size());
  for (size_t i = 0; i != dataLayers_.size(); ++i) {
    const_cast<Argument&>(inArgs[i]).deviceId = -1;
    dataLayers_[i]->setData(inArgs[i]);
  }

  for (auto& layer : layers_) {
    dispatchByDeviceId(layer->getDeviceId(), layer, TASK_FORWARD);
  }

  {
    REGISTER_TIMER("forwardTime");
    waitAllThread();
  }
  outArgs->clear();
  outArgs->reserve(outputLayers_.size());
  for (auto& layer : outputLayers_) {
    outArgs->push_back(layer->getOutput());
  }
}

void ParallelNeuralNetwork::backward(const UpdateCallback& callback) {
  for (auto& thread : threads_) {
    thread->setBackwardCallback(callback);
  }

  FOR_EACH_R(layer, layers_) {
    dispatchByDeviceId((*layer)->getDeviceId(), *layer, TASK_BACKWARD);
  }
  {
    REGISTER_TIMER("backwardTime");
    waitAllThread();
  }
}

void ParallelNeuralNetwork::forwardBackward(const std::vector<Argument>& inArgs,
                                            std::vector<Argument>* outArgs,
                                            PassType passType,
                                            const UpdateCallback& callback) {
  forward(inArgs, outArgs, passType);
  backward(callback);
}

void ParallelNeuralNetwork::start() {
  for (auto& thread : threads_) {
    thread->start();
  }
}

ParallelThread::ParallelThread(int threadId, int deviceId, bool useGpu)
    : threadId_(threadId), deviceId_(deviceId), useGpu_(useGpu) {}

ParallelThread::~ParallelThread() { stop(); }

void ParallelThread::stop() {
  if (computeThread_) {
    jobEnqueue(NULL, TASK_THREAD_FINISH);
    computeThread_->join();
    computeThread_.reset(nullptr);
  }
}

void ParallelThread::computeThread() {
  LOG(INFO) << "gradComputeThread " << threadId_;

  if (useGpu_) {
    hl_init(deviceId_);
  }

  while (true) {
    struct Job job_work = queue_.dequeue();

    if (job_work.task_ == TASK_END_LAYER) {
      continue;
    } else if (job_work.task_ == TASK_THREAD_FINISH) {
      break;
    }

    if (TASK_FORWARD == job_work.task_) {
      {
        REGISTER_TIMER_INFO("waitInputValue",
                            job_work.layer_->getName().c_str());
        job_work.layer_->waitInputValue();
      }
      {
        REGISTER_TIMER_INFO("threadForwardTimer",
                            job_work.layer_->getName().c_str());
        job_work.layer_->forward(passType_);
      }
      {
        REGISTER_TIMER_INFO("copyOutputToOtherDevice",
                            job_work.layer_->getName().c_str());
        job_work.layer_->copyOutputToOtherDevice();
      }
    } else {
      {
        REGISTER_TIMER_INFO("waitAndMergeOutputGrad",
                            job_work.layer_->getName().c_str());
        job_work.layer_->waitAndMergeOutputGrad();
      }
      {
        REGISTER_TIMER_INFO("threadBackwardTimer",
                            job_work.layer_->getName().c_str());
        job_work.layer_->backward(backwardCallback_);
      }
      hl_stream_synchronize(HPPL_STREAM_DEFAULT);
      job_work.layer_->markAllInputGrad();
    }
  }
}

void ParallelThread::start() {
  computeThread_.reset(new std::thread([this]() { computeThread(); }));
}

void ParallelThread::jobEnqueue(LayerPtr layer, TaskType task) {
  struct Job job_work;
  job_work.layer_ = layer;
  job_work.task_ = task;
  queue_.enqueue(job_work);
}

}  // namespace paddle
