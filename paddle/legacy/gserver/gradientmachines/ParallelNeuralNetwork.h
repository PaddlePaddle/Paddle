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

#pragma once

#include "NeuralNetwork.h"

namespace paddle {

class ParallelThread;

enum TaskType {
  TASK_FORWARD = 0,
  TASK_BACKWARD = 1,
  TASK_END_LAYER = 2,
  TASK_THREAD_FINISH = 3,
};

/**
 * A ParallelNeuralNetwork is capable of calculating a neural network through
 * multiple threads in parallel.
 */
class ParallelNeuralNetwork : public NeuralNetwork {
 public:
  ParallelNeuralNetwork(std::string subModelName = "",
                        NeuralNetwork *rootNetwork = nullptr)
      : NeuralNetwork(subModelName, rootNetwork) {}

  virtual void init(const ModelConfig &config,
                    ParamInitCallback callback = nullptr,
                    const std::vector<ParameterType> &parameterTypes =
                        std::vector<ParameterType>{PARAMETER_VALUE,
                                                   PARAMETER_GRADIENT,
                                                   PARAMETER_MOMENTUM},
                    bool useGpu = FLAGS_use_gpu);

  virtual void forward(const std::vector<Argument> &inArgs,
                       std::vector<Argument> *outArgs,
                       PassType passType);

  virtual void backward(const UpdateCallback &callback = nullptr);

  void forwardBackward(const std::vector<Argument> &inArgs,
                       std::vector<Argument> *outArgs,
                       PassType passType,
                       const UpdateCallback &callback = NULL);

  virtual void start();

  void addComputeThread(int deviceId);

  void dispatchByDeviceId(int deviceId, LayerPtr layer, TaskType task);

  void waitAllThread();

  // virtual void eval(Evaluator* evaluator);

 protected:
  bool useGpu_;
  /// number of gpu devices
  int numDevices_;
  std::vector<std::unique_ptr<ParallelThread>> threads_;
};

class ParallelThread {
 public:
  ParallelThread(int threadId, int deviceId, bool useGpu);
  ~ParallelThread();
  void jobEnqueue(LayerPtr layer, TaskType task);
  void start();
  void stop();
  int getDeviceId() const { return deviceId_; }

  void setBackwardCallback(const UpdateCallback &callback) {
    backwardCallback_ = callback;
  }
  void setForwardPassType(PassType passType) { passType_ = passType; }

 protected:
  void computeThread();

 public:
  struct Job {
    LayerPtr layer_;
    TaskType task_;
  };
  typedef Queue<Job> JobQueue;
  JobQueue queue_;

 protected:
  /// from 0 to threads-1
  int threadId_;
  /// the GPU device Id which the computeThread_ used
  int deviceId_;
  bool useGpu_;
  std::unique_ptr<std::thread> computeThread_;
  /// whether the thread should stop
  bool stopping_;
  UpdateCallback backwardCallback_;
  PassType passType_;
};
}  // namespace paddle
