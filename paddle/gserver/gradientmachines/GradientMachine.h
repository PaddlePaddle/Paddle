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

#include <iostream>
#include <vector>

#include "ModelConfig.pb.h"
#include "TrainerConfig.pb.h"
#include "paddle/gserver/dataproviders/DataProvider.h"
#include "paddle/gserver/layers/Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/ParameterUpdaterBase.h"
#include "paddle/utils/Thread.h"

#ifndef PADDLE_MOBILE_INFERENCE
#include "paddle/gserver/evaluators/Evaluator.h"
#endif

namespace paddle {
/**
 * @brief A gradient machine is capable of calculating some outputs given
 *        some inputs and performing gradient calculation based on the
 *        derivative from the outputs.
 *
 * A gradient machine can be either a full neural network or part of a neural
 * network.
 *
 * Usage for training:
 *
 *  1. Prepare inArgs. Put your input data into inArgs[i].value.
 *
 *  2. Call forward(inArgs, &outArgs)
 *
 *  3. Calculate gradient with respect to outArgs[i]->value
 *     and fill them into outArgs[i]->grad.
 *     This step can be skipped if your the outputs are from cost layers.
 *
 *  4. Call backward(). After backward, gradient of each parameter is
 *     accumulated to getParameters()[i]->getBuf(PARAMETER_GRADIENT)
 *
 *  5. Update parameter value getParameters()[i]->getBuf(PARAMETER_VALUE) using
 *     gradients.
 *
 *  6. Clear gradients to zero.
 *
 * Usage for prediction:
 *
 *  1. Prepare inArgs. Put your input data into inArgs[i].value.
 *
 *  2. Call forward(inArgs, &outArgs)
 *
 *  3. Obtain the prediction result from outArgs[i]
 */

typedef std::vector<LayerStatePtr> MachineState;

class GradientMachine;

typedef std::shared_ptr<GradientMachine> GradientMachinePtr;

class GradientMachine {
 public:
  enum CreateMode {
    kNormal = 0,
    kSgdSparseCpuTraining = 3,
    kTesting = 4,
    kCustom = 10
  };

  /**
   * Create a gradient machine from ModelConfig
   * Parameter will have parameterTypes
   */
  static GradientMachine* create(
      const ModelConfig& config,
      int mode = kNormal,
      const std::vector<ParameterType>& parameterTypes =
          std::vector<ParameterType>{
              PARAMETER_VALUE, PARAMETER_GRADIENT, PARAMETER_MOMENTUM});

  virtual ~GradientMachine() {}

  /**
   * Prefetch row ids of sparse parameter.
   */
  virtual void prefetch(const std::vector<Argument>& inArgs) { (void)inArgs; }

  /**
   * @brief Forward propagation.
   *
   * Calculate outputs (outArgs) based the inputs (inArgs)
   *
   * @note: if passType==PASS_TEST, then backward() should not be called
   */
  virtual void forward(const std::vector<Argument>& inArgs,
                       std::vector<Argument>* outArgs,
                       PassType passType) = 0;

  /**
   * @brief Backward propagation.
   *
   * Calculate the gradient of inArgs and parameter.
   *
   * This function should only be called after a corresponding forward() call.
   * The caller is responsible for filling the correct grad for the outArgs
   * obtained using forward().
   *
   * It may also change the grad field for the inArgs supplied at forward()
   */
  virtual void backward(const UpdateCallback& callback = nullptr) = 0;

  /**
   * Combine forward() and backward(). For multithread training, this
   * may be faster.
   *
   * @note: passType PASS_TEST is not allowed for forwardBackward().
   */
  virtual void forwardBackward(const std::vector<Argument>& inArgs,
                               std::vector<Argument>* outArgs,
                               PassType passType,
                               const UpdateCallback& callback = nullptr) {
    forward(inArgs, outArgs, passType);
    backward(callback);
  }

  virtual Argument getLayerOutput(const std::string& layerName) = 0;

  // see comment in Layer.h for the function with the same name
  virtual void resetState() {}

  // set machine state
  virtual void setState(const MachineState& machineState) {}

  // save machine state
  virtual void getState(MachineState& machineState) {}

  virtual void onPassEnd() = 0;

#ifndef PADDLE_MOBILE_INFERENCE
  /**
   * Create an evaluator which can be used for eval()
   */
  virtual Evaluator* makeEvaluator() const = 0;

  /**
   * evaluate using the given evaluator
   */
  virtual void eval(Evaluator* evaluator) const = 0;
#endif

  std::vector<ParameterPtr>& getParameters() { return parameters_; }

  std::vector<ParameterPtr>& getNonStaticParameters() {
    if (nonStaticParameters_.empty()) {
      for (auto para : parameters_) {
        if (!para->isStatic()) {
          nonStaticParameters_.push_back(para);
        }
      }
    }
    return nonStaticParameters_;
  }

  inline bool hasStaticParameters() {
    return parameters_.size() != getNonStaticParameters().size();
  }

  /**
   * @brief   Used before formal training, start work-threads and set
   *          trainer Parameters;
   *
   * @note    This function will only been implemented and used in a
   *          multithreaded environment.
   */
  virtual void start() {}

  /**
   * @brief   check  each work-thread whether is failed/error/finish,
   *          if not, return ture, and yes return false.
   *
   * @note    This function will only been implemented and used in a
   *          multithreaded environment.
   */
  virtual void finish() {}

  /**
   * @brief   set the training status a "finished" value, the sub_work_threads
   *          will option the change, and then exit.
   *
   * @note    This function will only been implemented and used in a
   *          multithreaded environment.
   */
  virtual bool trainIsOn() { return true; }

  /**
   * @brief   when all or some of the sub-workThreads are suspended to waiting
   *          controller's instructions, and after some processing done in the
   *          controller, it will call this function to wake up all the pending
   *          thread.
   *
   * @note    This function will only been implemented and used in a
   *          multithreaded environment.
   */
  virtual void restart() {}

  /// Set the gradient of the output from outside.
  virtual void setOutputGrad(const std::vector<Argument>& args) {
    LOG(FATAL) << "Not implemented!";
  }

  void saveParameters(const std::string& dir) const;

  void loadParameters(const std::string& dir);

  void randParameters();

  virtual void getStats(real& cost, int64_t& numProcessed) {
    (void)cost;
    (void)numProcessed;
  }

  /**
   * @brief   Release the middle layer's output memory.
   *
   * @note    This function is used for memory optimization in inference.
   */
  virtual void releaseOutput() {}

 protected:
  virtual void onLoadParameter() {}

  std::vector<ParameterPtr> parameters_;
  std::vector<ParameterPtr> nonStaticParameters_;
};

}  // namespace paddle
