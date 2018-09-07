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

#include "paddle/parameter/AverageOptimizer.h"
#include "paddle/parameter/FirstOrderOptimizer.h"
#include "paddle/parameter/OptimizerFunctions.h"
#include "paddle/parameter/OptimizerWithRegularizer.h"
#include "paddle/parameter/Parameter.h"
#include "paddle/parameter/Regularizer.h"
#include "paddle/utils/Util.h"

#include <memory>
#include <vector>

namespace paddle {

/**
 * \brief A parameter updater that uses multiple threads to update parameters.
   This parameter updater handles GPU and CPU updates differently,
   because at the current moment, the merging on CPU is happening on the
   main thread, and the its parameter size can be much larger than the one GPU.
   Thus, for GPU, the parameter updates happens in updateImpl() function, which
   is called by gradient machines as a callback function supplied to backward()
   and forwardBackward().
   For CPU, the parameter updates happens in separate threads maintained by this
   class.
 */
class SgdThreadUpdater : public ParameterUpdater {
 public:
  explicit SgdThreadUpdater(const OptimizationConfig& optConfig);
  virtual ~SgdThreadUpdater() {}

  // Use the startPass() function of the base optimizer.
  virtual void startPass();

  // Use the finishPass() function of the base optimizer.
  virtual bool finishPass();

  virtual void init(const std::vector<ParameterPtr>& parameters);
  virtual PassType startBatch(int64_t batchSize);
  // Call finishBatch for each optimizer.
  virtual void finishBatch(real cost);
  virtual void catchUpWith();
  virtual void apply();
  virtual void restore();

 protected:
  // This is the function that will be eventualy called by the GradientMachine.
  // used only for GPU update.
  virtual void updateImpl(Parameter* para);
  OptimizationConfig config_;
  int64_t numSamplesProcessed_;

  // One optimizers for each parameter.
  std::vector<std::unique_ptr<ParameterOptimizer>> optimizers_;

  // The update function for CPU sparse parameters.
  void threadUpdateSparse(int tid, size_t numThreads, Parameter* para);

  // The update function for CPU dense parameters.
  void threadUpdateDense(int tid, size_t numThreads, Parameter* para);
  // The update function for after update operations, such as averager.
  void threadTraverse(const ParameterOptimizer::TraverseCallback& callback,
                      int tid,
                      size_t numThreads,
                      Parameter* para);
  typedef std::function<const ParameterOptimizer::TraverseCallback(Parameter*)>
      GetTraverseCallback;
  void traverse(GetTraverseCallback getTraverseCallback);
};

}  // namespace paddle
