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
#include "paddle/function/KernelType.h"

namespace paddle {
class FuncConfig;
/**
 * @brief The WithFunction class. It mark a layer or function uses
 * paddle::function.
 */
class WithFunction {
protected:
  std::vector<function::KernelType> forward_;
  std::vector<function::KernelType> backward_;

  /**
   * Append function. Function is called in forward or backward.
   * \param functions. The kernels to be invoked.
   * \param name, function name
   * \param config, initialization configuration for the function
   * \param useGPU, true if append a GPU function.
   */
  void appendFunction(std::vector<function::KernelType>* functions,
                      const std::string& name,
                      const FuncConfig& config,
                      bool useGPU);
};

}  // namespace paddle
