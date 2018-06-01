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
#include <string>
#include <vector>
#include "paddle/utils/Error.h"

namespace paddle {

struct Argument;
/**
 * @brief Activation function is a function that transforms a set of input
 * signals into an output signals. The purpose of the activation function
 * is to introduce non-liearilty into the network.
 *
 * @note Common activation function are provieded, including linear,
 * sigmoid, softmax, sequence_max, relu, brelu, tanh, stanh,
 * softrelu, abs, square, exponential.
 *
 */
class ActivationFunction {
 public:
  static ActivationFunction* create(const std::string& type);
  static std::vector<std::string> getAllRegisteredTypes();

  ActivationFunction() {}

  virtual ~ActivationFunction() {}

  /**
   * @brief Foward propagation
   *
   * act.value <- f(act.value),
   * where f is the activation function.
   * Suppose that before calling forward(), act.value is x and
   * after forward() is called, act.value is y, then y = f(x).
   *
   * Usually, act is Layer::output_
   */
  virtual Error __must_check forward(Argument& act) = 0;

  /**
   * @brief Backward propagaion
   *
   * x and y are defined in the above comment for forward().
   * - Before calling backward(), act.grad = dE / dy, where E is the error/cost
   * - After backward() returns, act.grad = dE / dx = (dE/dy) * (dy/dx)
   */
  virtual Error __must_check backward(Argument& act) = 0;

  virtual const std::string& getName() const = 0;
};

}  // namespace paddle
