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

#include "ModelConfig.pb.h"
#include "paddle/parameter/Parameter.h"

#include "Layer.h"
#include "paddle/parameter/Argument.h"

namespace paddle {

// Macro for registering a operator type
// Example: REGISTER_OPERATOR(dot_mul, DotMulOperator);
#define REGISTER_OPERATOR(__type_name, __class_name)                \
  static InitFunction __reg_type_##__type_name([]() {               \
    Operator::registrar_.registerClass<__class_name>(#__type_name); \
  })

/**
 * Operator like Projection, but takes more than one Arguments as input.
 * @note: Operator can't have parameters.
 */
class Operator {
 public:
  static Operator* create(const OperatorConfig& config, bool useGpu);

  Operator(const OperatorConfig& config, bool useGpu)
      : config_(config), useGpu_(useGpu) {}

  virtual ~Operator() {}

  const OperatorConfig& getConfig() const { return config_; }

  static ClassRegistrar<Operator, OperatorConfig, bool> registrar_;

  /**
   * Forward propagation. If backward() will be called, in and out must be kept
   * valid until then.
   * @param ins inputs of operator
   * @param out output of operator
   * @param passType PASS_TRAIN of PASS_TEST
   */
  void forward(std::vector<const Argument*> ins,
               Argument* out,
               PassType passType) {
    ins_ = ins;
    out_ = out;
    passType_ = passType;
    forward();
  }

  virtual void prefetch(const Argument* in) {}
  virtual void forward() = 0;
  virtual void backward() = 0;

  /**
   * See comment in Layer.h for the function with the same name.
   */
  virtual void resetState() {}

  /**
   * Set layer state.
   */
  virtual void setState(LayerStatePtr state) {}

  /**
   * Set layer state.
   */
  virtual LayerStatePtr getState() { return nullptr; }

 protected:
  /// Config of operator
  OperatorConfig config_;
  bool useGpu_;

  /// Store `ins` passed to forward()
  std::vector<const Argument*> ins_;
  /// Store `out` passed to forward()
  Argument* out_;
  /// Store `passType` passed to forward()
  PassType passType_;
};
}  // namespace paddle
