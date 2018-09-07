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

#include "Layer.h"
#include "ModelConfig.pb.h"
#include "paddle/parameter/Parameter.h"

namespace paddle {

// Macro for registering a projection type
// Example: REGISTER_LAYER(fc, FullMatrixProjection);
#define REGISTER_PROJECTION(__type_name, __class_name)                \
  static InitFunction __reg_type_##__type_name([]() {                 \
    Projection::registrar_.registerClass<__class_name>(#__type_name); \
  })

#define REGISTER_PROJECTION_CREATE_FUNC(__type_name, createFunction)    \
  static InitFunction __reg_type_##__type_name([]() {                   \
    Projection::registrar_.registerClass(#__type_name, createFunction); \
  })

/**
 * A projection takes one Argument as input, calculate the result and add it
 * to output Argument.
 */
class Projection {
 public:
  static Projection* create(const ProjectionConfig& config,
                            ParameterPtr parameter,
                            bool useGpu);

  Projection(const ProjectionConfig& config,
             ParameterPtr parameter,
             bool useGpu)
      : config_(config), parameter_(parameter), useGpu_(useGpu) {}

  virtual ~Projection() {}

  const std::string& getName() const { return config_.name(); }

  /// Register a projection
  static ClassRegistrar<Projection, ProjectionConfig, ParameterPtr, bool>
      registrar_;

  /**
   * Forward propagation. If backward() will be called, in and out must be kept
   * valid until then.
   * @param in input of projection
   * @param out output of projection
   * @param passType PASS_TRAIN of PASS_TEST
   */
  void forward(const Argument* in, const Argument* out, PassType passType) {
    in_ = in;
    out_ = out;
    passType_ = passType;
    forward();
  }

  virtual void prefetch(const Argument* in) {}
  virtual void forward() = 0;
  virtual void backward(const UpdateCallback& callback) = 0;

  /**
   * See comment in Layer.h for the function with the same name.
   */
  virtual void resetState() {}

  /**
   * Set layer state.
   */
  virtual void setState(LayerStatePtr state) {}

  /**
   * Get layer state. A copy of internal state is returned.
   */
  virtual LayerStatePtr getState() { return nullptr; }

  /**
   * init forward_ and backward_ functions
   */
  virtual bool init() { return true; }

  /**
   * Get output size of projection.
   */
  size_t getOutputSize() const { return config_.output_size(); }

 protected:
  /**
   * Create layer function. Function is called in forward or backward.
   * \param function, Layer::forward_ or Layer::backward_
   * \param name, function name
   * \param config, initialization configuration for the function
   */
  void createFunction(std::vector<std::shared_ptr<FunctionBase>>& function,
                      const std::string& name,
                      const FuncConfig& config) {
    if (useGpu_) {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-GPU"));
    } else {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-CPU"));
    }
    auto& func = function.back();
    func->init(config);
  }

 protected:
  /// Config of projection
  ProjectionConfig config_;
  /// Parameter of projection
  ParameterPtr parameter_;
  bool useGpu_;

  /// Store `in` passed to forward()
  const Argument* in_;
  /// Store `out` passed to forward()
  const Argument* out_;
  /// Store `passType` passed to forward()
  PassType passType_;
  /// Layer forward function
  std::vector<std::shared_ptr<FunctionBase>> forward_;
  /// Layer backward function
  std::vector<std::shared_ptr<FunctionBase>> backward_;
};
}  // namespace paddle
