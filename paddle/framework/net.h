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

#include <paddle/framework/op_desc.pb.h>
#include <paddle/framework/operator.h>
#include "paddle/framework/net_proto.pb.h"
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {
using namespace paddle::platform;

// temporary put here for test
class PlainNetOpProtoAndCheckerMaker : public OpProtoAndCheckerMaker {
 public:
  PlainNetOpProtoAndCheckerMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddType("plainnet_operator");
    AddComment("This is test op");
  }
};

/**
 * @brief Network is also a type of Operator
 *
 * It will manage the operators it has.
 *
 * Network is the container and controller of a set of operators, user can build
 * a real network from a NetDesc which is a protobuf message and use
 * Network.Run() * to run all the operators in the network.

 * A network object knows all Operators belonging to this network. Variables,
 * which are inputs and outputs of these operators, are created and managed by a
 * hierarchy of Scope objects.
 *
 * This is the base class of network, all the networks should implement the apis
 * it defines.
 */
class Net : public OperatorBase {
 public:
  /**
   * @brief Add an Operator according to `def`.
   */
  virtual void AddOp(const OpDesc& def) = 0;

  /**
   * @brief Add optimizer operators acctording to `attrs`.
   */
  virtual void AddOptimizerOps() = 0;

  /**
   * @brief Add backward operators.
   */
  virtual void AddBackwardOps() = 0;
};

/**
 * @brief a basic implementation of Net.
 *
 * PlainNet is a very simple Net, it create a list of operators, and run them
 * sequentially following the order they added.
 */
class PlainNet : public Net {
 public:
  /**
   * Infer all the operators' input and output varialbes' shapes, will be called
   * before every mini-batch
   */
  void InferShape(const std::shared_ptr<Scope>& scope) const override;

  /**
   * @brief Run the network.
   *
   * Run all the operators with the `scope`, if no scope is provided, default
   * scope will be used instead. If no OpContext is provicded, default context
   * will be used.
   */
  void Run(const std::shared_ptr<Scope>& scope,
           const platform::DeviceContext& dev_ctx) const override;

  /**
   * @brief Add an operator to this network.
   */
  void AddOp(const OpDesc& def) override;

  /**
   * @brief Add all optimizer operators related into the network.
   */
  void AddOptimizerOps() override {}

  /**
   * @brief Add all backward operators related into the network.
   */
  void AddBackwardOps() override {}

 private:
  // the operators owned by `Network`.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
};

}  // namespace framework
}  // namespace paddle
