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
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {
/**
 * @brief Network is also a type of Operator
 *
 * It will manage the operators it has.
 *
 * Network is the container and controller of a set of operators.

 * A network object knows all Operators belonging to this network. Variables,
 * which are inputs and outputs of these operators, are created and managed by a
 * hierarchy of Scope objects.
 *
 * This is the base class of network, all the networks should implement the APIs
 * it defines.
 */
class NetOp : public OperatorBase {
 public:
  /**
   * Infer all the operators' input and output variables' shapes, will be called
   * before every mini-batch
   */
  void InferShape(const Scope& scope) const override {
    for (auto& op : ops_) {
      op->InferShape(scope);
    }
  }

  /**
   * @brief Run the network.
   *
   * Run all the operators with the `scope`, if no scope is provided, default
   * scope will be used instead. If no OpContext is provicded, default context
   * will be used.
   */
  void Run(const Scope& scope,
           const platform::DeviceContext& dev_ctx) const override {
    for (auto& op : ops_) {
      op->Run(scope, dev_ctx);
    }
  }

  /**
   * @brief Add an operator by ptr
   */
  void AddOp(const std::shared_ptr<OperatorBase>& op) {
    PADDLE_ENFORCE(!add_op_done_, "Cannot AddOp when this network is sealed");
    PADDLE_ENFORCE(op != nullptr, "Cannot Insert Null op");
    ops_.push_back(op);
  }

  void InsertOp(size_t pos, const std::shared_ptr<OperatorBase>& op) {
    PADDLE_ENFORCE(!add_op_done_,
                   "Cannot InsertOp when this network is sealed");
    PADDLE_ENFORCE(op != nullptr, "Cannot Insert Null op");
    PADDLE_ENFORCE(pos <= ops_.size(), "Out of range");
    ops_.insert(ops_.begin() + pos, op);
  }

  void CompleteAddOp(bool calculate = true);

  std::string DebugString() const override;

  bool IsNetOp() const override;

  std::vector<std::shared_ptr<OperatorBase>> ops_;

 private:
  bool add_op_done_{false};

  template <typename T, typename KeyType>
  static bool Contains(T container, KeyType key) {
    return container.find(key) != container.end();
  }
};

}  // namespace framework
}  // namespace paddle
