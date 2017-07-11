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

#include "paddle/framework/net_proto.pb.h"
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace framework {
using namespace paddle::platform;

// operator's index stored in a network.
typedef int OpIndex;
/**
 * NOTE following codes are some definitions of unimplemented concepts.
 * We write some basic implementation to make Net compilable. These APIs will
 * keep updating if the concepts related are implemented.
 */

struct OpDesc;
struct OpAttrs {};

class Operator {
 public:
  Operator(const OpDesc &def) {}
  void InferShape() {}
  void Run(DeviceContext *ctx) {}
};

/**
 * @brief Network that manage the operators it has.
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
class Net {
 public:
  /**
   * @brief Infer shapes of all inputs and outputs of operators.
   */
  virtual void InferShape(Scope *scope) = 0;
  /**
   * @brief Run the network.
   *
   * Run all the operators and return success(true) or not, with all the
   * variables are located in `scope`. `context` describes the detail execution
   * environment for ops. `begin` and `end` specify the scope of `ops_` to run,
   * If no positive indexes are provided, all operators in `ops_` will run.
   */
  virtual void Run(std::shared_ptr<Scope> scope, DeviceContext *ctx) = 0;

  /**
   * @brief Add an Operator according to `def`.
   */
  virtual OpIndex AddOp(const OpProto &def) = 0;

  /**
   * @brief Add optimizer operators acctording to `attrs`.
   */
  virtual void AddOptimizerOps(const OpAttrs &attrs) = 0;

  /**
   * @brief Add backward operators.
   */
  virtual void AddBackwardOps() = 0;

  /**
   * @brief Create a network.
   */
  static std::unique_ptr<Net> Create(const NetDesc &def = NetDesc());

  virtual ~Net() {}
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
   * @brief Initialize a PlainNet.
   *
   * Initialize from  a network describe by `def`. NetDesc is the definition of
   * a network.
   */
  PlainNet(const NetDesc &def);

  /**
   * Infer all the operators' input and output varialbes' shapes, will be called
   * before every mini-batch
   */
  virtual void InferShape(Scope *scope) override;

  /**
   * @brief Run the network.
   *
   * Run all the operators with the `scope`, if no scope is provided, default
   * scope will be used instead. If no OpContext is provicded, default context
   * will be used.
   */
  virtual void Run(std::shared_ptr<Scope> scope, DeviceContext *ctx) override;

  /**
   * @brief Add an operator to this network.
   */
  virtual OpIndex AddOp(const OpProto &def) override;

  /**
   * @brief Add all optimizer operators related into the network.
   */
  virtual void AddOptimizerOps(const OpAttrs &attrs) override;

  /**
   * @brief Add all backward operators related into the network.
   */
  virtual void AddBackwardOps() override;

  virtual ~PlainNet() override {}

 protected:
  /**
   * @brief Build the network.
   *
   * Create operators accordding to `def`, will be called by the constructor.
   */
  void BuildNet(const NetDesc &def);

  /**
   * @brief Add an operator into this network.
   *
   * Add a operator which is identified as `type` and has attributes described
   * in `attrs`, the `inputs` are the keys of readonly input variables,
   * `outputs` are keys of mutable output variables. An `OpIndex` will be
   * returned to indicate the offset of the new operator in `ops_`.
   */
  OpIndex AddOp(const std::string &type, const std::vector<std::string> &inputs,
                const std::vector<std::string> &outputs,
                const OpAttrs &attrs = OpAttrs());

 private:
  // the operators owned by `Network`.
  std::vector<Operator> ops_;
};

}  // namespace framework
}  // namespace paddle
