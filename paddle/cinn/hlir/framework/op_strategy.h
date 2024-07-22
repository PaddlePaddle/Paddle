// Copyright (c) 2021 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/schedule.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/utils/type_defs.h"

namespace cinn {
namespace hlir {
namespace framework {

using CINNCompute = lang::PackedFunc;
using CINNSchedule = lang::PackedFunc;

class OpStrategy;

/**
 * \brief Attributes of each node in graph.
 *  The attributes include the node's name, the corresponding operator
 *  and other parameters like axis.
 */
struct NodeAttr {
  using attr_t = utils::Attribute;

  /**
   * \brief The operator this node uses.
   */
  const Operator* op{nullptr};

  /**
   * \brief The name of this node.
   */
  std::string node_name;

  /**
   * \brief The attributes stored as string in dictionary.
   */
  absl::flat_hash_map<std::string, attr_t> attr_store;
};

std::ostream& operator<<(std::ostream& os, const NodeAttr& node_attr);

using StrategyFunction = std::function<std::shared_ptr<OpStrategy>(
    const NodeAttr&,
    const std::vector<ir::Tensor>&,
    const std::vector<Type>&,
    const std::vector<std::vector<int>>&,
    const cinn::common::Target&)>;

using StrategyFunctionSymbolic = std::function<std::shared_ptr<OpStrategy>(
    const NodeAttr&,
    const std::vector<ir::Tensor>&,
    const std::vector<Type>&,
    const std::vector<std::vector<ir::Dim>>&,
    const cinn::common::Target&)>;

//! Operator implementation that includes compute and schedule function.
class OpImpl : public cinn::common::Object {
 public:
  //! Compute function
  CINNCompute fcompute;
  //! Schedule function
  CINNSchedule fschedule;
  //! Name of the implementation
  std::string name;
  //! Priority level
  int plevel;
  /**
   * \brief Invoke the operator compute function.
   * @param attrs The attribute of the primitive
   * @param inputs The input tensors.
   * @param out_type The output type information.
   * @return The output compute description of the operator.
   */
  ir::Tensor Compute(const std::vector<ir::Tensor>& inputs,
                     const Type& out_type) {
    // TODO(haozech) : add support for packedfunc to return Tensor
    // Expected : return this->fcompute(inputs, out_type);
    ir::Tensor temp;
    return temp;
  }
  /**
   * \brief Build the computation schedule.
   * @param attrs The attribute of the node.
   * @param outs The output tensors.
   * @param target The build target.
   * @return The computation schedule.
   */
  cinn::common::Shared<Schedule> GetSchedule(
      const std::vector<ir::Tensor>& outs,
      const std::vector<ir::Tensor>& temp_tensors,
      const Target& target) {
    // TODO(haozech) : add support for packedfunc to return Schedule
    // Expected : return this->fschedule(outs, target);
    return nullptr;
  }

  const char* type_info() const override { return __type_info__; }

 private:
  static constexpr char* __type_info__ = "OpImplementation";
};

//! Specialized implementations for operators under certain conditions.
class OpSpec : public cinn::common::Object {
 public:
  //! List of implementations.
  std::vector<std::shared_ptr<OpImpl>> implementations;

  /** \brief Condition to enable the specialization.
   *    Could be undefined to represent generic case.
   *  TODO(haozech) : build a specified class SpecializedCondition to represent
   * the condition. Expected : SpecializedCondition condition;
   */
  std::string condition;

  const char* type_info() const override { return __type_info__; }

  void AddImpl(CINNCompute fcompute,
               CINNSchedule fschedule,
               std::string name,
               int plevel) {
    auto n = std::make_shared<OpImpl>();
    n->fcompute = fcompute;
    n->fschedule = fschedule;
    n->name = std::move(name);
    n->plevel = plevel;
    this->implementations.push_back(n);
  }

 private:
  static constexpr char* __type_info__ = "OpSpecialization";
};

//! Operator strategy class.
class OpStrategy : public cinn::common::Object {
 public:
  const char* type_info() const override { return __type_info__; }
  //! List of operator specializations.
  std::vector<std::shared_ptr<OpSpec>> specializations;

  /**
   * \brief Add an implementation.
   * @param fcompute Compute function
   * @param fschedule Schedule function
   * @param name Name of the implementation
   * @param plevel Priority level of the implementation
   */
  void AddImpl(CINNCompute fcompute,
               CINNSchedule fschedule,
               std::string name,
               int plevel);
  static std::shared_ptr<OpImpl> SelectImpl(
      const std::shared_ptr<OpStrategy>& strategy);

 private:
  static constexpr char* __type_info__ = "OpStrategy";
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
