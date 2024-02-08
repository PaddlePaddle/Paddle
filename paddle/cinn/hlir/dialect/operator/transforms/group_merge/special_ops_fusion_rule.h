// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/pir/include/core/operation.h"

namespace cinn {
namespace dialect {
namespace ir {

using OpPatternKind = hlir::framework::OpPatternKind;

class SpecialOpsFusionRule {
 public:
  typedef bool (*RuleFunc)(const ::pir::Operation*, OpPatternKind);

  static const SpecialOpsFusionRule& GetInstance() {
    thread_local static SpecialOpsFusionRule instance;
    return instance;
  }

  bool ProducerOpAllowsFusion(const ::pir::Operation* producer,
                              OpPatternKind consumer_group_pattern) const {
    auto iter = producer_op_rules_.find(producer->name());
    if (iter != producer_op_rules_.end()) {
      return iter->second(producer, consumer_group_pattern);
    }
    return true;
  }

  bool ConsumerOpAllowsFusion(const ::pir::Operation* consumer,
                              OpPatternKind producer_group_pattern) const {
    auto iter = consumer_op_rules_.find(consumer->name());
    if (iter != consumer_op_rules_.end()) {
      return iter->second(consumer, producer_group_pattern);
    }
    return true;
  }

 private:
  SpecialOpsFusionRule() { Init(); }

  SpecialOpsFusionRule(const SpecialOpsFusionRule&) = delete;
  SpecialOpsFusionRule(const SpecialOpsFusionRule&&) = delete;
  SpecialOpsFusionRule& operator=(const SpecialOpsFusionRule&) = delete;

  void Init();

  void RegisterProducerOpRule(const std::string& producer_op_name,
                              RuleFunc rule) {
    producer_op_rules_[producer_op_name] = rule;
  }

  void RegisterConsumerOpRule(const std::string& consumer_op_name,
                              RuleFunc rule) {
    consumer_op_rules_[consumer_op_name] = rule;
  }

  std::map<std::string, RuleFunc> producer_op_rules_;
  std::map<std::string, RuleFunc> consumer_op_rules_;
};

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
