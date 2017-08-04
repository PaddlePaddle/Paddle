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

#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/operator.h"

namespace paddle {
namespace framework {
class OpRegistry;

enum InOutType { IN, OUT };

struct OpInOutArg {
  explicit OpInOutArg(const std::string& proto_name, const InOutType& type,
                      bool needed_in_grad, size_t begin_idx, size_t end_idx)
      : proto_name_(proto_name),
        type_(type),
        needed_in_grad_(needed_in_grad),
        begin_idx_(begin_idx),
        end_idx_(end_idx) {}

  std::string proto_name_;
  InOutType type_;
  bool needed_in_grad_;
  size_t begin_idx_;
  size_t end_idx_;
};

class GradOpBuilder {
  using VarIndexMap = std::unordered_map<std::string, int>;

 public:
  explicit GradOpBuilder(const OperatorBase& op) : op_(op) {}
  OperatorBase* Build();

 private:
  OpInOutArg* BuildArg(const VarProto& var, const VarIndexMap& var_map,
                       const std::vector<int>& format, InOutType type);
  void BuildOpInOutArgList();
  void AddArgIntoGradOp(const OpInOutArg* arg, std::vector<std::string>& in_out,
                        std::vector<int>& format, VarIndexMap* varmap, int& idx,
                        bool is_grad) const;
  void CompleteGradOp(OperatorBase* grad_op) const;
  const OperatorBase& op_;
  std::vector<std::shared_ptr<OpInOutArg>> arg_list_;
};

}  // namespace framework
}  // namespace paddle
