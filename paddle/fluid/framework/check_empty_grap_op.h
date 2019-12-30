// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>
#include <string>
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

class EmptyGradOp {
 public:
  static EmptyGradOp& Instance();

  bool Has(const std::string& op_type, bool is_mkldnn_op) const {
    if (is_mkldnn_op) {
      return empty_grad_mkldnn_ops_.find(op_type) !=
             empty_grad_mkldnn_ops_.end();
    } else {
      return empty_grad_ops_.find(op_type) != empty_grad_ops_.end();
    }
  }

  void Insert(const std::string& op_type, bool is_mkldnn_op) {
    if (is_mkldnn_op) {
      empty_grad_mkldnn_ops_.insert(op_type);
    } else {
      empty_grad_ops_.insert(op_type);
    }
  }

  const std::set<std::string>& set() const { return empty_grad_ops_; }

  std::set<std::string>* mutable_set() { return &empty_grad_ops_; }

 private:
  EmptyGradOp() = default;
  std::set<std::string> empty_grad_ops_;
  std::set<std::string> empty_grad_mkldnn_ops_;

  DISABLE_COPY_AND_ASSIGN(EmptyGradOp);
};

class EmptyGradOperatorRegistra {
 public:
  explicit EmptyGradOperatorRegistra(const char* op_type,
                                     bool is_mkldnn_op = false) {
    EmptyGradOp::Instance().Insert(op_type, is_mkldnn_op);
  }
};

#define EMPTY_GRAD_OP(op_type)                          \
  static ::paddle::framework::EmptyGradOperatorRegistra \
      __op_check_grad_##op_type##__(#op_type, false)

#define EMPTY_GRAD_MKLDNN_OP(op_type)                   \
  static ::paddle::framework::EmptyGradOperatorRegistra \
      __op_check_grad_##op_type##__(#op_type, true)

}  // namespace framework
}  // namespace paddle
