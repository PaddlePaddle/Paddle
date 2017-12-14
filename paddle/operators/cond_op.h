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
#include <vector>
#include "glog/logging.h"
#include "paddle/framework/ddim.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/tensor.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

/*
 * @brief CondOp is a dynamic if-else Operator
 *
 * It has a input tensor named cond indicating which netop each instance will
 * run.
 *
 * if cond == 1, it will run true_net, which is a NetOp.
 *
 * if cond == 0, it will run false_net, which is another NetOp.
 */
class CondOp : public framework::OperatorBase {
 public:
  CondOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    sub_net_op_.resize(BRANCH_NUM);
  }

  CondOp(const CondOp& o)
      : framework::OperatorBase(
            static_cast<const framework::OperatorBase&>(o)) {
    // TODO(yuyang18): Implement copy ctor well.
    PADDLE_THROW("Not implemented");
  }

  framework::Scope& AddSubScope(const framework::Scope& scope) const;
  std::vector<framework::Scope*>& GetSubScopes(
      const framework::Scope& scope) const;

  framework::LoDTensor& AddIndexTensor(const framework::Scope& scope) const;
  std::vector<framework::LoDTensor>& GetIndexTensors(
      const framework::Scope& scope) const;

  void PrepareDataForSubnet(const framework::Scope& scope,
                            const platform::DeviceContext& dev_ctx) const;
  void MergeDataFromSubnet(const framework::Scope& scope,
                           const platform::DeviceContext& dev_ctx) const;

  /*
   * Set True Block
   */
  void set_truenet(std::unique_ptr<OperatorBase>&& net) {
    sub_net_op_[TRUE_BRANCH] = std::move(net);
  }

  /*
   * Set False Block
   */
  void set_falsenet(std::unique_ptr<OperatorBase>&& net) {
    sub_net_op_[FALSE_BRANCH] = std::move(net);
  }

  void Run(const framework::Scope& scope,
           const platform::DeviceContext& dev_ctx) const override;

 private:
  const int TRUE_BRANCH = 0;
  const int FALSE_BRANCH = 1;
  const int BRANCH_NUM = 2;

  // sub_net_op_[0]: subnet_t
  // sub_net_op_[1]: subnet_f
  std::vector<std::unique_ptr<framework::OperatorBase>> sub_net_op_;
};

}  // namespace operators
}  // namespace paddle
