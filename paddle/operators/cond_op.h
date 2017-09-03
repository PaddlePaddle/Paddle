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
#include "glog/logging.h"
#include "paddle/framework/eigen.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/ddim.h"
#include "paddle/operators/gather.h"
#include <vector>

namespace paddle {
namespace operators {

using namespace paddle::framework;

template <typename Place, typename T>
class CondOp final : public OperatorBase {
public:
  /**
   * InferShape must be called before Run.
   */
  void InferShape(const std::shared_ptr<Scope>& scope) const override;

  // Set True Block
  void set_truenet(std::unique_ptr<OperatorBase> net) {
    sub_net_op_[0] = std::move(net);
  }

  // Set False Block
  void set_falsenet(std::unique_ptr<OperatorBase> net) {
    sub_net_op_[1] = std::move(net);
  }

  virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const override {
    auto* cond = context.Input<Tensor>("Cond");
    // Step 1: get the true/false index at runtime
    // _index[0]: vector<int>, contains all index for cond[i] == true
    // _index[1]: vector<int>, contains all index for cond[i] == false
    for(int i = 0; i < 2; ++i)
      _index[i].clear();
    for(int i = 0; i < cond->dims()[0]; ++i) {
      if (cond->data<bool>()[i])
        _index[0].push_back(i);
      else
        _index[1].push_back(i);
    }
    // put _index[0] and _index[1] into two tensors
    // tensor_index[0] and tensor_index[1]
    framework::DDim dim_ = paddle::framework::make_ddim({0});
    for(int i = 0; i < 2; ++i) {
      dim_[0] = _index[i].size();
      int* tmp_ = _index[i]->mutable_data<int>(dim_, CPUPlace());
      tensor_index[i]->Resize(dim_);
      memcpy(tmp_, index_[i], dim_[0] * sizeof(int));
    }
    
    
    // Step 2: collect data by calling gather
    for (int i = 0; i < 2; ++i) { 
      // i= 0/i for True and False branches respectively
      for (auto& input : GetAttr<std::vector<std::string>>("sub_inputs")) {
        auto var_name = input.second;
        // find Tensor
        Tensor* Tensor_parent = scope.FindVar(var_name)->GetMutable<Tensor>();
        Tensor* Tensor_child = sub_scope_[i].FindVar(var_name)->GetMutable<Tensor>();
        Gather<T>(dev_ctx.GetPlace(), tensor_parent, tensor_index[i], tensor_child); 
      }
    }

    // Step 3: run
    for (int i = 0; i < 2; ++i)
      sub_net_op_[i]->Run(sub_scope_[i], dev_ctx);

    // Step 4: merge output results
    for (int i = 0; i < 2; ++i) {
      // i= 0/i for True and False branches respectively
      for (auto& output : GetAttr<std::vector<std::string>>("sub_outputs")) {
        auto var_name = output.second;
        // find Tensor
        Tensor* Tensor_parent = scope.FindVar(var_name)->GetMutable<Tensor>();
        Tensor* Tensor_child = sub_scope_[i].FindVar(var_name)->GetMutable<Tensor>();
        ScatterUpdate<T>(dev_ctx.GetPlace(), tensor_child, tensor_index[i], tensor_parent);
      }
    }
  }

private:
  // sub_scope_[0]: true scope
  // sub_scope_[1]: false scope
  std::vector<Scope*> sub_scope_;

  // sub_net_op_[0]: subnet_t
  // sub_net_op_[1]: subnet_f
  std::vector<std::unique_ptr<framework::OperatorBase>> sub_net_op_;

  // tensor_index[0]: True_index tensor
  // tensor_index[1]: False_index;
  std::vector<Tensor*> tensor_index;

  // _index[0]: True_index; 
  // _index[1]: False_index;
  vector<vector<int> > _index;
};

/*
class CondGradientOp final : public OperatorBase {
public:
	void Init() override;

	virtual void InferShape(const std::shared_ptr<Scope>& scope) const override;

	virtual void Run(const std::shared_ptr<Scope>& scope,
                   const platform::DeviceContext& dev_ctx) const override;
};*/

}  // namespace operators
}  // namespace paddle

