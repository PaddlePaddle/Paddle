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
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

class InferShapeContext;
class OpAttrChecker;

class InferShapeBase {
 public:
  virtual ~InferShapeBase() = default;
  virtual void operator()(InferShapeContext*) const = 0;
};

class OpInfo {
 public:
  OpCreator creator_;
  GradOpMakerFN grad_op_maker_;
  proto::OpProto* proto_{nullptr};
  OpAttrChecker* checker_{nullptr};
  InferVarTypeFN infer_var_type_;
  InferShapeFN infer_shape_;
  InferInplaceOpFN infer_inplace_;
  InferNoNeedBufferVarsFN infer_no_need_buffer_vars_;
  DygraphGradOpMakerFN dygraph_grad_op_maker_;

  // NOTE(zjl): this flag is added to check whether
  // the grad maker is the default one.
  bool use_default_grad_op_desc_maker_{false};

  // NOTE(huihuangzheng): this flag is added to check whether
  // the grad maker is the empty one.
  bool use_empty_grad_op_desc_maker_{false};

  bool HasOpProtoAndChecker() const {
    return proto_ != nullptr && checker_ != nullptr;
  }

  const proto::OpProto& Proto() const {
    PADDLE_ENFORCE_NOT_NULL(
        proto_,
        platform::errors::NotFound("Operator's Proto has not been registered"));
    PADDLE_ENFORCE_EQ(proto_->IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "Operator's Proto in op info is not initialized."));
    return *proto_;
  }

  const OpCreator& Creator() const {
    PADDLE_ENFORCE_NOT_NULL(creator_,
                            platform::errors::NotFound(
                                "Operator's Creator has not been registered."));
    return creator_;
  }

  const GradOpMakerFN& GradOpMaker() const {
    // Normally, proto_ should not be null, except some special operators, such
    // as LeaklyReluDoubleGrad op.
    std::string type = proto_ ? proto_->type() : "unknown";
    PADDLE_ENFORCE_NOT_NULL(
        grad_op_maker_,
        platform::errors::NotFound(
            "Operator %s's GradOpMaker has not been "
            "registered.\nPlease check whether (%s) operator has "
            "gradient operator.\nIf not, please set stop_gradient to be True "
            "for its input and output variables using var.stop_gradient=True.",
            type.c_str(), type.c_str()));
    return grad_op_maker_;
  }

  // some ops don't have grad_op_maker, add check before use GradOpMaker()
  bool HasGradOpMaker() const { return grad_op_maker_ != nullptr; }

  bool HasNonEmptyGradOpMaker() const {
    return grad_op_maker_ != nullptr && !use_empty_grad_op_desc_maker_;
  }

  const DygraphGradOpMakerFN& DygraphGradOpMaker() const {
    // Normally, proto_ should not be null, except some special operators, such
    // as LeaklyReluDoubleGrad op.
    std::string type = proto_ ? proto_->type() : "unknown";
    PADDLE_ENFORCE_NOT_NULL(
        dygraph_grad_op_maker_,
        platform::errors::NotFound(
            "Operator %s's DygraphGradOpMaker has not been "
            "registered.\nPlease check whether (%s) operator has "
            "gradient operator.\nIf not, please set stop_gradient to be True "
            "for its input and output variables using var.stop_gradient=True.",
            type.c_str(), type.c_str()));
    return dygraph_grad_op_maker_;
  }

  bool HasDygraphGradOpMaker() const {
    return dygraph_grad_op_maker_ != nullptr;
  }

  bool HasInferInplace() const { return infer_inplace_ != nullptr; }

  const OpAttrChecker* Checker() const { return checker_; }

  const InferNoNeedBufferVarsFN& NoNeedBufferVarsInferer() const {
    return infer_no_need_buffer_vars_;
  }
};

class OpInfoMap {
 public:
  static OpInfoMap& Instance();

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& type, const OpInfo& info) {
    PADDLE_ENFORCE_NE(Has(type), true,
                      platform::errors::AlreadyExists(
                          "Operator (%s) has been registered.", type));
    map_.insert({type, info});
  }

  const OpInfo& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
    PADDLE_ENFORCE_NOT_NULL(
        op_info_ptr,
        platform::errors::NotFound("Operator (%s) is not registered.", type));
    return *op_info_ptr;
  }

  const OpInfo* GetNullable(const std::string& type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  const std::unordered_map<std::string, OpInfo>& map() const { return map_; }

  std::unordered_map<std::string, OpInfo>* mutable_map() { return &map_; }

  std::vector<std::string> GetUseDefaultGradOpDescMakerOps() const;

 private:
  OpInfoMap() = default;
  std::unordered_map<std::string, OpInfo> map_;

  DISABLE_COPY_AND_ASSIGN(OpInfoMap);
};

}  // namespace framework
}  // namespace paddle
