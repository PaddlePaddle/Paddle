// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/utils/small_vector.h"

namespace paddle {
namespace imperative {
// TODO(chenweihang): now only check single var input
template <typename VarType>
static bool IsValidVar(const std::string& name,
                       const NameVarMap<VarType>& inputs) {
  auto it = inputs.find(name);
  if (it == inputs.end()) {
    return false;
  }
  if (it->second.empty()) {
    return false;
  }
  return it->second[0] != nullptr;
}

class KernelArgsNameMaker {
 public:
  virtual ~KernelArgsNameMaker() {}
  virtual const paddle::SmallVector<std::string>& GetInputArgsNames() = 0;
  virtual const paddle::SmallVector<std::string>& GetOutputArgsNames() = 0;
  virtual const paddle::SmallVector<
      std::pair<std::string, framework::proto::AttrType>>&
  GetAttrsArgsNamesAndTypes() = 0;
};

template <typename VarType>
class KernelArgsNameMakerByOpProto : public KernelArgsNameMaker {
 public:
  KernelArgsNameMakerByOpProto(framework::proto::OpProto* op_proto,
                               const imperative::NameVarMap<VarType>* inputs,
                               const imperative::NameVarMap<VarType>* outputs)
      : op_proto_(op_proto), inputs_(inputs), outputs_(outputs) {}

  ~KernelArgsNameMakerByOpProto() {}

  const paddle::SmallVector<std::string>& GetInputArgsNames() override {
    for (int i = 0; i < op_proto_->inputs_size(); ++i) {
      auto in = op_proto_->inputs()[i];

      // TODO(chenweihang): deal with diff param in vector
      if ((in.has_extra() && in.extra()) || (in.has_quant() && in.quant())) {
        VLOG(1) << "Dygraph PtKernel input: skip extra & quant input - "
                << in.name();
        continue;
      }

      std::string in_name = in.name();
      if (in.has_dispensable() && in.dispensable()) {
        if (this->contain_host_tensor_flags.count(in_name) > 0 &&
            IsValidVar<VarType>(in_name, *inputs_)) {
          VLOG(1) << "Dygraph PtKernel input: contain host input - " << in_name;
          this->contain_host_tensor_flags[in_name] = true;
        } else {
          VLOG(1) << "Dygraph PtKernel input: skip dispensable input - "
                  << in_name;
          continue;
        }
      }

      input_names.emplace_back(in.name());
    }
    return input_names;
  }

  const paddle::SmallVector<std::string>& GetOutputArgsNames() override {
    for (int i = 0; i < op_proto_->outputs_size(); ++i) {
      auto out_name = op_proto_->outputs()[i].name();
      VLOG(1) << "Dygraph PtKernel output: " << out_name;
      // TODO(chenweihang): outputs also need skip some cases

      output_names.emplace_back(out_name);
    }
    return output_names;
  }

  const paddle::SmallVector<std::pair<std::string, framework::proto::AttrType>>&
  GetAttrsArgsNamesAndTypes() override {
    for (int i = 0; i < op_proto_->attrs_size(); ++i) {
      auto attr = op_proto_->attrs()[i];
      if (attr.name() == "use_mkldnn" || attr.name() == "op_role" ||
          attr.name() == "op_role_var" || attr.name() == "op_namescope" ||
          attr.name() == "op_callstack" || attr.name() == "op_device") {
        VLOG(1) << "Dygraph PtKernel attribute: skip needless attr - "
                << attr.name();
        continue;
      }
      if ((attr.has_extra() && attr.extra()) ||
          (attr.has_quant() && attr.quant())) {
        VLOG(1) << "Dygraph PtKernel attribute: skip extra & quant attr - "
                << attr.name();
        continue;
      }
      if (attr_to_host_tensor.count(attr.name()) > 0 &&
          contain_host_tensor_flags.at(attr_to_host_tensor.at(attr.name())) ==
              true) {
        VLOG(1) << "Dygraph PtKernel attribute: skip dynaimc attr - "
                << attr.name() << ", because "
                << attr_to_host_tensor.at(attr.name()) << " exists.";
        continue;
      }
      // TODO(chenweihang): we need better methods to deal with special cases
      if (attr.name() == "dtype") {
        VLOG(1) << "Dygraph PtKernel attribute: skip " << op_proto_->type()
                << "'s dtype attr.";
        continue;
      }
      VLOG(1) << "Dygraph PtKernel attribute: " << attr.name();
      attr_names.emplace_back(
          std::pair<std::string, framework::proto::AttrType>(attr.name(),
                                                             attr.type()));
    }

    return attr_names;
  }

 private:
  framework::proto::OpProto* op_proto_;

  const imperative::NameVarMap<VarType>* inputs_;
  const imperative::NameVarMap<VarType>* outputs_;

  paddle::SmallVector<std::string> input_names;
  paddle::SmallVector<std::string> output_names;
  paddle::SmallVector<std::pair<std::string, framework::proto::AttrType>>
      attr_names;

  // TODO(chenweihang): For scale op, when the input has a `ScaleTensor`,
  // the following scale attribute should be skipped, and there are many
  // such ops, which require certain rules to process, now only for verify
  // scale op
  std::unordered_map<std::string, bool> contain_host_tensor_flags{
      {"ScaleTensor", false}};
  std::unordered_map<std::string, std::string> attr_to_host_tensor{
      {"scale", "ScaleTensor"}};
};

}  // namespace imperative
}  // namespace paddle
