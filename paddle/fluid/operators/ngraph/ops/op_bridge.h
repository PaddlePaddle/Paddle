/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>

#include "ngraph/node.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace ops {

class NgraphSingleton {
  NgraphSingleton() = default;
  NgraphSingleton(NgraphSingleton const&) = delete;
  void operator=(NgraphSingleton const) = delete;

  ~NgraphSingleton() = default;

  static std::map<
      std::string,
      std::function<void(const std::shared_ptr<framework::OperatorBase>&,
                         std::shared_ptr<std::unordered_map<
                             std::string, std::shared_ptr<ngraph::Node>>>)>>
      ng_node_maps_;

 public:
  template <typename TF>
  static void Register(TF&& tf, const std::string& name) {
    ng_node_maps_[name] = tf;
  }

  static bool Lookup(const std::string& name) {
    auto it = ng_node_maps_.find(name);
    if (it == ng_node_maps_.end()) {
      return true;
    }
    return false;
  }

  static void BuildNode(
      const std::shared_ptr<std::unordered_map<
          std::string, std::shared_ptr<ngraph::Node>>>& ng_maps,
      const std::shared_ptr<framework::OperatorBase>& op,
      const std::string& name) {
    ng_node_maps_[name](op, ng_maps);
  }
};

std::map<std::string,
         std::function<void(const std::shared_ptr<framework::OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphSingleton::ng_node_maps_;

}  // namespace ops
}  // namespace operators
}  // namespace paddle

#define REGISTER_NG_OP(op_type__, Converter__)                  \
  struct ng_##op_type__##_converter {                           \
    ng_##op_type__##_converter() {                              \
      paddle::operators::ops::NgraphSingleton::Register(        \
          paddle::operators::ngraphs::Converter__, #op_type__); \
    }                                                           \
  };                                                            \
  ng_##op_type__##_converter ng_##op_type__##_converter__;
