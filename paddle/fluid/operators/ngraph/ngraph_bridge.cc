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

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_ops.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {

std::map<std::string,
         std::function<void(const std::shared_ptr<framework::OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphBridge::NG_NODE_MAP = {
        {"fill_constant", paddle::operators::ngraphs::BuildFillConstantNode},
        {"mean", paddle::operators::ngraphs::BuildMeanNode},
        {"mean_grad", paddle::operators::ngraphs::BuildMeanGradNode},
        {"mul", paddle::operators::ngraphs::BuildMulNode},
        {"mul_grad", paddle::operators::ngraphs::BuildMulGradNode},
        {"scale", paddle::operators::ngraphs::BuildScaleNode},
        {"relu", paddle::operators::ngraphs::BuildUnaryNode<ngraph::op::Relu>},
        {"tanh", paddle::operators::ngraphs::BuildUnaryNode<ngraph::op::Tanh>},
        {"top_k", paddle::operators::ngraphs::BuildTopKNode}};

void NgraphBridge::BuildNgNode(
    const std::shared_ptr<framework::OperatorBase>& op) {
  auto& op_type = op->Type();
  NG_NODE_MAP[op_type](op, ngb_node_map_);
}

}  // namespace operators
}  // namespace paddle
