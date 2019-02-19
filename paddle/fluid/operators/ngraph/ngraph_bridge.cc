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

#include <algorithm>
#include <functional>
#include <vector>

#include "ngraph/ngraph.hpp"
#include "paddle/fluid/operators/ngraph/ngraph_bridge.h"
#include "paddle/fluid/operators/ngraph/ngraph_ops.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/ngraph_helper.h"

namespace paddle {
namespace operators {

namespace NG_OPS = paddle::operators::ngraphs;
std::map<std::string,
         std::function<void(const std::shared_ptr<framework::OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphBridge::NG_NODE_MAP = {
        {"accuracy", NG_OPS::BuildAccuracyNode},
        {"conv2d", NG_OPS::BuildConv2dNode},
        {"conv2d_grad", NG_OPS::BuildConv2dGradNode},
        {"batch_norm", NG_OPS::BuildBatchNormNode},
        {"batch_norm_grad", NG_OPS::BuildBatchNormGradNode},
        {"cross_entropy", NG_OPS::BuildCrossEntropyNode},
        {"cross_entropy_grad", NG_OPS::BuildCrossEntropyGradNode},
        {"elementwise_add", NG_OPS::BuildElementwiseAddNode},
        {"elementwise_add_grad", NG_OPS::BuildElementwiseAddGradNode},
        {"fill_constant", NG_OPS::BuildFillConstantNode},
        {"mean", NG_OPS::BuildMeanNode},
        {"mean_grad", NG_OPS::BuildMeanGradNode},
        {"mul", NG_OPS::BuildMulNode},
        {"mul_grad", NG_OPS::BuildMulGradNode},
        {"pool2d", NG_OPS::BuildPool2dNode},
        {"pool2d_grad", NG_OPS::BuildPool2dGradNode},
        {"softmax", NG_OPS::BuildSoftmaxNode},
        {"softmax_grad", NG_OPS::BuildSoftmaxGradNode},
        {"scale", NG_OPS::BuildScaleNode},
        {"sigmoid", NG_OPS::BuildUnaryNode<ngraph::op::Sigmoid>},
        {"sum", NG_OPS::BuildSumNode},
        {"relu", NG_OPS::BuildUnaryNode<ngraph::op::Relu>},
        {"relu_grad", NG_OPS::BuildReluGradNode},
        {"tanh", NG_OPS::BuildUnaryNode<ngraph::op::Tanh>},
        {"tanh_grad", NG_OPS::BuildTanhGradNode},
        {"top_k", NG_OPS::BuildTopKNode}};

void NgraphBridge::BuildNgNode(
    const std::shared_ptr<framework::OperatorBase>& op) {
  auto& op_type = op->Type();
  NG_NODE_MAP[op_type](op, ngb_node_map_);
}

}  // namespace operators
}  // namespace paddle
