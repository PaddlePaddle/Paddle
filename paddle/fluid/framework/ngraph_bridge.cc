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

#ifdef PADDLE_WITH_NGRAPH
#include <algorithm>
#include <functional>
#include <vector>

#include "paddle/fluid/framework/ngraph_bridge.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

#include "ngraph/ngraph.hpp"

namespace paddle {
namespace framework {

static ngraph::Shape FlattenTo2d(ngraph::Shape sh, int num) {
  auto x1 = std::accumulate(std::begin(sh), std::begin(sh) + num, 1,
                            std::multiplies<size_t>());
  auto x2 = std::accumulate(std::begin(sh) + num, std::end(sh), 1,
                            std::multiplies<size_t>());
  size_t x1_l = (size_t)x1;
  size_t x2_l = (size_t)x2;
  return ngraph::Shape{x1_l, x2_l};
}

static ngraph::Shape FlattenTo1d(ngraph::Shape sh, int num) {
  auto x1 = std::accumulate(std::begin(sh), std::end(sh) + num, 1,
                            std::multiplies<size_t>());
  size_t x1_l = (size_t)x1;
  return ngraph::Shape{x1_l};
}

static std::shared_ptr<ngraph::Node> Nhwc2Nchw(
    std::shared_ptr<ngraph::Node> in) {
  auto in_shape = in->get_shape();
  in_shape[0] = in->get_shape()[0];
  in_shape[1] = in->get_shape()[3];
  in_shape[2] = in->get_shape()[1];
  in_shape[3] = in->get_shape()[2];
  ngraph::AxisVector axis_vec = {0, 3, 1, 2};
  return std::make_shared<ngraph::op::Reshape>(in, axis_vec, in_shape);
}

static std::shared_ptr<ngraph::Node> Nchw2Nhwc(
    std::shared_ptr<ngraph::Node> in) {
  auto in_shape = in->get_shape();
  in_shape[0] = in->get_shape()[0];
  in_shape[1] = in->get_shape()[2];
  in_shape[2] = in->get_shape()[3];
  in_shape[3] = in->get_shape()[1];
  ngraph::AxisVector axis_vec = {0, 2, 3, 1};
  return std::make_shared<ngraph::op::Reshape>(in, axis_vec, in_shape);
}

static std::shared_ptr<ngraph::Node> NgReshaper(
    std::shared_ptr<ngraph::Node> input, ngraph::Shape shape) {
  std::vector<size_t> input_order(input->get_shape().size());
  std::iota(std::begin(input_order), std::end(input_order), 0);
  return std::make_shared<ngraph::op::Reshape>(
      input, ngraph::AxisVector(input_order), shape);
}

template <typename T>
std::shared_ptr<ngraph::Node> ElementwiseScalar(
    float scale, std::shared_ptr<ngraph::Node> node) {
  auto node_shape = node->get_shape();
  auto scale_const = ngraph::op::Constant::create(node->get_element_type(),
                                                  node_shape, {scale});
  return std::make_shared<T>(scale_const, node);
}

template <typename T>
std::shared_ptr<ngraph::Node> ElementwiseScalar(
    std::shared_ptr<ngraph::Node> scale_1d,
    std::shared_ptr<ngraph::Node> node) {
  auto scale_shape = scale_1d->get_shape();
  PADDLE_ENFORCE_EQ(scale_shape.size(), 1, "Supporting 1d scale node");
  PADDLE_ENFORCE_EQ(scale_shape.at(0), 1, "scale 1d in in shape {1}");

  auto node_shape = node->get_shape();
  ngraph::AxisSet axis_set;
  for (size_t i = 0; i < node_shape.size(); ++i) {
    axis_set.insert(i);
  }
  node_shape.push_back(1);

  auto scale_bcast =
      std::make_shared<ngraph::op::Broadcast>(scale_1d, node_shape, axis_set);

  auto scale_reshape = NgReshaper(scale_bcast, node->get_shape());

  return std::make_shared<T>(scale_reshape, node);
}

static std::shared_ptr<ngraph::Node> GroupedConvolution(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = filter_shape.at(0) / groups;
    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);
    auto ng_conv = std::make_shared<ngraph::op::Convolution>(
        data_slice, filter_slice, strides, dilations, paddings, paddings);
    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 1;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

static std::shared_ptr<ngraph::Node> GroupedGradConvolutionFilter(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters,
    const std::shared_ptr<ngraph::Node>& doutput, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  auto& out_shape = doutput->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = data_shape.at(0);

    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);

    const std::vector<size_t> olower_bound{0, i * filter_step, 0, 0};
    const std::vector<size_t> oupper_bound{out_shape.at(0),
                                           (i + 1) * filter_step,
                                           out_shape.at(2), out_shape.at(3)};
    auto out_slice = std::make_shared<ngraph::op::Slice>(doutput, olower_bound,
                                                         oupper_bound);

    auto ng_conv = std::make_shared<ngraph::op::ConvolutionBackpropFilters>(
        data_slice, filter_slice->get_shape(), out_slice, strides, dilations,
        paddings, paddings, ngraph::Strides{1, 1});

    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 0;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

static std::shared_ptr<ngraph::Node> GroupedGradConvolutionData(
    const std::shared_ptr<ngraph::Node>& data_batch,
    const std::shared_ptr<ngraph::Node>& filters,
    const std::shared_ptr<ngraph::Node>& doutput, const ngraph::Strides strides,
    const ngraph::Strides dilations, const ngraph::CoordinateDiff& paddings,
    size_t groups) {
  auto& data_shape = data_batch->get_shape();
  auto& filter_shape = filters->get_shape();
  auto& out_shape = doutput->get_shape();
  ngraph::NodeVector ng_slices;

  for (size_t i = 0; i < groups; ++i) {
    size_t channel_step = filter_shape.at(1);
    const std::vector<size_t> lower_bound{0, i * channel_step, 0, 0};
    const std::vector<size_t> upper_bound{data_shape.at(0),
                                          (i + 1) * channel_step,
                                          data_shape.at(2), data_shape.at(3)};
    auto data_slice = std::make_shared<ngraph::op::Slice>(
        data_batch, lower_bound, upper_bound);

    size_t filter_step = data_shape.at(0);

    const std::vector<size_t> filter_lower_bound{i * filter_step, 0, 0, 0};
    const std::vector<size_t> filter_upper_bound{
        (i + 1) * filter_step, filter_shape.at(1), filter_shape.at(2),
        filter_shape.at(3)};
    auto filter_slice = std::make_shared<ngraph::op::Slice>(
        filters, filter_lower_bound, filter_upper_bound);

    const std::vector<size_t> olower_bound{0, i * filter_step, 0, 0};
    const std::vector<size_t> oupper_bound{out_shape.at(0),
                                           (i + 1) * filter_step,
                                           out_shape.at(2), out_shape.at(3)};
    auto out_slice = std::make_shared<ngraph::op::Slice>(doutput, olower_bound,
                                                         oupper_bound);

    auto ng_conv = std::make_shared<ngraph::op::ConvolutionBackpropData>(
        data_slice->get_shape(), filter_slice, out_slice, strides, dilations,
        paddings, paddings, ngraph::Strides{1, 1});
    ng_slices.push_back(ng_conv);
  }

  size_t concat_axis = 1;
  return std::make_shared<ngraph::op::Concat>(ng_slices, concat_axis);
}

static std::shared_ptr<ngraph::Node> GetNode(
    const std::shared_ptr<OperatorBase>& op, const std::string prm,
    const VariableNameMap& var_map,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = var_map.at(prm);
  PADDLE_ENFORCE_EQ(var_names.size(), 1,
                    "op %s prm %s expects one associated var", op->Type(), prm);
  if (ngb_node_map->find(var_names[0]) != ngb_node_map->end()) {
    return (*ngb_node_map)[var_names[0]];
  } else {
    return nullptr;
  }
}

static std::shared_ptr<ngraph::Node> GetInputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string prm,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, prm, op->Inputs(), ngb_node_map);
}

static std::shared_ptr<ngraph::Node> GetOutputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string prm,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  return GetNode(op, prm, op->Outputs(), ngb_node_map);
}

static void SetOutputNode(
    const std::shared_ptr<OperatorBase>& op, const std::string prm,
    std::shared_ptr<ngraph::Node> node,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto& var_names = op->Outputs().at(prm);
  if (var_names.size() == 1) {
    (*ngb_node_map)[var_names[0]] = node;
  } else if (var_names.size() == 0) {
    (*ngb_node_map)[""] = node;
  } else {
    PADDLE_THROW("prm %s has more than 1 var_names.", prm);
  }
}

static bool HasOutput(const std::shared_ptr<OperatorBase>& op,
                      const std::string prm) {
  auto& outputs = op->Outputs();
  if (outputs.find(prm) == outputs.end()) return false;
  return outputs.at(prm).size() > 0;
}

template <typename T>
static void BuildBinaryNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto y = GetInputNode(op, "Y", ngb_node_map);
  auto out = std::make_shared<T>(x, y);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

template <typename T>
static void BuildUnaryNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto out = std::make_shared<T>(input);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static inline void GetMidDims(const ngraph::Shape& x_shape,
                              const ngraph::Shape& y_shape, int axis, int* pre,
                              int* n, int* post) {
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_shape[i];
  }

  for (size_t i = 0; i < y_shape.size(); ++i) {
    PADDLE_ENFORCE_EQ(x_shape[i + axis], y_shape[i],
                      "Broadcast dimension mismatch.");
    (*n) *= y_shape[i];
  }

  for (size_t i = axis + y_shape.size(); i < x_shape.size(); ++i) {
    (*post) *= x_shape[i];
  }
}

static inline void TrimTrailingSingularDims(ngraph::Shape* shape) {
  // Remove trailing dimensions of size 1 for y
  auto actual_shape_size = shape->size();
  for (; actual_shape_size != 0; --actual_shape_size) {
    if ((*shape)[actual_shape_size - 1] != 1) {
      break;
    } else {
      shape->pop_back();
    }
  }
}

static ngraph::NodeVector ElementwiseBinaryNodePrepare(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  int axis = op_attrs.Get<int>("axis");
  auto lhs = GetInputNode(op, "X", ngb_node_map);
  auto rhs = GetInputNode(op, "Y", ngb_node_map);

  auto lhs_shape = lhs->get_shape();
  auto rhs_shape = rhs->get_shape();

  PADDLE_ENFORCE_GE(lhs_shape.size(), rhs_shape.size(),
                    "Rank of first input must >= rank of second input.");
  if (lhs_shape == rhs_shape) {
    return ngraph::NodeVector{lhs, rhs};
  }
  axis = (axis == -1 ? lhs_shape.size() - rhs_shape.size() : axis);
  PADDLE_ENFORCE(axis >= 0 && axis < (int)(lhs_shape.size()),
                 "Axis should be in range [0, lhs_shape)");
  TrimTrailingSingularDims(&rhs_shape);
  axis = (rhs_shape.size() == 0) ? lhs_shape.size() : axis;

  int pre, n, post;
  GetMidDims(lhs_shape, rhs_shape, axis, &pre, &n, &post);

  ngraph::Shape l_shape{};
  l_shape.push_back(pre);
  l_shape.push_back(n);
  l_shape.push_back(post);

  std::vector<size_t> rhs_order(rhs->get_shape().size());
  std::iota(std::begin(rhs_order), std::end(rhs_order), 0);
  ngraph::Shape r_shape{};
  r_shape.push_back(n);
  auto rhs_reshape = std::make_shared<ngraph::op::Reshape>(
      rhs, ngraph::AxisVector(rhs_order), r_shape);
  auto rhs_bcast = std::make_shared<ngraph::op::Broadcast>(
      rhs_reshape, l_shape, ngraph::AxisSet{0, 2});
  std::vector<size_t> bcast_order(rhs_bcast->get_shape().size());
  std::iota(std::begin(bcast_order), std::end(bcast_order), 0);
  std::shared_ptr<ngraph::Node> rhs_bcast_reshape =
      std::make_shared<ngraph::op::Reshape>(
          rhs_bcast, ngraph::AxisVector(bcast_order), lhs_shape);
  return ngraph::NodeVector{lhs, rhs_bcast_reshape};
}

template <typename T>
static void BuildElementwiseBinaryNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto nodes = ElementwiseBinaryNodePrepare(op, ngb_node_map);
  std::shared_ptr<ngraph::Node>& x = nodes.at(0);
  std::shared_ptr<ngraph::Node>& y = nodes.at(1);

  if (x->get_element_type() != y->get_element_type()) {
    y = std::make_shared<ngraph::op::Convert>(y, x->get_element_type());
  }
  auto out = std::make_shared<T>(x, y);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

template <typename T>
static void BuildElementwiseCompareNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto nodes = ElementwiseBinaryNodePrepare(op, ngb_node_map);
  std::shared_ptr<ngraph::Node>& x = nodes.at(0);
  std::shared_ptr<ngraph::Node>& y = nodes.at(1);

  if (x->get_element_type() != y->get_element_type()) {
    x = std::make_shared<ngraph::op::Convert>(x, ngraph::element::f64);
    y = std::make_shared<ngraph::op::Convert>(y, ngraph::element::f64);
  }
  auto out = std::make_shared<T>(x, y);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildElementwiseAddGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  int axis = op_attrs.Get<int>("axis");

  auto dout = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto y = GetInputNode(op, "Y", ngb_node_map);
  auto dout_shape = dout->get_shape();
  auto y_shape = y->get_shape();

  if (dout_shape == y_shape) {
    SetOutputNode(op, "X@GRAD", dout, ngb_node_map);
    SetOutputNode(op, "Y@GRAD", dout, ngb_node_map);
  } else {
    axis = (axis == -1 ? dout_shape.size() - y_shape.size() : axis);
    TrimTrailingSingularDims(&y_shape);
    axis = (y_shape.size() == 0 ? dout_shape.size() : axis);

    int pre, n, post;
    GetMidDims(dout_shape, y_shape, axis, &pre, &n, &post);

    ngraph::Shape lhs_shape{};
    lhs_shape.push_back(pre);
    lhs_shape.push_back(n);
    if (post != 1) {
      lhs_shape.push_back(post);
    }

    std::vector<size_t> lhs_order(dout_shape.size());
    std::iota(std::begin(lhs_order), std::end(lhs_order), 0);
    auto dout_reshape = std::make_shared<ngraph::op::Reshape>(
        dout, ngraph::AxisVector(lhs_order), lhs_shape);

    ngraph::AxisSet axis_set{0};
    if (post != 1) {
      axis_set.insert(2);
    }

    auto dout_sum = std::make_shared<ngraph::op::Sum>(dout_reshape, axis_set);
    auto dy = std::make_shared<ngraph::op::Reshape>(
        dout_sum, ngraph::AxisVector{0}, y->get_shape());

    SetOutputNode(op, "X@GRAD", dout, ngb_node_map);
    SetOutputNode(op, "Y@GRAD", dy, ngb_node_map);
  }
}

static void BuildAccuracyNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto indices = GetInputNode(op, "Indices", ngb_node_map);
  auto label = GetInputNode(op, "Label", ngb_node_map);
  auto inference = GetInputNode(op, "Out", ngb_node_map);

  auto inference_shape = inference->get_shape();
  size_t num_samples = inference_shape.at(0);
  size_t k = inference_shape.at(1);

  std::shared_ptr<ngraph::Node> label_k = label;
  if (k > 1) {
    auto label_1d = std::make_shared<ngraph::op::Reshape>(
        label, ngraph::AxisVector{0, 1}, ngraph::Shape{num_samples});
    label_k = std::make_shared<ngraph::op::Broadcast>(label_1d, inference_shape,
                                                      ngraph::AxisSet{1});
  }

  auto node_equal = std::make_shared<ngraph::op::Equal>(indices, label_k);
  auto num_correct_0d =
      std::make_shared<ngraph::op::Sum>(node_equal, ngraph::AxisSet{0, 1});
  auto num_correct = std::make_shared<ngraph::op::Reshape>(
      num_correct_0d, ngraph::AxisVector{}, ngraph::Shape{1});
  auto n_samples = ngraph::op::Constant::create(
      ngraph::element::i64, ngraph::Shape{1}, {num_samples});
  auto accuracy =
      std::make_shared<ngraph::op::Convert>(num_correct, ngraph::element::f32) /
      std::make_shared<ngraph::op::Convert>(n_samples, ngraph::element::f32);

  SetOutputNode(op, "Accuracy", accuracy, ngb_node_map);
  SetOutputNode(op, "Correct", num_correct, ngb_node_map);
  SetOutputNode(op, "Total", n_samples, ngb_node_map);
}

static void BuildAdamNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto beta1pow = GetInputNode(op, "Beta1Pow", ngb_node_map);
  auto beta2pow = GetInputNode(op, "Beta2Pow", ngb_node_map);
  auto grad = GetInputNode(op, "Grad", ngb_node_map);
  auto learning_rate = GetInputNode(op, "LearningRate", ngb_node_map);
  auto moment1 = GetInputNode(op, "Moment1", ngb_node_map);
  auto moment2 = GetInputNode(op, "Moment2", ngb_node_map);
  auto param = GetInputNode(op, "Param", ngb_node_map);

  auto epsilon = op_attrs.Get<float>("epsilon");
  auto beta2 = op_attrs.Get<float>("beta2");
  auto beta1 = op_attrs.Get<float>("beta1");

  auto moment1_shape = moment1->get_shape();
  auto grad_shape = grad->get_shape();

  auto moment1out = std::make_shared<ngraph::op::Add>(
      ElementwiseScalar<ngraph::op::Multiply>(beta1, moment1),
      ElementwiseScalar<ngraph::op::Multiply>(1. - beta1, grad));

  auto grad_square = std::make_shared<ngraph::op::Multiply>(grad, grad);
  auto moment2out = std::make_shared<ngraph::op::Add>(
      ElementwiseScalar<ngraph::op::Multiply>(beta2, moment2),
      ElementwiseScalar<ngraph::op::Multiply>(1. - beta2, grad_square));
  auto node_sqrt = std::make_shared<ngraph::op::Sqrt>(
      ElementwiseScalar<ngraph::op::Subtract>(1., beta2pow));
  auto lr = std::make_shared<ngraph::op::Divide>(
      node_sqrt, ElementwiseScalar<ngraph::op::Subtract>(1., beta1pow));
  auto updated_lr = std::make_shared<ngraph::op::Multiply>(learning_rate, lr);

  auto moment2_sqrt = std::make_shared<ngraph::op::Sqrt>(moment2out);
  auto param_grad = std::make_shared<ngraph::op::Divide>(
      moment1out, ElementwiseScalar<ngraph::op::Add>(epsilon, moment2_sqrt));
  auto delta = ElementwiseScalar<ngraph::op::Multiply>(updated_lr, param_grad);
  auto param_out = std::make_shared<ngraph::op::Subtract>(param, delta);

  SetOutputNode(op, "Moment1Out", moment1out, ngb_node_map);
  SetOutputNode(op, "Moment2Out", moment2out, ngb_node_map);
  SetOutputNode(op, "ParamOut", param_out, ngb_node_map);
}

static void BuildBatchNormNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto& data_layout = op_attrs.Get<std::string>("data_layout");

  auto bias = GetInputNode(op, "Bias", ngb_node_map);
  auto mean = GetInputNode(op, "Mean", ngb_node_map);
  auto variance = GetInputNode(op, "Variance", ngb_node_map);
  auto scale = GetInputNode(op, "Scale", ngb_node_map);
  auto x = GetInputNode(op, "X", ngb_node_map);

  const bool is_test = op_attrs.Get<bool>("is_test");
  const float epsilon = op_attrs.Get<float>("epsilon");
  const float momentum = op_attrs.Get<float>("momentum");

  if (data_layout == "NHWC") {
    x = Nhwc2Nchw(x);
  }

  std::shared_ptr<ngraph::Node> mean_out, saved_mean, saved_variance,
      variance_out, y;

  if (!is_test) {
    auto BN = std::make_shared<ngraph::op::BatchNormTraining>(epsilon, scale,
                                                              bias, x);
    y = std::make_shared<ngraph::op::GetOutputElement>(BN, 0);
    saved_mean = std::make_shared<ngraph::op::GetOutputElement>(BN, 1);
    saved_variance = std::make_shared<ngraph::op::GetOutputElement>(BN, 2);

    mean_out = std::make_shared<ngraph::op::Add>(
        ElementwiseScalar<ngraph::op::Multiply>(momentum, mean),
        ElementwiseScalar<ngraph::op::Multiply>(1. - momentum, saved_mean));
    variance_out = std::make_shared<ngraph::op::Add>(
        ElementwiseScalar<ngraph::op::Multiply>(momentum, variance),
        ElementwiseScalar<ngraph::op::Multiply>(1. - momentum, saved_variance));

    if (data_layout == "NHWC") {
      y = Nchw2Nhwc(y);
    }

    SetOutputNode(op, "MeanOut", mean_out, ngb_node_map);
    SetOutputNode(op, "VarianceOut", variance_out, ngb_node_map);
    SetOutputNode(op, "SavedMean", saved_mean, ngb_node_map);
    SetOutputNode(op, "SavedVariance", saved_variance, ngb_node_map);
    SetOutputNode(op, "Y", y, ngb_node_map);
  } else {
    y = std::make_shared<ngraph::op::BatchNormInference>(epsilon, scale, bias,
                                                         x, mean, variance);
    SetOutputNode(op, "Y", y, ngb_node_map);
  }
}

static void BuildBatchNormGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto& data_layout = op_attrs.Get<std::string>("data_layout");

  auto bias = GetInputNode(op, "Bias", ngb_node_map);
  auto saved_mean = GetInputNode(op, "SavedMean", ngb_node_map);
  auto saved_variance = GetInputNode(op, "SavedVariance", ngb_node_map);
  auto scale = GetInputNode(op, "Scale", ngb_node_map);
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto dy = GetInputNode(op, "Y@GRAD", ngb_node_map);
  auto x_shape = x->get_shape();
  auto dy_shape = dy->get_shape();

  PADDLE_ENFORCE(x_shape.size() == 2 || x_shape.size() == 4,
                 "BN grap input size needs to be 2 or 4");
  PADDLE_ENFORCE_EQ(x_shape.size(), dy_shape.size(),
                    "BN grap input and delta size needs to be equal");

  if (x_shape.size() == 2) {
    x = std::make_shared<ngraph::op::Reshape>(
        x, ngraph::AxisVector{0, 1},
        ngraph::Shape{x_shape.at(0), x_shape.at(1), 1, 1});
    dy = std::make_shared<ngraph::op::Reshape>(
        dy, ngraph::AxisVector{0, 1},
        ngraph::Shape{dy_shape.at(0), dy_shape.at(1), 1, 1});
  }

  if (data_layout == "NHWC") {
    x = Nhwc2Nchw(dy);
    dy = Nhwc2Nchw(dy);
  }
  const float epsilon = op_attrs.Get<float>("epsilon");

  auto bn_bprop = std::make_shared<ngraph::op::BatchNormTrainingBackprop>(
      epsilon, scale, bias, x, saved_mean, saved_variance, dy);

  std::shared_ptr<ngraph::Node> dx =
      std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 0);
  auto dscale = std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 1);
  auto dbias = std::make_shared<ngraph::op::GetOutputElement>(bn_bprop, 2);
  SetOutputNode(op, "Bias@GRAD", dbias, ngb_node_map);
  SetOutputNode(op, "Scale@GRAD", dscale, ngb_node_map);
  if (x_shape.size() == 2) {
    SetOutputNode(op, "X@GRAD", NgReshaper(dx, x_shape), ngb_node_map);
  } else {
    if (data_layout == "NHWC") {
      dx = Nchw2Nhwc(dx);
    }
    SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
  }
}

static void BuildConv2dNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto filters = GetInputNode(op, "Filter", ngb_node_map);
  auto input = GetInputNode(op, "Input", ngb_node_map);

  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");
  std::vector<int> dilations = op_attrs.Get<std::vector<int>>("dilations");

  const ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                                   static_cast<size_t>(strides.at(1))};
  const ngraph::Strides ng_dilations{static_cast<size_t>(dilations.at(0)),
                                     static_cast<size_t>(dilations.at(1))};
  const ngraph::CoordinateDiff ng_paddings{
      static_cast<std::ptrdiff_t>(paddings.at(0)),
      static_cast<std::ptrdiff_t>(paddings.at(1))};

  int groups = static_cast<size_t>(op_attrs.Get<int>("groups"));
  PADDLE_ENFORCE_GE(groups, 1, "conv groups needs be no less than 1");

  std::shared_ptr<ngraph::Node> result;
  if (groups == 1) {
    result = std::make_shared<ngraph::op::Convolution>(
        input, filters, ng_strides, ng_dilations, ng_paddings, ng_paddings);
  } else {
    result = GroupedConvolution(input, filters, ng_strides, ng_dilations,
                                ng_paddings, groups);
  }
  SetOutputNode(op, "Output", result, ngb_node_map);
}

static void BuildConv2dGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto filter = GetInputNode(op, "Filter", ngb_node_map);
  auto input = GetInputNode(op, "Input", ngb_node_map);
  auto doutput = GetInputNode(op, "Output@GRAD", ngb_node_map);

  int groups = op_attrs.Get<int>("groups");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");
  std::vector<int> dilations = op_attrs.Get<std::vector<int>>("dilations");

  const ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                                   static_cast<size_t>(strides.at(1))};
  const ngraph::Strides ng_dilations{static_cast<size_t>(dilations.at(0)),
                                     static_cast<size_t>(dilations.at(1))};
  const ngraph::CoordinateDiff ng_paddings{
      static_cast<std::ptrdiff_t>(paddings.at(0)),
      static_cast<std::ptrdiff_t>(paddings.at(1))};

  std::shared_ptr<ngraph::Node> dfilter;
  std::shared_ptr<ngraph::Node> dinput;
  if (groups == 1) {
    dfilter = std::make_shared<ngraph::op::ConvolutionBackpropFilters>(
        input, filter->get_shape(), doutput, ng_strides, ng_dilations,
        ng_paddings, ng_paddings, ngraph::Strides{1, 1});

    dinput = std::make_shared<ngraph::op::ConvolutionBackpropData>(
        input->get_shape(), filter, doutput, ng_strides, ng_dilations,
        ng_paddings, ng_paddings, ngraph::Strides{1, 1});

  } else {
    dfilter = GroupedGradConvolutionFilter(input, filter, doutput, ng_strides,
                                           ng_dilations, ng_paddings, groups);
    dinput = GroupedGradConvolutionData(input, filter, doutput, ng_strides,
                                        ng_dilations, ng_paddings, groups);
  }

  SetOutputNode(op, "Filter@GRAD", dfilter, ngb_node_map);
  SetOutputNode(op, "Input@GRAD", dinput, ngb_node_map);
}

static void BuildConcatNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::vector<std::shared_ptr<ngraph::Node>> args;
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      auto& node0 = ngb_node_map->at(var_name);
      args.push_back(node0);
    }
  }
  auto op_attrs = AttrReader(op->Attrs());
  const size_t axis = op_attrs.Get<int>("axis");
  auto out = std::make_shared<ngraph::op::Concat>(args, axis);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildCrossEntropyNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  PADDLE_ENFORCE_EQ(is_soft_label, false,
                    "Not implemented for case soft label to be true");
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto label = std::make_shared<ngraph::op::Convert>(
      GetInputNode(op, "Label", ngb_node_map), x->get_element_type());
  auto label_shape = label->get_shape();
  auto x_shape = x->get_shape();
  if (!is_soft_label) {
    PADDLE_ENFORCE(label_shape.size() == 2 && label_shape.at(1) == 1,
                   "Lable is in N X 1 dimensios as it is not  soft label");
    PADDLE_ENFORCE(x_shape.size() == 2, "Input x should be in 2d");
    PADDLE_ENFORCE(x_shape.at(0) == label_shape.at(0),
                   "Input x and label"
                   " need to have the same dimension");
    auto label_1d = std::make_shared<ngraph::op::Reshape>(
        label, ngraph::AxisVector{0, 1}, ngraph::Shape{label_shape.at(0)});
    auto node_1_hot =
        std::make_shared<ngraph::op::OneHot>(label_1d, x_shape, 1);
    auto node_mul = std::make_shared<ngraph::op::Multiply>(x, node_1_hot);
    auto node_sum =
        std::make_shared<ngraph::op::Sum>(node_mul, ngraph::AxisSet{1});
    auto node_log = std::make_shared<ngraph::op::Log>(node_sum);
    auto node_clip = ngraph::op::Constant::create(
        node_log->get_element_type(), node_log->get_shape(), {-1e20});
    auto node_max = std::make_shared<ngraph::op::Maximum>(node_log, node_clip);
    auto node_neg = std::make_shared<ngraph::op::Negative>(node_max);
    auto xe = std::make_shared<ngraph::op::Reshape>(
        node_neg, ngraph::AxisVector{0}, label_shape);
    SetOutputNode(op, "Y", xe, ngb_node_map);
  } else {
    PADDLE_THROW("Not implemented");
  }
}

static void BuildCrossEntropyGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  const bool is_soft_label = op_attrs.Get<bool>("soft_label");
  PADDLE_ENFORCE_EQ(is_soft_label, false,
                    "Not implemented for case soft label to be true");
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto label = GetInputNode(op, "Label", ngb_node_map);
  auto dy = GetInputNode(op, "Y@GRAD", ngb_node_map);
  PADDLE_ENFORCE_EQ(is_soft_label, false,
                    "Not implemented for case soft label to be true");
  auto label_shape = label->get_shape();
  auto x_shape = x->get_shape();
  auto dy_shape = dy->get_shape();

  if (!is_soft_label) {
    PADDLE_ENFORCE(label_shape.size() == 2 && label_shape.at(1) == 1,
                   "Lable is in N X 1 dimensios as it is not  soft label");
    PADDLE_ENFORCE(dy_shape.size() == 2 && dy_shape.at(1) == 1,
                   "dy is in N X 1 dimensios");

    PADDLE_ENFORCE(x_shape.size() == 2, "Input x should be in 2d");
    PADDLE_ENFORCE(x_shape.at(0) == label_shape.at(0),
                   "Input x and label index 0"
                   " need to have the same dimension");
    auto label_1d = std::make_shared<ngraph::op::Reshape>(
        label, ngraph::AxisVector{0, 1}, ngraph::Shape{label_shape.at(0)});
    auto node_1hot = std::make_shared<ngraph::op::OneHot>(label_1d, x_shape, 1);

    auto dy_reshape = std::make_shared<ngraph::op::Reshape>(
        dy, ngraph::AxisVector{0, 1}, ngraph::Shape{dy_shape.at(0)});
    auto dy_bcast = std::make_shared<ngraph::op::Broadcast>(dy_reshape, x_shape,
                                                            ngraph::AxisSet{1});

    auto node_1hot_converted =
        std::make_shared<ngraph::op::Convert>(node_1hot, x->get_element_type());
    auto xe_grad = (-dy_bcast / x) * node_1hot_converted;
    SetOutputNode(op, "X@GRAD", xe_grad, ngb_node_map);
  } else {
    PADDLE_THROW("Not implemented");
  }
}

static void BuildDropoutNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = AttrReader(op->Attrs());
  auto dropout_prob = op_attrs.Get<float>("dropout_prob");
  auto dropout_implementation =
      op_attrs.Get<std::string>("dropout_implementation");
  if (!op_attrs.Get<bool>("is_test")) {
    if (dropout_implementation == "upscale_in_train") {
      // TODO(mozga) The operator is not supported for a training
    }
  } else {
    if (dropout_implementation == "upscale_in_train") {
      SetOutputNode(op, "Out", input, ngb_node_map);
    } else {
      float value = 1.0f - dropout_prob;
      auto out_val = ngraph::op::Constant::create(input->get_element_type(),
                                                  input->get_shape(), {value});
      auto mul_val = std::make_shared<ngraph::op::Multiply>(input, out_val);
      SetOutputNode(op, "Out", out_val, ngb_node_map);
    }
  }
}

static void BuildDropoutGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto dy = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto mask = GetInputNode(op, "Mask", ngb_node_map);
  SetOutputNode(op, "X@GRAD", dy * mask, ngb_node_map);
}

static void BuildFillConstantNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto vsp = op_attrs.Get<std::vector<int64_t>>("shape");
  ngraph::Shape shape;
  for (auto& sp : vsp) {
    shape.push_back(sp);
  }
  float value = op_attrs.Get<float>("value");
  ngraph::element::Type ng_dtype;
  auto data_type =
      static_cast<proto::VarType::Type>(op_attrs.Get<int>("dtype"));
  if (data_type == proto::VarType::FP32) {
    ng_dtype = ngraph::element::f32;
  } else if (data_type == proto::VarType::FP64) {
    ng_dtype = ngraph::element::f64;
  } else if (data_type == proto::VarType::INT64) {
    ng_dtype = ngraph::element::i64;
  } else if (data_type == proto::VarType::INT32) {
    ng_dtype = ngraph::element::i32;
  } else if (data_type == proto::VarType::BOOL) {
    ng_dtype = ngraph::element::boolean;
  } else {
    PADDLE_THROW("unsupported data type: %s", data_type);
  }
  auto out = ngraph::op::Constant::create(ng_dtype, shape, {value});
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildZerosLikeNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto out =
      ngraph::op::Constant::create(x->get_element_type(), x->get_shape(), {0});
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildLrnNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);

  auto op_attrs = AttrReader(op->Attrs());
  const int n = op_attrs.Get<int>("n");
  const float alpha = op_attrs.Get<float>("alpha");
  const float beta = op_attrs.Get<float>("beta");
  const float k = op_attrs.Get<float>("k");

  auto lrn_out = std::make_shared<ngraph::op::LRN>(input, alpha, beta, k, n);
  std::shared_ptr<ngraph::Node> mid_out = ngraph::op::Constant::create(
      input->get_element_type(), input->get_shape(), {k});

  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_shape() != mid_out->get_shape()) {
    mid_out = NgReshaper(mid_out, dummy_out->get_shape());
  }

  SetOutputNode(op, "MidOut", mid_out, ngb_node_map);
  SetOutputNode(op, "Out", lrn_out, ngb_node_map);
}

static void BuildMeanNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  ngraph::AxisSet axes;
  for (size_t i = 0; i < input->get_shape().size(); ++i) {
    axes.insert(i);
  }

  auto mean = ngraph::builder::mean(input, axes);
  auto mean_1d = std::make_shared<ngraph::op::Reshape>(
      mean, ngraph::AxisVector{}, ngraph::Shape{1});
  SetOutputNode(op, "Out", mean_1d, ngb_node_map);
}

static void BuildMeanGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto og = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto x_shape = x->get_shape();
  float x_size = std::accumulate(std::begin(x_shape), std::end(x_shape), 1,
                                 std::multiplies<float>());
  auto node_const = ngraph::op::Constant::create(og->get_element_type(),
                                                 ngraph::Shape{1}, {x_size});
  auto node_div = std::make_shared<ngraph::op::Divide>(og, node_const);

  auto result = ElementwiseScalar<ngraph::op::Add>(
      og / node_const,
      ngraph::op::Constant::create(og->get_element_type(), x_shape, {0}));
  SetOutputNode(op, "X@GRAD", result, ngb_node_map);
}

static void BuildMomentumNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto param = GetInputNode(op, "Param", ngb_node_map);
  auto grad = GetInputNode(op, "Grad", ngb_node_map);
  auto velocity = GetInputNode(op, "Velocity", ngb_node_map);
  auto learning_rate = GetInputNode(op, "LearningRate", ngb_node_map);

  auto mu = op_attrs.Get<float>("mu");
  bool use_nesterov = op_attrs.Get<bool>("use_nesterov");

  auto param_shape = param->get_shape();
  auto velocity_shape = velocity->get_shape();
  auto grad_shape = grad->get_shape();
  auto lr_shape = learning_rate->get_shape();

  auto shape_velocity = ngraph::Shape{velocity_shape};
  auto mu_create =
      ngraph::op::Constant::create(ngraph::element::f32, shape_velocity, {mu});

  auto vel_mul = std::make_shared<ngraph::op::Multiply>(velocity, mu_create);
  auto vel_out = std::make_shared<ngraph::op::Add>(vel_mul, grad);

  ngraph::NodeVector result;
  if (use_nesterov) {
    auto mul_res = std::make_shared<ngraph::op::Multiply>(vel_out, mu_create);
    auto add_res = std::make_shared<ngraph::op::Add>(grad, mul_res);

    auto add_2d = FlattenTo2d(add_res->get_shape(), 0);
    auto vel_reshape = NgReshaper(vel_out, add_2d);

    auto lr_bcast = std::make_shared<ngraph::op::Broadcast>(
        learning_rate, vel_reshape->get_shape(),
        ngraph::AxisSet{vel_reshape->get_shape().size() - 1});

    auto lr_1d = FlattenTo1d(lr_bcast->get_shape(), 0);
    auto lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_bcast, ngraph::AxisVector{0, 1}, lr_1d);

    lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_reshape, ngraph::AxisVector{0}, param->get_shape());

    auto mul_res1 = std::make_shared<ngraph::op::Multiply>(add_res, lr_reshape);
    auto res = std::make_shared<ngraph::op::Subtract>(param, mul_res1);
    SetOutputNode(op, "ParamOut", res, ngb_node_map);
  } else {
    auto vel_2d = FlattenTo2d(vel_out->get_shape(), 0);
    auto vel_reshape = NgReshaper(vel_out, vel_2d);

    auto lr_bcast = std::make_shared<ngraph::op::Broadcast>(
        learning_rate, vel_reshape->get_shape(),
        ngraph::AxisSet{vel_reshape->get_shape().size() - 1});

    auto lr_1d = FlattenTo1d(lr_bcast->get_shape(), 0);
    auto lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_bcast, ngraph::AxisVector{0, 1}, lr_1d);

    lr_reshape = std::make_shared<ngraph::op::Reshape>(
        lr_reshape, ngraph::AxisVector{0}, param->get_shape());

    auto mul_result =
        std::make_shared<ngraph::op::Multiply>(lr_reshape, vel_out);

    auto res = std::make_shared<ngraph::op::Subtract>(param, mul_result);
    SetOutputNode(op, "ParamOut", res, ngb_node_map);
  }
  SetOutputNode(op, "VelocityOut", vel_out, ngb_node_map);
}

static void BuildMulNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  int x_num_col_dims = op_attrs.Get<int>("x_num_col_dims");
  int y_num_col_dims = op_attrs.Get<int>("y_num_col_dims");
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto y = GetInputNode(op, "Y", ngb_node_map);

  auto x_reshape = x;
  auto y_reshape = y;

  if (x->get_shape().size() > 2) {
    auto x_2d = FlattenTo2d(x->get_shape(), x_num_col_dims);
    x_reshape = NgReshaper(x, x_2d);
  }

  if (y->get_shape().size() > 2) {
    auto y_2d = FlattenTo2d(y->get_shape(), y_num_col_dims);
    y_reshape = NgReshaper(y, y_2d);
  }

  std::shared_ptr<ngraph::Node> out =
      std::make_shared<ngraph::op::Dot>(x_reshape, y_reshape);

  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_shape() != out->get_shape()) {
    out = NgReshaper(out, dummy_out->get_shape());
  }
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildMulGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  int x_num_col_dims = op_attrs.Get<int>("x_num_col_dims");
  int y_num_col_dims = op_attrs.Get<int>("y_num_col_dims");
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto y = GetInputNode(op, "Y", ngb_node_map);
  auto dout = GetInputNode(op, "Out@GRAD", ngb_node_map);

  bool is_dx = HasOutput(op, "X@GRAD") ? true : false;
  bool is_dy = HasOutput(op, "Y@GRAD") ? true : false;

  auto x_shape = x->get_shape();
  auto y_shape = y->get_shape();

  auto x_reshape = x;
  auto y_reshape = y;

  if (x_shape.size() > 2) {
    auto x_2d_shape = FlattenTo2d(x_shape, x_num_col_dims);
    x_reshape = NgReshaper(x, x_2d_shape);
  }

  if (y_shape.size() > 2) {
    auto y_2d_shape = FlattenTo2d(y_shape, y_num_col_dims);
    y_reshape = NgReshaper(y, y_2d_shape);
  }

  auto x_reshape_shape = x_reshape->get_shape();
  std::reverse(x_reshape_shape.begin(), x_reshape_shape.end());
  auto x_transpose = std::make_shared<ngraph::op::Reshape>(
      x_reshape, ngraph::AxisVector{1, 0}, x_reshape_shape);

  auto y_reshape_shape = y_reshape->get_shape();
  std::reverse(y_reshape_shape.begin(), y_reshape_shape.end());
  auto y_transpose = std::make_shared<ngraph::op::Reshape>(
      y_reshape, ngraph::AxisVector{1, 0}, y_reshape_shape);

  if (is_dx) {
    auto dx = std::make_shared<ngraph::op::Dot>(dout, y_transpose);

    if (dx->get_shape() == x_shape) {
      SetOutputNode(op, "X@GRAD", dx, ngb_node_map);
    } else {
      auto dx_reshape = NgReshaper(dx, x_shape);
      SetOutputNode(op, "X@GRAD", dx_reshape, ngb_node_map);
    }
  }

  if (is_dy) {
    auto dy = std::make_shared<ngraph::op::Dot>(x_transpose, dout);

    if (dy->get_shape() == y_shape) {
      SetOutputNode(op, "Y@GRAD", dy, ngb_node_map);
    } else {
      auto dy_reshape = NgReshaper(dy, y_shape);
      SetOutputNode(op, "Y@GRAD", dy_reshape, ngb_node_map);
    }
  }
}

static void BuildPool2dNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto x_shape = x->get_shape();

  std::string pooling_type = op_attrs.Get<std::string>("pooling_type");
  std::vector<int> ksize = op_attrs.Get<std::vector<int>>("ksize");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");

  PADDLE_ENFORCE_EQ(x_shape.size() - 2, ksize.size(),
                    "Handling 2d pooling only");

  if (op_attrs.Get<bool>("global_pooling")) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(x_shape.at(i + 2));
    }
  }

  ngraph::Shape ng_padding_below{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_padding_above{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_ksize_shape{static_cast<size_t>(ksize.at(0)),
                               static_cast<size_t>(ksize.at(1))};
  ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                             static_cast<size_t>(strides.at(1))};

  auto ComputeCeiledOutput = [](int in, int k, int p, int s) {
    return (in - k + 2 * p) / s + 1;
  };

  if (op_attrs.Get<bool>("ceil_mode")) {
    auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
    auto dummpy_shape = dummy_out->get_shape();
    for (size_t i = 0; i < ng_padding_above.size(); ++i) {
      auto desired_size = ComputeCeiledOutput(x_shape[i + 2], ksize[i],
                                              paddings[i], strides[i]);
      if (desired_size != dummpy_shape[i + 2]) {
        ng_padding_above[i] += strides[i];
      }
    }
  }

  bool padding_exclusive = op_attrs.Get<bool>("exclusive");
  if (pooling_type == "max") {
    auto pool2d = std::make_shared<ngraph::op::MaxPool>(
        x, ng_ksize_shape, ng_strides, ng_padding_below, ng_padding_above);
    SetOutputNode(op, "Out", pool2d, ngb_node_map);
  } else if (pooling_type == "avg") {
    auto pool2d = std::make_shared<ngraph::op::AvgPool>(
        x, ng_ksize_shape, ng_strides, ng_padding_below, ng_padding_above,
        !padding_exclusive);
    SetOutputNode(op, "Out", pool2d, ngb_node_map);
  } else {
    PADDLE_THROW("Support max and avg pooling only");
  }
}

static void BuildPool2dGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  auto out = GetInputNode(op, "Out", ngb_node_map);
  auto dout = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto x_shape = x->get_shape();

  std::string pooling_type = op_attrs.Get<std::string>("pooling_type");
  std::vector<int> ksize = op_attrs.Get<std::vector<int>>("ksize");
  std::vector<int> strides = op_attrs.Get<std::vector<int>>("strides");
  std::vector<int> paddings = op_attrs.Get<std::vector<int>>("paddings");

  PADDLE_ENFORCE_EQ(x_shape.size() - 2, ksize.size(),
                    "Handling 2d pooling only");

  if (op_attrs.Get<bool>("global_pooling")) {
    for (size_t i = 0; i < ksize.size(); ++i) {
      paddings[i] = 0;
      ksize[i] = static_cast<int>(x_shape.at(i + 2));
    }
  }

  ngraph::Shape ng_padding_below{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_padding_above{static_cast<size_t>(paddings.at(0)),
                                 static_cast<size_t>(paddings.at(1))};
  ngraph::Shape ng_ksize_shape{static_cast<size_t>(ksize.at(0)),
                               static_cast<size_t>(ksize.at(1))};
  ngraph::Strides ng_strides{static_cast<size_t>(strides.at(0)),
                             static_cast<size_t>(strides.at(1))};

  bool padding_exclusive = op_attrs.Get<bool>("exclusive");
  if (pooling_type == "max") {
    auto pool2d_grad = std::make_shared<ngraph::op::MaxPoolBackprop>(
        x, dout, ng_ksize_shape, ng_strides, ng_padding_below, ng_padding_above,
        std::dynamic_pointer_cast<ngraph::op::MaxPool>(out));
    SetOutputNode(op, "X@GRAD", pool2d_grad, ngb_node_map);
  } else if (pooling_type == "avg") {
    auto pool2d_grad = std::make_shared<ngraph::op::AvgPoolBackprop>(
        x->get_shape(), dout, ng_ksize_shape, ng_strides, ng_padding_below,
        ng_padding_above, !padding_exclusive);
    SetOutputNode(op, "X@GRAD", pool2d_grad, ngb_node_map);
  } else {
    PADDLE_THROW("Support max and avg pooling only");
  }
}
static void BuildReluGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto out = GetInputNode(op, "Out", ngb_node_map);
  auto dout = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto relu_grad = std::make_shared<ngraph::op::ReluBackprop>(out, dout);
  SetOutputNode(op, "X@GRAD", relu_grad, ngb_node_map);
}

static void BuildReshapeNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::shared_ptr<ngraph::Node> input = GetInputNode(op, "X", ngb_node_map);
  // TODO(mozga-intel) The vector of shape is not supported yet, that's
  // asDispensable() operator"
  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_shape() != input->get_shape()) {
    input = NgReshaper(input, dummy_out->get_shape());
  } else {
    auto add_2d = FlattenTo2d(input->get_shape(), 3);
    input = NgReshaper(input, add_2d);
  }

  SetOutputNode(op, "Out", input, ngb_node_map);
}

static void BuildScaleNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  float scale = op_attrs.Get<float>("scale");
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto out = ElementwiseScalar<ngraph::op::Multiply>(scale, x);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildSoftmaxNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto x = GetInputNode(op, "X", ngb_node_map);
  auto x_shape = x->get_shape();
  auto x_max =
      std::make_shared<ngraph::op::Max>(x, ngraph::AxisSet{x_shape.size() - 1});
  auto x_max_bcast = std::make_shared<ngraph::op::Broadcast>(
      x_max, x_shape, ngraph::AxisSet{x_shape.size() - 1});
  auto x_clipped = ElementwiseScalar<ngraph::op::Maximum>(-64., x_max_bcast);
  auto x_shifted = std::make_shared<ngraph::op::Subtract>(x, x_clipped);
  auto softmax = std::make_shared<ngraph::op::Softmax>(
      x_shifted, ngraph::AxisSet{x_shape.size() - 1});
  auto min_clip = ElementwiseScalar<ngraph::op::Maximum>(1.e-30, softmax);
  SetOutputNode(op, "Out", min_clip, ngb_node_map);
}

static void BuildSoftmaxGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto softmax = GetInputNode(op, "Out", ngb_node_map);
  auto softmax_grad = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto softmax_shape = softmax->get_shape();
  auto node_sum = std::make_shared<ngraph::op::Sum>(softmax * softmax_grad,
                                                    ngraph::AxisSet{1});
  auto node_bcast = std::make_shared<ngraph::op::Broadcast>(
      node_sum, softmax_shape, ngraph::AxisSet{1});
  auto result = (softmax_grad - node_bcast) * softmax;
  SetOutputNode(op, "X@GRAD", result, ngb_node_map);
}

static void BuildSumNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  std::vector<std::string> op_inputs;
  for (auto& var_name_item : op->Inputs()) {
    for (auto& var_name : var_name_item.second) {
      op_inputs.push_back(var_name);
      if (ngb_node_map->find(var_name) == ngb_node_map->end()) {
        PADDLE_THROW("op % input varname %s is not found in var_node_map",
                     op->Type(), var_name);
      }
    }
  }
  std::shared_ptr<ngraph::Node>& sum = ngb_node_map->at(op_inputs[0]);
  for (size_t k = 1; k < op_inputs.size(); ++k) {
    std::shared_ptr<ngraph::Node>& nodek = ngb_node_map->at(op_inputs[k]);
    if (nodek->get_element_type() != sum->get_element_type()) {
      nodek =
          std::make_shared<ngraph::op::Convert>(nodek, sum->get_element_type());
    }
    sum = sum + nodek;
  }
  SetOutputNode(op, "Out", sum, ngb_node_map);
}

static void BuildTanhGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto out = GetInputNode(op, "Out", ngb_node_map);
  auto dout = GetInputNode(op, "Out@GRAD", ngb_node_map);
  auto shape = out->get_shape();
  auto node_const =
      ngraph::op::Constant::create(ngraph::element::f32, shape, {1});
  auto result = dout * (node_const - out * out);
  SetOutputNode(op, "X@GRAD", result, ngb_node_map);
}

static void BuildTopKNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto op_attrs = AttrReader(op->Attrs());
  int k = op_attrs.Get<int>("k");
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto top_k = std::make_shared<ngraph::op::TopK>(
      input, input->get_shape().size() - 1, ngraph::element::i64, k);
  std::shared_ptr<ngraph::Node> indices =
      std::make_shared<ngraph::op::GetOutputElement>(top_k, 0);
  std::shared_ptr<ngraph::Node> out =
      std::make_shared<ngraph::op::GetOutputElement>(top_k, 1);
  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_element_type() != out->get_element_type()) {
    out = std::make_shared<ngraph::op::Convert>(out,
                                                dummy_out->get_element_type());
  }
  SetOutputNode(op, "Indices", indices, ngb_node_map);
  SetOutputNode(op, "Out", out, ngb_node_map);
}

static void BuildTransposeNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = AttrReader(op->Attrs());
  std::vector<int> axis = op_attrs.Get<std::vector<int>>("axis");
  auto x_reshape_shape = input->get_shape();
  std::reverse(x_reshape_shape.begin(), x_reshape_shape.end());
  ngraph::AxisVector axis_vec;
  for (auto& v : axis) {
    axis_vec.push_back(v);
  }
  std::shared_ptr<ngraph::Node> x_transpose =
      std::make_shared<ngraph::op::Reshape>(input, axis_vec, x_reshape_shape);
  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_shape() != x_transpose->get_shape()) {
    x_transpose = NgReshaper(x_transpose, dummy_out->get_shape());
  }
  SetOutputNode(op, "Out", x_transpose, ngb_node_map);
}

static void BuildTransposeGradNode(
    const std::shared_ptr<OperatorBase>& op,
    std::shared_ptr<
        std::unordered_map<std::string, std::shared_ptr<ngraph::Node>>>
        ngb_node_map) {
  auto input = GetInputNode(op, "X", ngb_node_map);
  auto op_attrs = AttrReader(op->Attrs());
  std::vector<int> axis = op_attrs.Get<std::vector<int>>("axis");
  auto x_reshape_shape = input->get_shape();
  std::reverse(x_reshape_shape.begin(), x_reshape_shape.end());
  ngraph::AxisVector axis_vec(axis.size());
  for (size_t i = 0; i < axis.size(); ++i) {
    axis_vec[axis.at(i)] = i;
  }
  std::shared_ptr<ngraph::Node> x_transpose =
      std::make_shared<ngraph::op::Reshape>(input, axis_vec, x_reshape_shape);
  auto dummy_out = GetOutputNode(op, "Out", ngb_node_map);
  if (dummy_out && dummy_out->get_shape() != x_transpose->get_shape()) {
    x_transpose = NgReshaper(x_transpose, dummy_out->get_shape());
  }
  SetOutputNode(op, "Out", x_transpose, ngb_node_map);
}

std::map<std::string,
         std::function<void(const std::shared_ptr<OperatorBase>&,
                            std::shared_ptr<std::unordered_map<
                                std::string, std::shared_ptr<ngraph::Node>>>)>>
    NgraphBridge::NG_NODE_MAP = {
        {"accuracy", BuildAccuracyNode},
        {"adam", BuildAdamNode},
        {"batch_norm", BuildBatchNormNode},
        {"batch_norm_grad", BuildBatchNormGradNode},
        {"conv2d", BuildConv2dNode},
        {"conv2d_grad", BuildConv2dGradNode},
        {"concat", BuildConcatNode},
        {"cross_entropy", BuildCrossEntropyNode},
        {"cross_entropy_grad", BuildCrossEntropyGradNode},
        {"depthwise_conv2d", BuildConv2dNode},
        // {"dropout", BuildDropoutNode},
        {"dropout_grad", BuildDropoutGradNode},
        {"elementwise_add", BuildElementwiseBinaryNode<ngraph::op::Add>},
        {"elementwise_add_grad", BuildElementwiseAddGradNode},
        {"elementwise_sub", BuildElementwiseBinaryNode<ngraph::op::Subtract>},
        {"elementwise_mul", BuildElementwiseBinaryNode<ngraph::op::Multiply>},
        {"elementwise_div", BuildElementwiseBinaryNode<ngraph::op::Divide>},
        {"elementwise_max", BuildElementwiseBinaryNode<ngraph::op::Maximum>},
        {"elementwise_min", BuildElementwiseBinaryNode<ngraph::op::Minimum>},
        {"elementwise_pow", BuildElementwiseBinaryNode<ngraph::op::Power>},
        {"equal", BuildElementwiseCompareNode<ngraph::op::Equal>},
        {"not_equal", BuildElementwiseCompareNode<ngraph::op::NotEqual>},
        {"less_than", BuildElementwiseCompareNode<ngraph::op::Less>},
        {"less_equal", BuildElementwiseCompareNode<ngraph::op::LessEq>},
        {"greater_equal", BuildElementwiseCompareNode<ngraph::op::GreaterEq>},
        {"greater_than", BuildElementwiseCompareNode<ngraph::op::Greater>},
        {"fill_constant", BuildFillConstantNode},
        {"fill_zeros_like", BuildZerosLikeNode},
        {"logical_not", BuildUnaryNode<ngraph::op::Not>},
        {"logical_and", BuildBinaryNode<ngraph::op::And>},
        {"logical_or", BuildBinaryNode<ngraph::op::Or>},
        {"lrn", BuildLrnNode},
        {"mean", BuildMeanNode},
        {"mean_grad", BuildMeanGradNode},
        {"momentum", BuildMomentumNode},
        {"mul", BuildMulNode},
        {"mul_grad", BuildMulGradNode},
        {"pool2d", BuildPool2dNode},
        {"pool2d_grad", BuildPool2dGradNode},
        {"relu", BuildUnaryNode<ngraph::op::Relu>},
        {"relu_grad", BuildReluGradNode},
        {"reshape", BuildReshapeNode},
        {"scale", BuildScaleNode},
        {"sigmoid", BuildUnaryNode<ngraph::op::Sigmoid>},
        {"softmax", BuildSoftmaxNode},
        {"softmax_grad", BuildSoftmaxGradNode},
        {"sum", BuildSumNode},
        {"tanh", BuildUnaryNode<ngraph::op::Tanh>},
        {"tanh_grad", BuildTanhGradNode},
        {"top_k", BuildTopKNode},
        {"transpose", BuildTransposeNode},
        {"transpose_grad", BuildTransposeGradNode}};


void NgraphBridge::BuildNgNode(const std::shared_ptr<OperatorBase>& op) {
  auto& op_type = op->Type();
  NG_NODE_MAP[op_type](op, ngb_node_map_);
}

}  // namespace framework
}  // namespace paddle
#endif
