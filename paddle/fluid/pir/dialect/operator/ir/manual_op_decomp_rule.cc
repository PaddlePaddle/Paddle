// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/primitive/base/lazy_tensor.h"
#include "paddle/fluid/primitive/decomp_rule/decomp_rule/composite.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_base.h"

// TODO(chenzhuo)
// this file will be generated in pd_op_decomp.cc

namespace paddle::dialect {
using IntArray = paddle::experimental::IntArray;

std::vector<std::vector<pir::Value>> BatchNormOp::Decomp(pir::Operation* op) {
  VLOG(4) << "Decomp call batch_norm's decomp interface begin";
  BatchNormOp op_obj = op->dyn_cast<BatchNormOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of batch_norm";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor mean(std::make_shared<primitive::LazyTensor>(op_obj.mean()));
  Tensor variance(std::make_shared<primitive::LazyTensor>(op_obj.variance()));
  paddle::optional<Tensor> scale;
  if (!IsEmptyValue(op_obj.scale())) {
    scale = paddle::make_optional<Tensor>(
        Tensor(std::make_shared<primitive::LazyTensor>(op_obj.scale())));
  }
  paddle::optional<Tensor> bias;
  if (!IsEmptyValue(op_obj.bias())) {
    bias = paddle::make_optional<Tensor>(
        Tensor(std::make_shared<primitive::LazyTensor>(op_obj.bias())));
  }

  VLOG(6) << "Decomp prepare attributes of batch_norm";
  bool is_test = op->attribute("is_test").dyn_cast<pir::BoolAttribute>().data();
  float momentum =
      op->attribute("momentum").dyn_cast<pir::FloatAttribute>().data();
  float epsilon =
      op->attribute("epsilon").dyn_cast<pir::FloatAttribute>().data();
  const std::string& data_layout =
      op->attribute("data_format").dyn_cast<pir::StrAttribute>().AsString();
  bool use_global_stats =
      op->attribute("use_global_stats").dyn_cast<pir::BoolAttribute>().data();
  bool trainable_statistics = op->attribute("trainable_statistics")
                                  .dyn_cast<pir::BoolAttribute>()
                                  .data();

  VLOG(6) << "Decomp call batch_norm's forward composite rule prepare";

  auto org_res = op->results();
  std::vector<std::vector<pir::Value>> res(org_res.size());

  VLOG(6) << "Decomp call batch_norm's forward composite rule begin";

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> op_res =
      paddle::primitive::details::batch_norm_decomp<primitive::LazyTensor>(
          x,
          mean,
          variance,
          scale,
          bias,
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics);

  VLOG(6) << "Decomp call batch_norm's forward composite rule end";

  res[0].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<0>(op_res).impl())
                       ->value());
  res[1].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<1>(op_res).impl())
                       ->value());
  res[2].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<2>(op_res).impl())
                       ->value());
  if (std::get<3>(op_res).initialized()) {
    res[3].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                         std::get<3>(op_res).impl())
                         ->value());
  } else {
    pir::Value saved_mean;
    res[3].push_back(saved_mean);
  }
  if (std::get<4>(op_res).initialized()) {
    res[4].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                         std::get<4>(op_res).impl())
                         ->value());
  } else {
    pir::Value saved_var;
    res[4].push_back(saved_var);
  }
  pir::Value reserve_space;
  res[5].push_back(reserve_space);

  VLOG(4) << "Decomp call batch_norm's decomp interface end";
  return res;
}

std::vector<std::vector<pir::Value>> BatchNorm_Op::Decomp(pir::Operation* op) {
  VLOG(4) << "Decomp call batch_norm_'s decomp interface begin";
  BatchNorm_Op op_obj = op->dyn_cast<BatchNorm_Op>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of batch_norm_";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor mean(std::make_shared<primitive::LazyTensor>(op_obj.mean()));
  Tensor variance(std::make_shared<primitive::LazyTensor>(op_obj.variance()));
  paddle::optional<Tensor> scale;
  if (!IsEmptyValue(op_obj.scale())) {
    scale = paddle::make_optional<Tensor>(
        Tensor(std::make_shared<primitive::LazyTensor>(op_obj.scale())));
  }
  paddle::optional<Tensor> bias;
  if (!IsEmptyValue(op_obj.bias())) {
    bias = paddle::make_optional<Tensor>(
        Tensor(std::make_shared<primitive::LazyTensor>(op_obj.bias())));
  }

  VLOG(6) << "Decomp prepare attributes of batch_norm_";
  bool is_test = op->attribute("is_test").dyn_cast<pir::BoolAttribute>().data();
  float momentum =
      op->attribute("momentum").dyn_cast<pir::FloatAttribute>().data();
  float epsilon =
      op->attribute("epsilon").dyn_cast<pir::FloatAttribute>().data();
  const std::string& data_layout =
      op->attribute("data_format").dyn_cast<pir::StrAttribute>().AsString();
  bool use_global_stats =
      op->attribute("use_global_stats").dyn_cast<pir::BoolAttribute>().data();
  bool trainable_statistics = op->attribute("trainable_statistics")
                                  .dyn_cast<pir::BoolAttribute>()
                                  .data();

  VLOG(6) << "Decomp call batch_norm_'s forward composite rule prepare";

  auto org_res = op->results();
  std::vector<std::vector<pir::Value>> res(org_res.size());

  VLOG(6) << "Decomp call batch_norm_'s forward composite rule begin";

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> op_res =
      paddle::primitive::details::batch_norm_decomp<primitive::LazyTensor>(
          x,
          mean,
          variance,
          scale,
          bias,
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics);

  VLOG(6) << "Decomp call batch_norm_'s forward composite rule end";

  res[0].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<0>(op_res).impl())
                       ->value());
  res[1].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<1>(op_res).impl())
                       ->value());
  res[2].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                       std::get<2>(op_res).impl())
                       ->value());
  if (std::get<3>(op_res).initialized()) {
    res[3].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                         std::get<3>(op_res).impl())
                         ->value());
  } else {
    pir::Value saved_mean;
    res[3].push_back(saved_mean);
  }
  if (std::get<4>(op_res).initialized()) {
    res[4].push_back(std::static_pointer_cast<primitive::LazyTensor>(
                         std::get<4>(op_res).impl())
                         ->value());
  } else {
    pir::Value saved_var;
    res[4].push_back(saved_var);
  }

  pir::Value reserve_space;
  res[5].push_back(reserve_space);

  VLOG(4) << "Decomp call batch_norm_'s decomp interface end";
  return res;
}

std::vector<std::vector<pir::Value>> ClipOp::Decomp(pir::Operation* op) {
  VLOG(4) << "Decomp call clip's decomp interface begin";

  ClipOp op_obj = op->dyn_cast<ClipOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of clip";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor min(std::make_shared<primitive::LazyTensor>(op_obj.min()));
  Tensor max(std::make_shared<primitive::LazyTensor>(op_obj.max()));

  VLOG(6) << "Decomp call clip's forward composite rule prepare";

  auto org_res = op->results();
  std::vector<std::vector<pir::Value>> res(org_res.size());

  VLOG(6) << "Decomp call clip's forward composite rule begin";
  Tensor op_res =
      paddle::primitive::details::clip_decomp<primitive::LazyTensor>(
          x, min, max);
  VLOG(6) << "Decomp call clip's forward composite rule end";
  res[0].push_back(
      std::static_pointer_cast<primitive::LazyTensor>(op_res.impl())->value());

  VLOG(4) << "Decomp call clip's decomp interface end";
  return res;
}

std::vector<std::vector<pir::Value>> OneHotOp::Decomp(pir::Operation* op) {
  VLOG(4) << "Decomp call one_hot's decomp interface begin";

  OneHotOp op_obj = op->dyn_cast<OneHotOp>();
  (void)op_obj;

  FLAGS_tensor_operants_mode = "static";

  VLOG(6) << "Decomp Prepare inputs of one_hot";

  Tensor x(std::make_shared<primitive::LazyTensor>(op_obj.x()));
  Tensor num_classes(
      std::make_shared<primitive::LazyTensor>(op_obj.num_classes()));

  VLOG(6) << "Decomp prepare attributes of one_hot";

  VLOG(6) << "Decomp call one_hot's forward composite rule prepare";

  auto org_res = op->results();
  std::vector<std::vector<pir::Value>> res(org_res.size());

  VLOG(6) << "Decomp call one_hot's forward composite rule begin";
  Tensor op_res =
      paddle::primitive::details::one_hot_decomp<primitive::LazyTensor>(
          x, num_classes);
  VLOG(6) << "Decomp call one_hot's forward composite rule end";

  res[0].push_back(
      std::static_pointer_cast<primitive::LazyTensor>(op_res.impl())->value());

  VLOG(4) << "Decomp call one_hot's decomp interface end";
  return res;
}

}  // namespace paddle::dialect
