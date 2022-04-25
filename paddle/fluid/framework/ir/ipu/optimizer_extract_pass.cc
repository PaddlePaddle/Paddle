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

#include "paddle/fluid/framework/ir/ipu/optimizer_extract_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

std::set<std::string> ignored_ops = {
    "sign",
    "sum",
    "clip",
    "clip_by_norm",
    "reduce_sum",
    "sqrt",
    "elementwise_max",
    "elementwise_div",
    "elementwise_mul",
    "scale",            // adamax
    "assign",           // adamw
    "squared_l2_norm",  // gradient_clip_norm
    "cast",             // mix-precision support
};

const bool startswith(const std::string& str, const std::string& pre) {
  if (str.rfind(pre, 0) == 0) {
    return true;
  } else {
    return false;
  }
}

const bool is_grad_clip_op(const std::string& op_namescope) {
  return startswith(op_namescope, "/gradient_clip");
}

const bool is_optimizer_op(const std::string& op_namescope) {
  return startswith(op_namescope, "/optimizer");
}

const bool is_regularization_op(const std::string& op_namescope) {
  return startswith(op_namescope, "/regularization");
}

void IpuOptimizerExtractPass::ApplyImpl(ir::Graph* graph) const {
  // optimizer values will be extracted when lowering optimizer in ipu_backend
  OpDesc new_op("popart_optimizer", {}, {}, {});
  new_op.SetAttr("op_role", 0);
  new_op.SetAttr("with_lr_sched", false);

  std::set<std::string> set_ops{};
  // save the weight decay tensor_name and weight_decay_value for Lamb
  std::vector<std::string> weight_decay_vars{};
  std::vector<float> weight_decay_values{};

  // use map store <op_type, op_ptr> ?
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    auto op = node->Op();
    auto op_type = op->Type();
    int op_role_ = BOOST_GET_CONST(
        int, op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    auto op_role = static_cast<OpRole>(op_role_);

    if (op_role == OpRole::kOptimize) {
      // save weight decay value from every lamb optimizer op
      if (op_type == "lamb" && op->HasAttr("weight_decay")) {
        auto weight_decay_value =
            BOOST_GET_CONST(float, op->GetAttr("weight_decay"));
        auto params = op->Output("ParamOut");
        weight_decay_vars.push_back(params[0]);
        weight_decay_values.push_back(weight_decay_value);
      }

      if (set_ops.count(op_type)) {
        continue;
      }

      auto op_namescope =
          BOOST_GET_CONST(std::string, op->GetAttr("op_namescope"));
      bool is_grad_clip = is_grad_clip_op(op_namescope);
      // bool is_optimizer = is_optimizer_op(op_namescope);
      bool is_regularization = is_regularization_op(op_namescope);

      VLOG(10) << "found optimizer releated op: " << op_type;
      // initial larning_rate will be set in ipu_backend
      set_ops.insert(op_type);
      if (op_type == "sgd") {
        auto type = std::string{"sgd"};
        auto lr_var = op->Input("LearningRate").front();
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("weight_decay", 0.0f);
        new_op.SetAttr("momentum", 0.0f);
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "momentum") {
        auto type = std::string{"sgd"};
        // auto LearningRate = op->Input("LearningRate");
        auto use_nesterov = BOOST_GET_CONST(bool, op->GetAttr("use_nesterov"));
        PADDLE_ENFORCE_EQ(use_nesterov, false,
                          platform::errors::Unimplemented(
                              "ipu does not support nesterov mode."));
        auto regularization_method =
            BOOST_GET_CONST(std::string, op->GetAttr("regularization_method"));
        PADDLE_ENFORCE_NE(regularization_method, "l1_decay",
                          platform::errors::Unimplemented(
                              "ipu does not support l1_decay mode."));
        auto multi_precision =
            BOOST_GET_CONST(bool, op->GetAttr("multi_precision"));
        PADDLE_ENFORCE_EQ(multi_precision, false,
                          platform::errors::Unimplemented(
                              "ipu does not support multi_precision mode."));
        auto rescale_grad = BOOST_GET_CONST(float, op->GetAttr("rescale_grad"));
        PADDLE_ENFORCE_EQ(rescale_grad, 1.0,
                          platform::errors::Unimplemented(
                              "ipu does not support rescale_grad mode."));
        auto regularization_coeff =
            BOOST_GET_CONST(float, op->GetAttr("regularization_coeff"));
        auto lr_var = op->Input("LearningRate").front();
        auto momentum = BOOST_GET_CONST(float, op->GetAttr("mu"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("momentum", momentum);
        new_op.SetAttr("weight_decay", regularization_coeff);
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "adam" || op_type == "adamw") {
        auto type = std::string{"adam"};
        auto lr_var = op->Input("LearningRate").front();
        auto beta1 = BOOST_GET_CONST(float, op->GetAttr("beta1"));
        auto beta2 = BOOST_GET_CONST(float, op->GetAttr("beta2"));
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        auto lazy_mode = BOOST_GET_CONST(bool, op->GetAttr("lazy_mode"));
        auto multi_precision =
            BOOST_GET_CONST(bool, op->GetAttr("multi_precision"));
        PADDLE_ENFORCE_EQ(lazy_mode, false,
                          platform::errors::Unimplemented(
                              "ipu does not support lazy_mode mode."));
        PADDLE_ENFORCE_EQ(multi_precision, false,
                          platform::errors::Unimplemented(
                              "ipu does not support multi_precision mode."));
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("weight_decay", 0.0f);
        new_op.SetAttr("beta1", beta1);
        new_op.SetAttr("beta2", beta2);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("adam_mode", std::string{"adam"});
        // adam or adamw
        if (op_type == "adam") {
          new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
          new_op.SetAttr("raw_type", std::string{"adam"});
        } else {
          new_op.SetAttr("weight_decay_mode", std::string{"decay"});
          new_op.SetAttr("raw_type", std::string{"adamw"});
        }
      } else if (op_type == "adamax") {
        auto type = std::string{"adam"};
        auto lr_var = op->Input("LearningRate").front();
        auto beta1 = BOOST_GET_CONST(float, op->GetAttr("beta1"));
        auto beta2 = BOOST_GET_CONST(float, op->GetAttr("beta2"));
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("weight_decay", 0.0f);
        new_op.SetAttr("beta1", beta1);
        new_op.SetAttr("beta2", beta2);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("adam_mode", std::string{"adamax"});
        new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "lamb") {
        // use decay mode
        auto type = std::string{"adam"};
        auto lr_var = op->Input("LearningRate").front();
        auto weight_decay = BOOST_GET_CONST(float, op->GetAttr("weight_decay"));
        auto beta1 = BOOST_GET_CONST(float, op->GetAttr("beta1"));
        auto beta2 = BOOST_GET_CONST(float, op->GetAttr("beta2"));
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("weight_decay", weight_decay);
        new_op.SetAttr("beta1", beta1);
        new_op.SetAttr("beta2", beta2);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("adam_mode", std::string{"lamb"});
        new_op.SetAttr("weight_decay_mode", std::string{"decay"});
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "adadelta") {
        // NO LearningRate
        auto type = std::string{"adaptive"};
        auto rho = BOOST_GET_CONST(float, op->GetAttr("rho"));
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("weight_decay", 0.0f);
        new_op.SetAttr("alpha", rho);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("momentum", 0.0f);
        new_op.SetAttr("adaptive_mode", std::string{"adadelta"});
        new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "adagrad") {
        auto type = std::string{"adaptive"};
        auto lr_var = op->Input("LearningRate").front();
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("lr_var", lr_var);
        new_op.SetAttr("weight_decay", 0.0f);
        // `alpha` use default
        new_op.SetAttr("alpha", 0.99f);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("momentum", 0.0f);
        new_op.SetAttr("adaptive_mode", std::string{"adagrad"});
        new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
        new_op.SetAttr("raw_type", op_type);
      } else if (op_type == "rmsprop") {
        auto type = std::string{"adaptive"};
        auto lr_var = op->Input("LearningRate").front();
        auto epsilon = BOOST_GET_CONST(float, op->GetAttr("epsilon"));
        auto decay = BOOST_GET_CONST(float, op->GetAttr("decay"));
        auto momentum = BOOST_GET_CONST(float, op->GetAttr("momentum"));
        auto centered = BOOST_GET_CONST(bool, op->GetAttr("centered"));
        new_op.SetAttr("type", type);
        new_op.SetAttr("weight_decay", 0.0f);
        new_op.SetAttr("alpha", decay);
        new_op.SetAttr("eps", epsilon);
        new_op.SetAttr("momentum", momentum);
        new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
        if (centered) {
          new_op.SetAttr("adaptive_mode", std::string{"centered_rmsprop"});
          new_op.SetAttr("raw_type", op_type);
        } else {
          new_op.SetAttr("adaptive_mode", std::string{"rmsprop"});
          new_op.SetAttr("raw_type", op_type);
        }
      } else if (is_regularization && op_type == "scale") {
        // set weight_decay for L2Decay
        auto scale = BOOST_GET_CONST(float, op->GetAttr("scale"));
        new_op.SetAttr("weight_decay", scale);
      } else if (is_grad_clip && op_type == "fill_constant") {
        // set clip_norm for ClipGradByGlobalNorm
        auto value = BOOST_GET_CONST(float, op->GetAttr("value"));
        new_op.SetAttr("clip_norm", value);
      } else if (ignored_ops.count(op_type)) {
        VLOG(10) << "Ignore optimizer releated op: " << op_type;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unknown optimizer releated op_type: %s", op_type));
      }
    } else if (op_role == OpRole::kLoss) {
      VLOG(10) << "found loss op type: " << op->Type();
      auto outputs = op->Outputs();
      PADDLE_ENFORCE_EQ(
          outputs.size(), 1,
          platform::errors::InvalidArgument("Can only support one loss key"));
      auto losses = outputs.begin()->second;
      PADDLE_ENFORCE_EQ(
          losses.size(), 1,
          platform::errors::InvalidArgument("Can only support one loss name"));
      auto loss_var = losses.front();
      new_op.SetAttr("loss_var", loss_var);
    } else if (op_role == OpRole::kLRSched) {
      // op_role == OpRole::kLRSched | OpRole::kOptimize
      new_op.SetAttr("with_lr_sched", true);
    }
  }

  // seems with_lr_sched is always true
  new_op.SetAttr("with_lr_sched", true);

  // setup weight decay for Lamb
  new_op.SetAttr("weight_decay_vars", weight_decay_vars);
  new_op.SetAttr("weight_decay_values", weight_decay_values);

  // weight_decay/coeff is "scale" attr of scale_op
  if (set_ops.count("scale") && set_ops.count("sum")) {
    if (set_ops.count("sign")) {
      // L1Decay
      // sign + scale + sum
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported L1Decay regularizer"));
    } else {
      // L2Decay
      // scale + sum
      new_op.SetAttr("weight_decay_mode", std::string{"l2_regularization"});
    }
  } else {
    VLOG(10) << "No weight deacy setting found";
  }

  // setup grad clip
  if (set_ops.count("clip")) {
    // ClipGradByValue
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported ClipGradByValue"));
  } else if (set_ops.count("clip_by_norm")) {
    // ClipGradByNorm
    PADDLE_THROW(platform::errors::Unimplemented("Unsupported ClipGradByNorm"));
  }

  // ClipGradByGlobalNorm
  // use graph pattern match ClipGradByGlobalNorm
  // square + reduce_sum + sum + sqrt + fill_constant
  // + elementwise_max + elementwise_div + elementwise_mul
  // clip_norm from fill_constant`s attr `value` dtype float

  if (new_op.HasAttr("type")) {
    auto new_node = graph->CreateOpNode(&new_op);
    VLOG(10) << "New Optimizer Node:";
    VLOG(10) << DebugString(new_node);
  } else {
    PADDLE_THROW(platform::errors::NotFound(
        "No optimizer found, optimizer must be one of these types: sgd, "
        "momentum, adam, adamw, adamax, lamb, adadelta, adagrad or rmsprop"));
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(optimizer_extract_pass,
              paddle::framework::ir::IpuOptimizerExtractPass);
