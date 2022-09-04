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

#include "paddle/fluid/platform/device/ipu/ipu_compiler.h"

#include <popart/adam.hpp>
#include <popart/adaptive.hpp>
#include <popart/optimizer.hpp>
#include <popart/sgd.hpp>
#include <popart/voiddata.hpp>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_names.h"
#include "paddle/fluid/platform/device/ipu/ipu_strategy.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"
#include "paddle/utils/blank.h"

namespace paddle {
namespace platform {
namespace ipu {

namespace {

struct CustomOpAttrVisitor {
  CustomOpAttrVisitor(std::map<std::string, popart::any>* attr,
                      const std::string& attr_name)
      : attrs_(attr), attr_name_(attr_name) {}

  mutable std::map<std::string, popart::any>* attrs_;
  std::string attr_name_;

  void operator()(int v) const { attrs_->emplace(attr_name_, v); }
  void operator()(float v) const { attrs_->emplace(attr_name_, v); }
  void operator()(double v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::string& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<int>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<float>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<std::string>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(bool v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::vector<bool>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(BlockDesc* desc) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `BlockDesc` type when extracting "
        "custom operator attributes."));
  }
  void operator()(const std::vector<BlockDesc*>& v) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `BlockDesc` type when extracting  "
        "custom operator attributes."));
  }
  void operator()(int64_t v) const { attrs_->emplace(attr_name_, v); }
  void operator()(const std::vector<int64_t>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(const std::vector<double>& v) const {
    attrs_->emplace(attr_name_, v);
  }
  void operator()(paddle::blank) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `paddle::blank` type when extracting "
        "custom operator attributes."));
  }
  void operator()(framework::VarDesc*) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `VarDesc*` type when extracting "
        "custom operator attributes."));
  }
  void operator()(const std::vector<framework::VarDesc*>&) const {
    PADDLE_THROW(platform::errors::Unavailable(
        "Unsupported calling method for `std::vector<framework::VarDesc*>` "
        "type when extracting custom operator attributes."));
  }
};

struct ConstantOpAttrVisitor {
  ConstantOpAttrVisitor(framework::LoDTensor* tensor, VarType::Type dtype)
      : tensor_(tensor), dtype_(dtype) {}

  framework::LoDTensor* tensor_;
  VarType::Type dtype_;

  void operator()(const std::vector<int>& vec) const {
    framework::TensorFromVector<int>(vec, tensor_);
  }
  void operator()(const std::vector<float>& vec) const {
    if (dtype_ == VarType::FP16) {
      std::vector<float16> vec_fp16;
      std::transform(vec.begin(),
                     vec.end(),
                     std::back_inserter(vec_fp16),
                     [](float f) -> float16 { return float16(f); });
      framework::TensorFromVector<float16>(vec_fp16, tensor_);
    } else {
      framework::TensorFromVector<float>(vec, tensor_);
    }
  }
  void operator()(const std::vector<bool>& vec) const {
    framework::TensorFromVector<bool>(vec, tensor_);
  }
  void operator()(const std::vector<int64_t>& vec) const {
    framework::TensorFromVector<int64_t>(vec, tensor_);
  }
  void operator()(const std::vector<double>& vec) const {
    // popart do not support float64 constant
    std::vector<float> vec_fp32;
    std::transform(vec.begin(),
                   vec.end(),
                   std::back_inserter(vec_fp32),
                   [](double f) -> float { return static_cast<float>(f); });
    framework::TensorFromVector<float>(vec_fp32, tensor_);
  }
#define RAISE_ERROR \
  PADDLE_THROW(     \
      platform::errors::InvalidArgument("Constant value must be a vector"))
  void operator()(int v) const { RAISE_ERROR; }
  void operator()(float v) const { RAISE_ERROR; }
  void operator()(double v) const { RAISE_ERROR; }
  void operator()(const std::string& v) const { RAISE_ERROR; }
  void operator()(const std::vector<std::string>& v) const { RAISE_ERROR; }
  void operator()(bool v) const { RAISE_ERROR; }
  void operator()(BlockDesc* desc) const { RAISE_ERROR; }
  void operator()(const std::vector<BlockDesc*>& v) const { RAISE_ERROR; }
  void operator()(int64_t v) const { RAISE_ERROR; }
  void operator()(paddle::blank) const { RAISE_ERROR; }
  void operator()(framework::VarDesc*) const { RAISE_ERROR; }
  void operator()(const std::vector<framework::VarDesc*>&) const {
    RAISE_ERROR;
  }
#undef RAISE_ERROR
};

popart::AdamMode AdamModeFromStr(const std::string& str,
                                 const bool& use_no_bias_optimizer) {
  if (str == "adam") {
    if (!use_no_bias_optimizer)
      return popart::AdamMode::Adam;
    else
      return popart::AdamMode::AdamNoBias;
  } else if (str == "adamax") {
    return popart::AdamMode::AdaMax;
  } else if (str == "lamb") {
    if (!use_no_bias_optimizer)
      return popart::AdamMode::Lamb;
    else
      return popart::AdamMode::LambNoBias;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Uknown AdamMode: %s, AdamMode must be one of these values: adam, "
        "adamax or lamb",
        str));
  }
}

popart::AdaptiveMode AdaptiveModeFromStr(const std::string& str) {
  if (str == "adadelta") {
    return popart::AdaptiveMode::AdaDelta;
  } else if (str == "adagrad") {
    return popart::AdaptiveMode::AdaGrad;
  } else if (str == "rmsprop") {
    return popart::AdaptiveMode::RMSProp;
  } else if (str == "centered_rmsprop") {
    return popart::AdaptiveMode::CenteredRMSProp;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Uknown AdaptiveMode: %s, AdaptiveMode must be one of these values: "
        "adadelta, adagrad, rmsprop or centered_rmsprop",
        str));
  }
}

popart::WeightDecayMode WeightDecayModeFromStr(const std::string& str) {
  if (str == "decay") {
    return popart::WeightDecayMode::Decay;
  } else if (str == "l2_regularization") {
    return popart::WeightDecayMode::L2Regularization;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Uknown WeightDecayMode: %s, WeightDecayMode must be decay or "
        "l2_regularization",
        str));
  }
}

popart::DataType DataTypeFromStr(const std::string& str) {
  if (str == "FLOAT") {
    return popart::DataType::FLOAT;
  } else if (str == "FLOAT16") {
    return popart::DataType::FLOAT16;
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported DataType: %s", str));
  }
}

template <typename T>
T GetAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return PADDLE_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename T>
nonstd::optional<T> GetOptAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return PADDLE_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename TI, typename TO>
TO GetCastSigAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    auto x = PADDLE_GET_CONST(TI, op_desc->GetAttr(attr));
    return static_cast<TO>(x);
  } else {
    return {};
  }
}

// Helper for adding namescope info
struct NameScopeHelper {
  NameScopeHelper(const OpDesc* op, popart::Builder* builder);

  ~NameScopeHelper() {
    if (pushed_) {
      builder_->popNameScope();
    }
  }

  bool pushed_ = false;
  popart::Builder* builder_;
};

NameScopeHelper::NameScopeHelper(const OpDesc* op, popart::Builder* builder)
    : builder_(builder) {
  auto op_namescope = PADDLE_GET_CONST(std::string, op->GetAttr(sOpNamescope));
  if (op_namescope.empty() || op_namescope == "/") {
    return;
  }
  op_namescope.pop_back();
  op_namescope.erase(op_namescope.begin());
  builder->pushNameScope(op_namescope);
  pushed_ = true;
}

}  // namespace

GraphHelper::GraphHelper(const Graph* g) {
  graph = g;
  sorted_ops = framework::ir::TopologySortOperations(*g);
  for (auto* node : g->Nodes()) {
    nodes_id_map[node->id()] = node;
    if (node->IsVar()) {
      vars_name_map[node->Name()] = node;
      sorted_vars_id.push_back(node->id());
    }
  }
  std::sort(sorted_vars_id.begin(), sorted_vars_id.end());
}

Compiler::Compiler() { RegisterOpFunc(); }

Compiler::~Compiler() {
  builder_.reset();
  resources_.reset();
}

void Compiler::Prepare(const Graph* graph) {
  builder_ = popart::Builder::create();
  resources_ = std::make_unique<CompilerResources>();
  graph_helper_ = std::make_unique<GraphHelper>(graph);
  // Set the flag of set_amp_for_all_
  for (auto* node : graph_helper_->sorted_ops) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    if (op_type == "popart_matmul") {
      if (op_desc->HasAttr(sAvailMemAttribute)) {
        set_amp_for_all_ = false;
        return;
      }
    }
  }
}

void Compiler::RegisterOpFunc() {
  VLOG(10) << "enter Compiler::RegisterOpFunc";
#define INT_VEC std::vector<std::int64_t>
#define INT32_VEC std::vector<std::int32_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define INT32 std::int32_t
#define BOOL bool
#define STRING std::string
#define STRING_VEC std::vector<std::string>
#define NONE

#define ARG(Type, Name) , GetAttrAllowNull<Type>(#Name, op_desc)
#define OPT_ARG(Type, Name) , GetOptAttrAllowNull<Type>(#Name, op_desc)
#define SIG_ARG(TI, TO, Name) , GetCastSigAttrAllowNull<TI, TO>(#Name, op_desc)
#define POPART_CONST_ARG(Name) , const PopartConstant& Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant& Name
#define POPART_ATTRIB_VEC_ARG(Name)
#define BODY_ARG(Name) NONE

  name_function_ = {
#define OP_DECL(FuncName, OnnxImpl, Args)                     \
  {#FuncName, [&](OpDesc* op_desc) {                          \
     auto op_type = op_desc->Type();                          \
     VLOG(10) << "build op:" << op_type << " args " << #Args; \
     auto inputs = GetOpInputs(op_desc);                      \
     auto debug_context = BuildDebugContext(op_desc);         \
     auto aiGraphcoreOpset = builder_->aiGraphcoreOpset1();   \
     auto aiOnnxOpset = builder_->aiOnnxOpset11();            \
     NameScopeHelper ns_helper(op_desc, builder_.get());      \
     auto output_ids = OnnxImpl(inputs Args, debug_context);  \
     PostLower(output_ids, op_desc);                          \
   }},  // NOLINT
#include "paddle/fluid/platform/device/ipu/supported_ops_autogen.h"
#include "paddle/fluid/platform/device/ipu/supported_ops_custom.h"
  };

#undef OP_DECL
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef SIG_ARG
#undef OPT_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT32
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT32_VEC
#undef INT_VEC
}

void Compiler::InitInputs(const std::vector<std::string>& feed_list) {
  for (const auto& feed_name : feed_list) {
    auto* node = graph_helper_->vars_name_map[feed_name];
    auto* var_desc = node->Var();
    VLOG(10) << "feed_name= " << var_desc->Name();
    auto data_type = VarType2PopartDType(var_desc->GetDataType());
    popart::TensorInfo input_info{data_type, var_desc->GetShape()};
    VLOG(10) << "popart input_info = " << input_info;
    popart::TensorId tensor_id =
        builder_->addInputTensor(input_info, feed_name);
    VLOG(10) << "popart input tensor id = " << tensor_id;
    resources_->inputs.push_back(tensor_id);
    resources_->tensors.emplace(var_desc->Name(), tensor_id);
  }
}

void Compiler::InitOutputs(const std::vector<std::string>& fetch_list) {
  for (const auto& fetch_name : fetch_list) {
    auto tensor = resources_->tensors.find(fetch_name);
    PADDLE_ENFORCE_NE(
        tensor,
        resources_->tensors.end(),
        platform::errors::NotFound(
            "Output tensor %s is not found, please check the model.",
            fetch_name));
    VLOG(10) << "fetch_name= " << fetch_name;
    VLOG(10) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    resources_->outputs.push_back(tensor->second);
  }
}

void Compiler::LowerConstants(const Scope* scope) {
  auto& kid_scope = scope->NewScope();
  VLOG(10) << "enter Compiler::LowerConstants";
  for (auto* node : graph_helper_->sorted_ops) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    if (op_type == "popart_constant") {
      auto shape =
          PADDLE_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("dims"));
      auto dtype_ = PADDLE_GET_CONST(int, op_desc->GetAttr("dtype"));
      auto dtype = PopartDType2VarType(
          OnnxDType2PopartType(static_cast<ONNXDataType>(dtype_)));
      auto tensor_name = GetOpOutputs(op_desc).front();
      auto* var = kid_scope.Var(tensor_name);
      VLOG(10) << "lowering constant: " << tensor_name;
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      ConstantOpAttrVisitor visitor(tensor, dtype);
      auto value = op_desc->GetAttr("value");
      paddle::visit(visitor, value);
      auto ddim = phi::make_ddim(shape);
      tensor->Resize(ddim);

      auto const_data = std::unique_ptr<popart::ConstVoidData>();
      popart::TensorInfo tensor_info(PhiDType2PopartDType(tensor->dtype()),
                                     shape);
      const_data.reset(new popart::ConstVoidData(tensor->data(), tensor_info));
      NameScopeHelper ns_helper(op_desc, builder_.get());
      popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data);
      PostLower(result, op_desc);
      resources_->tensors.emplace(tensor_name, result);
    }
  }
  VLOG(10) << "leave Compiler::LowerConstants";
}

void Compiler::LowerWeights(const Scope* scope) {
  VLOG(10) << "enter Compiler::LowerWeights";
  // At this step, the graph doesn't contains optimizer related states
  for (auto id : graph_helper_->sorted_vars_id) {
    auto* node = graph_helper_->nodes_id_map[id];
    // Weights are var node and Persistable
    if (node->IsVar() && !node->IsCtrlVar() && node->Var() &&
        node->Var()->Persistable() && node->inputs.empty()) {
      // Weights are Parameter in training mode
      if (ipu_strategy_->is_training && !node->Var()->IsParameter()) {
        continue;
      }
      auto var_name = node->Var()->Name();
      // Some op has same input and output tensor, like batchnorm
      if (resources_->tensors.count(var_name) != 0) {
        VLOG(10) << "found existed one, skip lowering Weight: " << var_name;
        continue;
      }
      VLOG(10) << "lowering weight: " << var_name;
      auto var = scope->FindVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(
          var,
          platform::errors::NotFound("Tensor %s is not found in the scope",
                                     var_name));
      auto tensor = var->Get<framework::LoDTensor>();
      auto dtype = PhiDType2PopartDType(tensor.dtype());
      auto shape = std::vector<int64_t>();
      for (size_t i = 0; i < tensor.dims().size(); ++i) {
        shape.push_back(tensor.dims().at(i));
      }

      popart::TensorInfo tensor_info(dtype, shape);
      popart::ConstVoidData const_data{tensor.data(), tensor_info};
      if (!node->outputs.empty()) {
        auto op_node = node->outputs[0];
        NameScopeHelper ns_helper(op_node->Op(), builder_.get());
        popart::TensorId result =
            builder_->addInitializedInputTensor(const_data, var_name);
        resources_->tensors.emplace(var_name, result);
        resources_->weights.push_back(var_name);
      }
    }
  }
  VLOG(10) << "leave Compiler::LowerWeights";
}

void Compiler::LowerBody() {
  VLOG(10) << "enter Compiler::LowerBody";
  for (auto* node : graph_helper_->sorted_ops) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    VLOG(10) << "lowering op: " << op_type;

    if (op_type == "popart_constant") {
      // pass
    } else if (op_type == "popart_optimizer") {
      // pass
    } else if (op_type == "popart_checkpointoutput") {
      auto inputs = GetOpInputs(op_desc);
      NameScopeHelper ns_helper(op_desc, builder_.get());
      auto output_ids = builder_->checkpointOutput(inputs);
      PostLower(output_ids, op_desc);
    } else if (op_type == "popart_custom_op") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto attributes = std::map<std::string, popart::any>{};
      for (auto& attr : op_desc->GetAttrMap()) {
        CustomOpAttrVisitor visitor(&attributes, attr.first);
        paddle::visit(visitor, attr.second);
      }
      auto __op_type =
          PADDLE_GET_CONST(std::string, op_desc->GetAttr("__op_type"));
      VLOG(10) << "Build graph from custom op: " << __op_type;
      auto it = custom_ops_.find(__op_type);
      NameScopeHelper ns_helper(op_desc, builder_.get());
      auto output_ids = builder_->customOp(it->second.popart_op,
                                           it->second.popart_op.version,
                                           inputs,
                                           outputs.size(),
                                           attributes,
                                           debug_context);
      PostLower(output_ids, op_desc);
    } else if (op_type == "popart_printtensor") {
      auto inputs = GetOpInputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto print_gradient =
          PADDLE_GET_CONST(int64_t, op_desc->GetAttr("print_gradient"));
      auto title = PADDLE_GET_CONST(std::string, op_desc->GetAttr("title"));
      NameScopeHelper ns_helper(op_desc, builder_.get());
      auto output_ids = builder_->aiGraphcoreOpset1().printtensor(
          inputs, print_gradient, debug_context, title);
      PostLower(output_ids, op_desc);
    } else {
      auto itr = name_function_.find(op_type);
      if (itr != name_function_.end()) {
        itr->second(node->Op());
      } else {
        PADDLE_THROW(platform::errors::NotFound(
            "%s is not registered, please check for unsupported operators for "
            "running on IPU",
            op_type));
      }
    }
  }
  VLOG(10) << "leave Compiler::LowerBody";
}

void Compiler::LowerOptimizer(const Scope* scope) {
  for (auto* node : graph_helper_->sorted_ops) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    if (op_type == "popart_optimizer") {
      auto raw_type =
          PADDLE_GET_CONST(std::string, op_desc->GetAttr("raw_type"));
      resources_->optimizer_type = raw_type;
      resources_->with_lr_sched =
          PADDLE_GET_CONST(bool, op_desc->GetAttr("with_lr_sched"));
      if (ipu_strategy_->is_dynamic) {
        // loss_var in dy2static is set by identity_loss. And lr is
        // passed by ipu_strategy.
        resources_->lr = ipu_strategy_->lr;
      } else {
        auto loss_var =
            PADDLE_GET_CONST(std::string, op_desc->GetAttr("loss_var"));
        resources_->loss_var = resources_->tensors[loss_var];
        if (op_desc->HasAttr("lr_var")) {
          auto lr_var =
              PADDLE_GET_CONST(std::string, op_desc->GetAttr("lr_var"));
          resources_->lr_var = lr_var;
          resources_->lr = GetSingleVarFromScope<float>(scope, lr_var);
        } else {
          // adadelta has no lr
          resources_->lr = 0.01f;
          resources_->with_lr_sched = false;
        }
      }
      VLOG(10) << "Set initial lr: " << resources_->lr;

      // Get the type of optimizer
      auto type = PADDLE_GET_CONST(std::string, op_desc->GetAttr("type"));
      // Set weight decay by tensor names for Lamb
      auto weight_decay_vars = PADDLE_GET_CONST(
          std::vector<std::string>, op_desc->GetAttr("weight_decay_vars"));
      auto weight_decay_values = PADDLE_GET_CONST(
          std::vector<float>, op_desc->GetAttr("weight_decay_values"));
      // Get the maximum permissible value for gradient clipping
      std::vector<popart::ClipNormSettings> clip_norm_settings = {};
      if (op_desc->HasAttr("clip_norm")) {
        auto clip_norm = PADDLE_GET_CONST(float, op_desc->GetAttr("clip_norm"));
        clip_norm_settings.push_back(
            popart::ClipNormSettings::clipAllWeights(clip_norm));
        VLOG(10) << "Set the global gradient clipping with the maximum "
                    "permissible value: "
                 << clip_norm;
      }

      // Values from ipu_strategy
      auto loss_scaling = ipu_strategy_->loss_scaling;
      auto accl1_type = DataTypeFromStr(ipu_strategy_->accl1_type);
      auto accl2_type = DataTypeFromStr(ipu_strategy_->accl2_type);
      auto accl3_type = DataTypeFromStr(ipu_strategy_->accl3_type);

      if (type == "sgd") {
        auto weight_decay =
            PADDLE_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto momentum = PADDLE_GET_CONST(float, op_desc->GetAttr("momentum"));
        resources_->optimizer_fn = [=](float lr) {
          return std::make_unique<popart::SGD>(
              popart::OptimizerValue(lr, false),
              popart::OptimizerValue(weight_decay, false),
              popart::OptimizerValue(momentum, true),
              popart::SGD::getUnsetDampening(),
              popart::SGD::getUnsetVelocityScaling(),
              popart::OptimizerValue(loss_scaling, true),
              clip_norm_settings);
        };
        resources_->eval_optimizer = std::make_unique<popart::SGD>(
            popart::OptimizerValue(0.0, false),
            popart::OptimizerValue(0.0, false),
            popart::OptimizerValue(0.0, true),
            popart::SGD::getUnsetDampening(),
            popart::SGD::getUnsetVelocityScaling(),
            popart::OptimizerValue(loss_scaling, true),
            clip_norm_settings);
      } else if (type == "adam") {
        auto weight_decay =
            PADDLE_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto beta1 = PADDLE_GET_CONST(float, op_desc->GetAttr("beta1"));
        auto beta2 = PADDLE_GET_CONST(float, op_desc->GetAttr("beta2"));
        auto eps = PADDLE_GET_CONST(float, op_desc->GetAttr("eps"));
        auto mwn = ipu_strategy_->max_weight_norm;
        VLOG(10) << "set max_weight_norm: " << mwn;
        auto adam_mode_ =
            PADDLE_GET_CONST(std::string, op_desc->GetAttr("adam_mode"));
        auto adam_mode =
            AdamModeFromStr(adam_mode_, ipu_strategy_->use_no_bias_optimizer);
        auto weight_decay_mode_ = ipu_strategy_->weight_decay_mode;
        auto scaled_optimizer_state_ = ipu_strategy_->scaled_optimizer_state;
        if (weight_decay_mode_.empty()) {
          weight_decay_mode_ = PADDLE_GET_CONST(
              std::string, op_desc->GetAttr("weight_decay_mode"));
        }
        auto weight_decay_mode = WeightDecayModeFromStr(weight_decay_mode_);
        resources_->optimizer_fn = [=](float lr) {
          if (adam_mode == popart::AdamMode::Lamb ||
              adam_mode == popart::AdamMode::LambNoBias) {
            const std::map<std::string, std::pair<float, bool>>
                optimizer_value = {{"defaultLearningRate", {lr, false}},
                                   {"defaultBeta1", {beta1, false}},
                                   {"defaultBeta2", {beta2, false}},
                                   {"defaultEps", {eps, true}},
                                   {"lossScaling", {loss_scaling, true}},
                                   {"defaultMaxWeightNorm", {mwn, true}}};
            auto optimizer_instance =
                std::make_unique<popart::Adam>(optimizer_value,
                                               adam_mode,
                                               weight_decay_mode,
                                               popart::DataType::UNDEFINED,
                                               accl1_type,
                                               accl2_type,
                                               clip_norm_settings,
                                               scaled_optimizer_state_);
            for (int i = 0; i < weight_decay_vars.size(); i++) {
              optimizer_instance->insertSpecific(
                  weight_decay_vars[i],
                  {{"weightDecay", {weight_decay_values[i], false}}});
              VLOG(10) << "Set Tensor " << weight_decay_vars[i]
                       << " weight decay as " << weight_decay_values[i];
            }
            return optimizer_instance;
          } else {
            return std::make_unique<popart::Adam>(
                popart::OptimizerValue(lr, false),
                popart::OptimizerValue(weight_decay, false),
                popart::OptimizerValue(beta1, false),
                popart::OptimizerValue(beta2, false),
                popart::OptimizerValue(eps, true),
                popart::OptimizerValue(loss_scaling, true),
                popart::OptimizerValue(mwn, true),
                adam_mode,
                weight_decay_mode,
                popart::DataType::UNDEFINED,
                accl1_type,
                accl2_type,
                clip_norm_settings,
                scaled_optimizer_state_);
          }
        };
        if (adam_mode == popart::AdamMode::Lamb) {
          const std::map<std::string, std::pair<float, bool>> optimizer_value =
              {{"defaultLearningRate", {0.0, false}},
               {"defaultBeta1", {beta1, false}},
               {"defaultBeta2", {beta2, false}},
               {"defaultEps", {eps, true}},
               {"lossScaling", {loss_scaling, true}},
               {"defaultMaxWeightNorm", {mwn, true}}};
          auto eval_optimizer =
              std::make_unique<popart::Adam>(optimizer_value,
                                             adam_mode,
                                             weight_decay_mode,
                                             popart::DataType::UNDEFINED,
                                             popart::DataType::FLOAT,
                                             popart::DataType::FLOAT,
                                             clip_norm_settings,
                                             scaled_optimizer_state_);
          for (int i = 0; i < weight_decay_vars.size(); i++) {
            eval_optimizer->insertSpecific(weight_decay_vars[i],
                                           {{"weightDecay", {0.0, false}}});
          }
          resources_->eval_optimizer = std::move(eval_optimizer);
        } else if (adam_mode == popart::AdamMode::LambNoBias) {
          const std::map<std::string, std::pair<float, bool>> optimizer_value =
              {{"defaultLearningRate", {0.0, false}},
               {"defaultBeta1", {1.0, false}},
               {"defaultBeta2", {1.0, false}},
               {"defaultEps", {eps, true}},
               {"lossScaling", {loss_scaling, true}},
               {"defaultMaxWeightNorm", {mwn, true}}};
          auto eval_optimizer =
              std::make_unique<popart::Adam>(optimizer_value,
                                             adam_mode,
                                             weight_decay_mode,
                                             popart::DataType::UNDEFINED,
                                             popart::DataType::FLOAT,
                                             popart::DataType::FLOAT,
                                             clip_norm_settings,
                                             scaled_optimizer_state_);
          for (int i = 0; i < weight_decay_vars.size(); i++) {
            eval_optimizer->insertSpecific(weight_decay_vars[i],
                                           {{"weightDecay", {0.0, false}}});
          }
          resources_->eval_optimizer = std::move(eval_optimizer);
        } else {
          resources_->eval_optimizer = std::make_unique<popart::Adam>(
              popart::OptimizerValue(0.0, false),
              popart::OptimizerValue(0.0, false),
              popart::OptimizerValue(beta1, false),
              popart::OptimizerValue(beta2, false),
              popart::OptimizerValue(eps, true),
              popart::OptimizerValue(loss_scaling, true),
              popart::OptimizerValue(mwn, true),
              adam_mode,
              weight_decay_mode,
              popart::DataType::UNDEFINED,
              popart::DataType::FLOAT,
              popart::DataType::FLOAT,
              clip_norm_settings,
              scaled_optimizer_state_);
        }
      } else if (type == "adaptive") {
        auto alpha = PADDLE_GET_CONST(float, op_desc->GetAttr("alpha"));
        auto momentum = PADDLE_GET_CONST(float, op_desc->GetAttr("momentum"));
        auto eps = PADDLE_GET_CONST(float, op_desc->GetAttr("eps"));
        auto weight_decay =
            PADDLE_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto adaptive_mode_ =
            PADDLE_GET_CONST(std::string, op_desc->GetAttr("adaptive_mode"));
        auto adaptive_mode = AdaptiveModeFromStr(adaptive_mode_);
        auto weight_decay_mode_ = ipu_strategy_->weight_decay_mode;
        if (weight_decay_mode_.empty()) {
          weight_decay_mode_ = PADDLE_GET_CONST(
              std::string, op_desc->GetAttr("weight_decay_mode"));
        }
        auto weight_decay_mode = WeightDecayModeFromStr(weight_decay_mode_);
        resources_->optimizer_fn = [=](float lr) {
          return std::make_unique<popart::Adaptive>(
              popart::OptimizerValue(lr, false),
              popart::OptimizerValue(weight_decay, false),
              popart::OptimizerValue(alpha, true),
              popart::OptimizerValue(momentum, true),
              popart::OptimizerValue(eps, true),
              popart::OptimizerValue(loss_scaling, true),
              adaptive_mode,
              weight_decay_mode,
              popart::DataType::UNDEFINED,
              accl1_type,
              accl2_type,
              accl3_type);
        };
        resources_->eval_optimizer = std::make_unique<popart::Adaptive>(
            popart::OptimizerValue(0.0, false),
            popart::OptimizerValue(0.0, false),
            popart::OptimizerValue(alpha, true),
            popart::OptimizerValue(momentum, true),
            popart::OptimizerValue(eps, true),
            popart::OptimizerValue(loss_scaling, true),
            adaptive_mode,
            weight_decay_mode,
            popart::DataType::UNDEFINED,
            popart::DataType::FLOAT,
            popart::DataType::FLOAT,
            popart::DataType::UNDEFINED);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "optimizer %s is not implemented", type));
      }
    } else if (op_type == "popart_identity_loss") {
      auto outputs = op_desc->Outputs();
      PADDLE_ENFORCE_EQ(
          outputs.size(),
          1,
          platform::errors::InvalidArgument("Can only support one loss key"));
      auto losses = outputs.begin()->second;
      PADDLE_ENFORCE_EQ(
          losses.size(),
          1,
          platform::errors::InvalidArgument("Can only support one loss name"));
      auto loss_var = losses.front();
      resources_->loss_var = resources_->tensors[loss_var];
    }
  }
}

void Compiler::PostLower(const std::vector<std::string>& tensor_ids,
                         const OpDesc* op_desc) {
  // Set pipline
  // Due to the limitation of popart, if an op has multiple outputs,
  // pipline settings needs to be set at the same time
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());
  if (op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = PADDLE_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_ids_set, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = PADDLE_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_ids_set, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << " = " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  // Record output tensors
  auto pd_outs = GetOpOutputs(op_desc);
  PADDLE_ENFORCE_EQ(
      pd_outs.size(),
      tensor_ids.size(),
      platform::errors::Fatal("paddle and popart op have different outputs"));
  for (int i = 0; i < tensor_ids.size(); ++i) {
    resources_->tensors.emplace(pd_outs[i], tensor_ids[i]);
  }
  for (auto& tensor_id : tensor_ids) {
    PostLower(tensor_id, op_desc, true);
  }
}

void Compiler::PostLower(const std::string& tensor_id, const OpDesc* op_desc) {
  // Record output tensor
  auto pd_outs = GetOpOutputs(op_desc);
  PADDLE_ENFORCE_EQ(
      pd_outs.size(),
      1,
      platform::errors::Fatal("paddle and popart op have different outputs"));
  resources_->tensors.emplace(pd_outs[0], tensor_id);
  PostLower(tensor_id, op_desc, false);
}

void Compiler::PostLower(const std::string& tensor_id,
                         const OpDesc* op_desc,
                         bool skip_pipline) {
  // Set pipline
  if (!skip_pipline && op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = PADDLE_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_id, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = PADDLE_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_id, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << " = " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  // Set amp
  if (op_desc->Type() == "popart_matmul") {
    if (set_amp_for_all_) {
      auto amp = ipu_strategy_->available_memory_proportion;
      if (amp < 0.0f || amp > 1.0) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "AvailableMemoryProportion %f is invalid, which should be in "
            "range [0.0, 1.0]",
            amp));
      }
      if (amp > 0.0f) {
        builder_->setAvailableMemoryProportion(tensor_id, amp);
      }
    } else {
      if (op_desc->HasAttr(sAvailMemAttribute)) {
        auto amp =
            PADDLE_GET_CONST(float, op_desc->GetAttr(sAvailMemAttribute));
        if (amp < 0.0f || amp > 1.0) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "AvailableMemoryProportion %f is invalid, which should be in "
              "range [0.0, 1.0]",
              amp));
        }
        if (amp > 0.0f) {
          builder_->setAvailableMemoryProportion(tensor_id, amp);
          VLOG(10) << "set available_memory_proportion for tensor: "
                   << tensor_id << " as " << amp;
        }
      }
    }
    // Set serialize matmul
    if (op_desc->HasAttr(sMatmulSerializeFactor)) {
      auto factor =
          PADDLE_GET_CONST(int, op_desc->GetAttr(sMatmulSerializeFactor));
      std::string mode = "output_channels";
      if (op_desc->HasAttr(sMatmulSerializeMode)) {
        mode = PADDLE_GET_CONST(std::string,
                                op_desc->GetAttr(sMatmulSerializeMode));
      }
      builder_->setSerializeMatMul({tensor_id}, mode, factor, true);
    }
  }
}

void Compiler::SetCustomOps(
    const std::vector<IpuCustomOpIdentifier>& custom_ops) {
  for (auto x : custom_ops) {
    custom_ops_.emplace(x.paddle_op, x);
  }
}

std::string Compiler::GetFP16ModelProto() {
  popart::GraphTransformer graph_transformer(builder_->getModelProto());
  graph_transformer.convertFloatsToHalfs();
  return graph_transformer.getModelProto();
}

std::string Compiler::GetModelProto() { return builder_->getModelProto(); }

void Compiler::SaveModelProto(const std::string& path) {
  builder_->saveModelProto(path);
}

void Compiler::SaveModelProtoNoCheck(const std::string& path) {
  auto proto = GetModelProto();
  std::ofstream onnxfile(path, std::ios_base::binary);
  onnxfile.write(proto.data(), proto.size());
  onnxfile.close();
}

std::vector<std::string> Compiler::GetOpInputs(const OpDesc* op) {
  auto ins = op->Input("__inputs__");
  std::vector<std::string> inputs;
  for (const auto& in : ins) {
    if (resources_->tensors.find(in) != resources_->tensors.end()) {
      inputs.push_back(resources_->tensors[in]);
    } else {
      inputs.push_back(in);
    }
  }
  return inputs;
}

const std::vector<std::string>& Compiler::GetOpOutputs(const OpDesc* op) {
  return op->Output("__outputs__");
}

popart::DebugContext Compiler::BuildDebugContext(const OpDesc* op) {
  auto op_identify_id =
      PADDLE_GET_CONST(std::string, op->GetAttr(sOpIdentifyIdAttr));
  VLOG(10) << "op_identify_id of op: " << op->Type() << " is "
           << op_identify_id;
  return popart::DebugContext(op_identify_id);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
