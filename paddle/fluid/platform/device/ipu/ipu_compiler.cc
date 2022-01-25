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
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

popart::AdamMode AdamModeFromStr(const std::string& str) {
  if (str == "adam") {
    return popart::AdamMode::Adam;
  } else if (str == "adamax") {
    return popart::AdamMode::AdaMax;
  } else if (str == "lamb") {
    return popart::AdamMode::Lamb;
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

template <typename T>
T GetAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename T>
nonstd::optional<T> GetOptAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename TI, typename TO>
TO GetCastSigAttrAllowNull(std::string attr, OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    auto x = BOOST_GET_CONST(TI, op_desc->GetAttr(attr));
    return static_cast<TO>(x);
  } else {
    return {};
  }
}

Compiler::Compiler() { RegisterOpFunc(); }

Compiler::~Compiler() {
  builder_.reset();
  resources_.reset();
}

void Compiler::Prepare() {
  builder_ = popart::Builder::create();
  resources_ = std::make_unique<CompilerResources>();
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
#define STRING_VEC std::vector<std::string*>
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
     auto output_names = GetOpOutputs(op_desc);               \
     auto debug_context = BuildDebugContext(op_desc);         \
     auto aiGraphcoreOpset = builder_->aiGraphcoreOpset1();   \
     auto aiOnnxOpset = builder_->aiOnnxOpset11();            \
     auto output_ids = OnnxImpl(inputs Args, debug_context);  \
     SetIpuIndexStage(output_ids, op_desc);                   \
     SetAMPAttributes(output_ids, op_desc);                   \
     SetSerializeAttributes(output_ids, op_desc);             \
     InsertTensors(output_names, output_ids);                 \
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

void Compiler::LowerBody(const Graph* graph) {
  VLOG(10) << "enter Compiler::LowerBody";
  auto nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* node : nodes) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    VLOG(10) << "lowering op: " << op_type;

    if (op_type == "popart_constant") {
      // pass
    } else if (op_type == "popart_optimizer") {
      // pass
    } else if (op_type == "popart_checkpointoutput") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto output_ids = builder_->checkpointOutput(inputs);
      InsertTensors(outputs, output_ids);
    } else if (op_type == "popart_custom_op") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto attributes = std::map<std::string, popart::any>{};
      for (auto& attr : op_desc->GetAttrMap()) {
        CustomOpAttrVisitor visitor(&attributes, attr.first);
        boost::apply_visitor(visitor, attr.second);
      }
      auto __op_type =
          BOOST_GET_CONST(std::string, op_desc->GetAttr("__op_type"));
      VLOG(10) << "Build graph from custom op: " << __op_type;
      auto it = custom_ops_.find(__op_type);
      auto output_ids =
          builder_->customOp(it->second.popart_op, it->second.popart_op.version,
                             inputs, outputs.size(), attributes, debug_context);
      SetIpuIndexStage(output_ids, op_desc);
      InsertTensors(outputs, output_ids);
    } else if (op_type == "popart_printtensor") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto debug_context = BuildDebugContext(op_desc);
      auto print_gradient =
          BOOST_GET_CONST(int64_t, op_desc->GetAttr("print_gradient"));
      auto title = BOOST_GET_CONST(std::string, op_desc->GetAttr("title"));
      auto output_ids = builder_->aiGraphcoreOpset1().printtensor(
          inputs, print_gradient, debug_context, title);
      SetIpuIndexStage(output_ids, op_desc);
      InsertTensors(outputs, output_ids);
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

void Compiler::InitInputs(Graph* graph,
                          const std::vector<std::string>& feed_list) {
  for (const auto& feed_name : feed_list) {
    feed_list_.push_back(feed_name);
    for (const Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        auto* var_desc = n->Var();
        if (feed_name == var_desc->Name()) {
          VLOG(10) << "feed_name= " << var_desc->Name();
          auto data_type = VarType2PopartType(var_desc->GetDataType());
          popart::TensorInfo input_info{data_type, var_desc->GetShape()};
          VLOG(10) << "popart input_info = " << input_info;
          popart::TensorId tensor_id =
              builder_->addInputTensor(input_info, feed_name);
          VLOG(10) << "popart input tensor id = " << tensor_id;
          resources_->inputs.push_back(tensor_id);
          resources_->tensors.emplace(var_desc->Name(), tensor_id);
        }
      }
    }
  }
}

void Compiler::InitOutputs(const std::vector<std::string>& fetch_list) {
  for (const auto& fetch_name : fetch_list) {
    fetch_list_.push_back(fetch_name);
    auto tensor = resources_->tensors.find(fetch_name);
    PADDLE_ENFORCE_NE(
        tensor, resources_->tensors.end(),
        platform::errors::NotFound(
            "Output tensor %s is not found, please check the model.",
            fetch_name));
    VLOG(10) << "fetch_name= " << fetch_name;
    VLOG(10) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    resources_->outputs.push_back(tensor->second);
  }
}

void Compiler::LowerConstants(const Graph* graph, const Scope* scope) {
  auto& kid_scope = scope->NewScope();
  VLOG(10) << "enter Compiler::LowerConstants";
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    if (op_type == "popart_constant") {
      auto shape =
          BOOST_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("dims"));
      auto dtype_ = BOOST_GET_CONST(int, op_desc->GetAttr("dtype"));
      auto dtype = PopartType2VarType(OnnxDtype2PopartType(dtype_));
      auto tensor_name = op_desc->Output("__outputs__")[0];
      auto* var = kid_scope.Var(tensor_name);
      VLOG(10) << "lowering constant: " << tensor_name;
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      ConstantOpAttrVisitor visitor(tensor, dtype);
      auto value = op_desc->GetAttr("value");
      boost::apply_visitor(visitor, value);
      auto ddim = framework::make_ddim(shape);
      tensor->Resize(ddim);

      auto const_data = std::unique_ptr<popart::ConstVoidData>();
      popart::TensorInfo tensor_info(VarType2PopartType(tensor->type()), shape);
      const_data.reset(new popart::ConstVoidData(tensor->data(), tensor_info));
      popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data);
      SetIpuIndexStage(result, op_desc);
      resources_->tensors.emplace(tensor_name, result);
    }
  }
  VLOG(10) << "leave Compiler::LowerConstants";
}

void Compiler::LowerWeights(const Graph* graph, const Scope* scope) {
  VLOG(10) << "enter Compiler::LowerWeights";
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::PreconditionNotMet(
                              "You should call set_scope before LowerWeights"));
  // at this step, the graph doesn't contains optimizer related states
  for (const auto* node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      if (node->Var()->Persistable() && node->inputs.empty()) {
        auto var_name = node->Var()->Name();
        if (resources_->tensors.count(var_name) != 0) {
          continue;
        }
        VLOG(10) << "lowering weight: " << var_name;

        auto var = scope->FindVar(var_name);
        if (var) {
          auto tensor = var->Get<framework::LoDTensor>();
          auto dtype = VarType2PopartType(tensor.type());
          auto shape = std::vector<int64_t>();
          for (size_t i = 0; i < tensor.dims().size(); ++i) {
            shape.push_back(tensor.dims().at(i));
          }
          popart::TensorInfo tensor_info(dtype, shape);
          popart::ConstVoidData const_data{tensor.data(), tensor_info};
          popart::TensorId result =
              builder_->addInitializedInputTensor(const_data, var_name);
          resources_->tensors.emplace(var_name, result);
          resources_->weights.push_back(result);
        }
      }
    }
  }
  VLOG(10) << "leave Compiler::LowerWeights";
}

void Compiler::LowerOptimier(const Graph* graph, const Scope* scope) {
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp()) {
      continue;
    }

    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    if (op_type == "popart_optimizer") {
      auto raw_type =
          BOOST_GET_CONST(std::string, op_desc->GetAttr("raw_type"));
      resources_->optimizer_type = raw_type;
      auto loss_var =
          BOOST_GET_CONST(std::string, op_desc->GetAttr("loss_var"));
      resources_->loss_var = resources_->tensors[loss_var];
      resources_->with_lr_sched =
          BOOST_GET_CONST(bool, op_desc->GetAttr("with_lr_sched"));
      if (op_desc->HasAttr("lr_var")) {
        auto lr_var = BOOST_GET_CONST(std::string, op_desc->GetAttr("lr_var"));
        resources_->lr_var = lr_var;
        resources_->lr = GetSingleVarFromScope<float>(scope, lr_var);
      } else {
        // adadelta has no lr
        resources_->lr = 0.01f;
        resources_->with_lr_sched = false;
      }
      VLOG(10) << "Set initial lr: " << resources_->lr;
      auto loss_scaling = ipu_strategy_->loss_scaling;
      auto type = BOOST_GET_CONST(std::string, op_desc->GetAttr("type"));
      if (type == "sgd") {
        auto weight_decay =
            BOOST_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto momentum = BOOST_GET_CONST(float, op_desc->GetAttr("momentum"));
        resources_->optimizer_fn = [=](float lr) {
          return std::make_unique<popart::SGD>(
              popart::OptimizerValue(lr, false),
              popart::OptimizerValue(weight_decay, true),
              popart::OptimizerValue(momentum, true),
              popart::SGD::getUnsetDampening(),
              popart::SGD::getUnsetVelocityScaling(),
              popart::OptimizerValue(loss_scaling, true));
        };
      } else if (type == "adam") {
        auto weight_decay =
            BOOST_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto beta1 = BOOST_GET_CONST(float, op_desc->GetAttr("beta1"));
        auto beta2 = BOOST_GET_CONST(float, op_desc->GetAttr("beta2"));
        auto eps = BOOST_GET_CONST(float, op_desc->GetAttr("eps"));
        auto mwn = ipu_strategy_->max_weight_norm;
        VLOG(10) << "set max_weight_norm: " << mwn;
        auto adam_mode_ =
            BOOST_GET_CONST(std::string, op_desc->GetAttr("adam_mode"));
        auto adam_mode = AdamModeFromStr(adam_mode_);
        auto weight_decay_mode_ =
            BOOST_GET_CONST(std::string, op_desc->GetAttr("weight_decay_mode"));
        auto weight_decay_mode = WeightDecayModeFromStr(weight_decay_mode_);
        resources_->optimizer_fn = [=](float lr) {
          return std::make_unique<popart::Adam>(
              popart::OptimizerValue(lr, false),
              popart::OptimizerValue(weight_decay, true),
              popart::OptimizerValue(beta1, true),
              popart::OptimizerValue(beta2, true),
              popart::OptimizerValue(eps, true),
              popart::OptimizerValue(loss_scaling, true),
              popart::OptimizerValue(mwn, true), adam_mode, weight_decay_mode,
              popart::DataType::UNDEFINED, popart::DataType::FLOAT,
              popart::DataType::FLOAT);
        };
      } else if (type == "adaptive") {
        auto alpha = BOOST_GET_CONST(float, op_desc->GetAttr("alpha"));
        auto momentum = BOOST_GET_CONST(float, op_desc->GetAttr("momentum"));
        auto eps = BOOST_GET_CONST(float, op_desc->GetAttr("eps"));
        auto weight_decay =
            BOOST_GET_CONST(float, op_desc->GetAttr("weight_decay"));
        auto adaptive_mode_ =
            BOOST_GET_CONST(std::string, op_desc->GetAttr("adaptive_mode"));
        auto adaptive_mode = AdaptiveModeFromStr(adaptive_mode_);
        auto weight_decay_mode_ =
            BOOST_GET_CONST(std::string, op_desc->GetAttr("weight_decay_mode"));
        auto weight_decay_mode = WeightDecayModeFromStr(weight_decay_mode_);
        resources_->optimizer_fn = [=](float lr) {
          return std::make_unique<popart::Adaptive>(
              popart::OptimizerValue(lr, false),
              popart::OptimizerValue(weight_decay, true),
              popart::OptimizerValue(alpha, true),
              popart::OptimizerValue(momentum, true),
              popart::OptimizerValue(eps, true),
              popart::OptimizerValue(loss_scaling, true), adaptive_mode,
              weight_decay_mode, popart::DataType::UNDEFINED,
              popart::DataType::FLOAT, popart::DataType::FLOAT,
              popart::DataType::FLOAT);
        };
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "optimizer %s is not implemented", type));
      }
    }
  }
}

void Compiler::InsertTensors(const std::vector<std::string>& output_names,
                             const std::vector<std::string>& tensor_ids) {
  PADDLE_ENFORCE_EQ(output_names.size(), tensor_ids.size(),
                    platform::errors::Fatal("InsertTensors size mismatch"));
  for (int i = 0; i < tensor_ids.size(); i++) {
    std::string tensor_id = tensor_ids[i];
    resources_->tensors.emplace(output_names[i], tensor_ids[i]);
  }
}

void Compiler::InsertTensors(const std::vector<std::string>& output_names,
                             const std::string& tensor_id) {
  PADDLE_ENFORCE_EQ(output_names.size(), 1,
                    platform::errors::Fatal("InsertTensors size mismatch"));
  resources_->tensors.emplace(output_names[0], tensor_id);
}

void Compiler::SetIpuIndexStage(const std::vector<std::string>& tensor_ids,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetIpuIndexStage";
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());

  if (op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_ids_set, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_ids_set, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << "= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

void Compiler::SetIpuIndexStage(const std::string& tensor_id,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetIpuIndexStage";

  if (op_desc->HasAttr(sIpuIndexAttr)) {
    auto ipu_index = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuIndexAttr));
    builder_->virtualGraph(tensor_id, ipu_index);
    VLOG(10) << "set " << sIpuIndexAttr << " = " << ipu_index
             << " for op: " << op_desc->Type();
    if (op_desc->HasAttr(sIpuStageAttr)) {
      auto ipu_stage = BOOST_GET_CONST(int, op_desc->GetAttr(sIpuStageAttr));
      builder_->pipelineStage(tensor_id, ipu_stage);
      VLOG(10) << "set " << sIpuStageAttr << "= " << ipu_stage
               << " for op: " << op_desc->Type();
    }
  }
  VLOG(10) << "leave Compiler::SetIpuIndexStage";
}

void Compiler::SetAMPAttributes(const std::vector<std::string>& tensor_ids,
                                const OpDesc* op_desc) {
  if (op_desc->Type() == "popart_matmul") {
    for (const auto& tensor_id : tensor_ids) {
      SetAMPAttributes(tensor_id, op_desc);
    }
  }
}

void Compiler::SetAMPAttributes(const std::string& tensor_id,
                                const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetAMPAttributes";
  if (op_desc->Type() == "popart_matmul") {
    auto amp = ipu_strategy_->available_memory_proportion;
    if (amp > 0.0f && amp <= 1.0) {
      builder_->setAvailableMemoryProportion(tensor_id, amp);
    }
  }
  VLOG(10) << "leave Compiler::SetAMPAttributes";
}

void Compiler::SetSerializeAttributes(
    const std::vector<std::string>& tensor_ids, const OpDesc* op_desc) {
  VLOG(10) << "enter Compiler::SetSerializeAttributes";
  auto tensor_ids_set =
      std::set<std::string>(tensor_ids.begin(), tensor_ids.end());

  if (op_desc->Type() == "popart_matmul") {
    if (op_desc->HasAttr(sMatmulSerializeFactor)) {
      auto factor =
          BOOST_GET_CONST(int, op_desc->GetAttr(sMatmulSerializeFactor));
      std::string mode = "output_channels";
      if (op_desc->HasAttr(sMatmulSerializeMode)) {
        mode = BOOST_GET_CONST(std::string,
                               op_desc->GetAttr(sMatmulSerializeMode));
      }
      builder_->setSerializeMatMul(tensor_ids_set, mode, (int64_t)factor, true);
    }
  }
  VLOG(10) << "leave Compiler::SetSerializeAttributes";
}

void Compiler::SetSerializeAttributes(const std::string& tensor_id,
                                      const OpDesc* op_desc) {
  std::vector<std::string> tensor_ids = {tensor_id};
  SetSerializeAttributes(tensor_ids, op_desc);
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

std::string Compiler::GetModelProto() {
  if (ipu_strategy_->enable_fp16) {
    return GetFP16ModelProto();
  } else {
    return builder_->getModelProto();
  }
}

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
      BOOST_GET_CONST(std::string, op->GetAttr(sOpIdentifyIdAttr));
  VLOG(10) << "op_identify_id of op: " << op->Type() << " is "
           << op_identify_id;
  return popart::DebugContext(op_identify_id);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
