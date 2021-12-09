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

#include "paddle/fluid/platform/ipu/ipu_compiler.h"

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/platform/ipu/ipu_utils.h"

namespace paddle {
namespace platform {
namespace ipu {

template <typename T>
T GetAttrAllowNull(std::string attr, framework::OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

template <typename T>
nonstd::optional<T> GetOptAttrAllowNull(std::string attr,
                                        framework::OpDesc* op_desc) {
  if (op_desc->HasAttr(attr)) {
    return BOOST_GET_CONST(T, op_desc->GetAttr(attr));
  } else {
    return {};
  }
}

Compiler::Compiler() {
  builder_ = popart::Builder::create();
  RegisterOpFunc();
}

Compiler::~Compiler() {}

void Compiler::RegisterOpFunc() {
  VLOG(10) << "enter Compiler::RegisterOpFunc";
#define INT_VEC std::vector<std::int64_t>
#define FLOAT_VEC std::vector<float>
#define FLOAT float
#define INT std::int64_t
#define BOOL bool
#define STRING std::string
#define STRING_VEC std::vector<std::string*>
#define NONE

#define ARG(Type, Name) , GetAttrAllowNull<Type>(#Name, op_desc)
#define OPT_ARG(Type, Name) , GetOptAttrAllowNull<Type>(#Name, op_desc)
#define POPART_CONST_ARG(Name) , const PopartConstant& Name
#define HOST_SIDE_CONST_ARG(Name) , const HostSideConstant& Name
#define POPART_ATTRIB_VEC_ARG(Name)
#define BODY_ARG(Name) NONE

  name_function_ = {
#define OP_DECL(FuncName, OnnxImpl, Args)                     \
  {#FuncName, [&](framework::OpDesc* op_desc) {               \
     auto op_type = op_desc->Type();                          \
     VLOG(10) << "build op:" << op_type << " args " << #Args; \
     auto inputs = GetOpInputs(op_desc);                      \
     auto output_names = GetOpOutputs(op_desc);               \
     auto debug_context = BuildDebugContext(op_desc);         \
     auto aiGraphcoreOpset = builder_->aiGraphcoreOpset1();   \
     auto aiOnnxOpset = builder_->aiOnnxOpset11();            \
     auto output_ids = OnnxImpl(inputs Args, debug_context);  \
     SetIpuIndexStage(output_ids, op_desc);                   \
     InsertTensors(output_names, output_ids);                 \
   }},  // NOLINT
#include "paddle/fluid/platform/ipu/supported_ops_autogen.h"
  };

#undef OP_DECL
#undef BODY_ARG
#undef POPART_ATTRIB_VEC_ARG
#undef HOST_SIDE_CONST_ARG
#undef POPART_CONST_ARG
#undef OPT_ARG
#undef ARG
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
}

void Compiler::LowerBody(const framework::ir::Graph* graph) {
  VLOG(10) << "enter Compiler::LowerBody";
  auto nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* node : nodes) {
    auto* op_desc = node->Op();
    auto op_type = op_desc->Type();
    VLOG(10) << "node->type: " << op_type;

    if (op_type == "popart_constant") {
      auto dims =
          BOOST_GET_CONST(std::vector<int64_t>, op_desc->GetAttr("dims"));
      auto dtype_ = BOOST_GET_CONST(int, op_desc->GetAttr("dtype"));
      auto dtype = OnnxDtype2PopartType(dtype_);
      popart::TensorInfo tensor_info{dtype, dims};
      auto value_attr = op_desc->GetAttr("value");
      auto const_data = std::unique_ptr<popart::ConstVoidData>{};
      switch (dtype) {
        case popart::DataType::FLOAT:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<float>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::INT32:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<int>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::DOUBLE:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<double>, value_attr).data(),
              tensor_info));
          break;
        case popart::DataType::INT64:
          const_data.reset(new popart::ConstVoidData(
              BOOST_GET_CONST(std::vector<int64_t>, value_attr).data(),
              tensor_info));
          break;
        default:
          PADDLE_THROW(platform::errors::Unimplemented(
              "The popart datatype is not supported, popart::DataType is %d",
              dtype));
      }
      popart::TensorId result = builder_->aiOnnxOpset11().constant(*const_data);
      SetIpuIndexStage(result, op_desc);
      InsertTensors(GetOpOutputs(op_desc), result);
    } else if (op_type == "popart_batchnormalization") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      auto num_outputs = outputs.size();
      auto epsilon = BOOST_GET_CONST(float, op_desc->GetAttr("epsilon"));
      auto momentum = BOOST_GET_CONST(float, op_desc->GetAttr("momentum"));
      auto result = builder_->aiOnnxOpset11().batchnormalization(
          inputs, num_outputs, epsilon, momentum);
      SetIpuIndexStage(result, op_desc);
      InsertTensors(GetOpOutputs(op_desc), result);
    } else if (op_type == "popart_nllloss") {
      auto inputs = GetOpInputs(op_desc);
      auto ignoreIndex = BOOST_GET_CONST(int, op_desc->GetAttr("ignoreIndex"));
      auto result = builder_->aiGraphcoreOpset1().nllloss(
          inputs, popart::ReductionType::NoReduction, ignoreIndex);
      SetIpuIndexStage(result, op_desc);
      InsertTensors(GetOpOutputs(op_desc), result);
    } else if (op_type == "popart_topk") {
      auto inputs = GetOpInputs(op_desc);
      auto outputs = GetOpOutputs(op_desc);
      int64_t axis = BOOST_GET_CONST(int64_t, op_desc->GetAttr("axis"));
      int sorted_INT32 = BOOST_GET_CONST(int, op_desc->GetAttr("sorted"));
      int64_t sorted = int64_t{sorted_INT32};

      auto aiOnnxOpset = builder_->aiOnnxOpset11();

      popart::ConvInputs result;
      if (inputs.size() == 2) {
        VLOG(10)
            << "[Compiler::LowerBody] size of inputs for <popart_topk> is 2";
        result = aiOnnxOpset.topk(inputs, axis, sorted);
      } else if (inputs.size() == 1) {
        VLOG(10)
            << "[Compiler::LowerBody] size of inputs for <popart_topk> is 1";
        int64_t k = BOOST_GET_CONST(int64_t, op_desc->GetAttr("k"));
        popart::TensorInfo kShape{"INT64", std::vector<int64_t>{1}};
        popart::ConstVoidData kData = {&k, kShape};
        auto K_t = aiOnnxOpset.constant(kData);
        result = aiOnnxOpset.topk({inputs[0], K_t}, axis, sorted);
      }
      result[1] = aiOnnxOpset.cast({result[1]}, "INT32");
      SetIpuIndexStage(result, op_desc);
      VLOG(10) << "[Compiler::LowerBody] output[1]: " << outputs[1];
      VLOG(10) << "[Compiler::LowerBody] output[1]: "
               << GetOpOutputs(op_desc)[1] << " -> " << result[1];
      tensors_.emplace(GetOpOutputs(op_desc)[1], result[1]);  // topk indices
      VLOG(10) << "[Compiler::LowerBody] output[0]: " << outputs[0];
      VLOG(10) << "[Compiler::LowerBody] output[0]: "
               << GetOpOutputs(op_desc)[0] << " -> " << result[0];
      tensors_.emplace(GetOpOutputs(op_desc)[0], result[0]);  // topk values
    } else {
      auto itr = name_function_.find(op_type);
      if (itr != name_function_.end()) {
        itr->second(node->Op());
      } else {
        PADDLE_THROW(platform::errors::NotFound(
            "Op %s is not registered in popart canonicalization", op_type));
      }
    }
  }
  VLOG(10) << "leave Compiler::LowerBody";
}

void Compiler::InitInputs(framework::ir::Graph* graph,
                          const std::vector<std::string>& feed_list) {
  for (const auto& feed_name : feed_list) {
    feed_list_.push_back(feed_name);
    for (const framework::ir::Node* n : graph->Nodes()) {
      if (n->IsVar()) {
        auto* var_desc = n->Var();
        if (feed_name == var_desc->Name()) {
          VLOG(10) << "feed_name= " << var_desc->Name();
          auto data_type = VarType2PopartType(var_desc->GetDataType());
          if (ipu_strategy_->enable_fp16) {
            data_type = popart::DataType::FLOAT16;
          }
          popart::TensorInfo input_info{data_type, var_desc->GetShape()};
          VLOG(10) << "popart input_info = " << input_info;
          popart::TensorId tensor_id =
              builder_->addInputTensor(input_info, feed_name);
          VLOG(10) << "popart input tensor id = " << tensor_id;
          inputs_.push_back(tensor_id);
          tensors_.emplace(var_desc->Name(), tensor_id);
        }
      }
    }
  }
}

void Compiler::InitOutputs(const std::vector<std::string>& fetch_list) {
  for (const auto& fetch_name : fetch_list) {
    fetch_list_.push_back(fetch_name);
    auto tensor = tensors_.find(fetch_name);
    PADDLE_ENFORCE_NE(tensor, tensors_.end(),
                      platform::errors::NotFound(
                          "output tensor %s does not exist.", fetch_name));
    VLOG(10) << "fetch_name= " << fetch_name;
    VLOG(10) << "popart output tensor id = " << tensor->second;
    builder_->addOutputTensor(tensor->second);
    outputs_.push_back(tensor->second);
  }
}

void Compiler::LowerWeights(const framework::ir::Graph* graph,
                            const framework::Scope* scope_) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "You should call set_scope before LowerWeights"));
  // at this step, the graph doesn't contains optimizer related states
  for (const auto* node : graph->Nodes()) {
    if (node->IsVar() && !node->IsCtrlVar() && node->Var()) {
      if (node->Var()->Persistable() && node->inputs.empty()) {
        auto var_name = node->Var()->Name();
        // workround: https://github.com/graphcore/Paddle/issues/151
        if (tensors_.count(var_name) != 0) {
          continue;
        }

        auto var = scope_->FindVar(var_name);
        if (var) {
          auto tensor = var->Get<framework::LoDTensor>();
          auto dtype = VarType2PopartType(tensor.type());
          auto shape = std::vector<int64_t>();
          for (size_t i = 0; i < tensor.dims().size(); ++i) {
            shape.push_back(tensor.dims().at(i));
          }
          popart::TensorInfo tensor_info(dtype, shape);
          popart::ConstVoidData const_data{tensor.data<void>(), tensor_info};
          popart::TensorId result =
              builder_->addInitializedInputTensor(const_data, var_name);
          tensors_.emplace(var_name, result);
          weights_.push_back(result);
        }
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
    tensors_.emplace(output_names[i], tensor_ids[i]);
  }
}

void Compiler::InsertTensors(const std::vector<std::string>& output_names,
                             const std::string& tensor_id) {
  PADDLE_ENFORCE_EQ(output_names.size(), 1,
                    platform::errors::Fatal("InsertTensors size mismatch"));
  tensors_.emplace(output_names[0], tensor_id);
}

void Compiler::SetIpuIndexStage(const std::vector<std::string>& tensor_ids,
                                const framework::OpDesc* op_desc) {
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
                                const framework::OpDesc* op_desc) {
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

std::vector<popart::TensorId>& Compiler::GetWeights() { return weights_; }

// convertFloatsToHalfs
void Compiler::ConvertProtoToFp16() {
  popart::GraphTransformer graph_transformer(builder_->getModelProto());
  graph_transformer.convertFloatsToHalfs();
  converted_proto_ = graph_transformer.getModelProto();
}

std::string Compiler::GetModelProto() {
  if (converted_proto_.length()) {
    return converted_proto_;
  }
  return builder_->getModelProto();
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

std::vector<std::string> Compiler::GetOpInputs(const framework::OpDesc* op) {
  auto ins = op->Input("__inputs__");
  std::vector<std::string> inputs;
  for (const auto& in : ins) {
    if (tensors_.find(in) != tensors_.end()) {
      inputs.push_back(tensors_[in]);
    } else {
      inputs.push_back(in);
    }
  }
  return inputs;
}

const std::vector<std::string>& Compiler::GetOpOutputs(
    const framework::OpDesc* op) {
  return op->Output("__outputs__");
}

popart::DebugContext Compiler::BuildDebugContext(const framework::OpDesc* op) {
  auto op_identify_id =
      BOOST_GET_CONST(std::string, op->GetAttr(sOpIdentifyIdAttr));
  VLOG(10) << "op_identify_id of op: " << op->Type() << " is "
           << op_identify_id;
  return popart::DebugContext(op_identify_id);
}

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
