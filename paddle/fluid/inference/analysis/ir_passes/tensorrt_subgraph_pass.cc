
// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"
#include <cstddef>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace {

bool IsFloat(framework::proto::VarType::Type t) {
  if (t == framework::proto::VarType::FP16 ||
      t == framework::proto::VarType::FP32 ||
      t == framework::proto::VarType::FP64 ||
      t == framework::proto::VarType::BF16)
    return true;
  return false;
}

// if in mixed model precision, we should make all tensorrt_engine's output
// floats dtype to float32 dtype.
void OutputProcess(framework::ir::Graph *graph,
                   const std::unordered_set<framework::ir::Node *> &trt_outputs,
                   phi::Backend backend,
                   phi::DataType precision,
                   const std::unordered_set<std::string> &blacklist) {
  framework::BlockDesc *block_desc{nullptr};
  int suffix = 0;
  std::unordered_map<framework::ir::Node *, framework::ir::Node *>
      var_to_cast_op_map;

  framework::proto::VarType::Type to_type;
  if (precision == phi::DataType::FLOAT16) {
    to_type = framework::proto::VarType::FP16;
  } else if (precision == phi::DataType::BFLOAT16) {
    to_type = framework::proto::VarType::BF16;
  } else if (precision == phi::DataType::FLOAT32) {
    return;
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only support "
        "fp16 and bf16.",
        static_cast<int>(precision)));
  }

  for (auto *op_node : framework::ir::TopologySortOperations(*graph)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed") block_desc = op_node->Op()->Block();
    if (op_type != "tensorrt_engine") continue;
    for (auto *var_node : op_node->outputs) {
      if (!trt_outputs.count(var_node)) continue;
      if (!var_node->Var()->Persistable() &&
          IsFloat(var_node->Var()->GetDataType()) &&
          var_node->Var()->GetDataType() != framework::proto::VarType::FP32) {
        for (auto *next_op : var_node->outputs) {
          // if next_op support mixed_precision, we need to add cast op.
          if (OpSupportPrecision(
                  phi::TransToPhiKernelName(next_op->Op()->Type()),
                  backend,
                  precision,
                  blacklist)) {
            AddCastOp(graph,
                      var_node,
                      next_op,
                      framework::proto::VarType::FP32,
                      to_type,
                      &suffix,
                      block_desc,
                      &var_to_cast_op_map);
            var_node->Var()->SetDataType(framework::proto::VarType::FP32);
          }
        }
      }
    }
  }
}

}  // namespace

using framework::ir::Node;

void analysis::TensorRtSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph);

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  if (model_precision == phi::DataType::BFLOAT16) {
    LOG(WARNING)
        << "Paddle-TRT not support bf16 mixed precison, just fallback.";
    return;
  }

  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  bool no_calib_int8 = enable_int8 && !(use_calib_mode);
  auto trt_disabled_ops = Get<std::vector<std::string>>("trt_disabled_ops");
  auto with_dynamic_shape = Get<bool>("with_dynamic_shape");
  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    if (find(trt_disabled_ops.begin(),
             trt_disabled_ops.end(),
             node->Op()->Type()) != trt_disabled_ops.end()) {
      VLOG(3) << node->Op()->Type().c_str()
              << " is diabled by config in TensorRT";
      return false;
    }
    bool is_ok = tensorrt::OpTeller::Global().Tell(
        node, no_calib_int8, with_dynamic_shape);
    if (!is_ok)
      VLOG(3) << node->Op()->Type().c_str() << " op is not in TensorRT";
    return is_ok;
  };

  framework::ir::SubGraphFuser fuser(
      graph,
      teller,
      Get<int>("min_subgraph_size") /*min subgraph size*/,
      "tensorrt_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in trt, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      CreateTensorRTOp(node, graph, graph_param_names, &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          framework::ir::Agent(node).subgraph()->begin(),
          framework::ir::Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id,
                              const std::string &max_batch_size,
                              const std::string &precision,
                              const bool for_calibration) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
    engine_hash_key += "#";
  }
  engine_hash_key += predictor_id;
  if (!for_calibration) {
    engine_hash_key += "#";
    engine_hash_key += max_batch_size;
  }
  engine_hash_key += "#";
  engine_hash_key += precision;

  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  VLOG(2) << "TRT engine hash key: " << engine_hash_key;
  VLOG(2) << "TRT engine key: " << engine_key;
  return engine_key;
}

void TensorRtSubgraphPass::CreateTensorRTOp(
    framework::ir::Node *node,
    framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The subgraph should not be empty."));

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for TensorRTEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  // const framework::BlockDesc& main_block = program_desc->Block(0);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";
  for (auto node : subgraph) {
    if (node->NodeType() == Node::Type::kOperation) {
      VLOG(5) << "trt subgraph has op: " << (node->Op()->Type());
    }
  }

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  // if we delete fluid copy of params shared by more than 1 ops, there will be
  // problem, so we filter them out.
  std::vector<std::string> params_not_shared;

  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0 &&
        x->outputs.size() <= 1) {
      params_not_shared.push_back(x->Name());
    }
  }

  auto model_precision =
      static_cast<phi::DataType>(Get<int>("model_precision"));
  auto mixed_black_list =
      Get<std::unordered_set<std::string>>("mixed_black_list");

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  std::map<std::string, int> origin_name_output_dims;
  std::unordered_set<Node *> trt_outputs;
  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
    origin_name_output_dims[x->Name()] = x->Var()->GetShape().size();
    trt_outputs.insert(x);
  }

  OutputProcess(
      graph, trt_outputs, phi::Backend::GPU, model_precision, mixed_black_list);

  std::unordered_map<std::string, std::string> output_name_map;
  std::unordered_map<std::string, framework::ir::Node *> graph_var_map;

  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      graph_var_map[node->Name()] = node;
    }
  }
  auto precision_mode = Get<AnalysisConfig::Precision>("precision_mode");
  bool enable_fp16 = false;
  if (precision_mode == AnalysisConfig::Precision::kHalf) enable_fp16 = true;
  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  auto &subgraph_nodes = *framework::ir::Agent(node).subgraph();
  auto min_input_shape =
      Get<std::map<std::string, std::vector<int>>>("min_input_shape");
  auto max_input_shape =
      Get<std::map<std::string, std::vector<int>>>("max_input_shape");
  auto opt_input_shape =
      Get<std::map<std::string, std::vector<int>>>("optim_input_shape");

  auto allow_build_at_runtime = Get<bool>("trt_allow_build_at_runtime");
  auto shape_range_info_path = Get<std::string>("trt_shape_range_info_path");
  auto trt_tuned_dynamic_shape = Get<bool>("trt_tuned_dynamic_shape");
  int max_batch_size = Get<int>("max_batch_size");
  if (trt_tuned_dynamic_shape) {
    VLOG(1) << "trt dynamic_shape deserialize from " << shape_range_info_path;
    inference::DeserializeShapeRangeInfo(shape_range_info_path,
                                         &min_input_shape,
                                         &max_input_shape,
                                         &opt_input_shape);
  }

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.
  RenameAndGetOutputs(subgraph_nodes,
                      &block_desc,
                      input_names_with_id,
                      &output_names_with_id,
                      &output_names,
                      &output_name_map,
                      graph_var_map,
                      !enable_int8);

  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  std::vector<int> renamed_output_dims;
  for (auto name : output_names) {
    PADDLE_ENFORCE_NE(output_name_map.count(name),
                      0,
                      platform::errors::PreconditionNotMet(
                          "The output_name_map should have %s", name));
    output_mapping.push_back(output_name_map[name]);
    renamed_output_dims.push_back(origin_name_output_dims[name]);
  }
  PADDLE_ENFORCE_EQ(output_mapping.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The output_mapping should not be empty."));
  PADDLE_ENFORCE_EQ(
      !block_desc.Proto()->vars().empty(),
      true,
      platform::errors::PreconditionNotMet("the block has no var-desc"));

  // Set attrs
  op_desc->SetType("tensorrt_engine");
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));

  op_desc->SetBlockAttr("sub_block", new_block);
  op_desc->SetAttr("subgraph", block_desc.Proto()->SerializeAsString());
  op_desc->SetAttr("max_batch_size", max_batch_size);
  op_desc->SetAttr("workspace_size", Get<int64_t>("workspace_size"));
  op_desc->SetAttr("gpu_id", Get<int>("gpu_device_id"));
  op_desc->SetAttr("output_name_mapping", output_mapping);
  op_desc->SetAttr("origin_output_dims", renamed_output_dims);
  op_desc->SetAttr("parameters", params);
  op_desc->SetAttr("allow_build_at_runtime", allow_build_at_runtime);
  op_desc->SetAttr("shape_range_info_path", shape_range_info_path);
  op_desc->SetAttr("use_inspector", Get<bool>("use_inspector"));
  op_desc->SetAttr("model_precision", Get<int>("model_precision"));

  // we record all inputs' shapes in attr to check if they are consistent
  // with the real inputs' shapes retrieved from scope when trt runs.
  for (auto *x : node->inputs) {
    if (x->IsVar() && x->Var()) {
      framework::VarDesc *var = x->Var();
      op_desc->SetAttr(var->Name() + "_shape", var->GetShape());
    }
  }

  auto use_static_engine = Get<bool>("use_static_engine");
  op_desc->SetAttr("use_static_engine", use_static_engine);
  if (use_static_engine)
    op_desc->SetAttr("model_opt_cache_dir",
                     Get<std::string>("model_opt_cache_dir"));

  // TODO(NHZlX)
  // There are models with the same structure but the different parameters,
  // when running in the 'use_serialize' mode, there is a bug.
  // serialization is affected by max_batch_size, but calibration is not.
  // So we use separate engine keys in serialization and calibration.
  auto engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        false);
  auto calibration_engine_key =
      GenerateEngineKey(input_names_with_id,
                        output_names_with_id,
                        std::to_string(0),
                        std::to_string(max_batch_size),
                        std::to_string(static_cast<int>(precision_mode)),
                        true);
  auto predictor_id = Get<int>("predictor_id");

  // Get "" when there is no cached calibration table data.
  std::string calibration_data = "";
  if (enable_int8 && use_calib_mode) {
    calibration_data =
        GetTrtCalibTableData(Get<std::string>("model_opt_cache_dir"),
                             calibration_engine_key,
                             enable_int8);
  }
  op_desc->SetAttr("calibration_data", calibration_data);
  op_desc->SetAttr("enable_int8", enable_int8);
  op_desc->SetAttr("enable_fp16", enable_fp16);
  op_desc->SetAttr("use_calib_mode", use_calib_mode);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("calibration_engine_key", calibration_engine_key);
  op_desc->SetAttr("predictor_id", predictor_id);

  std::string trt_engine_serialized_data = "";
  op_desc->SetAttr("engine_serialized_data", trt_engine_serialized_data);
  op_desc->Flush();

  std::unique_ptr<tensorrt::TRTInt8Calibrator> calibrator;
  if (enable_int8 && calibration_data.size() != 0) {
    calibrator.reset(new tensorrt::TRTInt8Calibrator(calibration_data));
    LOG(INFO) << "RUN Paddle TRT int8 calibration mode...";
  }
  // When in int8 mode and calibration_mode, the program just produce the
  // calibration table data.
  bool calibration_mode =
      (enable_int8 && calibration_data.size() == 0 && use_calib_mode);
  if (calibration_mode) {
    // calibraion mode means generate int8 calibration table data process.
    return;
  }

  std::copy(params_not_shared.begin(),
            params_not_shared.end(),
            std::back_inserter(*repetitive_params));

  // Check trt version for dynamic shape input.

  if (min_input_shape.size() > 0 && TRT_VERSION < 6000) {
    LOG_FIRST_N(WARNING, 1) << "You are using the dynamic size input mode of "
                               "Paddle-TRT, but we found that the version of "
                               "the TensorRT is less than 6.0, so we use the "
                               "static shape mode instead.";
    min_input_shape = {};
    max_input_shape = {};
    opt_input_shape = {};
  }

  auto to_major_version = [&](int full_version) -> float {
    return (full_version / 100) / 10.0;
  };
  const float compile_time_trt_version = to_major_version(TRT_VERSION);
  const float run_time_trt_version =
      to_major_version(tensorrt::GetInferLibVersion());
  if (compile_time_trt_version != run_time_trt_version) {
    LOG_FIRST_N(WARNING, 1)
        << "The Paddle Inference library is compiled with "
        << compile_time_trt_version << " version TensorRT, "
        << "but the runtime TensorRT you are using is " << run_time_trt_version
        << " version. "
           "This might cause serious compatibility issues. We strongly "
           "recommend using the same TRT version at runtime.";
  }

  // Setting the disable_trt_plugin_fp16 to true means that TRT plugin will not
  // run fp16.
  // When running fp16, the output accuracy of the model will be affected,
  // closing the plugin fp16 may bring some improvement on accuracy.
  bool disable_trt_plugin_fp16 = Get<bool>("disable_trt_plugin_fp16");
  tensorrt::TensorRTEngine *trt_engine =
      inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
          .Create(engine_key + std::to_string(predictor_id),
                  max_batch_size,
                  Get<int64_t>("workspace_size"),
                  precision_mode,
                  calibrator.get(),
                  Get<int>("gpu_device_id"),
                  min_input_shape,
                  max_input_shape,
                  opt_input_shape,
                  disable_trt_plugin_fp16,
                  static_cast<phi::DataType>(Get<int>("model_precision")));
  trt_engine->SetUseOSS(Get<bool>("use_varseqlen"));
  trt_engine->SetWithInterleaved(Get<bool>("with_interleaved"));
  trt_engine->SetTransformerPosid(
      Get<std::string>("tensorrt_transformer_posid"));
  trt_engine->SetTransformerMaskid(
      Get<std::string>("tensorrt_transformer_maskid"));
  trt_engine->SetUseDLA(Get<bool>("trt_use_dla"));
  trt_engine->SetDLACore(Get<int>("trt_dla_core"));
  trt_engine->SetUseInspector(Get<bool>("use_inspector"));
  trt_engine->SetWithErnie(
      graph->Has(framework::ir::kEmbEltwiseLayernormPass) &&
      graph->Has(framework::ir::kMultiheadMatmulPass));

  if (use_static_engine) {
    trt_engine_serialized_data = GetTrtEngineSerializedData(
        Get<std::string>("model_opt_cache_dir"), engine_key);
    // we can load the engine info serialized before from the disk.
    if (!trt_engine_serialized_data.empty()) {
      trt_engine->Deserialize(trt_engine_serialized_data);
      LOG(INFO) << "Load TRT Optimized Info from "
                << GetTrtEngineSerializedPath(
                       Get<std::string>("model_opt_cache_dir"), engine_key);
      return;
    }
  }

  // the following code will NOT run in following situation:
  // 1. calibraion mode (generate trt int8 calibraiton table data)
  // 2. already load serialized trt engine info.
  LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
               "kernel etc). This process may cost a lot of time.";

  auto *scope = param_scope();
  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());
  std::unordered_set<std::string> param_set(params.begin(), params.end());
  inference::Singleton<inference::tensorrt::OpConverter>::Global()
      .ConvertBlockToTRTEngine(
          &block_desc_temp,
          *scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          param_set,
          output_mapping,
          trt_engine);

  if (use_static_engine) {
    nvinfer1::IHostMemory *serialized_engine_data = trt_engine->Serialize();
    trt_engine_serialized_data =
        std::string((const char *)serialized_engine_data->data(),
                    serialized_engine_data->size());
    SaveTrtEngineSerializedDataToFile(
        GetTrtEngineSerializedPath(Get<std::string>("model_opt_cache_dir"),
                                   engine_key),
        trt_engine_serialized_data);
    LOG(INFO) << "Save TRT Optimized Info to "
              << GetTrtEngineSerializedPath(
                     Get<std::string>("model_opt_cache_dir"), engine_key);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");

REGISTER_PASS_CAPABILITY(tensorrt_subgraph_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("pool2d", 0)
            .EQ("relu", 0)
            .EQ("softmax", 0)
            .EQ("sigmoid", 0)
            .EQ("hard_swish", 0)
            .LE("depthwise_conv2d", 1)
            .EQ("batch_norm", 0)
            .EQ("concat", 0)
            .EQ("tanh", 0)
            .EQ("pad", 0)
            .LE("elementwise_add", 1)
            .LE("elementwise_mul", 1)
            .EQ("prelu", 0)
            .LE("conv2d_transpose", 2)
            .LE("leaky_relu", 1)
            .EQ("fc", 0)
            .EQ("shuffle_channel", 0)
            .EQ("swish", 0)
            .EQ("split", 0)
            .LE("instance_norm", 1)
            .EQ("gelu", 0)
            .EQ("layer_norm", 0)
            .EQ("scale", 0)
            .LE("matmul", 1));
