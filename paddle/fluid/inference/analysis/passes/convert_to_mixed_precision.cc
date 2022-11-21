// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/passes/ir_graph_clean_pass.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace inference {
namespace analysis {

namespace {
using VarType = framework::proto::VarType;

bool PhiKernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto kernels = phi::KernelFactory::Instance().kernels();
  if (kernels.find(op_type) == kernels.end()) {
    return false;
  }
  phi::KernelKey kernel_key(backend, layout, data_type);
  return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
}

bool GpuKernelSupportPrecision(
    const std::string& op_type,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool res = PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPU, data_type, layout);
  res |= PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPUDNN, data_type, layout);

  if (!res) {
    auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (auto& kern_pair : it->second) {
        if (platform::is_gpu_place(kern_pair.first.place_) &&
            kern_pair.first.data_type_ == VarType::FP16) {
          res = true;
          break;
        }
      }
    }
  }
  return res;
}

class ConvertToMixedPrecisionPass {
  using BlockID = size_t;

 public:
  explicit ConvertToMixedPrecisionPass(
      const std::string& model_file,
      const std::string& params_file,
      const std::string& mixed_model_file,
      const std::string& mixed_params_file,
      phi::DataType mixed_precision,
      phi::Backend backend,
      bool keep_io_types,
      const std::unordered_set<std::string>& black_list)
      : model_file_(model_file),
        params_file_(params_file),
        mixed_model_file_(mixed_model_file),
        mixed_params_file_(mixed_params_file),
        mixed_precision_(mixed_precision),
        backend_(backend),
        keep_io_types_(keep_io_types),
        black_list_(black_list),
        place_(paddle::CPUPlace()),
        executor_(place_) {
    VLOG(4) << "black_list has ";
    for (auto& name : black_list_) {
      VLOG(4) << " - " << name;
    }
  }

  void Run();

 private:
  void LoadAndPrepare();
  inline bool VarNodeHasDtype(framework::ir::Node* node);
  void ConvertAllFp64ToFp32(framework::ir::Graph* graph);
  void FixCastAttr(framework::ir::Graph* graph);
  void SaveMixedModel();
  void ConvertTensorDtype(BlockID block_idx);
  void ProcessInputNode(bool support_precision,
                        framework::ir::Node* in_node,
                        framework::ir::Node* op_node,
                        int* suffix,
                        framework::BlockDesc* block_desc,
                        VarType::Type to_type,
                        BlockID block_idx);

  void ProcessOutputNode(BlockID block_idx,
                         framework::ir::Node* var_node,
                         VarType::Type to_type);
  inline bool IsFloatVarType(VarType::Type type);

  bool OutShouldNotConvert(framework::ir::Node* var_node);
  // Just process special cases for weights conversion.
  bool WeightsShouldNotConvert(framework::ir::Node* var_node);

  // Return Node* which first appers in block.
  framework::ir::Node* GetRealVarNode(framework::ir::Node* node);

  // Fallback to fp32 dtype when encounter circle (Not a DAG graph).
  void ProcessCircleCases();

 private:
  std::string model_file_;
  std::string params_file_;
  std::string mixed_model_file_;
  std::string mixed_params_file_;
  phi::DataType mixed_precision_;
  phi::Backend backend_;
  bool keep_io_types_;
  std::unordered_set<std::string> black_list_;
  paddle::CPUPlace place_;
  framework::Executor executor_;
  framework::Scope scope_;

  std::unordered_map<std::string, framework::ir::Node*> name2node_;
  std::unordered_map<framework::ir::Node*, framework::ir::Node*> cast_map_;
  int suffix_{0};

  std::set<std::string> var_names_in_circles_;

  std::unique_ptr<framework::ProgramDesc> program_desc_{nullptr};
  std::unique_ptr<framework::ir::Graph> main_graph_{nullptr};
  std::vector<framework::ir::Graph*> graphes_;
};

framework::ir::Node* ConvertToMixedPrecisionPass::GetRealVarNode(
    framework::ir::Node* var_node) {
  CHECK_EQ(var_node->IsVar(), true);
  if (name2node_.count(var_node->Name())) return name2node_[var_node->Name()];
  return var_node;
}

inline bool ConvertToMixedPrecisionPass::VarNodeHasDtype(
    framework::ir::Node* var_node) {
  CHECK_EQ(var_node->IsVar(), true);
  auto type = var_node->Var()->GetType();
  return (type == VarType::SELECTED_ROWS) || (type == VarType::LOD_TENSOR) ||
         (type == VarType::LOD_TENSOR_ARRAY) || (type == VarType::STRINGS) ||
         (type == VarType::VOCAB);
}

void ConvertToMixedPrecisionPass::ProcessInputNode(
    bool support_precision,
    framework::ir::Node* in_node,
    framework::ir::Node* op_node,
    int* suffix,
    framework::BlockDesc* block_desc,
    VarType::Type to_type,
    BlockID block_idx) {
  if (!in_node->IsVar()) return;
  auto* real_node = GetRealVarNode(in_node);
  if (!VarNodeHasDtype(real_node)) return;
  auto* graph = graphes_[block_idx];
  auto* in_var = real_node->Var();
  auto in_var_type = in_var->GetDataType();
  auto prev_type = in_var_type;

  if (support_precision) {
    if (in_var->Persistable() && in_var_type == VarType::FP32) {
      if (WeightsShouldNotConvert(in_node)) return;
      in_var->SetDataType(to_type);
      in_var_type = to_type;
      VLOG(3) << "   in_node name " << in_var->Name() << " from " << prev_type
              << " to " << to_type;
    } else if (!in_var->Persistable() && IsFloatVarType(in_var_type) &&
               in_var_type != to_type) {
      AddCastOp(graph,
                in_node,
                op_node,
                in_var_type,
                to_type,
                suffix,
                block_desc,
                &cast_map_);
      VLOG(3) << "   in_node name " << in_var->Name() << "(" << prev_type
              << ") to " << cast_map_[in_node]->Name() << "(" << to_type << ")";
    }
  } else {
    if (!in_var->Persistable() && IsFloatVarType(in_var_type) &&
        in_var_type != to_type) {
      AddCastOp(graph,
                in_node,
                op_node,
                in_var_type,
                to_type,
                suffix,
                block_desc,
                &cast_map_);
      VLOG(3) << "   in_node name " << in_var->Name() << "(" << prev_type
              << ") to " << cast_map_[in_node]->Name() << "(" << to_type << ")";
    }
  }
}

void ConvertToMixedPrecisionPass::ProcessOutputNode(
    BlockID block_idx, framework::ir::Node* var_node, VarType::Type to_type) {
  if (!var_node->IsVar()) return;
  auto* real_node = GetRealVarNode(var_node);
  if (!VarNodeHasDtype(real_node)) return;
  auto* out_var = real_node->Var();
  auto prev_type = out_var->GetDataType();
  if (out_var->GetDataType() == VarType::FP32) {
    if (OutShouldNotConvert(var_node)) return;
    out_var->SetDataType(to_type);
  }
  VLOG(3) << "   out_node name " << var_node->Name() << " from dtype "
          << prev_type << " to " << out_var->GetDataType();
}

// Just process special cases.
bool ConvertToMixedPrecisionPass::OutShouldNotConvert(
    framework::ir::Node* var_node) {
  auto op_node = var_node->inputs[0];
  auto* op_desc = op_node->Op();

  // batch_norm's input and output (variance and mean) are the same.
  if (op_desc->Type() == "batch_norm") {
    auto vecs = op_desc->Output("MeanOut");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("VarianceOut");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedMean");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedVariance");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
  }

  return false;
}

bool ConvertToMixedPrecisionPass::WeightsShouldNotConvert(
    framework::ir::Node* var_node) {
  auto op_nodes = var_node->outputs;
  for (auto* op_node : op_nodes) {
    auto* op_desc = op_node->Op();
    // batch_norm op's bias, mean, scale and variance just be float32, so we can
    // not convert the dtype.
    if (op_desc->Type() == "batch_norm") {
      auto vecs = op_desc->Input("Bias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Mean");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Scale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Variance");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
    } else if (op_desc->Type() == "fused_multi_transformer") {
      auto vecs = op_desc->Input("LnScale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("LnBias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("FFNLnScale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("FFNLnBias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
    }
  }

  return false;
}

inline bool ConvertToMixedPrecisionPass::IsFloatVarType(VarType::Type type) {
  return (type == VarType::FP16) || (type == VarType::FP32) ||
         (type == VarType::BF16);
}

void ConvertToMixedPrecisionPass::LoadAndPrepare() {
  program_desc_ =
      inference::Load(&executor_, &scope_, model_file_, params_file_);
  main_graph_ = std::unique_ptr<framework::ir::Graph>(
      new framework::ir::Graph(*program_desc_));

  for (size_t i = 0; i < main_graph_->SubGraphsSize(); ++i) {
    auto* graph = main_graph_->GetSubGraph(i);
    graphes_.push_back(graph);

    for (auto* node : graph->Nodes()) {
      if (!node->IsVar()) continue;
      if (!name2node_.count(node->Name())) {
        name2node_[node->Name()] = node;
      }
    }
  }

  // Remove all control var
  IrInferCleanGraphPass pass;
  Argument arg;
  arg.SetMainGraphNotOwned(main_graph_.get());
  pass.Run(&arg);

  ProcessCircleCases();
}

// Find var names which in circles.
void ConvertToMixedPrecisionPass::ProcessCircleCases() {
  std::vector<std::string> vars_in_circles;
  for (size_t idx = 0; idx < program_desc_->Size(); ++idx) {
    for (auto* op : program_desc_->Block(idx).AllOps()) {
      // TODO(inference): batch_norm has circle, but we need to fuse it in conv
      // op.
      if (op->Type() == "batch_norm") continue;
      const auto& in_names = op->InputArgumentNames();
      const auto& out_names = op->OutputArgumentNames();
      std::set<std::string> in_names_set(in_names.begin(), in_names.end());
      std::set<std::string> out_names_set(out_names.begin(), out_names.end());
      std::set_intersection(in_names_set.begin(),
                            in_names_set.end(),
                            out_names_set.begin(),
                            out_names_set.end(),
                            std::back_inserter(vars_in_circles));
    }
  }

  for (auto& name : vars_in_circles) {
    var_names_in_circles_.insert(name);
  }
  for (auto& name : var_names_in_circles_) {
    LOG(INFO) << name
              << " in circles, so we will skip process those vars and ops.";
  }
}

inline void ProcessConstantOpAttr(framework::ir::Node* op_node,
                                  VarType::Type from_type,
                                  VarType::Type to_type) {
  if (!op_node->IsOp()) return;
  auto op_type = op_node->Op()->Type();
  if (op_type == "feed" || op_type == "fetch") return;

  if (op_type == "fill_constant") {
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("dtype", static_cast<int>(to_type));
  } else if (op_type == "assign_value") {
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("dtype", static_cast<int>(to_type));
  } else if (op_type == "eye") {
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("dtype", static_cast<int>(to_type));
  } else if (op_type == "fill_any_like") {
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("dtype", static_cast<int>(to_type));
  } else if (op_type == "cast") {
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("in_dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("in_dtype", static_cast<int>(to_type));
    if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("out_dtype")) ==
        static_cast<int>(from_type))
      op_node->Op()->SetAttr("out_dtype", static_cast<int>(to_type));
  }
}

void ConvertToMixedPrecisionPass::ConvertAllFp64ToFp32(
    framework::ir::Graph* graph) {
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    ProcessConstantOpAttr(op_node, VarType::FP64, VarType::FP32);
    auto inputs = op_node->inputs;
    for (auto* in_node : inputs) {
      auto* in_var = in_node->Var();
      if (!in_var->Persistable() && in_var->GetDataType() == VarType::FP64) {
        in_var->SetDataType(VarType::FP32);
      }
    }
  }
}

void ConvertToMixedPrecisionPass::Run() {
  LoadAndPrepare();

  for (size_t i = 0; i < graphes_.size(); ++i) {
    auto* graph = graphes_[i];
    VLOG(2) << " --------  handle subgraph " << i << ", has "
            << graph->Nodes().size() << " nodes --------";

    ConvertAllFp64ToFp32(graph);
    ConvertTensorDtype(i);
    FixCastAttr(graph);

    CHECK_EQ(framework::ir::VarDescIsConsistency(*graph), true);
  }

  SaveMixedModel();
}

void ConvertToMixedPrecisionPass::ConvertTensorDtype(BlockID block_idx) {
  auto* graph = graphes_[block_idx];
  VarType::Type to_type;
  if (mixed_precision_ == phi::DataType::FLOAT16) {
    to_type = VarType::FP16;
  } else if (mixed_precision_ == phi::DataType::BFLOAT16) {
    to_type = VarType::BF16;
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only "
        "support fp16 and bf16.",
        static_cast<int>(mixed_precision_)));
  }

  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  auto* block_desc = op_nodes[0]->Op()->Block();
  int num_low_precision = 0;
  std::vector<framework::ir::Node*> output_nodes;

  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    VLOG(3) << "-------------------- op_type " << op_type << ", phi_type "
            << phi::TransToPhiKernelName(op_type);
    // 1. set input dtype.
    if (op_type == "feed") {
      auto feed_var = op_node->outputs[0]->Var();
      if (!keep_io_types_ && feed_var->GetDataType() == VarType::FP32) {
        feed_var->SetDataType(to_type);
      }
    } else if (op_type == "fetch") {
      auto* fetch_var = op_node->inputs[0];
      output_nodes.push_back(fetch_var);
      continue;
    } else if (op_type == "cast") {
      continue;
    }

    // We can not add cast operator before ops who have sub_block, as in
    // sub_block we may get a var which may be transformer by cast op.
    else if (op_node->Op()->HasAttr("sub_block")) {  // NOLINT
      continue;
    }

    // 2. if op support fp16/bf16 and not in blacklist.
    //      - cast weight to fp16/bf16.
    //      - add cast op if the input dtype is not fp16/bf16.
    //      - set output dtype.
    else if (black_list_.count(op_type) == 0) {  // NOLINT
      bool support_precision =
          OpSupportPrecision(op_type, backend_, mixed_precision_, black_list_);

      // If op's output in circle, we should not convert to fp16.
      for (auto* out_node : op_node->outputs) {
        if (var_names_in_circles_.count(out_node->Name())) {
          support_precision = false;
          VLOG(2) << " op's output " << out_node->Name()
                  << " is in circle, we can not support this case, just skip.";
          break;
        }
      }

      // If the op has no input or output of float type, we will not choose the
      // low precision kernel.
      if (support_precision) {
        bool has_float_in_out{false};
        for (auto* in_node : op_node->inputs) {
          if (!in_node->IsVar()) continue;
          if (in_node->Var()->GetType() != VarType::LOD_TENSOR) {
            support_precision = false;
            VLOG(2) << " op has tensor array input[" << in_node->Name()
                    << "], just skip.";
            break;
          }
          auto* real_node = GetRealVarNode(in_node);
          if (real_node->Var()->GetDataType() == VarType::FP16 ||
              real_node->Var()->GetDataType() == VarType::FP32 ||
              real_node->Var()->GetDataType() == VarType::FP64 ||
              real_node->Var()->GetDataType() == VarType::BF16) {
            has_float_in_out = true;
            break;
          }
        }
        for (auto* out_node : op_node->outputs) {
          if (!out_node->IsVar()) continue;
          auto* real_node = GetRealVarNode(out_node);
          if (real_node->Var()->GetDataType() == VarType::FP16 ||
              real_node->Var()->GetDataType() == VarType::FP32 ||
              real_node->Var()->GetDataType() == VarType::FP64 ||
              real_node->Var()->GetDataType() == VarType::BF16) {
            has_float_in_out = true;
            break;
          }
        }

        if (!has_float_in_out) {
          support_precision = false;
          VLOG(2) << " op doesn't has float input and output, just skip.";
        }
      }

      VLOG(2) << "op type: " << op_type
              << " support low precision: " << support_precision;

      if (support_precision) {
        ProcessConstantOpAttr(op_node, VarType::FP32, to_type);
        VLOG(2) << " process input nodes:";
        ++num_low_precision;
        auto inputs = op_node->inputs;
        for (auto* in_node : inputs) {
          ProcessInputNode(
              true, in_node, op_node, &suffix_, block_desc, to_type, block_idx);
        }

        VLOG(2) << " process output nodes:";
        auto outputs = op_node->outputs;
        for (auto* out_node : outputs) {
          ProcessOutputNode(block_idx, out_node, to_type);
        }
      } else {
        auto inputs = op_node->inputs;
        for (auto* in_node : inputs) {
          ProcessInputNode(false,
                           in_node,
                           op_node,
                           &suffix_,
                           block_desc,
                           VarType::FP32,
                           block_idx);
        }
      }
    }

    // 3. check op not support fp16/bf16 or in blacklist.
    //      - add cast op if the input dtype is not fp32.
    else {  // NOLINT
      VLOG(3) << "not to run fp16 op_type: " << op_type << ", node input size "
              << op_node->inputs.size();
      auto in_nodes = op_node->inputs;
      for (auto* in_node : in_nodes) {
        auto* in_var = in_node->Var();
        if (in_var->GetDataType() == to_type) {
          AddCastOp(graph,
                    in_node,
                    op_node,
                    to_type,
                    VarType::FP32,
                    &suffix_,
                    block_desc,
                    &cast_map_);
          VLOG(3) << "-- " << in_node->Name() << "(" << to_type << ") to "
                  << cast_map_[in_node]->Name() << "(" << VarType::FP32 << ")";
        }
      }
    }
  }

  // 4. if output_op's dtype is not compatible to output dtype, then just
  // insert cast.
  for (auto* node : output_nodes) {
    framework::ir::Node* fetch_op{nullptr};
    for (auto* op_node : node->outputs) {
      if (op_node->IsOp() && op_node->Op()->Type() == "fetch") {
        fetch_op = op_node;
      }
    }
    CHECK_NOTNULL(fetch_op);
    auto* var = node->Var();
    if (keep_io_types_ && var->GetDataType() == to_type) {
      // fp16/bf16 -> fp32.
      AddCastOp(graph,
                node,
                fetch_op,
                to_type,
                VarType::FP32,
                &suffix_,
                block_desc,
                &cast_map_);
    } else if (!keep_io_types_ && var->GetDataType() == VarType::FP32) {
      // fp32 -> fp16/bf16
      AddCastOp(graph,
                node,
                fetch_op,
                VarType::FP32,
                to_type,
                &suffix_,
                block_desc,
                &cast_map_);
    }
  }

  if (num_low_precision)
    LOG(INFO) << "---  detected " << num_low_precision
              << " low precision ops in " << block_idx << " subgraph";
}

// We modify op's input output precision, and we need to fix cast op in_dtype
// and out_dtype attribute.
// TODO(inference): we need a cast elimination pass.
void ConvertToMixedPrecisionPass::FixCastAttr(framework::ir::Graph* graph) {
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type != "cast") continue;
    auto input = op_node->inputs[0];
    auto output = op_node->outputs[0];
    op_node->Op()->SetAttr("in_dtype",
                           static_cast<int>(input->Var()->GetDataType()));
    op_node->Op()->SetAttr("out_dtype",
                           static_cast<int>(output->Var()->GetDataType()));
  }
}

void ConvertToMixedPrecisionPass::SaveMixedModel() {
  framework::ProgramDesc mixed_program_desc;
  framework::ir::GraphToProgram(*main_graph_, &mixed_program_desc);

  auto parameters = scope_.LocalVarNames();
  std::sort(parameters.begin(), parameters.end());

  std::unordered_set<std::string> weights_should_be_fp32;
  for (auto* node : main_graph_->Nodes()) {
    if (!node->IsVar()) continue;
    if (VarNodeHasDtype(node)) {
      if (node->Var()->Persistable() &&
          node->Var()->GetDataType() == VarType::FP32) {
        VLOG(2) << "weights keep to fp32: " << node->Name() << ", ptr "
                << reinterpret_cast<void*>(node->Var());
        weights_should_be_fp32.insert(node->Name());
      }
    }
  }

#define CONVERT_TENSOR_DTYPE(DTYPE, dtype)                                   \
  mixed_tensor.set_type(DTYPE);                                              \
  auto* mixed_data = mixed_tensor.mutable_data<dtype>(platform::CPUPlace()); \
  for (int64_t i = 0; i < origin_tensor->numel(); i++) {                     \
    mixed_data[i] = static_cast<dtype>(origin_data[i]);                      \
  }                                                                          \
  origin_tensor->clear();                                                    \
  paddle::framework::TensorCopySync(                                         \
      mixed_tensor, platform::CPUPlace(), origin_tensor)

  for (const auto& param_name : parameters) {
    if (weights_should_be_fp32.count(param_name)) continue;
    auto* var = scope_.FindLocalVar(param_name);
    if (var->IsType<phi::DenseTensor>()) {
      auto* origin_tensor = var->GetMutable<phi::DenseTensor>();
      if (origin_tensor->dtype() != phi::DataType::FLOAT32) continue;
      phi::DenseTensor mixed_tensor;
      mixed_tensor.Resize(origin_tensor->dims());
      auto* origin_data =
          origin_tensor->mutable_data<float>(platform::CPUPlace());
      if (mixed_precision_ == phi::DataType::FLOAT16) {
        CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::FLOAT16,
                             phi::dtype::float16);
      } else if (mixed_precision_ == phi::DataType::BFLOAT16) {
        CONVERT_TENSOR_DTYPE(paddle::experimental::DataType::BFLOAT16,
                             phi::dtype::bfloat16);
      }
    }
  }

#undef CONVERT_TENSOR_DTYPE

  auto SerializeParams = [&]() -> std::string {
    std::ostringstream os;
    phi::CPUContext ctx;
    for (const auto& param : parameters) {
      PADDLE_ENFORCE_NOT_NULL(
          scope_.FindVar(param),
          platform::errors::NotFound(
              "Block should already have a '%s' variable", param));
      auto* tensor = scope_.FindVar(param)->GetMutable<phi::DenseTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    return os.str();
  };

  auto StrToBinary = [](const std::string& path, const std::string& str) {
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());
    file.close();
  };

  StrToBinary(mixed_model_file_,
              mixed_program_desc.Proto()->SerializeAsString());
  StrToBinary(mixed_params_file_, SerializeParams());
}
}  // namespace

void AddCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* node,
    framework::ir::Node* next_op,
    VarType::Type from_type,
    VarType::Type to_type,
    int* suffix,
    framework::BlockDesc* block_desc,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* map) {
  auto update_cast_desc = [&](framework::OpDesc& desc,
                              const std::string& x_name,
                              const std::string& out_name,
                              const int in_dtype,
                              const int out_dtype) {
    desc.SetType("cast");
    desc.SetInput("X", {x_name});
    desc.SetOutput("Out", {out_name});
    desc.SetAttr("in_dtype", in_dtype);
    desc.SetAttr("out_dtype", out_dtype);
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  if (map->count(node) == 0) {
    // insert cast op before node.
    std::string cast_input_name = node->Var()->Name();
    std::string cast_output_name =
        node->Var()->Name() + "_cast.tmp_" + std::to_string((*suffix)++);
    CHECK_NOTNULL(block_desc);
    framework::OpDesc cast_op_desc(block_desc);
    update_cast_desc(cast_op_desc,
                     cast_input_name,
                     cast_output_name,
                     static_cast<int>(from_type),
                     static_cast<int>(to_type));
    auto* cast_op_node = graph->CreateOpNode(&cast_op_desc);
    auto* cast_output_vardesc = block_desc->Var(cast_output_name);
    cast_output_vardesc->SetPersistable(false);
    cast_output_vardesc->SetDataType(to_type);
    cast_output_vardesc->SetShape(node->Var()->GetShape());
    auto* cast_output_node = graph->CreateVarNode(cast_output_vardesc);
    IR_NODE_LINK_TO(cast_op_node, cast_output_node);
    (*map)[node] = cast_output_node;
  }
  next_op->Op()->Rename(node->Name(), map->at(node)->Name());
  IR_NODE_LINK_TO(node, map->at(node)->inputs[0]);
  IR_NODE_UNLINK(node, next_op);
  IR_NODE_LINK_TO(map->at(node), next_op);
}

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool support_precision = false;
  if (blacklist.count(op_type) == 0) {
    if (backend == phi::Backend::GPU)
      support_precision = GpuKernelSupportPrecision(op_type, precision);
    else
      support_precision =
          PhiKernelSupportPrecision(phi_op_type, backend, precision);
  }
  return support_precision;
}

void ConvertToMixedPrecision(
    const std::string& model_file,
    const std::string& params_file,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    phi::DataType mixed_precision,
    phi::Backend backend,
    bool keep_io_types,
    const std::unordered_set<std::string>& black_list) {
  ConvertToMixedPrecisionPass pass(model_file,
                                   params_file,
                                   mixed_model_file,
                                   mixed_params_file,
                                   mixed_precision,
                                   backend,
                                   keep_io_types,
                                   black_list);
  pass.Run();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
