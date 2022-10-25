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
#include "paddle/phi/core/tensor_meta.h"

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
    black_list_.insert("assign");
    black_list_.insert("fill_constant");
    black_list_.insert("assign_value");
    black_list_.insert("eye");
    black_list_.insert("fill_any_like");
    black_list_.insert("fill_constant_batch_size_like");
  }

  void Run();

 private:
  void LoadAndPrepare();
  inline bool NodeVarHasDtype(framework::ir::Node* node);
  void ConvertAllFp64ToFp32(framework::ir::Graph* graph);
  void FixCastAttr(framework::ir::Graph* graph);
  void SaveMixedModel();
  void ConvertTensorDtype(int block_idx);
  void ProcessInputNode(bool support_precision,
                        framework::ir::Node* in_node,
                        framework::ir::Node* op_node,
                        int* suffix,
                        framework::BlockDesc* block_desc,
                        VarType::Type to_type,
                        int block_idx);

  void ProcessOutputNode(int block_idx,
                         framework::ir::Node* var_node,
                         VarType::Type to_type);
  inline bool IsFloatVarType(VarType::Type type);

  bool OutShouldNotConvert(framework::ir::Node* var_node);
  // Just process special cases for weights conversion.
  bool WeightsShouldNotConvert(framework::ir::Node* var_node);

  // To support multi block, we need to consider a lot of special cases.
  // Return Node* which first appers in block.
  framework::ir::Node* GetRealNode(int block_idx, framework::ir::Node* node);
  void FindVarsInMultiBlock();
  inline bool VarIsMultiPrecisionOpsOut(int block_idx,
                                        framework::ir::Node* op_node);

 private:
  // A trick. Patch for strange op, which input name equal to output name, such
  // as `fused_multi_transformer`
  void PatchForStrangeOp();

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

  std::unordered_map<framework::ir::Node*, framework::ir::Node*> cast_map_;
  std::unordered_map<std::string, std::pair<VarType::Type, int>>
      vars_in_multi_block_map_;
  std::vector<std::unordered_map<std::string, std::vector<std::string>>>
      vars_appear_multi_in_one_block_;
  int suffix_{0};

  std::unique_ptr<framework::ProgramDesc> program_desc_{nullptr};
  std::unique_ptr<framework::ir::Graph> main_graph_{nullptr};
  std::vector<framework::ir::Graph*> graphes_;
};

framework::ir::Node* ConvertToMixedPrecisionPass::GetRealNode(
    int block_idx, framework::ir::Node* node) {
  if (vars_in_multi_block_map_.count(node->Name())) {
    int var_origin_block_id = vars_in_multi_block_map_.at(node->Name()).second;
    if (block_idx != var_origin_block_id) {
      auto* graph = graphes_[var_origin_block_id];
      for (auto* nd : graph->Nodes()) {
        if (nd->Name() == node->Name()) {
          return nd;
        }
      }
    }
  }

  return node;
}

inline bool ConvertToMixedPrecisionPass::NodeVarHasDtype(
    framework::ir::Node* node) {
  if (!node->IsVar()) return false;
  auto type = node->Var()->GetType();
  return (type == VarType::SELECTED_ROWS) || (type == VarType::LOD_TENSOR) ||
         (type == VarType::LOD_TENSOR_ARRAY) || (type == VarType::STRINGS) ||
         (type == VarType::VOCAB);
}

// op1(fp32) -> var1, op2(fp16) -> var1
// if and only if op1 and op2 both support fp16, we convert op1 and op2's
// precision.
inline bool ConvertToMixedPrecisionPass::VarIsMultiPrecisionOpsOut(
    int block_idx, framework::ir::Node* op_node) {
  CHECK_EQ(op_node->IsOp(), true);

  bool ret{false};
  for (auto* var_node : op_node->outputs) {
    auto* real_var_node = GetRealNode(block_idx, var_node);
    if (!real_var_node->Var()->Persistable() &&
        vars_appear_multi_in_one_block_[block_idx].count(var_node->Name())) {
      for (const auto& op_type :
           vars_appear_multi_in_one_block_[block_idx].at(var_node->Name())) {
        if (OpSupportPrecision(
                op_type, backend_, mixed_precision_, black_list_)) {
          ret = true;
          VLOG(2) << var_node->Name()
                  << " is multi precision op's out, so we skip convert to fp16";
          break;
        }
      }
    }
    if (ret) break;
  }
  return ret;
}

void ConvertToMixedPrecisionPass::ProcessInputNode(
    bool support_precision,
    framework::ir::Node* in_node,
    framework::ir::Node* op_node,
    int* suffix,
    framework::BlockDesc* block_desc,
    VarType::Type to_type,
    int block_idx) {
  auto* real_node = GetRealNode(block_idx, in_node);
  if (!NodeVarHasDtype(real_node)) return;
  auto graph = graphes_[block_idx];
  bool is_main_block = block_idx == 0;
  auto* in_var = real_node->Var();
  auto in_var_type = in_var->GetDataType();
  auto prev_type = in_var_type;
  bool is_in_multi_block = vars_in_multi_block_map_.count(in_var->Name());

  if (!is_main_block && is_in_multi_block) {
    in_var_type = vars_in_multi_block_map_.at(in_var->Name()).first;
  }
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
    int block_idx, framework::ir::Node* var_node, VarType::Type to_type) {
  auto* real_node = GetRealNode(block_idx, var_node);
  if (!NodeVarHasDtype(real_node)) return;
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

  // Remove all control var
  IrInferCleanGraphPass pass;
  Argument arg;
  arg.SetMainGraphNotOwned(main_graph_.get());
  pass.Run(&arg);

  vars_appear_multi_in_one_block_.resize(program_desc_->Size());
  FindVarsInMultiBlock();
}

void ConvertToMixedPrecisionPass::FindVarsInMultiBlock() {
  std::vector<std::set<std::string>> block_var_names_set(program_desc_->Size());
  for (size_t i = 0; i < program_desc_->Size(); ++i) {
    for (auto* op : program_desc_->Block(i).AllOps()) {
      const auto& in_names = op->InputArgumentNames();
      block_var_names_set[i].insert(in_names.begin(), in_names.end());
      const auto& out_names = op->OutputArgumentNames();
      if (op->HasAttr("sub_block") == false) {
        for (const auto& n : out_names) {
          if (block_var_names_set[i].count(n)) {
            vars_appear_multi_in_one_block_[i][n].push_back(op->Type());
          }
        }
      }
      block_var_names_set[i].insert(out_names.begin(), out_names.end());
    }
  }

  for (size_t i = 0; i < program_desc_->Size() - 1; ++i) {
    for (size_t j = i + 1; j < program_desc_->Size(); ++j) {
      std::set<std::string> vars_in_multi_block;
      std::set_intersection(
          block_var_names_set[i].begin(),
          block_var_names_set[i].end(),
          block_var_names_set[j].begin(),
          block_var_names_set[j].end(),
          std::inserter(vars_in_multi_block, vars_in_multi_block.begin()));

      for (const auto& name : vars_in_multi_block) {
        vars_in_multi_block_map_.emplace(name,
                                         std::make_pair(VarType::FP32, i));
      }
    }
  }
}

void ConvertToMixedPrecisionPass::ConvertAllFp64ToFp32(
    framework::ir::Graph* graph) {
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed" || op_type == "fetch") continue;

    if (op_type == "fill_constant") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("dtype", static_cast<int>(VarType::FP32));
    } else if (op_type == "assign_value") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("dtype", static_cast<int>(VarType::FP32));
    } else if (op_type == "eye") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("dtype", static_cast<int>(VarType::FP32));
    } else if (op_type == "fill_any_like") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("dtype", static_cast<int>(VarType::FP32));
    } else if (op_type == "cast") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("in_dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("in_dtype", static_cast<int>(VarType::FP32));
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("out_dtype")) ==
          static_cast<int>(VarType::FP64))
        op_node->Op()->SetAttr("out_dtype", static_cast<int>(VarType::FP32));
    }

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

  for (size_t i = 0; i < main_graph_->SubGraphsSize(); ++i) {
    auto graph = main_graph_->GetSubGraph(i);
    graphes_.push_back(graph);
    VLOG(2) << " --------  handle subgraph " << i << ", has "
            << graph->Nodes().size() << " nodes --------";

    ConvertAllFp64ToFp32(graph);
    ConvertTensorDtype(i);
    FixCastAttr(graph);

    // A trick
    PatchForStrangeOp();

    CHECK_EQ(framework::ir::VarDescIsConsistency(*graph), true);
  }

  SaveMixedModel();
}

void ConvertToMixedPrecisionPass::ConvertTensorDtype(int block_idx) {
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

    else if (op_node->Op()->HasAttr("sub_block")) {  // NOLINT
      // sub_block op's output dtype should be same as input dtype, if have the
      // same name.
      std::unordered_map<std::string, framework::ir::Node*> in_name_to_node;
      for (auto* in : op_node->inputs) {
        auto* real_node = GetRealNode(block_idx, in);
        if (NodeVarHasDtype(real_node)) {
          in_name_to_node[in->Name()] = in;
        }
      }

      for (auto* out : op_node->outputs) {
        auto* real_node = GetRealNode(block_idx, out);
        if (NodeVarHasDtype(real_node)) {
          if (in_name_to_node.count(out->Name()))
            real_node->Var()->SetDataType(
                in_name_to_node[out->Name()]->Var()->GetDataType());
        }
      }

      continue;
    }

    // 2. if op support fp16/bf16 and not in blacklist.
    //      - cast weight to fp16/bf16.
    //      - add cast op if the input dtype is not fp16/bf16.
    //      - set output dtype.
    //
    // If a var(op's out var) appears multiple times in a block, we should not
    // convert to fp16.
    else if (black_list_.count(op_type) == 0 &&  // NOLINT
             !VarIsMultiPrecisionOpsOut(block_idx, op_node)) {
      bool support_precision =
          OpSupportPrecision(op_type, backend_, mixed_precision_, black_list_);

      // if op not has float input, we will not choose the low precision kernel.
      {
        bool has_float_input{false};
        for (auto in_node : op_node->inputs) {
          auto* real_node = GetRealNode(block_idx, in_node);
          if (real_node->Var()->GetDataType() == VarType::FP16 ||
              real_node->Var()->GetDataType() == VarType::FP32 ||
              real_node->Var()->GetDataType() == VarType::FP64 ||
              real_node->Var()->GetDataType() == VarType::BF16) {
            has_float_input = true;
            break;
          }
        }
        if (!has_float_input) {
          support_precision = false;
          VLOG(2) << " op doesn't has float input, just skip.";
        }
      }
      VLOG(2) << " support low precision " << support_precision;

      if (support_precision) {
        VLOG(2) << " process input nodes:";
        ++num_low_precision;
        auto inputs = op_node->inputs;

        // Just for paddle's terriable case: op's input and output has the same
        // name.
        std::unordered_map<std::string, std::string> names_map;
        for (auto* out_node : op_node->outputs) {
          for (auto* in_node : op_node->inputs) {
            if (out_node->Name() == in_node->Name()) {
              names_map[out_node->Name()] = in_node->Name();
            }
          }
        }

        // Process inputs.
        for (auto* in_node : inputs) {
          ProcessInputNode(
              true, in_node, op_node, &suffix_, block_desc, to_type, block_idx);
          if (names_map.count(in_node->Name()) && cast_map_.count(in_node)) {
            names_map[in_node->Name()] = cast_map_[in_node]->Name();
          }
        }
        VLOG(2) << " process output nodes:";
        // Process outputs.
        for (auto* out_node : op_node->outputs) {
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
      VLOG(3) << "not to run fp16 op_type: " << op_type;
      for (auto* in_node : op_node->inputs) {
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

  for (auto* node : graph->Nodes()) {
    auto* real_node = GetRealNode(block_idx, node);
    if (!NodeVarHasDtype(real_node)) continue;

    if (vars_in_multi_block_map_.count(real_node->Name()) &&
        vars_in_multi_block_map_.at(real_node->Name()).second == block_idx) {
      vars_in_multi_block_map_.at(real_node->Name()).first =
          real_node->Var()->GetDataType();
    }
  }

  if (num_low_precision)
    LOG(INFO) << "---  detected " << num_low_precision
              << " low precision ops in " << block_idx << " subgraph";
}

// We modify op's input output precision, and we need to fix cast op in_dtype
// and out_dtype attribute.
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
    if (!(node->IsVar())) continue;
    if (NodeVarHasDtype(node)) {
      if (node->Var()->Persistable() &&
          node->Var()->GetDataType() == VarType::FP32) {
        VLOG(2) << "weights keep to fp32: " << node->Name();
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
      VLOG(3) << "Serialize param: " << param;
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

void ConvertToMixedPrecisionPass::PatchForStrangeOp() {
  for (auto* graph : graphes_) {
    for (auto op_node : framework::ir::TopologySortOperations(*graph)) {
      if (op_node->Name() == "fused_multi_transformer") {
        auto cache_kv_inputs = op_node->Op()->Input("CacheKV");
        auto cache_kv_outputs = op_node->Op()->Output("CacheKVOut");
        CHECK_EQ(cache_kv_inputs.size(), cache_kv_outputs.size());
        for (size_t i = 0; i < cache_kv_inputs.size(); ++i) {
          op_node->Op()->RenameOutput(cache_kv_outputs[i], cache_kv_inputs[i]);
        }
      }
    }
  }
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
