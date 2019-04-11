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

/*
 * This file defines the class Argument, which is the input and output of the
 * analysis module. All the fields that needed either by Passes or PassManagers
 * are contained in Argument.
 *
 * TODO(Superjomn) Find some way better to contain the fields when it grow too
 * big.
 */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Graph;

#ifdef PADDLE_WITH_MKLDNN
using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, framework::LoDTensor>>;
#endif

/*
 * The argument definition of both Pass and PassManagers.
 *
 * All the fields should be registered here for clearness.
 */
struct Argument {
  Argument() = default;
  explicit Argument(const std::string& model_dir) { SetModelDir(model_dir); }

  using unique_ptr_t = std::unique_ptr<void, std::function<void(void*)>>;
  using fusion_statis_t = std::unordered_map<std::string, int>;
  using engine_opt_info_t = std::map<std::string, std::string>;
  using anakin_max_shape_t = std::map<std::string, std::vector<int>>;

  bool Has(const std::string& key) const { return valid_fields_.count(key); }

#define DECL_ARGUMENT_FIELD(field__, Field, type__)          \
 public:                                                     \
  type__& field__() {                                        \
    PADDLE_ENFORCE(Has(#field__), "There is no such field"); \
    return field__##_;                                       \
  }                                                          \
  void Set##Field(const type__& x) {                         \
    field__##_ = x;                                          \
    valid_fields_.insert(#field__);                          \
  }                                                          \
  DECL_ARGUMENT_FIELD_VALID(field__);                        \
  type__* field__##_ptr() { return &field__##_; }            \
                                                             \
 private:                                                    \
  type__ field__##_;

#define DECL_ARGUMENT_FIELD_VALID(field__) \
  bool field__##_valid() { return Has(#field__); }

#define DECL_ARGUMENT_UNIQUE_FIELD(field__, Field, type__)                \
 public:                                                                  \
  type__& field__() {                                                     \
    PADDLE_ENFORCE_NOT_NULL(field__##_);                                  \
    PADDLE_ENFORCE(Has(#field__));                                        \
    return *static_cast<type__*>(field__##_.get());                       \
  }                                                                       \
  void Set##Field(type__* x) {                                            \
    field__##_ =                                                          \
        unique_ptr_t(x, [](void* x) { delete static_cast<type__*>(x); }); \
    valid_fields_.insert(#field__);                                       \
  }                                                                       \
  void Set##Field##NotOwned(type__* x) {                                  \
    valid_fields_.insert(#field__);                                       \
    field__##_ = unique_ptr_t(x, [](void* x) {});                         \
  }                                                                       \
  DECL_ARGUMENT_FIELD_VALID(field__);                                     \
  type__* field__##_ptr() {                                               \
    PADDLE_ENFORCE(Has(#field__));                                        \
    return static_cast<type__*>(field__##_.get());                        \
  }                                                                       \
  type__* Release##Field() {                                              \
    PADDLE_ENFORCE(Has(#field__));                                        \
    valid_fields_.erase(#field__);                                        \
    return static_cast<type__*>(field__##_.release());                    \
  }                                                                       \
                                                                          \
 private:                                                                 \
  unique_ptr_t field__##_;

  DECL_ARGUMENT_FIELD(predictor_id, PredictorID, int);
  // Model path
  DECL_ARGUMENT_FIELD(model_dir, ModelDir, std::string);
  // Model specified with program and parameters files.
  DECL_ARGUMENT_FIELD(model_program_path, ModelProgramPath, std::string);
  DECL_ARGUMENT_FIELD(model_params_path, ModelParamsPath, std::string);
  DECL_ARGUMENT_FIELD(model_from_memory, ModelFromMemory, bool);
  DECL_ARGUMENT_FIELD(engine_opt_info, EngineOptInfo, engine_opt_info_t);

  // The overall graph to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(main_graph, MainGraph, framework::ir::Graph);
  // The overall Scope to work on.
  DECL_ARGUMENT_UNIQUE_FIELD(scope, Scope, framework::Scope);

  // The default program, loaded from disk.
  DECL_ARGUMENT_UNIQUE_FIELD(main_program, MainProgram, framework::ProgramDesc);

  // The ir passes to perform in analysis phase.
  DECL_ARGUMENT_FIELD(ir_analysis_passes, IrAnalysisPasses,
                      std::vector<std::string>);
  DECL_ARGUMENT_FIELD(analysis_passes, AnalysisPasses,
                      std::vector<std::string>);

  // Pass a set of op types to enable its mkldnn kernel
  DECL_ARGUMENT_FIELD(mkldnn_enabled_op_types, MKLDNNEnabledOpTypes,
                      std::unordered_set<std::string>);

#ifdef PADDLE_WITH_MKLDNN
  // A set of op types to enable their quantized kernels
  DECL_ARGUMENT_FIELD(quantize_enabled_op_types, QuantizeEnabledOpTypes,
                      std::unordered_set<std::string>);

  // A set of op IDs to exclude from enabling their quantized kernels
  DECL_ARGUMENT_FIELD(quantize_excluded_op_ids, QuantizeExcludedOpIds,
                      std::unordered_set<int>);

  // Scales for variables to be quantized
  DECL_ARGUMENT_FIELD(quant_var_scales, QuantVarScales, VarQuantScale);
#endif

  // Passed from config.
  DECL_ARGUMENT_FIELD(use_gpu, UseGPU, bool);
  DECL_ARGUMENT_FIELD(gpu_device_id, GPUDeviceId, int);
  DECL_ARGUMENT_FIELD(use_tensorrt, UseTensorRT, bool);
  DECL_ARGUMENT_FIELD(tensorrt_max_batch_size, TensorRtMaxBatchSize, int);
  DECL_ARGUMENT_FIELD(tensorrt_workspace_size, TensorRtWorkspaceSize, int);
  DECL_ARGUMENT_FIELD(tensorrt_min_subgraph_size, TensorRtMinSubgraphSize, int);
  DECL_ARGUMENT_FIELD(tensorrt_precision_mode, TensorRtPrecisionMode,
                      AnalysisConfig::Precision);
  DECL_ARGUMENT_FIELD(tensorrt_use_static_engine, TensorRtUseStaticEngine,
                      bool);

  DECL_ARGUMENT_FIELD(anakin_max_input_shape, AnakinMaxInputShape,
                      anakin_max_shape_t);
  DECL_ARGUMENT_FIELD(anakin_max_batch_size, AnakinMaxBatchSize, int);
  DECL_ARGUMENT_FIELD(anakin_min_subgraph_size, AnakinMinSubgraphSize, int);
  DECL_ARGUMENT_FIELD(use_anakin, UseAnakin, bool);

  // Memory optimized related.
  DECL_ARGUMENT_FIELD(enable_memory_optim, EnableMemoryOptim, bool);
  DECL_ARGUMENT_FIELD(static_memory_optim, StaticMemoryOptim, bool);
  DECL_ARGUMENT_FIELD(static_memory_optim_force_update,
                      StaticMemoryOptimForceUpdate, bool);
  // Indicate which kind of sort algorithm is used for operators, the memory
  // optimization relays on the sort algorithm.
  DECL_ARGUMENT_FIELD(memory_optim_sort_kind, MemoryOptimSortKind, int);

  // The program transformed by IR analysis phase.
  DECL_ARGUMENT_UNIQUE_FIELD(ir_analyzed_program, IrAnalyzedProgram,
                             framework::proto::ProgramDesc);

  DECL_ARGUMENT_FIELD(fusion_statis, FusionStatis, fusion_statis_t);

 private:
  std::unordered_set<std::string> valid_fields_;
};

#define ARGUMENT_CHECK_FIELD(argument__, fieldname__) \
  PADDLE_ENFORCE(argument__->Has(#fieldname__),       \
                 "the argument field [%s] should be set", #fieldname__);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
