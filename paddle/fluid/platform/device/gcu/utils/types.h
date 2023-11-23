/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_GCU

#include <cstddef>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/phi/common/data_type.h"

#include "dtu/hlir/builder/hlir_builder.h"
#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_memory.h"

namespace builder {
class Op;
class Builder;
class PrimitiveType;
}  // namespace builder

namespace hlir {
class Module;
}  // namespace hlir

namespace paddle {
namespace platform {
namespace gcu {
#ifndef GCU_MEM_SECUREC_MAX_LEN
#define GCU_MEM_SECUREC_MAX_LEN (0x7fffffffUL)
#endif

class GcuTransInfo;

using Graph = paddle::framework::ir::Graph;
using GraphPtr = const paddle::framework::ir::Graph*;
using Tensor = phi::DenseTensor;
using Scope = paddle::framework::Scope;
using ExecutionContext = paddle::framework::ExecutionContext;
using Node = paddle::framework::ir::Node;
using OpDesc = paddle::framework::OpDesc;
using ProgramDesc = paddle::framework::ProgramDesc;
using NodePtr = Node*;
using DataType = phi::DataType;
using PaddleVarType = framework::proto::VarType::Type;
// using TransToProtoVarType = paddle::framework::TransToProtoVarType;

using GcuOp = ::builder::Op;
using GcuOpPtr = std::shared_ptr<GcuOp>;
using GcuPrimitiveType = builder::PrimitiveType;
using GcuType = builder::Type;
using GcuShape = std::vector<int64_t>;
using GcuBuilder = builder::Builder;
using GcuBuilderPtr = std::shared_ptr<builder::Builder>;
using GcuGraphPtr = std::shared_ptr<hlir::Module>;
using GcuMemPtr = std::shared_ptr<paddle::platform::gcu::runtime::Memory>;

// ENV DEFINE
const char* const kDumpTraceBack = "ENFLAME_ENABLE_DUMP_TRACEBACK";
const char* const kProfiler = "PADDLE_GCU_PROFILE";
const char* const kRunningMode = "PADDLE_GCU_RUNNING_MODE";
const char* const kFpBpNoTrans = "PADDLE_GCU_FP_BP_NO_TRANS";

// ATTR DEFINE
const char* const kAttrOpOutVarName = "_op_out_var_name";
const char* const kGcuProgramKey = "_gcu_program_key";
const char* const kGcuGraphOpCategory = "_gcu_graph_op_category";
const char* const kGraphType = "gcu_graph_type";  // for dynamic to static
const char* const kGcuLayoutType = "gcu_layout_type";
const char* const kGcuMainFormat = "gcu_main_format";

// ATTR VALUE
namespace GcuGraphOpCategory {
const char* const OPTIMIZER = "gcu_optimizer";
}

namespace RunningMode {
const char* const SERIAL = "serial";
const char* const ADAPTIVE = "adaptive";
const char* const FORCE_SERIAL = "force_serial";
}  // namespace RunningMode

enum GraphType { FP = 0, BP = 1, OTHER = 2 };

struct PaddleVarDesc {
  PaddleVarDesc() = default;
  PaddleVarDesc(const std::string& name,
                const std::vector<int64_t>& dims,
                PaddleVarType dtype,
                int64_t size)
      : var_name(name), shapes(dims), data_type(dtype), data_size(size) {}
  std::string var_name;
  std::vector<int64_t> shapes;
  PaddleVarType data_type = paddle::framework::proto::VarType::FP32;
  int64_t data_size = 0;
};

struct GcuTensorDesc {
  GcuTensorDesc() = default;
  GcuTensorDesc(const GcuMemPtr& ptr,
                int64_t len,
                const std::vector<int64_t>& dims,
                PaddleVarType dtype,
                bool sub_mem = false,
                int64_t count = 0)
      : data(ptr),
        var_desc("", dims, dtype, len),
        numel(count),
        is_sub_memory(sub_mem) {
    if (numel == 0) {
      numel = 1;
      std::for_each(
          dims.begin(), dims.end(), [&](int64_t dim) { numel *= dim; });
    }
  }
  GcuTensorDesc(const GcuMemPtr& ptr,
                int64_t len,
                int64_t count = 0,
                const std::vector<int64_t>& dims = {})
      : data(ptr),
        var_desc("", dims, paddle::framework::proto::VarType::FP32, len),
        numel(count),
        is_sub_memory(false) {
    if (numel == 0) {
      numel = 1;
      std::for_each(
          dims.begin(), dims.end(), [&](int64_t dim) { numel *= dim; });
    }
  }
  GcuTensorDesc(const GcuMemPtr& ptr,
                const PaddleVarDesc& desc,
                int64_t count = 0)
      : data(ptr), var_desc(desc), numel(count), is_sub_memory(false) {
    if (numel == 0) {
      numel = 1;
      std::for_each(desc.shapes.begin(), desc.shapes.end(), [&](int64_t dim) {
        numel *= dim;
      });
    }
  }
  GcuMemPtr data = nullptr;
  PaddleVarDesc var_desc;
  int64_t numel = 0;
  bool is_sub_memory = false;
};

struct ResourceReflection {
  std::map<size_t, PaddleVarDesc> map_inputs_to_pd_var;
  std::map<size_t, PaddleVarDesc> map_outputs_to_pd_var;
  // key: output idx value:{input idx, size}
  std::map<size_t, std::tuple<size_t, int64_t>> map_ref_out_to_weight;
  void Clear() {
    map_inputs_to_pd_var.clear();
    map_outputs_to_pd_var.clear();
    map_ref_out_to_weight.clear();
  }
};

struct ParamDesc {
  ParamDesc() = default;
  ParamDesc(const std::string& name,
            const std::string& sym,
            const PaddleVarDesc& desc)
      : var_name(name), symbol(sym), var_desc(desc) {}
  std::string var_name;
  std::string symbol;
  PaddleVarDesc var_desc;
};

struct CollectiveParams {
  CollectiveParams() = default;
  CollectiveParams(const std::string& in_name,
                   const std::string& in_sym,
                   const std::string& out_name,
                   const std::string& out_sym,
                   const PaddleVarDesc& in_desc,
                   const PaddleVarDesc& out_desc,
                   bool reuse = true)
      : reuse_input(reuse) {
    in_out_desc[0] = ParamDesc(in_name, in_sym, in_desc);
    in_out_desc[1] = ParamDesc(out_name, out_sym, out_desc);
  }
  ParamDesc in_out_desc[2];
  bool reuse_input = true;
};

using WeightUpdateParams = ParamDesc;

struct GlobalMemoryRef {
  std::unordered_map<size_t, std::vector<std::string>> input_keys;
  std::unordered_map<size_t, std::vector<std::string>> output_keys;
  std::vector<std::string> leaf_outputs;
  std::vector<std::string> leaf_output_keys;
  std::pair<std::vector<std::string>, std::vector<std::string>>
      global_in_out_keys;
  std::vector<std::string> weights;
  std::unordered_set<std::string> weights_to_trans;
  std::unordered_map<std::string, std::string> weight_to_symbol;
  std::unordered_map<std::string, std::string> var_to_symbol;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      symbol_to_vars;
  std::vector<CollectiveParams> allreduce_params;
  std::map<std::string, WeightUpdateParams> weight_update_params;
  std::map<std::string, GcuTransInfo> weights_trans_info;
  bool is_training_graph = true;
  std::string running_mode = RunningMode::SERIAL;
};
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
#endif
