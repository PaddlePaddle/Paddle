/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/framework/variable.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/onednn_helper.h"
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/operators/cuda_graph_with_in_out.h"
#endif
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_mkldnn);

namespace paddle {
namespace operators {

using StepScopeVar = std::vector<framework::Scope *>;
using BlockDesc = framework::BlockDesc;
using ProgramDesc = framework::ProgramDesc;

using Variable = framework::Variable;
using SelectedRows = phi::SelectedRows;

namespace details {

// all input vars should be phi::DenseTensor & is initialized
static void CheckInputVarStatus(const Variable &var,
                                const std::string &var_name) {
  PADDLE_ENFORCE_EQ(var.IsType<phi::DenseTensor>(),
                    true,
                    phi::errors::InvalidArgument(
                        "The input variable %s of "
                        "RunProgram(Grad)Op holds "
                        "wrong type. Expect type is phi::DenseTensor, but "
                        "receive type is %s.",
                        var_name,
                        platform::demangle(framework::ToTypeName(var.Type()))));
  PADDLE_ENFORCE_EQ(
      var.Get<phi::DenseTensor>().IsInitialized(),
      true,
      phi::errors::InvalidArgument("The tensor in input variable %s of "
                                   "RunProgram(Grad)Op "
                                   "is not initialized.",
                                   var_name));
}

static void CheckOutputVarStatus(const Variable &src_var,
                                 const Variable &dst_var,
                                 const std::string &var_name) {
  if (dst_var.IsType<phi::DenseTensor>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<phi::DenseTensor>(),
        true,
        phi::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is phi::DenseTensor, but receive type is "
            "%s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<phi::DenseTensor>().IsInitialized(),
                      true,
                      phi::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          var_name));
  } else if (dst_var.IsType<phi::SelectedRows>()) {
    PADDLE_ENFORCE_EQ(
        src_var.IsType<phi::SelectedRows>(),
        true,
        phi::errors::InvalidArgument(
            "The output variable %s get from "
            "RunProgram(Grad)Op's internal scope holds "
            "wrong type. Expect type is SelectedRows, but receive type is %s.",
            var_name,
            platform::demangle(framework::ToTypeName(src_var.Type()))));
    PADDLE_ENFORCE_EQ(src_var.Get<phi::SelectedRows>().value().IsInitialized(),
                      true,
                      phi::errors::InvalidArgument(
                          "The tensor in output variable %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          var_name));

  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type phi::DenseTensor or SelectedRows, "
        "but received variable %s's type is %s",
        var_name,
        platform::demangle(framework::ToTypeName(dst_var.Type()))));
  }
}

static void VariableShare(const Variable &src_var, Variable *dst_var) {
  // The previous check ensures that the variable type can only be
  // phi::DenseTensor or SelectedRows.
  if (src_var.IsType<phi::DenseTensor>()) {
    auto *lod_tensor = dst_var->GetMutable<phi::DenseTensor>();
    lod_tensor->ShareDataWith(src_var.Get<phi::DenseTensor>());
    lod_tensor->set_lod(src_var.Get<phi::DenseTensor>().lod());
  } else if (src_var.IsType<phi::SelectedRows>()) {
    auto *selected_rows = dst_var->GetMutable<phi::SelectedRows>();
    selected_rows->mutable_value()->ShareDataWith(
        src_var.Get<phi::SelectedRows>().value());
    selected_rows->set_rows(src_var.Get<phi::SelectedRows>().rows());
    selected_rows->set_height(src_var.Get<phi::SelectedRows>().height());
  }
}

static void ShareVarsIntoScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    if (var_names[i] == framework::kFakeVarName ||
        var_names[i] == paddle::framework::kEmptyVarName) {
      continue;
    }
    auto *var = scope->Var(var_names[i]);
    CheckInputVarStatus(*vars[i], var_names[i]);
    VariableShare(*vars[i], var);
  }
}

static void ShareVarsFromScope(const std::vector<Variable *> &vars,
                               const std::vector<std::string> &var_names,
                               const BlockDesc &global_block,
                               framework::Scope *scope) {
  for (size_t i = 0; i < vars.size(); ++i) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't findthem in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    if (var_names[i] == framework::kEmptyVarName ||
        var_names[i] == framework::kFakeVarName ||
        !global_block.HasVar(var_names[i])) {
      VLOG(2) << "find variable name is " << var_names[i] << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(var_names[i]);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        phi::errors::NotFound("The output variable %s is not in "
                              "RunProgram(Grad)Op'"
                              "s internal scope.",
                              var_names[i]));
    CheckOutputVarStatus(*var, *vars[i], var_names[i]);
    VariableShare(*var, vars[i]);
  }
}

#ifdef PADDLE_WITH_CUDA
static cudaStreamCaptureMode StringToCUDAGraphCaptureMode(
    const std::string &mode) {
  if (mode == "global") {
    return cudaStreamCaptureModeGlobal;
  } else if (mode == "thread_local") {
    return cudaStreamCaptureModeThreadLocal;
  } else if (mode == "relaxed") {
    return cudaStreamCaptureModeRelaxed;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported CUDA Graph capture mode %s", mode));
  }
}
#elif defined(PADDLE_WITH_HIP)
static hipStreamCaptureMode StringToCUDAGraphCaptureMode(
    const std::string &mode) {
  if (mode == "global") {
    return hipStreamCaptureModeGlobal;
  } else if (mode == "thread_local") {
    return hipStreamCaptureModeThreadLocal;
  } else if (mode == "relaxed") {
    return hipStreamCaptureModeRelaxed;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Unsupported CUDA Graph capture mode %s", mode));
  }
}
#endif

}  // namespace details

template <typename T, typename DeviceContext>
class RunProgramOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(phi::errors::InvalidArgument("Not supported yet!"));
    const auto &capture_mode = ctx.Attr<std::string>("cuda_graph_capture_mode");
    if (capture_mode.empty()) {
      return;
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto is_test = ctx.Attr<bool>("is_test");
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU,
        true,
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
    auto *graph_var = ctx.OutputVar("CUDAGraph");
    PADDLE_ENFORCE_NOT_NULL(
        graph_var,
        phi::errors::InvalidArgument("Output(CUDAGraph) must exist when "
                                     "cuda_graph_capture_mode is valid."));
    using GraphVecType = std::vector<std::unique_ptr<CUDAGraphWithInOuts>>;
    auto &inner_graphs = *(graph_var->GetMutable<GraphVecType>());
    inner_graphs.resize(std::max<size_t>(3, inner_graphs.size()));
    size_t graph_idx = is_test ? 0 : 1;
    if (inner_graphs[graph_idx].get() == nullptr) {
      VLOG(10) << "Capture Forward CUDA Graph";
    } else {
      VLOG(10) << "Run Forward CUDA Graph directly";
      ExecuteCUDAGraph(
          ctx, {"X"}, {"Out", "DOut"}, inner_graphs[graph_idx].get());
    }
#else
    PADDLE_THROW(
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
#endif
  }
};

template <typename T, typename DeviceContext>
class RunProgramGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW(phi::errors::InvalidArgument("Not supported yet!"));
    const auto &capture_mode = ctx.Attr<std::string>("cuda_graph_capture_mode");
    if (capture_mode.empty()) {
      return;
    }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU,
        true,
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
    auto *graph_var =
        const_cast<framework::Variable *>(ctx.InputVar("CUDAGraph"));
    PADDLE_ENFORCE_NOT_NULL(
        graph_var,
        phi::errors::InvalidArgument("Output(CUDAGraph) must exist when "
                                     "cuda_graph_capture_mode is valid."));
    auto &inner_graphs = *(
        graph_var
            ->GetMutable<std::vector<std::unique_ptr<CUDAGraphWithInOuts>>>());
    const size_t graph_idx = 2;
    if (inner_graphs[graph_idx].get() == nullptr) {
      VLOG(10) << "Capture Backward CUDA Graph";
    } else {
      ExecuteCUDAGraph(ctx,
                       {framework::GradVarName("Out")},
                       {framework::GradVarName("X")},
                       inner_graphs[graph_idx].get());
      VLOG(10) << "Run Backward CUDA Graph directly";
    }
#else
    PADDLE_THROW(
        phi::errors::InvalidArgument("The cuda_graph_capture_mode is only "
                                     "valid when using NVIDIA GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
