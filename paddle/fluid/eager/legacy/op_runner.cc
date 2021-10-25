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

#include "paddle/fluid/eager/legacy/op_runner.h"
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/eager/legacy/amp_auto_cast.h"
#include "paddle/fluid/eager/legacy/prepared_operator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/denormal.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(use_mkldnn);
DECLARE_string(tracer_mkldnn_ops_on);
DECLARE_string(tracer_mkldnn_ops_off);

namespace egr {

void OpRunImpl(const framework::OperatorBase& op, const NameTensorMap& ins,
               const NameTensorMap& outs, const framework::AttributeMap& attrs,
               const framework::AttributeMap& default_attrs,
               const platform::Place& place) {
  auto* op_kernel = dynamic_cast<const framework::OperatorWithKernel*>(&op);
  PADDLE_ENFORCE_NOT_NULL(
      op_kernel, platform::errors::PermissionDenied(
                     "Only support operator with kernel in Dygraph mode."));
  auto& info = op.Info();
  if (info.infer_var_type_) {
    TensorRuntimeInferVarTypeContext infer_var_type_ctx(ins, outs, attrs,
                                                        default_attrs);
    info.infer_var_type_(&infer_var_type_ctx);
  }

  // Initialize output tensor
  for (auto& tensor_pair : outs) {
    for (auto& tensor : tensor_pair.second) {
      if (tensor) {
        InitializeTensor(tensor);
      }
    }
  }

  /**
   * [ Why need temporary inputs here? ]
   *
   * PrepareData should not change original input tensor inplace.
   * Suppose the user defines a tensor(int), enters an op to execute,
   * and then this op rewrites GetExpectedKernelForVar, and converts
   * this tensor to float type during execution. After the dynamic
   * graph is executed, the user-defined variable will be lost, and
   * the user cannot get the originally defined int tensor, because
   * it has been converted to float, this should be regarded as a bug
   * in certain usage scenarios
   *
   * In static graph mode, when op is executed, a temporary scope
   * `transfer_scope` is created before PrepareData, the data after
   * transform is stored in the temporary scope, and then discarded
   * after the execution of op, but the original input is directly
   * overwritten in the previous dynamic graph implemention.
   */
  auto prepared_op =
      PreparedOp::Prepare(ins, outs, *op_kernel, place, attrs, default_attrs);
  auto tmp_ins_ptr =
      PrepareData<VarType>(*op_kernel, ins, prepared_op.kernel_type());
  if (tmp_ins_ptr == nullptr) {
    prepared_op.Run(ins, outs, attrs, default_attrs);
  } else {
    prepared_op.Run(*tmp_ins_ptr, outs, attrs, default_attrs);
  }

  VLOG(4) << LayerDebugString(op.Type(), ins, outs);

  // set the output var
  for (auto& var_pair : outs) {
    for (auto& var : var_pair.second) {
      // NOTE(zhiqu): The ouput may be NULL because of pruning.
      if (var) {
        SetForwardDataTypeOfGradVar(var);
      }
    }
  }
}

void RunOp(const std::string& type, const NameTensorMap& ins,
           const NameTensorMap& outs, framework::AttributeMap attrs,
           const platform::Place& place,
           const std::map<std::string, std::string>& inplace_map = {}) {
  VLOG(1) << "Run Op: " << type;
  if (FLAGS_use_mkldnn) {
    // if both lists are empty all ops are enabled (default for
    // FLAGS_use_mkldnn=1)
    // if ops_on list is not empty only ops from that list are enabled
    if (!FLAGS_tracer_mkldnn_ops_on.empty()) {
      auto is_on = FLAGS_tracer_mkldnn_ops_on.find(type) != std::string::npos;
      attrs["use_mkldnn"] = is_on;
    } else {
      // if ops_on list is empty all ops are enabled except types from off_list
      auto is_off = FLAGS_tracer_mkldnn_ops_off.find(type) != std::string::npos;
      attrs["use_mkldnn"] = !is_off;
    }
  }
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
  }

  static paddle::framework::AttributeMap empty_attrs_map = {};
  const paddle::framework::AttributeMap& default_attrs =
      attr_checker == nullptr ? empty_attrs_map
                              : attr_checker->GetDefaultAttrMap();

  NameTensorMap new_ins = ins;
  if (amp_level_ == 1) {
    VLOG(5) << "Auto mixed precision run operator: " << type;
    new_ins = AutoCastInputs(type, ins);
  } else if (amp_level_ == 2) {
    VLOG(5) << "Pure fp16 run operator: " << type;
    new_ins = CastPureFp16Inputs(type, ins);
  }

  try {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::SetDeviceId(BOOST_GET_CONST(platform::CUDAPlace, place).device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
      platform::SetXPUDeviceId(
          BOOST_GET_CONST(platform::XPUPlace, place).device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    } else if (platform::is_npu_place(place)) {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::SetNPUDeviceId(
          BOOST_GET_CONST(platform::NPUPlace, place).device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with NPU if use NPUPlace."));
#endif
    }

    OpRunImpl(*op, new_ins, outs, attrs, default_attrs, place);
  } catch (platform::EnforceNotMet& exception) {
    framework::AppendErrorOpHint(type, &exception);
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::Fatal(
        "Operator %s raises an %s exception.\n"
        "The exception content is\n:%s.",
        type, platform::demangle(typeid(ex).name()), ex.what()));
  } catch (...) {
    // NOTE: this branch represents a very serious bug with
    // low probability of occurrence, and we can't get its
    // exception content here.
    PADDLE_THROW(platform::errors::Fatal(
        "Operator %s raises an unknown exception.", type));
  }

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, new_ins, outs, attrs);
  }
}
}  // namespace egr
