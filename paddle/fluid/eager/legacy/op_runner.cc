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
#include "paddle/fluid/eager/legacy/infer_var_type_context.h"
#include "paddle/fluid/eager/legacy/prepared_operator.h"
#include "paddle/fluid/eager/legacy/tensor_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/denormal.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(use_mkldnn);
DECLARE_string(tracer_mkldnn_ops_on);
DECLARE_string(tracer_mkldnn_ops_off);

namespace egr {
namespace legacy {

void OpRunImpl(const paddle::framework::OperatorBase& op,
               const NameTensorMap& ins, const NameTensorMap& outs,
               const paddle::framework::AttributeMap& attrs,
               const paddle::framework::AttributeMap& default_attrs,
               const paddle::platform::Place& place) {
  VLOG(6) << "Get Opertor With Kernel";
  auto* op_kernel =
      dynamic_cast<const paddle::framework::OperatorWithKernel*>(&op);
  PADDLE_ENFORCE_NOT_NULL(
      op_kernel, paddle::platform::errors::PermissionDenied(
                     "Only support operator with kernel in Dygraph mode."));
  auto& info = op.Info();
  if (info.infer_var_type_) {
    VLOG(6) << "Run InferVarType";
    egr::legacy::TensorRuntimeInferVarTypeContext infer_var_type_ctx(
        ins, outs, attrs, default_attrs);
    VLOG(9) << "Actual Run InferVarType";
    info.infer_var_type_(&infer_var_type_ctx);
  }
  VLOG(6) << "Initialize output tensor";
  // Initialize output tensor
  for (auto& tensor_pair : outs) {
    for (auto& tensor : tensor_pair.second) {
      if (tensor && tensor.get() && (!tensor->Var().IsInitialized())) {
        InitializeVariable(tensor->MutableVar(),
                           paddle::framework::proto::VarType::LOD_TENSOR);
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
  VLOG(6) << "Prepare Op";
  auto prepared_op = egr::legacy::PreparedOp::Prepare(
      ins, outs, *op_kernel, place, attrs, default_attrs);
  VLOG(6) << "Prepare Data";
  auto tmp_ins_ptr =
      egr::legacy::PrepareData(*op_kernel, ins, prepared_op.kernel_type());
  VLOG(6) << "Run Prepared Op";
  if (tmp_ins_ptr == nullptr) {
    prepared_op.Run(ins, outs, attrs, default_attrs);
  } else {
    prepared_op.Run(*tmp_ins_ptr, outs, attrs, default_attrs);
  }

  // TODO(jiabin): Set the output var's grad Forward DataType
}

void RunOp(const std::string& type, const NameTensorMap& ins,
           const NameTensorMap& outs, paddle::framework::AttributeMap attrs,
           const paddle::platform::Place& place,
           paddle::framework::AttributeMap* default_attrs,
           bool override_default_attr_map,
           const std::map<std::string, std::string>& inplace_map) {
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
  auto op = paddle::framework::OpRegistry::CreateOp(type, {}, {}, {}, false);

  PADDLE_ENFORCE_NOT_NULL(default_attrs,
                          paddle::platform::errors::PermissionDenied(
                              "Detected default_attrs = nullptr."));

  if (override_default_attr_map) {
    const auto& op_info = op->Info();
    auto* attr_checker = op_info.Checker();
    if (attr_checker) {
      attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
    }

    static paddle::framework::AttributeMap empty_attrs_map = {};
    *default_attrs = attr_checker == nullptr
                         ? empty_attrs_map
                         : attr_checker->GetDefaultAttrMap();
  }

  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  VLOG(6) << "Check AMP status";
  NameTensorMap new_ins = ins;
  if (amp_level == paddle::imperative::AmpLevel::O1) {
    VLOG(5) << "Auto mixed precision run operator: " << type;
    new_ins = AutoCastInputs(type, ins);
  } else if (amp_level == paddle::imperative::AmpLevel::O2) {
    VLOG(5) << "Pure fp16 run operator: " << type;
    new_ins = CastPureFp16Inputs(type, ins);
  }

  try {
    VLOG(6) << "Get Device id";
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      paddle::platform::SetDeviceId(
          BOOST_GET_CONST(paddle::platform::CUDAPlace, place).device);
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    } else if (paddle::platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
      paddle::platform::SetXPUDeviceId(
          BOOST_GET_CONST(paddle::platform::XPUPlace, place).device);
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    } else if (paddle::platform::is_npu_place(place)) {
#ifdef PADDLE_WITH_ASCEND_CL
      paddle::platform::SetNPUDeviceId(
          BOOST_GET_CONST(paddle::platform::NPUPlace, place).device);
#else
      PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with NPU if use NPUPlace."));
#endif
    }
    VLOG(6) << "Step in OpRunImpl";
    OpRunImpl(*op, new_ins, outs, attrs, *default_attrs, place);
  } catch (paddle::platform::EnforceNotMet& exception) {
    paddle::framework::AppendErrorOpHint(type, &exception);
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Operator %s raises an %s exception.\n"
        "The exception content is\n:%s.",
        type, paddle::platform::demangle(typeid(ex).name()), ex.what()));
  } catch (...) {
    // NOTE: this branch represents a very serious bug with
    // low probability of occurrence, and we can't get its
    // exception content here.
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Operator %s raises an unknown exception.", type));
  }
  VLOG(6) << "Finish Run Op";
  // TODO(jiabin): Support this later
  // if (enable_program_desc_tracing_) {
  //   VLOG(5) << "Trace op " << type << " into ProgramDesc";
  //   program_desc_tracer_->InsertOp(type, new_ins, outs, attrs);
  // }
}

}  // namespace legacy
}  // namespace egr
