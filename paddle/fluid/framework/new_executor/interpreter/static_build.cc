// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/static_build.h"

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"

// These Ops is OperatorBase, but we have been handle them in static build
std::set<std::string> OperatorBasesHandledInStaticBuild = {"read"};

std::set<std::string> OperatorBasesMustRunInStaticBuild = {
    "create_double_buffer_reader", "create_py_reader"};

std::set<std::string> OpsCanSkipedFakeAllocInStaticBuild = {
    "c_comm_init",
    "c_comm_init_all",
    "c_comm_init_multitrainer",
    "c_gen_bkcl_id",
    "c_gen_nccl_id",
    "c_sync_calc_stream",
    "c_sync_comm_stream",
    "c_wait_comm",
    "c_wait_compute",
    "create_double_buffer_reader",
    "create_py_reader",
    "depend",
    "fetch_v2",
    "send_v2",
    "nop"};

std::set<std::string> StaticBuildBlackList = {
    "batch_norm" /*: to handle reserve_space output*/,
    "cinn_instruction_run" /*: to handle subgraph infermeta*/,
    "cinn_launch" /*: to handle subgraph infermeta*/,
    "run_program" /*: to handle scope output*/,
    "sparse_sparse_coo_tensor" /*: to handle sparse output*/,
    "distributed_fused_lamb_init"};

namespace paddle {
namespace framework {
namespace interpreter {

bool BlockCanBeStaticBuilt(const framework::BlockDesc& block) {
  // in_black_list = (kernelCode >> 7) & 1
  // is_operator_base = (kernelCode >> 6) & 1
  // is_custom_op = (kernelCode >> 5) & 1
  // use_mkldnn = (kernelCode >> 4) & 1
  using KernelCode = int8_t;
  std::set<std::pair<std::string, KernelCode>> invalid_ops;
  for (auto& op : block.AllOps()) {
    auto op_type = op->Type();
    const framework::OpInfo& info = OpInfoMap::Instance().Get(op_type);
    auto op_base =
        info.Creator()(op_type, op->Inputs(), op->Outputs(), op->GetAttrMap());

    bool in_black_list = StaticBuildBlackList.count(op_type);
    bool is_operator_base =
        (dynamic_cast<framework::OperatorWithKernel*>(op_base) == nullptr);
    bool is_custom_op =
        egr::Controller::Instance().GetOpMetaInfoMap().count(op_type);
    bool use_mkldnn = false;
    if (op->HasAttr("use_mkldnn")) {
      Attribute attr = op->GetAttr("use_mkldnn");
      use_mkldnn = attr.index() == 1 ? PADDLE_GET_CONST(int, attr)
                                     : PADDLE_GET_CONST(bool, attr);
    }
    bool has_structured_kernel =
        phi::KernelFactory::Instance().HasStructuredKernel(op_type);

    KernelCode kernel_code = static_cast<KernelCode>(
        (in_black_list << 7) + (is_operator_base << 6) + (is_custom_op << 5) +
        (use_mkldnn << 4) + (has_structured_kernel << 2));
    if (!OpsCanSkipedFakeAllocInStaticBuild.count(op_type)) {
      if (in_black_list ||
          (is_operator_base &&
           !OperatorBasesHandledInStaticBuild.count(op_type)) ||
          is_custom_op || use_mkldnn) {
        invalid_ops.insert(std::make_pair(op_type, kernel_code));
      }
    }
  }

  if (!invalid_ops.empty()) {
    std::stringstream ss;
    ss << "The following OPs are unable to static build:\n";
    for (auto& item : invalid_ops) {
      ss << item.first << " [in_black_list = " << (item.second >> 7 & 1)
         << ", is_operator_base = " << (item.second >> 6 & 1)
         << ", is_custom_op = " << (item.second >> 5 & 1)
         << ", use_mkldnn = " << (item.second >> 4 & 1)
         << (item.second >> 2 & 1) << "]\n";
    }
    VLOG(1) << ss.str();
  }

  return invalid_ops.empty();
}

inline bool IsExtendedTensor(const phi::TensorBase& tensor) {
  return framework::RawTensor::classof(&tensor) ||
         framework::Strings::classof(&tensor) ||
         framework::Vocab::classof(&tensor);
}

bool TensorShouldBeFakeInitialized(const OperatorBase& op,
                                   const std::string& parameter_name,
                                   const phi::TensorBase* tensor) {
  const std::string& op_type = op.Type();
  if (OpsCanSkipedFakeAllocInStaticBuild.count(op_type)) {
    return false;
  }

  if (op_type == "adam" || op_type == "adamw" || op_type == "merged_adam") {
    if (op.Attr<bool>("use_global_beta_pow") &&
        (parameter_name == "Beta1PowOut" || parameter_name == "Beta2PowOut")) {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "coalesce_tensor" && parameter_name == "Output") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "dgc" && parameter_name == "k") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "distributed_fused_lamb" && parameter_name == "ParamOut") {
    VLOG(2) << "Skip fake initialization for: " << parameter_name;
    return false;
  }

  if (op_type == "fused_bias_residual_layernorm" &&
      parameter_name == "residual_out") {
    if (op.HasInputs("residual")) {
      bool is_residual_empty = op.Input("residual") == kEmptyVarName;
      bool is_norm_weight_empty = op.Input("norm_weight") == kEmptyVarName;
      bool is_norm_bias_empty = op.Input("norm_bias") == kEmptyVarName;
      if (!is_residual_empty) {
        if (is_norm_weight_empty && is_norm_bias_empty) {
          VLOG(2) << "Skip fake initialization for: " << parameter_name;
          return false;
        }
      } else {
        VLOG(2) << "Skip fake initialization for: " << parameter_name;
        return false;
      }
    } else {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "fake_quantize_range_abs_max") {
    if (op.Attr<bool>("is_test") &&
        (parameter_name == "OutScale" || parameter_name == "OutScales")) {
      VLOG(2) << "Skip fake initialization for: " << parameter_name;
      return false;
    }
  }

  if (op_type == "segment_pool" && parameter_name == "SummedIds") {
    return op.Attr<std::string>("pooltype") == "MEAN" &&
           dynamic_cast<const OperatorWithKernel*>(&op)
                   ->kernel_type()
                   ->place_ != phi::CPUPlace();
  }

  return tensor && !IsExtendedTensor(*tensor);
}

phi::TensorBase* GetTensorFormVar(framework::Variable* var) {
  if (var) {
    if (var->template IsType<phi::DenseTensor>()) {
      return var->template GetMutable<phi::DenseTensor>();
    } else if (var->template IsType<phi::SelectedRows>()) {
      return var->template GetMutable<phi::SelectedRows>();
    } else if (var->template IsType<phi::SparseCooTensor>()) {
      return var->template GetMutable<phi::SparseCooTensor>();
    } else if (var->template IsType<phi::TensorArray>()) {
      return var->template GetMutable<phi::TensorArray>();
    } else if (var->template IsType<framework::Strings>()) {
      return var->template GetMutable<framework::Strings>();
    } else if (var->template IsType<paddle::framework::RawTensor>()) {
      return var->template GetMutable<paddle::framework::RawTensor>();
    } else if (!var->IsInitialized()) {
      // The following is for RAW type of var
      return var->template GetMutable<paddle::framework::RawTensor>();
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported `%s` type when get tensor.",
          framework::ToTypeName(var->Type())));
    }
  } else {
    VLOG(4) << "Var is nullptr";
    return nullptr;
  }
}

template <class TensorType>
void FakeInitializeTensor(const platform::DeviceContext& dev_ctx,
                          const phi::Place& place,
                          const phi::DataType& dtype,
                          const phi::DataLayout& layout,
                          TensorType* tensor) {
  PADDLE_ENFORCE_NE(place.GetType(),
                    phi::AllocationType::UNDEFINED,
                    phi::errors::InvalidArgument(
                        "The place %s to fake intialize is not valid.", place));
  PADDLE_ENFORCE_NE(dtype,
                    phi::DataType::UNDEFINED,
                    phi::errors::InvalidArgument(
                        "The dtype %s to fake intialize is not valid.", dtype));
  PADDLE_ENFORCE_NE(
      layout,
      phi::DataLayout::UNDEFINED,
      phi::errors::InvalidArgument(
          "The layout %s to fake intialize is not valid.", layout));
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      phi::errors::InvalidArgument(
          "The tensor to fake intialize should not be null."));

  if (tensor->initialized() && place == tensor->place() &&
      dtype == tensor->dtype() && tensor->layout() == layout) {
    return;
  }

  // set place
  if (tensor->initialized()) {  // avoid overwriting valid data
    platform::DeviceContext* dev_ctx_for_copy;
    if (place.GetType() != AllocationType::CPU) {
      dev_ctx_for_copy = platform::DeviceContextPool::Instance().Get(place);
    } else {
      dev_ctx_for_copy =
          platform::DeviceContextPool::Instance().Get(tensor->place());
    }
    phi::Copy(*dev_ctx_for_copy, *tensor, place, /*blocking=*/true, tensor);
  } else {
    if (place == phi::CPUPlace()) {
      dev_ctx.HostAlloc(tensor,
                        dtype,
                        /*requested_size=*/0,
                        /*fake_alloc=*/true);
    } else {
      PADDLE_ENFORCE_EQ(place,
                        dev_ctx.GetPlace(),
                        phi::errors::Unavailable(
                            "The place %s for fack alloc is not equal to "
                            "the place %s of DeviceContext.",
                            place,
                            dev_ctx.GetPlace()));
      dev_ctx.Alloc(tensor,
                    dtype,
                    /*requested_size=*/0,
                    /*pinned=*/false,
                    /*fake_alloc=*/true);
    }
  }

  // set dtype and layout
  tensor->set_type(dtype);
  tensor->set_layout(layout);

  VLOG(4) << "Tensor " << tensor << " fake alloc with type = " << dtype
          << ", place = " << place << ", layout = " << layout;
}

void FakeInitializeTensorBase(const platform::DeviceContext& dev_ctx,
                              const phi::Place& place,
                              const phi::DataType& dtype,
                              const phi::DataLayout& layout,
                              phi::TensorBase* tensor) {
  if (phi::DenseTensor::classof(tensor)) {
    FakeInitializeTensor(
        dev_ctx, place, dtype, layout, dynamic_cast<phi::DenseTensor*>(tensor));
  } else if (phi::SelectedRows::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SelectedRows*>(tensor));
  } else if (phi::SparseCooTensor::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SparseCooTensor*>(tensor));
  } else if (phi::SparseCsrTensor::classof(tensor)) {
    FakeInitializeTensor(dev_ctx,
                         place,
                         dtype,
                         layout,
                         dynamic_cast<phi::SparseCsrTensor*>(tensor));
  } else if (phi::TensorArray::classof(tensor)) {
    FakeInitializeTensor(
        dev_ctx, place, dtype, layout, dynamic_cast<phi::TensorArray*>(tensor));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported `%s` type when fake initialize tensor.",
        tensor->type_info().name()));
  }
}

void FakeInitializeOutputsForOperatorBase(const OperatorBase& op,
                                          const phi::Place& place,
                                          Scope* scope) {
  const std::string& op_type = op.Type();
  if (OpsCanSkipedFakeAllocInStaticBuild.count(op_type)) {
    return;
  }

  phi::DeviceContext* dev_ctx =
      platform::DeviceContextPool::Instance().Get(place);

  if (op_type == "read") {
    const std::string& reader_name = op.Input("Reader");
    framework::ReaderHolder* reader =
        GET_DATA_SAFELY(scope->FindVar(reader_name), "Input", "Reader", "Read")
            .GetMutable<framework::ReaderHolder>();

    std::shared_ptr<operators::reader::BufferedReader> buffered_reader =
        std::dynamic_pointer_cast<operators::reader::BufferedReader>(
            reader->Get());
    phi::Place target_place =
        buffered_reader ? buffered_reader->GetPlace() : phi::CPUPlace();

    auto& outputs = op.Outputs("Out");
    auto& var_types = reader->VarTypes();
    PADDLE_ENFORCE_EQ(
        outputs.size(),
        var_types.size(),
        phi::errors::Unavailable("The output size of read_op (%d) should equal "
                                 "to the var_types size of ReaderHolder (%d).",
                                 outputs.size(),
                                 var_types.size()));

    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string& parameter_name = outputs[i];
      phi::TensorBase* out_tensor =
          GetTensorFormVar(scope->FindVar(parameter_name));
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::DataType dtype = phi::TransToPhiDataType(var_types[i]);
        FakeInitializeTensorBase(
            *dev_ctx, target_place, dtype, out_tensor->layout(), out_tensor);
      }
    }
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Can not static build for op: %s", op_type));
  }
}

phi::DataType GetInputDType(const RuntimeContext& runtime_ctx,
                            const std::string parameter_name) {
  phi::TensorBase* in_tensor =
      GetTensorFormVar(runtime_ctx.inputs.find(parameter_name)->second.at(0));
  return in_tensor->dtype();
}

bool InputExisted(const RuntimeContext& runtime_ctx,
                  const std::string& parameter_name) {
  auto it = runtime_ctx.inputs.find(parameter_name);
  if (it == runtime_ctx.inputs.end() || it->second.empty()) {
    return false;
  }
  return true;
}

phi::DataType InferDTypeFromAttr(const framework::OperatorBase& op,
                                 const RuntimeContext& runtime_ctx,
                                 const std::string& attr_name) {
  int dtype_attr = op.Attr<int>(attr_name);
  if (dtype_attr == -1) {  // -1 means the dtype is same as intput
    return GetInputDType(runtime_ctx, "X");
  }
  return phi::TransToPhiDataType(dtype_attr);
}

phi::DataType InferMPDType(const RuntimeContext& runtime_ctx,
                           const std::string parameter_name) {
  phi::DataType in_dtype = GetInputDType(runtime_ctx, parameter_name);
  return (in_dtype == phi::DataType::BFLOAT16 ||
          in_dtype == phi::DataType::FLOAT16)
             ? phi::DataType::FLOAT32
             : in_dtype;
}

void FakeInitializeOutputsForFunctionKernel(
    const framework::OperatorBase& op,
    const phi::Kernel& phi_kernel,
    const phi::KernelSignature& kernel_sig,
    const RuntimeContext& runtime_ctx,
    const platform::DeviceContext& dev_ctx) {
  std::string op_type = op.Type();
  auto output_names = kernel_sig.output_names;
  auto output_defs = phi_kernel.args_def().output_defs();
  PADDLE_ENFORCE_EQ(output_names.size(),
                    output_defs.size(),
                    platform::errors::InvalidArgument(
                        "The size of outputs_args names (%d) must be equal to "
                        "the size of kernel output_defs (%d).",
                        output_names.size(),
                        output_defs.size()));
  size_t start_idx = 0;
  for (size_t i = 0; i < output_names.size(); ++i) {
    const std::string& parameter_name = output_names[i];
    auto it = runtime_ctx.outputs.find(parameter_name);
    // Deal with the case that some outputs are not found or be NULL when run
    // the kernel. For example : the outputs of matmul_grad are dx and dy,
    // sometimes dx or dy may be NULL.
    if (it == runtime_ctx.outputs.end() || it->second.empty()) {
      VLOG(4) << "Output " << parameter_name << " not found";
      ++start_idx;
      continue;
    }
    auto& outs_vector = it->second;
    for (auto out_var : outs_vector) {
      phi::TensorBase* out_tensor = GetTensorFormVar(out_var);
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::TensorArgDef& tensor_arg_def = output_defs[i];

        // analyze place
        phi::Backend backend = tensor_arg_def.backend;
        if (backend == phi::Backend::UNDEFINED) {
          if (op_type == "adam" || op_type == "adamw" ||
              op_type == "merged_adam") {
            phi::TensorBase* beta1_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta1Pow")->second.at(0));
            phi::TensorBase* beta2_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta2Pow")->second.at(0));
            if (beta1_pow->place() == beta2_pow->place()) {
              backend = phi::TransToPhiBackend(beta1_pow->place());
            }
          } else if (op_type == "reshape2") {
            phi::TensorBase* x =
                GetTensorFormVar(runtime_ctx.inputs.find("X")->second.at(0));
            backend = phi::TransToPhiBackend(x->place());
          } else {
            PADDLE_THROW(phi::errors::Unimplemented(
                "Unsupported UNDEFINED backend for op: %s, parameter: %s",
                op_type,
                parameter_name));
          }
        }
        phi::Place place = backend == phi::Backend::CUSTOM
                               ? dev_ctx.GetPlace()
                               : phi::TransToPhiPlace(backend);

        // analyze dtype
        phi::DataType dtype = tensor_arg_def.dtype;
        if (dtype == DataType::UNDEFINED) {
          // Some OP's InferMeta is sensitive to DDim, so we cannot get their
          // output dtype from InferMeta
          if (op_type == "adam" || op_type == "adamw") {
            dtype = InferMPDType(runtime_ctx, "Param");
          } else if (op_type == "arg_min" || op_type == "arg_max" ||
                     op_type == "coalesce_tensor" || op_type == "one_hot_v2" ||
                     op_type == "unique") {
            dtype = InferDTypeFromAttr(op, runtime_ctx, "dtype");
          } else if (op_type == "bincount" || op_type == "reduce_sum_grad") {
            dtype = GetInputDType(runtime_ctx, "X");
          } else if (op_type == "lamb") {
            bool multi_precision = op.Attr<bool>("multi_precision");
            dtype = GetInputDType(runtime_ctx, "Moment1");
            if (multi_precision && dtype == phi::DataType::FLOAT16) {
              dtype = phi::DataType::FLOAT32;
            }
          } else if (op_type == "layer_norm") {
            dtype = InferMPDType(runtime_ctx, "X");
          } else if (op_type == "reduce_sum") {
            phi::DataType in_dtype = GetInputDType(runtime_ctx, "X");
            int dtype_attr = op.Attr<int>("out_dtype");
            if (dtype_attr != -1) {
              dtype = phi::TransToPhiDataType(dtype_attr);
              if (dtype == DataType::UNDEFINED) {
                dtype = in_dtype;
              }
            } else {
              dtype =
                  (in_dtype == DataType::BOOL || in_dtype == DataType::INT32)
                      ? DataType::INT64
                      : in_dtype;
            }
          } else if (op_type == "searchsorted") {
            bool out_int32 = op.Attr<bool>("out_int32");
            if (out_int32) {
              dtype = DataType::INT32;
            } else {
              dtype = DataType::INT64;
            }
          } else if (op_type == "fused_bias_residual_layernorm") {
            auto in_dtype = GetInputDType(runtime_ctx, "x");
            float quant_scale = op.Attr<float>("quant_scale");
            if (InputExisted(runtime_ctx, "residual") &&
                !InputExisted(runtime_ctx, "norm_weight") &&
                !InputExisted(runtime_ctx, "norm_bias")) {
              dtype = in_dtype;
            } else {
              if (quant_scale > 0.0f) {
                dtype = DataType::INT8;
              } else {
                dtype = in_dtype;
              }
            }
          } else {
            VLOG(4) << "Get dtype result from InferMeta";
            RuntimeInferShapeContext infer_shape_ctx(op, runtime_ctx);
            dynamic_cast<const framework::OperatorWithKernel*>(&op)
                ->Info()
                .infer_shape_(&infer_shape_ctx);
            dtype = out_tensor->dtype();  // dtype from InferMeta
          }
        }

        // analyze layout
        phi::DataLayout layout = tensor_arg_def.layout;
        FakeInitializeTensorBase(dev_ctx, place, dtype, layout, out_tensor);
      }
    }
    start_idx += outs_vector.size();
  }
}

void FakeInitializeOutputsForStructureKernel(
    const framework::OpKernelType& op_kernel_type,
    ExecutionContext* execution_context) {
  const framework::OperatorBase& op = execution_context->GetOp();
  if (OpsCanSkipedFakeAllocInStaticBuild.count(op.Type())) {
    return;
  }

  const VariableNameMap& outputs = op.Outputs();
  for (auto& item : outputs) {
    const std::string& parameter_name = item.first;
    auto multi_output_var = execution_context->MultiOutputVar(parameter_name);
    for (Variable* var : multi_output_var) {
      phi::TensorBase* out_tensor = GetTensorFormVar(var);
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::Place place = execution_context->GetPlace();
        phi::DataType dtype =
            phi::TransToPhiDataType(op_kernel_type.data_type_);
        phi::DataLayout layout = out_tensor->layout();
        FakeInitializeTensorBase(execution_context->device_context(),
                                 place,
                                 dtype,
                                 layout,
                                 out_tensor);
      }
    }
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
