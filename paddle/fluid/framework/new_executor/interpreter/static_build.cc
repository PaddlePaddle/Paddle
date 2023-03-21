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

// These Ops is OperatorBase, but we have been handle them in static build
std::set<std::string> OperatorBasesHandledInStaticBuild = {"read"};

std::set<std::string> OperatorBasesMustRunInStaticBuild = {
    "create_double_buffer_reader", "create_py_reader"};

std::set<std::string> OpsCanSkipedFakeAllocInStaticBuild = {
    "create_double_buffer_reader", "create_py_reader", "fetch_v2"};

// These Op needs set output dtype when register phi kernel, but they didn't
std::set<std::string> OpsNeedSetOutputDtypeWhenRegisterPhiKernel = {
    "clip_by_norm",
    "eig",
    "eig_grad",
    "eigh",
    "generate_proposals",
    "graph_sample_neighbors",
    "group_norm",
    "instance_norm",
    "lamb",
    "less_equal",
    "less_than",
    "momentum",
    "multiclass_nms3",
    "multinomial",
    "nanmedian",
    "rnn",
    "search_sort",
    "sync_batch_norm_grad",
    "unique",
    "unique_consecutive_flattened_tensor",
    "unique_raw"};

// Cannot static analysis these Ops' output dtype or backend because their
// kernels have not moved to PHI yet.
std::set<std::string> OpsWithFluidKernelNeedMoveToPhi = {
    "fused_attention",
    "fused_attention_grad",
    "fused_batch_norm_act",
    "fused_batch_norm_act_grad",
    "sequence_pool",
    "stft"};

std::set<std::string> StaticBuildBlackList = {
    "batch_norm" /*: to handle reserve_space output*/,
    "sparse_sparse_coo_tensor" /*: to handle sparse output*/};

namespace paddle {
namespace framework {
namespace interpreter {

bool BlockCanBeStaticBuilt(const framework::BlockDesc& block) {
  // in_black_list = (kernelCode >> 6) & 1
  // is_operator_base = (kernelCode >> 5) & 1
  // is_custom_op = (kernelCode >> 4) & 1
  // has_fluid_kernel = (kernelCode >> 3) & 1
  // has_structed_kernel = (kernelCode >> 2) & 1
  // need_move_to_phi = (kernelCode >> 1) & 1
  // need_set_dtype =  KernelCode & 1
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
    bool has_fluid_kernel = OperatorWithKernel::AllOpKernels().count(op_type);
    bool has_structured_kernel =
        phi::KernelFactory::Instance().HasStructuredKernel(op_type);
    bool need_move_to_phi = (has_fluid_kernel || has_structured_kernel) &&
                            OpsWithFluidKernelNeedMoveToPhi.count(op_type);
    bool need_set_dtype =
        !has_fluid_kernel && !has_structured_kernel &&
        OpsNeedSetOutputDtypeWhenRegisterPhiKernel.count(op_type);

    KernelCode kernel_code = (in_black_list << 6) + (is_operator_base << 5) +
                             (is_custom_op << 4) + (has_fluid_kernel << 3) +
                             (has_structured_kernel << 2) +
                             (need_move_to_phi << 1) + need_set_dtype;
    if (!OpsCanSkipedFakeAllocInStaticBuild.count(op_type)) {
      if (in_black_list ||
          (is_operator_base &&
           !OperatorBasesHandledInStaticBuild.count(op_type)) ||
          is_custom_op || need_move_to_phi || need_set_dtype) {
        invalid_ops.insert(std::make_pair(op_type, kernel_code));
      }
    }
  }

  if (!invalid_ops.empty()) {
    std::stringstream ss;
    ss << "The following OPs are unable to static build:\n";
    for (auto& item : invalid_ops) {
      ss << item.first << " [in_black_list = " << (item.second >> 6 & 1)
         << ", is_operator_base = " << (item.second >> 5 & 1)
         << ", is_custom_op = " << (item.second >> 4 & 1)
         << ", has_fluid_kernel = " << (item.second >> 3 & 1)
         << ", has_structed_kerenl = " << (item.second >> 2 & 1)
         << ", need_move_to_phi = " << (item.second >> 1 & 1)
         << ", need_set_dtype = " << (item.second & 1) << "]\n";
    }
    VLOG(0) << ss.str();
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

  if (op_type == "adam" || op_type == "adamw") {
    if (op.Attr<bool>("use_global_beta_pow") &&
        (parameter_name == "Beta1PowOut" || parameter_name == "Beta2PowOut")) {
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

  return tensor && !IsExtendedTensor(*tensor) && !tensor->initialized();
}

phi::TensorBase* GetTensorFormVar(framework::Variable* var) {
  if (var) {
    if (var->template IsType<phi::DenseTensor>()) {
      return var->template GetMutable<phi::DenseTensor>();
    } else if (var->template IsType<phi::SelectedRows>()) {
      return var->template GetMutable<phi::SelectedRows>();
    } else if (var->template IsType<phi::SparseCooTensor>()) {
      return var->template GetMutable<phi::SparseCooTensor>();
    } else if (var->template IsType<framework::LoDTensorArray>()) {
      return var->template GetMutable<framework::LoDTensorArray>();
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

void FakeInitializeTensor(const platform::DeviceContext& dev_ctx,
                          const phi::DataType& dtype,
                          const phi::Place& place,
                          phi::TensorBase* tensor) {
  PADDLE_ENFORCE_NOT_NULL(
      tensor,
      phi::errors::InvalidArgument(
          "The tensor to fake intialize should not be null."));
  if (place == phi::CPUPlace()) {
    dev_ctx.HostAlloc(tensor,
                      dtype,
                      /*requested_size=*/0,
                      /*fake_alloc=*/true);
  } else {
    PADDLE_ENFORCE_EQ(
        place,
        dev_ctx.GetPlace(),
        phi::errors::Unavailable("The place %s for fack alloc is not equal to "
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

void FakeInitializeOutputsForOperatorBase(const OperatorBase& op,
                                          const platform::Place& place,
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

        VLOG(4) << parameter_name << " fake alloc with type " << dtype
                << " on place " << place << " " << out_tensor;
        FakeInitializeTensor(*dev_ctx, dtype, place, out_tensor);
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
    for (size_t offset = 0; offset < outs_vector.size(); ++offset) {
      phi::TensorBase* out_tensor = GetTensorFormVar(outs_vector[offset]);
      if (TensorShouldBeFakeInitialized(op, parameter_name, out_tensor)) {
        phi::TensorArgDef& tensor_arg_def = output_defs[i];
        phi::DataType dtype = tensor_arg_def.dtype;
        if (dtype == DataType::UNDEFINED ||
            OpsNeedSetOutputDtypeWhenRegisterPhiKernel.count(
                std::string(op_type))) {
          // Some OP's InferMeta is sensitive to DDim, so we cannot get their
          // output dtype from InferMeta
          if (op_type == "adam" || op_type == "adamw") {
            dtype = InferMPDType(runtime_ctx, "Param");
          } else if (op_type == "arg_min" || op_type == "arg_max" ||
                     op_type == "one_hot_v2") {
            dtype = InferDTypeFromAttr(op, runtime_ctx, "dtype");
          } else if (op_type == "bincount") {
            dtype = GetInputDType(runtime_ctx, "X");
          } else if (op_type == "layer_norm" || op_type == "reduce_sum_grad") {
            dtype = InferMPDType(runtime_ctx, "X");
          } else if (op_type == "reduce_sum") {
            int dtype_attr = op.Attr<int>("out_dtype");
            if (dtype_attr != -1) {
              dtype = phi::TransToPhiDataType(dtype_attr);
            } else {
              phi::DataType in_dtype = GetInputDType(runtime_ctx, "X");
              dtype =
                  (in_dtype == DataType::BOOL || in_dtype == DataType::INT32)
                      ? DataType::INT64
                      : in_dtype;
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

        phi::Backend backend = tensor_arg_def.backend;

        if (backend == phi::Backend::UNDEFINED) {
          if (op_type == "adam" || op_type == "adamw") {
            phi::TensorBase* beta1_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta1Pow")->second.at(0));
            phi::TensorBase* beta2_pow = GetTensorFormVar(
                runtime_ctx.inputs.find("Beta2Pow")->second.at(0));
            if (beta1_pow->place() == CPUPlace() &&
                beta2_pow->place() == CPUPlace()) {
              backend = phi::TransToPhiBackend(CPUPlace());
            } else {
              backend = phi::TransToPhiBackend(GPUPlace());
            }
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

        VLOG(4) << parameter_name << " fake alloc with type " << dtype
                << " on place " << place << " " << out_tensor;

        FakeInitializeTensor(dev_ctx, dtype, place, out_tensor);
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
        phi::DataType dtype =
            phi::TransToPhiDataType(op_kernel_type.data_type_);
        phi::Place place = execution_context->GetPlace();

        VLOG(4) << parameter_name << " fake alloc with type " << dtype
                << " on place " << place << " " << out_tensor;

        FakeInitializeTensor(
            execution_context->device_context(), dtype, place, out_tensor);
      }
    }
  }
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
