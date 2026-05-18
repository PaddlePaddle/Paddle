// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/kernel_op_function_manual.h"

#include <Python.h>

#include <memory>
#include <string>
#include <vector>

#include "paddle/common/errors.h"
#include "paddle/common/overloaded.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/utils/pybind.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/core/dense_tensor.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/cinn/runtime/cinn_runtime.h"
#endif

PD_DECLARE_bool(cinn_measure_kernel_time);
PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_kernel_execution_label);
PD_DECLARE_bool(cinn_check_jit_instruction_shape);
#endif

namespace paddle {
namespace pybind {
namespace {

using DenseTensor = phi::DenseTensor;
using Tensor = paddle::Tensor;

std::string GetRequiredStringAttribute(pir::Operation* op,
                                       const std::string& attr_name) {
  PADDLE_ENFORCE_NOT_NULL(
      op, common::errors::InvalidArgument("Operation must not be nullptr."));
  PADDLE_ENFORCE_EQ(op->HasAttribute(attr_name),
                    true,
                    common::errors::InvalidArgument(
                        "Operation %s does not have string attribute %s.",
                        op->name().c_str(),
                        attr_name.c_str()));
  auto attr = op->attribute(attr_name);
  PADDLE_ENFORCE_EQ(attr.isa<pir::StrAttribute>(),
                    true,
                    common::errors::InvalidArgument(
                        "Operation %s attribute %s must be StrAttribute.",
                        op->name().c_str(),
                        attr_name.c_str()));
  return attr.dyn_cast<pir::StrAttribute>().AsString();
}

std::unique_ptr<paddle::dialect::OpYamlInfoParser> CreateOpYamlInfoParser(
    const std::string& op_name) {
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  PADDLE_ENFORCE_NOT_NULL(
      yaml_interface,
      common::errors::PreconditionNotMet(
          "Can not find OpYamlInfoInterface from [%s].", op_name));
  return std::make_unique<paddle::dialect::OpYamlInfoParser>(
      yaml_interface->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));
}

pybind11::list ToPyStringList(const std::vector<std::string>& names) {
  pybind11::list result;
  for (const auto& name : names) {
    result.append(name);
  }
  return result;
}

pybind11::dict GetPhiKernelOpInfo(pir::Operation* op) {
  PADDLE_ENFORCE_NOT_NULL(
      op,
      common::errors::InvalidArgument(
          "get_phi_kernel_op_info op must not be nullptr."));
  PADDLE_ENFORCE_EQ(
      op->name(),
      "pd_kernel.phi_kernel",
      common::errors::InvalidArgument(
          "get_phi_kernel_op_info only accepts pd_kernel.phi_kernel, but got "
          "%s.",
          op->name().c_str()));

  const std::string op_name = GetRequiredStringAttribute(op, "op_name");
  const std::string kernel_name = GetRequiredStringAttribute(op, "kernel_name");
  auto parser = CreateOpYamlInfoParser(op_name);

  pybind11::dict attrs;
  for (const auto& attr_name : parser->AttributeNames()) {
    if (!op->HasAttribute(attr_name)) {
      continue;
    }
    attrs[attr_name.c_str()] =
        paddle::dialect::GetAttributeData(op->attribute(attr_name));
  }

  pybind11::dict info;
  info["op_name"] = op_name;
  info["kernel_name"] = kernel_name;
  info["input_names"] = ToPyStringList(parser->InputNames());
  info["attr_names"] = ToPyStringList(parser->AttributeNames());
  info["attrs"] = attrs;
  return info;
}

#ifdef PADDLE_WITH_CINN

using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

typedef void (*lower_func_ptr_g)(void*, int32_t, void*);
typedef void (*infer_shape_func_ptr_g)(void*, int32_t, int64_t**);

class DirectCinnKernelLauncher {
 public:
  explicit DirectCinnKernelLauncher(const CINNKernelInfo& cinn_kernel_info)
      : cinn_kernel_info_(cinn_kernel_info) {}

  ~DirectCinnKernelLauncher() { FreeFuncArgs(); }

  void InitFuncArgs(const std::vector<DenseTensor*>& kernel_tensor_args) {
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      func_args_.emplace_back(buffer);
    }

    const auto& GetSymbolArg = common::Overloaded{
        [&](const CINNKernelInfo::ArgDimIdx& binding_info) -> int64_t {
          return static_cast<int64_t>(
              kernel_tensor_args[binding_info.arg_idx]->dims().at(
                  binding_info.dim_idx));
        },
        [&](const CINNKernelInfo::ArgValueIdx& binding_info) -> int64_t {
          const auto& tensor = [&]() -> DenseTensor {
            DenseTensor new_tensor =
                *(kernel_tensor_args[binding_info.arg_idx]);
            if (new_tensor.place() == phi::CPUPlace()) {
              return new_tensor;
            }
            framework::TensorCopySync(
                *(kernel_tensor_args[binding_info.arg_idx]),
                phi::CPUPlace(),
                &new_tensor);
            return new_tensor;
          }();
          if (tensor.dtype() == phi::DataType::INT32) {
            std::vector<int> tensor_data;
            framework::TensorToVector(tensor, &tensor_data);
            return tensor_data[binding_info.value_idx];
          }
          if (tensor.dtype() == phi::DataType::INT64) {
            std::vector<int64_t> tensor_data;
            framework::TensorToVector(tensor, &tensor_data);
            return tensor_data[binding_info.value_idx];
          }
          PADDLE_THROW(common::errors::Fatal(
              "Only int32 and int64 tensors are supported "
              "for CINN dynamic shape symbol args."));
        }};

    for (const auto& pair : cinn_kernel_info_.symbol_args_map) {
      func_args_.emplace_back(std::visit(GetSymbolArg, pair.second));
    }
  }

  void InferShape(const std::vector<DenseTensor*>& kernel_tensor_args,
                  const std::vector<phi::DDim>& ir_dims,
                  int32_t input_tensor_size,
                  int32_t output_tensor_size) {
    std::vector<int64_t*> output_tensor_shapes(output_tensor_size);
    for (int i = 0; i < output_tensor_size; ++i) {
      output_tensor_shapes[i] = reinterpret_cast<int64_t*>(
          calloc(kernel_tensor_args[input_tensor_size + i]->dims().size(),
                 sizeof(int64_t*)));
    }

    ((infer_shape_func_ptr_g)cinn_kernel_info_.infer_shape_fn_ptr)(
        static_cast<void*>(func_args_.data()),
        func_args_.size(),
        output_tensor_shapes.data());

    for (int i = 0; i < output_tensor_size; ++i) {
      phi::DDim dim(output_tensor_shapes[i],
                    kernel_tensor_args[input_tensor_size + i]->dims().size());
      if (static_cast<size_t>(i) < ir_dims.size() &&
          FLAGS_cinn_check_jit_instruction_shape) {
        CheckDims(ir_dims[i], dim);
        CheckDimGTZero(dim);
      }
      kernel_tensor_args[input_tensor_size + i]->Resize(dim);
      free(output_tensor_shapes[i]);
    }
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE)
  void Run(const std::vector<DenseTensor*>& kernel_tensor_args,
           void* stream,
           bool is_gpu) {
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      if (!kernel_tensor_args[i]->has_allocation()) {
        cinn_pod_value_to_buffer_p(&(func_args_[i]))->memory = nullptr;
      } else {
        cinn_pod_value_to_buffer_p(&(func_args_[i]))->memory =
            reinterpret_cast<uint8_t*>(kernel_tensor_args[i]->data());
      }
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (FLAGS_cinn_measure_kernel_time ||
        FLAGS_tile_config_policy == "search") {
      common::PerformanceStatistician& ps =
          common::PerformanceStatistician::Instance();
      phi::gpuStream_t stream;
      phi::InitStream(&stream);
      phi::backends::gpu::GpuDeviceSync();
      if (is_gpu) {
        ps.SetGraphNodesNum(25);
        int graph_nodes_num = ps.GetGraphNodesNum();
        phi::gpuGraph_t graph;
        phi::gpuGraphExec_t instance;
        phi::gpuStreamBeginCapture(stream, gpuStreamCaptureMode(0));
        for (int i = 0; i < graph_nodes_num; ++i) {
          ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
              static_cast<void*>(func_args_.data()), func_args_.size(), stream);
        }
        phi::gpuStreamEndCapture(stream, &graph);
#ifdef PADDLE_WITH_CUDA
        cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
#elif defined(PADDLE_WITH_HIP)
        hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
#endif
        ps.CudaStart(FLAGS_cinn_kernel_execution_label);
        phi::gpuGraphLaunch(instance, stream);
        ps.CudaEnd(FLAGS_cinn_kernel_execution_label);
        phi::gpuGraphDestroy(graph);
        phi::gpuGraphExecDestroy(instance);
        phi::DestroyStream(stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
      phi::backends::gpu::GpuDeviceSync();
    } else {
#endif
      if (is_gpu) {
        ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    }
#endif
  }
#endif

 private:
  void FreeFuncArgs() {
    for (auto& arg : func_args_) {
      if (arg.type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
        delete cinn_pod_value_to_buffer_p(&arg);
      }
    }
    func_args_.clear();
  }

  void CheckDims(const phi::DDim& first, const phi::DDim& second) const {
    PADDLE_ENFORCE_EQ(
        first.size(),
        second.size(),
        common::errors::PreconditionNotMet("The rank of dim MUST be same. "
                                           "But get [%d] and [%d]",
                                           first.size(),
                                           second.size()));
    for (int i = 0; i < first.size(); ++i) {
      if (first[i] > 0) {
        PADDLE_ENFORCE_EQ(
            first[i],
            second[i],
            common::errors::PreconditionNotMet(
                "Dim MUST be equal, but first[%d] is [%d], second[%d] is [%d]",
                i,
                first[i],
                i,
                second[i]));
      }
    }
  }

  void CheckDimGTZero(const phi::DDim& dim) const {
    for (int i = 0; i < dim.size(); ++i) {
      PADDLE_ENFORCE_GE(
          dim.at(i),
          0,
          common::errors::PreconditionNotMet(
              "The dim of CINN direct kernel output must be >= 0. "
              "Jit kernel name: %s. Tensor dim: %s",
              cinn_kernel_info_.fn_name,
              dim.to_str()));
    }
  }

  CINNKernelInfo cinn_kernel_info_;
  std::vector<cinn_pod_value_t> func_args_;
};

DenseTensor* MutableDenseTensor(const Tensor& tensor, size_t index) {
  PADDLE_ENFORCE_EQ(
      tensor.is_dense_tensor(),
      true,
      common::errors::InvalidArgument(
          "CINN direct kernel input %d must be DenseTensor.", index));
  return static_cast<DenseTensor*>(tensor.impl().get());
}

std::vector<Tensor> ParseTensorList(PyObject* obj) {
  PADDLE_ENFORCE_EQ(PyList_Check(obj) || PyTuple_Check(obj),
                    true,
                    common::errors::InvalidArgument(
                        "run_cinn_jit_kernel inputs must be list or tuple."));
  Py_ssize_t size = PyList_Check(obj) ? PyList_Size(obj) : PyTuple_Size(obj);
  std::vector<Tensor> tensors;
  tensors.reserve(static_cast<size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item =
        PyList_Check(obj) ? PyList_GetItem(obj, i) : PyTuple_GetItem(obj, i);
    PADDLE_ENFORCE_EQ(
        PyCheckTensor(item),
        true,
        common::errors::InvalidArgument(
            "run_cinn_jit_kernel inputs[%d] must be Tensor, but got %s.",
            i,
            reinterpret_cast<PyTypeObject*>(item->ob_type)->tp_name));
    tensors.emplace_back(UnSafeGetTensorFromPyObject(item));
  }
  return tensors;
}

phi::Place InferKernelPlace(pir::Operation* op,
                            const std::vector<Tensor>& inputs) {
  if (op->HasAttribute("exec_backend")) {
    return op->attribute("exec_backend")
        .dyn_cast<paddle::dialect::PlaceAttribute>()
        .data();
  }
  if (!inputs.empty()) {
    return inputs[0].place();
  }
  if (op->num_results() > 0) {
    auto type = op->result(0).type();
    if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
      return type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>().place();
    }
  }
  return phi::CPUPlace();
}

struct DenseTensorMetaFromIr {
  phi::DataType dtype;
  phi::DDim dims;
};

DenseTensorMetaFromIr GetDenseTensorMetaFromIr(pir::Type type) {
  if (type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
    auto tensor_type =
        type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    return {paddle::dialect::TransToPhiDataType(tensor_type.dtype()),
            tensor_type.dims()};
  }
  if (type.isa<paddle::dialect::DenseTensorType>()) {
    auto tensor_type = type.dyn_cast<paddle::dialect::DenseTensorType>();
    return {paddle::dialect::TransToPhiDataType(tensor_type.dtype()),
            tensor_type.dims()};
  }
  PADDLE_THROW(common::errors::Unimplemented(
      "CINN direct kernel only supports DenseTensorType results."));
}

std::vector<Tensor> RunCinnJitKernel(pir::Operation* op,
                                     const std::vector<Tensor>& inputs) {
  PADDLE_ENFORCE_EQ(
      op != nullptr,
      true,
      common::errors::InvalidArgument("run_cinn_jit_kernel op is nullptr."));
  PADDLE_ENFORCE_EQ(
      op->name(),
      "cinn_runtime.jit_kernel",
      common::errors::InvalidArgument(
          "run_cinn_jit_kernel only accepts cinn_runtime.jit_kernel, but got "
          "%s.",
          op->name().c_str()));
  PADDLE_ENFORCE_EQ(inputs.size(),
                    op->num_operands(),
                    common::errors::InvalidArgument(
                        "CINN direct kernel expects %d inputs, but got %d.",
                        op->num_operands(),
                        inputs.size()));

  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  DirectCinnKernelLauncher launcher(jit_kernel_op.cinn_kernel_info());
  int32_t input_tensor_size = static_cast<int32_t>(op->num_operands());
  int32_t output_tensor_size = static_cast<int32_t>(op->num_results());
  bool need_update_shape = false;
  std::vector<DenseTensor*> tensor_args;
  std::vector<DenseTensor*> alloc_tensors;
  std::vector<phi::DDim> ir_dims;
  std::vector<Tensor> outputs;
  std::vector<DenseTensor> temp_space_tensors;

  tensor_args.reserve(inputs.size() + op->num_results() +
                      jit_kernel_op.cinn_kernel_info().temp_space_sizes.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensor_args.push_back(MutableDenseTensor(inputs[i], i));
  }

  outputs.reserve(op->num_results());
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    auto meta = GetDenseTensorMetaFromIr(op->result(i).type());
    auto dense_tensor = std::make_shared<DenseTensor>();
    dense_tensor->set_type(meta.dtype);
    dense_tensor->Resize(meta.dims);
    for (int j = 0; j < meta.dims.size(); ++j) {
      if (meta.dims[j] < 0) {
        need_update_shape = true;
      }
    }
    ir_dims.push_back(meta.dims);
    outputs.emplace_back(dense_tensor);
    tensor_args.push_back(dense_tensor.get());
    alloc_tensors.push_back(dense_tensor.get());
  }

  for (int64_t size : jit_kernel_op.cinn_kernel_info().temp_space_sizes) {
    auto& tensor = temp_space_tensors.emplace_back();
    tensor.set_type(phi::DataType::UINT8);
    tensor.Resize({size});
    if (size < 0) {
      need_update_shape = true;
    }
  }
  for (auto& tensor : temp_space_tensors) {
    tensor_args.push_back(&tensor);
    alloc_tensors.push_back(&tensor);
  }
  output_tensor_size += static_cast<int32_t>(temp_space_tensors.size());

  phi::Place place = InferKernelPlace(op, inputs);
  auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE)
  void* running_stream = nullptr;
  bool is_gpu = false;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (place.GetType() == phi::AllocationType::GPU) {
    is_gpu = true;
    running_stream =
        static_cast<void*>(static_cast<phi::GPUContext*>(dev_ctx)->stream());
  }
#elif defined(PADDLE_WITH_CUSTOM_DEVICE)
  if (place.GetType() == phi::AllocationType::CUSTOM) {
    is_gpu = true;
    running_stream =
        static_cast<void*>(static_cast<phi::CustomContext*>(dev_ctx)->stream());
  }
#endif

  launcher.InitFuncArgs(tensor_args);
  if (need_update_shape) {
    launcher.InferShape(
        tensor_args, ir_dims, input_tensor_size, output_tensor_size);
  }
  for (auto* tensor : alloc_tensors) {
    dev_ctx->Alloc(tensor, tensor->dtype());
  }
  launcher.Run(tensor_args, running_stream, is_gpu);

  for (auto& tensor : temp_space_tensors) {
    tensor.clear();
  }
  return outputs;
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "run_cinn_jit_kernel requires Paddle built with CINN and "
      "CUDA/HIP/custom-device support."));
#endif
}

PyObject* RunCinnJitKernelPyObject(PyObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  try {
    PyObject* op_obj = nullptr;
    PyObject* inputs_obj = nullptr;
    static const char* kwlist[] = {"op", "inputs", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "OO:run_cinn_jit_kernel",
                                     const_cast<char**>(kwlist),
                                     &op_obj,
                                     &inputs_obj)) {
      return nullptr;
    }
    auto* op = pybind11::handle(op_obj).cast<pir::Operation*>();
    auto inputs = ParseTensorList(inputs_obj);

    std::vector<Tensor> outputs = RunCinnJitKernel(op, inputs);

    if (outputs.empty()) {
      return PyTuple_New(0);
    }
    if (outputs.size() == 1) {
      return ToPyObject(outputs[0]);
    }
    return ToPyObject(outputs, false);
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

#else

PyObject* RunCinnJitKernelPyObject(PyObject* self,
                                   PyObject* args,
                                   PyObject* kwargs) {
  try {
    PADDLE_THROW(common::errors::Unimplemented(
        "run_cinn_jit_kernel requires Paddle built with CINN. "
        "Fast kernel runtime does not fall back to run_program."));
  } catch (...) {
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

#endif

PyMethodDef ManualDirectKernelMethods[] = {
    {"run_cinn_jit_kernel",
     (PyCFunction)(void (*)(void))RunCinnJitKernelPyObject,
     METH_VARARGS | METH_KEYWORDS,
     "Launch a compiled CINN jit_kernel op without PirInterpreter."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace

void BindManualDirectKernelOpFunctions(pybind11::module* module) {
  module->def(
      "get_phi_kernel_op_info", &GetPhiKernelOpInfo, pybind11::arg("op"));
  if (PyModule_AddFunctions(module->ptr(), ManualDirectKernelMethods) < 0) {
    PADDLE_THROW(common::errors::Fatal(
        "Add manual functions to core.eager.kernel_ops failed!"));
  }
}

}  // namespace pybind
}  // namespace paddle
