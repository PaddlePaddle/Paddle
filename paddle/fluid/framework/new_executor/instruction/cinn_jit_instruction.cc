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

#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/errors.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/cinn/runtime/cinn_runtime.h"
#endif
PD_DECLARE_bool(cinn_bucket_compile);
PD_DECLARE_bool(cinn_measure_kernel_time);
PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_kernel_execution_label);

namespace paddle {
namespace framework {

typedef void (*lower_func_ptr_g)(void*, int32_t, void*);
typedef void (*infer_shape_func_ptr_g)(void*, int32_t, int64_t**);

class CinnJitInstruction::FnPtrImpl {
  using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

 public:
  explicit FnPtrImpl(const CINNKernelInfo& cinn_kernel_info)
      : cinn_kernel_info_(cinn_kernel_info) {}

  void InitFuncArgs(const std::vector<phi::DenseTensor*>& kernel_tensor_args) {
    func_args_.clear();

    // 1. Create placeholders for tensor args
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      func_args_.emplace_back(buffer);
    }

    // 2. Convert symbol args about dynamic shape to cinn_pod_value_t
    const auto& GetSymbolArg = common::Overloaded{
        [&](const CINNKernelInfo::ArgDimIdx& binding_info) -> int64_t {
          return static_cast<int64_t>(
              kernel_tensor_args[binding_info.arg_idx]->dims().at(
                  binding_info.dim_idx));
        },
        [&](const CINNKernelInfo::ArgValueIdx& binding_info) -> int64_t {
          const auto& tensor = [&]() -> phi::DenseTensor {
            phi::DenseTensor new_tensor =
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
          } else if (tensor.dtype() == phi::DataType::INT64) {
            std::vector<int64_t> tensor_data;
            framework::TensorToVector(tensor, &tensor_data);
            return tensor_data[binding_info.value_idx];
          }
          PADDLE_THROW(
              ::common::errors::Fatal("Dead code, only support int32 and int64 "
                                      "for dynamic shape arg now"));
        }};

    for (const auto& [_, binding_info] : cinn_kernel_info_.symbol_args_map) {
      func_args_.emplace_back(std::visit(GetSymbolArg, binding_info));
    }

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Run func_args_ size: " << func_args_.size();
      for (const auto& args : func_args_) {
        VLOG(4) << " args type_code: " << args.type_code();
      }
    }
  }

  void Run(const std::vector<phi::DenseTensor*>& kernel_tensor_args,
           void* stream,
           bool is_gpu) {
    VLOG(6) << "Start Run: " << cinn_kernel_info_.fn_name;

    // Pass real tensor data to cinn_buffer_t func args placeholder
    for (size_t i = 0; i < kernel_tensor_args.size(); ++i) {
      cinn_pod_value_to_buffer_p(&(func_args_[i]))->memory =
          reinterpret_cast<uint8_t*>(kernel_tensor_args[i]->data());
    }

    // Launch host kernel
    if (FLAGS_cinn_measure_kernel_time ||
        FLAGS_tile_config_policy == "search") {
      VLOG(3) << "enter searching config branch";
      ::common::PerformanceStatistician& ps =
          ::common::PerformanceStatistician::Instance();
      auto data_p = static_cast<void*>(func_args_.data());
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      cudaDeviceSynchronize();
      if (is_gpu) {
        ps.SetGraphNodesNum(25);
        int graph_nodes_num = ps.GetGraphNodesNum();
        cudaGraph_t graph;
        cudaGraphExec_t instance;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int ikrnl = 0; ikrnl < graph_nodes_num; ikrnl++) {
          ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
              static_cast<void*>(func_args_.data()), func_args_.size(), stream);
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        ps.CudaStart(FLAGS_cinn_kernel_execution_label);
        cudaGraphLaunch(instance, stream);
        ps.CudaEnd(FLAGS_cinn_kernel_execution_label);
        cudaGraphDestroy(graph);
        cudaGraphExecDestroy(instance);
        cudaStreamDestroy(stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
      cudaDeviceSynchronize();
    } else {
      if (is_gpu) {
        ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      } else {
        ((lower_func_ptr_g)cinn_kernel_info_.CX86_fn_ptr)(
            static_cast<void*>(func_args_.data()), func_args_.size(), stream);
      }
    }
    VLOG(6) << "End Run: " << cinn_kernel_info_.fn_name;

    std::cerr << "run func " << cinn_kernel_info_.fn_name << std::endl;
  }

  void InferShape(const std::vector<phi::DenseTensor*>& kernel_tensor_args,
                  int32_t input_tensor_size,
                  int32_t output_tensor_size) {
    VLOG(6) << "Start InferShape: " << cinn_kernel_info_.fn_name;
    // Define an array of Pointers to hold the output tensor shape
    std::vector<int64_t*> output_tensor_shapes(output_tensor_size);
    for (int i = 0; i < output_tensor_size; ++i) {
      output_tensor_shapes[i] = reinterpret_cast<int64_t*>(
          malloc(kernel_tensor_args[input_tensor_size + i]->dims().size() *
                 sizeof(int64_t*)));
    }

    // Launch infer_shape_fn_ptr to infer shape of output tensor
    ((infer_shape_func_ptr_g)cinn_kernel_info_.infer_shape_fn_ptr)(
        static_cast<void*>(func_args_.data()),
        func_args_.size(),
        output_tensor_shapes.data());

    // Resize shape of output tensor
    for (int i = 0; i < output_tensor_size; ++i) {
      DDim dim(output_tensor_shapes[i],
               kernel_tensor_args[input_tensor_size + i]->dims().size());
      kernel_tensor_args[input_tensor_size + i]->Resize(dim);
      free(output_tensor_shapes[i]);
    }
    VLOG(6) << "End InferShape: " << cinn_kernel_info_.fn_name;
  }

 private:
  CINNKernelInfo cinn_kernel_info_;

  std::vector<cinn_pod_value_t> func_args_;
};

CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const phi::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place) {
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cinn_kernel_info());
  op_ = op;
  input_tensor_size = op->num_operands();
  output_tensor_size = op->num_results();

  place_ = place;

  InitInputsOutputsIds(op, *value_exec_info);

  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);

    auto var_name = value_exec_info->GetVarName(in);
    auto tensor = value_exec_info->GetScope()
                      ->FindVar(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);
  }

  if (op->HasAttribute("exec_backend")) {
    place_ = op->attribute("exec_backend")
                 .dyn_cast<paddle::dialect::PlaceAttribute>()
                 .data();
  }
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value result = op->result(i);
    auto var_name = value_exec_info->GetVarName(result);

    auto tensor = value_exec_info->GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);
    auto alloc_tensor_type =
        result.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    tensor->set_type(
        paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    for (size_t j = 0; j < alloc_tensor_type.dims().size(); ++j) {
      if (alloc_tensor_type.dims()[j] < 0) {
        need_update_shape = true;
        continue;
      }
    }
    tensor->Resize(alloc_tensor_type.dims());
  }
}

void CinnJitInstruction::Run() {
#if defined(PADDLE_WITH_CUDA)
  void* running_stream = nullptr;
  bool is_gpu = false;

  if (place_.GetType() == phi::AllocationType::GPU) {
    is_gpu = true;
    running_stream =
        static_cast<void*>(static_cast<phi::GPUContext*>(dev_ctx_)->stream());
  }

  fn_ptr_impl_->InitFuncArgs(tensor_args_);

  if (FLAGS_cinn_bucket_compile && need_update_shape) {
    fn_ptr_impl_->InferShape(
        tensor_args_, input_tensor_size, output_tensor_size);
  }
  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    dev_ctx_->Alloc(tensor_args_[i], tensor_args_[i]->dtype());
  }

  // 2. exexute kernel
  fn_ptr_impl_->Run(tensor_args_, running_stream, is_gpu);

  // std::cerr << "fin run cinn jit\n";
  dev_ctx_->Wait();

  for (int i = 0; i < input_tensor_size; ++i) {
    std::cerr << "input i " << i << "\t" << *tensor_args_[i] << std::endl;
    std::cerr << tensor_args_[i] << std::endl;
  }
  std::cerr << "=============\n";
  for (int i = input_tensor_size; i < tensor_args_.size(); ++i) {
    std::cerr << "out i " << i - input_tensor_size << "\t" << *tensor_args_[i]
              << std::endl;
    std::cerr << tensor_args_[i] << std::endl;
  }
#else
  VLOG(0) << "Not Supported: cinn jit instruction currently does not "
             "support non-CUDA kernel";
#endif
}

const std::string& CinnJitInstruction::Name() const {
  static const std::string name = "cinn_jit";
  return name;
}

}  // namespace framework
}  // namespace paddle
