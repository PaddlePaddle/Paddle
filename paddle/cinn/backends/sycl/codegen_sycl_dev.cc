// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/sycl/codegen_sycl_dev.h"
#include <glog/logging.h>
#include <paddle/cinn/utils/string.h>

#include <fstream>
#include <set>
#include <unordered_set>

#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify.h"

#include "paddle/cinn/backends/sycl/compiler_sycl.h"
using cinn::backends::syclrtc::NUM;

namespace cinn {
namespace backends {

const std::string &CodeGenSYCL_Dev::GetSourceHeader() {
  std::string source_header =
      R"(#include <sycl/sycl.hpp>
#include "cinn_sycl_runtime_source.h"
typedef sycl::half float16;
)";
  return source_header;
}

CodeGenSYCL_Dev::CodeGenSYCL_Dev(Target target) : CodeGenC(target) {}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module,
                                     bool for_syclrtc) {
  for_syclrtc_ = for_syclrtc;
  auto source = Compile(module, OutputKind::CImpl);

  return source;
}

void CodeGenSYCL_Dev::Compile(const ir::Module &module,
                              const Outputs &outputs) {
  PADDLE_THROW(::common::errors::Fatal("CINN_SYCL_codegen_NOT_IMPLEMENTED"));
}

void CodeGenSYCL_Dev::Compile(const ir::LoweredFunc &func) {
  IrPrinter::Visit(Expr(func));
}

std::vector<Expr> CodeGenSYCL_Dev::GenerateBufferAliasExprs(
    const ir::_LoweredFunc_ *op, const std::vector<ir::Buffer> &temp_buffers) {
  std::set<ir::Buffer> temp_buffer_set(temp_buffers.begin(),
                                       temp_buffers.end());
  // prepare temp buffer alias
  std::vector<Expr> buffer_alias;
  auto tensors = ir::ir_utils::CollectIRNodes(op->body, [&](const Expr *x) {
    return x->as_tensor() && x->as_tensor()->buffer.defined() &&
           temp_buffer_set.count(x->as_tensor()->buffer);
  });

  // unique tensors
  std::set<ir::Tensor> unique_tensors;
  for (auto &e : tensors) {
    unique_tensors.insert(e.as_tensor_ref());
  }

  for (auto &t : unique_tensors) {
    auto data_type = t->type();
    auto data_ptr_type = data_type;
    data_ptr_type.set_cpp_handle();

    Var t_var(t->name, data_ptr_type);
    Var buf_var(t->buffer->name, data_ptr_type);
    buffer_alias.push_back(ir::Let::Make(t_var, buf_var));
  }

  return buffer_alias;
}

void CodeGenSYCL_Dev::Visit(const ir::_LoweredFunc_ *op) {
  // clear names valid within scope when enter a new function
  vectorized_tensor_names_.clear();

  // Print the packed function
  str_ += "// CodeGenSYCL: NOTE: Auto-generated packed function\n";
  str_ += "void ";
  str_ += op->name;
  str_ +=
      "(sycl::queue &Q, sycl::range<3> dimGrid, sycl::range<3> dimBlock, "
      "void** void_args) {\n";
  IncIndent();
  // read void_args
  PrintFunctionDeclaration(op);
  DoIndent();
  str_ += "Q.submit([&](sycl::handler &h) {\n";
  IncIndent();
  DoIndent();
  str_ += "h.parallel_for<class " + GenerateKernelName(op) +
          ">(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), "
          "[=](sycl::nd_item<3> item) "
          "[[intel::kernel_args_restrict]]";
  if (op->cuda_axis_info.valid()) {
    bool has_symbol_in_thread_num = false;
    std::string launch_bounds_max_work_group_size =
        "[[intel::max_work_group_size(";
    for (int i = 0; i < 3; i++) {
      ir::Expr block_dim = op->cuda_axis_info.block_dim(i);
      if (block_dim.is_constant()) {
        launch_bounds_max_work_group_size +=
            std::to_string(block_dim.get_constant());
        if (i < 2) {
          launch_bounds_max_work_group_size += ", ";
        }
      } else {
        has_symbol_in_thread_num = true;
        break;
      }
    }
    launch_bounds_max_work_group_size += ")]]";
    if (!has_symbol_in_thread_num) {
      str_ += launch_bounds_max_work_group_size;
    }
  }
  str_ += "\n";
  // function body
  PrintFunctionBody(op);

  str_ += ");\n";
  DecIndent();
  DoIndent();
  str_ += "});\n";
  DecIndent();
  str_ += "}\n";
}

void CodeGenSYCL_Dev::Visit(const ir::_Var_ *op) {
  if (utils::Startswith(op->name, "threadIdx") ||
      utils::Startswith(op->name, "blockIdx")) {
    if (utils::Startswith(op->name, "threadIdx")) {
      str_ += "(int)item.get_local_id(";
    } else {
      str_ += "(int)item.get_group(";
    }
    if (utils::Endswith(op->name, "x")) {
      str_ += std::to_string(2);
    } else if (utils::Endswith(op->name, "y")) {
      str_ += std::to_string(1);
    } else if (utils::Endswith(op->name, "z")) {
      str_ += std::to_string(0);
    }
    str_ += ")";
  } else {
    str_ += op->name;
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Alloc *op) {
  PADDLE_ENFORCE_NE(
      op->destination.as_buffer(),
      nullptr,
      ::common::errors::InvalidArgument("ir::Alloc's buffer cannot nullptr."));
  PrintTempBufferCreation(op->destination.as_buffer_ref());
}

void CodeGenSYCL_Dev::Visit(const ir::Min *op) {
  str_ += "sycl::min(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Max *op) {
  str_ += "sycl::max(";
  IrPrinter::Visit(op->a());
  str_ += ", ";
  IrPrinter::Visit(op->b());
  str_ += ")";
}

void CodeGenSYCL_Dev::PrintFunctionBody(const ir::_LoweredFunc_ *op) {
  DoIndent();

  std::vector<Expr> new_body;

  auto alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  auto temp_buffer_alias = GenerateBufferAliasExprs(op, op->temp_bufs);
  auto alis_var_exprs = op->CudaAliasVarExprs();

#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(std::end(new_body), std::begin(field__), std::end(field__));
  APPEND_TO_NEW_BODY(alloca_temp_buffers)
  APPEND_TO_NEW_BODY(temp_buffer_alias)
  APPEND_TO_NEW_BODY(alis_var_exprs)

  new_body.push_back(op->body);

  Expr func_body = ir::Block::Make(new_body);

  optim::SimplifyBlocks(&func_body);
  // Make sure that the function's body is wrapped by a block
  if (!func_body.As<ir::Block>()) {
    func_body = ir::Block::Make({func_body});
  }
  IrPrinter::Visit(func_body);
}

void CodeGenSYCL_Dev::PrintFunctionDeclaration(const ir::_LoweredFunc_ *op) {
  for (int i = 0; i < op->args.size(); i++) {
    DoIndent();
    auto &arg = op->args[i];
    if (arg.is_buffer()) {
      // In CUDA kernel, only primitive type is supported, so we replace the
      // buffer with T*j
      if (arg.is_input()) str_ += "const ";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
      // str_ += kCKeywordRestrict;
      str_ += " ";
      str_ += ir::BufferGetTensorName(arg.buffer_arg().As<ir::_Buffer_>());
      str_ += " = (";
      str_ += GetTypeRepr(arg.buffer_arg()->dtype);
      str_ += "* ";
    } else if (arg.is_var()) {
      if (arg.var_arg()->type().is_cpp_handle()) {
        // str_ += kCKeywordRestrict;
      }
      str_ += GetTypeRepr(arg.type());
      str_ += " ";
      str_ += arg.name();
      str_ += " = (";
      str_ += GetTypeRepr(arg.type());
    } else {
      CINN_NOT_IMPLEMENTED
    }
    str_ += ")(*(void **)(void_args[";
    str_ += std::to_string(i);
    str_ += "]));\n";
  }
}

void CodeGenSYCL_Dev::PrintBuiltinCodes() {}

std::string CodeGenSYCL_Dev::Compile(const ir::Module &module,
                                     CodeGenC::OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (for_syclrtc_) {
      str_ += "#ifdef __cplusplus\n";
      str_ += "extern \"C\" {\n";
      str_ += "#endif\n";
    }

    PrintBuiltinCodes();

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    PADDLE_THROW(::common::errors::Fatal("Not supported OutputKind"));
  }

  if (for_syclrtc_) {
    str_ += "\n#ifdef __cplusplus\n";
    str_ += "}\n";
    str_ += "#endif\n";
  }
  return str_;
}

void CodeGenSYCL_Dev::PrintIncludes() { str_ += GetSourceHeader(); }

void CodeGenSYCL_Dev::PrintTempBufferCreation(const ir::Buffer &buffer) {
  PADDLE_ENFORCE_NE(buffer->type(),
                    Void(),
                    ::common::errors::InvalidArgument(
                        "Buffer type cannot be void in CodeGenSYCL_Dev"));
  auto print_gpu_memory = [&](const std::string &mark) {
    str_ += mark;
    str_ += GetTypeRepr(buffer->dtype);
    str_ += " ";
    str_ += buffer->name;
    str_ += " ";

    str_ += "[ ";
    Expr buffer_size(1);
    for (int i = 0; i < buffer->shape.size(); i++) {
      buffer_size = buffer_size * buffer->shape[i];
    }
    optim::Simplify(&buffer_size);
    IrPrinter::Visit(buffer_size);
    str_ += " ]";
  };
  switch (buffer->memory_type) {
    case ir::MemoryType::GPUShared: {
      str_ += "auto ";
      str_ += buffer->name;
      str_ += " = *sycl::ext::oneapi::group_local_memory<";
      str_ += GetTypeRepr(buffer->dtype);
      str_ += "[ ";
      Expr buffer_size(1);
      for (int i = 0; i < buffer->shape.size(); i++) {
        buffer_size = buffer_size * buffer->shape[i];
      }
      optim::Simplify(&buffer_size);
      IrPrinter::Visit(buffer_size);
      str_ += " ]>(item.get_group())";
      break;
    }

    case ir::MemoryType::GPULocal:
      print_gpu_memory("");
      break;

    default:
      PADDLE_THROW(::common::errors::Fatal(
          "SYCL device codegen not support memory %s, %s",
          buffer->name,
          buffer->memory_type));
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Call *op) {
  if (op->name == "__syncthreads") {
    str_ += "sycl::group_barrier(item.get_group())";
    return;
  }
  str_ += op->name;
  str_ += "(";

  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      auto &arg = op->read_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->read_args.back().as_tensor()) {
      str_ += op->read_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->read_args.back());
    }
  }

  if (!op->write_args.empty()) {
    str_ += ", ";
    for (int i = 0; i < op->write_args.size() - 1; i++) {
      auto &arg = op->write_args[i];
      if (arg.as_tensor()) {
        str_ += arg.as_tensor()->name;
        str_ += ", ";
      } else {
        IrPrinter::Visit(arg);
        str_ += ", ";
      }
    }
    if (op->write_args.back().as_tensor()) {
      str_ += op->write_args.back().as_tensor()->name;
    } else {
      IrPrinter::Visit(op->write_args.back());
    }
  }
  // sycl need parameter nd_item
  if ((op->name.find("cinn_block_reduce") != std::string::npos) ||
      (op->name.find("cinn_warp_reduce") != std::string::npos)) {
    str_ += ", item";
  }

  str_ += ")";
}

void CodeGenSYCL_Dev::Visit(const ir::Let *op) {
  PADDLE_ENFORCE_EQ(
      op->type().valid(),
      true,
      ::common::errors::InvalidArgument(
          "ir::Let's op type cannot be valid in CodeGenSYCL_Dev"));

  // identify vectorized tensors by checking their dtypes are customized_type
  // with customized_type::kcuda_builtin_vector_t prefix, and save their names
  if (op->type().is_customized() &&
      utils::Startswith(op->type().customized_type(),
                        common::customized_type::kcuda_builtin_vector_t)) {
    str_ += GetTypeRepr(op->type());
    if (op->type().is_cpp_handle()) {
      str_ += " ";
      // str_ += kCKeywordRestrict;
    }
    str_ += " ";
    IrPrinter::Visit(op->symbol);
    vectorized_tensor_names_.insert(utils::GetStreamCnt(op->symbol));
    // skip "=0" in "half8 temp = 0;" sincethe operator= of half8 may not
    // overloaded.
    if (op->body.As<ir::IntImm>() && op->body.As<ir::IntImm>()->value == 0) {
      return;
    }
    str_ += " = ";
    IrPrinter::Visit(op->body);
  } else {
    CodeGenC::Visit(op);
  }
}

bool CodeGenSYCL_Dev::PrintBuiltinVectorAccess(const ir::LoadStoreAddrMnger *op,
                                               ir::Expr index_expr,
                                               bool is_store) {
  static constexpr char index2suffix[8] = {
      'x', 'y', 'z', 'w', 'v', 'u', 't', 's'};

  // addr of op should be a place of tensor and the index is simple int number
  if (!op->is_addr_tensor() || !index_expr.As<ir::IntImm>()) {
    return false;
  }
  auto *tensor = op->tensor.As<ir::_Tensor_>();
  PADDLE_ENFORCE_NE(tensor,
                    nullptr,
                    ::common::errors::InvalidArgument(
                        "Tensor in CodeGenSYCL_Dev::PrintBuiltinVectorAccess "
                        "cannot be NULL."));

  // identify vectorized tensors by their names
  if (!vectorized_tensor_names_.count(tensor->name)) {
    return false;
  }

  // the index can't exceed the range of cuda built-in vector type
  int index = index_expr.As<ir::IntImm>()->value;
  if (index < 0 || index >= 8) {
    return false;
  }
  if (is_store && tensor->type().is_cpp_handle()) {
    str_ += tensor->name;
    str_ += "[";
    str_ += std::to_string(index);
    str_ += "]";
  } else {
    str_ += tensor->name;
    str_ += (tensor->type().is_cpp_handle() ? "->" : ".");
    str_ += index2suffix[index];
  }
  return true;
}

void CodeGenSYCL_Dev::Visit(const ir::Load *op) {
  // overload this visit function to especially deal with the case when it
  // accesses element at a cuda built-in vector, others still resolve to
  // CodeGenC
  if (!PrintBuiltinVectorAccess(op, op->index(), false)) {
    CodeGenC::Visit(op);
  }
}

void CodeGenSYCL_Dev::Visit(const ir::Store *op) {
  // overload this visit function to especially deal with the case when it
  // accesses element at a cuda built-in vector, others still resolve to
  // CodeGenC
  if (PrintBuiltinVectorAccess(op, op->index(), true)) {
    str_ += " = ";
    IrPrinter::Visit(op->value);
  } else {
    CodeGenC::Visit(op);
  }
}

std::string CodeGenSYCL_Dev::GenerateKernelName(const ir::_LoweredFunc_ *op) {
  std::string kernel_name = "space" + std::to_string(NUM::getNum());
  kernel_name += "_";
  kernel_name += op->name;
  return kernel_name;
}

}  // namespace backends
}  // namespace cinn
