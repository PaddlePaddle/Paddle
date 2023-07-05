// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/codegen_c.h"

#include <fstream>
#include <string>

#include "paddle/cinn/backends/extern_func_emitter.h"
#include "paddle/cinn/backends/extern_func_emitter_builtin.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/utils/ir_verify.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/remove_nested_block.h"
#include "paddle/cinn/runtime/cpu/thread_backend.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/cinn/utils/string.h"

//! Root of the builtin code.
DECLARE_string(cinn_x86_builtin_code_root);

namespace cinn {
namespace backends {
using namespace utils;  // NOLINT
using cinn::common::float16;

const char *kCKeywordRestrict = "__restrict__";

void CodeGenC::Compile(const ir::Module &module, const Outputs &outputs) {
  ir::IrVerify(Expr(module));

  if (!outputs.c_header_name.empty()) {
    auto source = Compile(module, OutputKind::CHeader);
    std::ofstream file(outputs.c_header_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_header_name;
    file << source;
    file.close();
    LOG(WARNING) << "Output C header to file " << outputs.c_header_name;
  }

  if (!outputs.c_source_name.empty()) {
    auto source = Compile(module, OutputKind::CImpl);
    std::ofstream file(outputs.c_source_name);
    CHECK(file.is_open()) << "failed to open file " << outputs.c_source_name;
    file << source;
    file.close();
    LOG(WARNING) << "Output C source to file " << outputs.c_source_name;
  }
}

CodeGenC::CodeGenC(Target target) : ir::IrPrinter(ss_) {}

std::string CodeGenC::Compile(const ir::Module &module,
                              OutputKind output_kind) {
  if (output_kind == OutputKind::CHeader) {
    GenerateHeaderFile(module);
  } else if (output_kind == OutputKind::CImpl) {
    PrintIncludes();

    if (inline_builtin_codes_) PrintBuiltinCodes();

    std::vector<ir::Buffer> buffers;
    for (auto &buffer : module->buffers) {
      buffers.emplace_back(buffer.as_buffer_ref());
    }

    for (auto &func : module.functions()) {
      Compile(func);
    }
  } else {
    LOG(FATAL) << "Not supported OutputKind";
  }
  return ss_.str();
}
std::string CodeGenC::Compile(const ir::LoweredFunc &function) {
  CHECK(function.defined());
  Print(function);
  os() << "\n\n";
  return ss_.str();
}

std::string CodeGenC::GetTypeName(Type type) {
  // common scalar type
#define GET_SCALAR_TYPE(pred_expr, scalar_name) \
  if (pred_expr) {                              \
    return scalar_name;                         \
  }

  GET_SCALAR_TYPE(type.is_void(), "void");
  GET_SCALAR_TYPE(type.is_bool(), "bool");

  GET_SCALAR_TYPE(type.is_int(8), "int8_t");
  GET_SCALAR_TYPE(type.is_int(16), "int16_t");
  GET_SCALAR_TYPE(type.is_int(32), "int32_t");
  GET_SCALAR_TYPE(type.is_int(64), "int64_t");

  GET_SCALAR_TYPE(type.is_uint(8), "uint8_t");
  GET_SCALAR_TYPE(type.is_uint(16), "uint16_t");
  GET_SCALAR_TYPE(type.is_uint(32), "uint32_t");
  GET_SCALAR_TYPE(type.is_uint(64), "uint64_t");

  GET_SCALAR_TYPE(type.is_bfloat16(), "bfloat16");
  GET_SCALAR_TYPE(type.is_float16(), "float16");
  GET_SCALAR_TYPE(type.is_float(32), "float")
  GET_SCALAR_TYPE(type.is_float(64), "double")
#undef GET_SCALAR_TYPE

  // customized_type
  if (type.is_customized_type()) {
    CHECK(!type.customized_type().empty()) << "customized_type can't be empty.";
    auto customized_name = type.customized_type();
    // get name of a cuda built-in vector type, it is started with a
    // 'CudaVectorType::' prefix
    if (utils::Startswith(customized_name,
                          common::customized_type::kcuda_builtin_vector_t)) {
      customized_name.erase(
          0, strlen(common::customized_type::kcuda_builtin_vector_t));
    }
    return customized_name;
  }

  // other types are not implemented yet
  CINN_NOT_IMPLEMENTED
  return "";
}

std::string CodeGenC::GetTypeRepr(Type type) {
  std::string str;
  if (type.is_cpp_const()) {
    str = "const ";
  }

  str += GetTypeName(type);
  if (type.is_cpp_handle()) {
    str += "*";
  } else if (type.is_cpp_handle2()) {
    str += "**";
  }
  return str;
}
void CodeGenC::Visit(const ir::IntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::UIntImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::FloatImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::StringImm *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Add *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Sub *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mul *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Div *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Mod *op) {
  auto copied = op->b();
  optim::Simplify(&copied);
  if (copied.is_constant()) {
    int temp = static_cast<int>(copied.get_constant());
    if ((temp & (temp - 1)) == 0) {
      os() << "(";
      Print(op->a());
      os() << " & ";
      os() << std::to_string(temp - 1);
      os() << ")";
      return;
    }
  }
  PrintBinaryOp("%", op);
}
void CodeGenC::Visit(const ir::EQ *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::NE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::LE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GT *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::GE *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::And *op) { PrintBinaryOp("&&", op); }
void CodeGenC::Visit(const ir::Or *op) { PrintBinaryOp("||", op); }
void CodeGenC::Visit(const ir::Min *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Max *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Minus *op) { IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Not *op) {
  os() << "(!";
  IrPrinter::Print(op->v());
  os() << ")";
}
void CodeGenC::Visit(const ir::Cast *op) { PrintCastExpr(op->type(), op->v()); }
void CodeGenC::Visit(const ir::For *op) {
  Expr extent = op->extent;
  Expr min = op->min;
  int num_task = 1;
  if (op->is_parallel()) {
    os() << "int num_task = max_concurrency();\n";
    DoIndent();
    os() << "omp_set_num_threads(num_task);\n";
    DoIndent();
    os() << "auto flambda = [=](int task_id, int num_task) -> int {\n";
    IncIndent();
    DoIndent();
    os() << "int n_per_task = ";
    Expr num_task_var = Var("num_task");
    Print((op->extent + num_task_var - 1) / num_task_var);
    os() << ";\n";
    CHECK_EQ(min.as_int32(), 0);
    auto task_id = Var("task_id");
    auto n_per_task = Var("n_per_task");
    min = task_id * n_per_task;
    extent = (task_id + 1) * n_per_task;
    DoIndent();
  }
  os() << "for (";
  os() << GetTypeRepr(Int(32));
  os() << " " << op->loop_var->name;
  os() << " = ";
  Print(min);
  os() << "; ";
  os() << op->loop_var->name;
  os() << " < ";
  Print(op->extent);
  if (op->is_parallel()) {
    os() << " && ";
    os() << op->loop_var->name;
    os() << " < ";
    Print(extent);
  }
  os() << "; ";

  os() << op->loop_var->name;
  os() << " += 1";
  os() << ") ";

  Print(op->body);
  if (op->is_parallel()) {
    os() << "\n";
    DoIndent();
    os() << "return 0;\n";
    DecIndent();
    DoIndent();
    os() << "};\n";
    os() << "#pragma omp parallel num_threads(num_task)\n";
    DoIndent();
    os() << "{\n";
    IncIndent();
    DoIndent();
    os() << "int task_id = omp_get_thread_num();\n";
    DoIndent();
    os() << "flambda(task_id, num_task);\n";
    DecIndent();
    DoIndent();
    os() << "}";
  }
}
void CodeGenC::Visit(const ir::PolyFor *op) {
  os() << "for (";
  os() << GetTypeRepr(Int(32));
  os() << " " << op->iterator->name;
  os() << " = ";
  Print(op->init);
  os() << "; ";
  Print(op->condition);
  os() << "; ";

  os() << op->iterator->name;
  os() << " += ";
  Print(op->inc);
  os() << ") ";

  Print(op->body);
}
void CodeGenC::Visit(const ir::Select *op) {
  os() << "(";
  os() << "(";
  Print(op->condition);
  os() << ") ? ";
  Print(op->true_value);
  os() << " : ";
  Print(op->false_value);
  os() << ")";
}
void CodeGenC::Visit(const ir::IfThenElse *op) {
  os() << "if (";
  Print(op->condition);
  os() << ") {\n";

  if (!op->true_case.As<ir::Block>()) IncIndent();
  DoIndent();
  Print(op->true_case);
  if (!op->true_case.As<ir::Block>()) os() << ";";
  os() << "\n";

  if (!op->true_case.As<ir::Block>()) DecIndent();

  DoIndent();
  os() << "}";

  if (op->false_case.defined()) {
    os() << " else {\n";

    if (!op->true_case.As<ir::Block>()) IncIndent();
    DoIndent();
    Print(op->false_case);
    if (!op->false_case.As<ir::Block>()) os() << ";";
    os() << "\n";
    if (!op->true_case.As<ir::Block>()) DecIndent();

    DoIndent();
    os() << "}";
  }
}
void CodeGenC::Visit(const ir::Block *op) {
  os() << "{\n";

  IncIndent();

  for (int i = 0; i < op->stmts.size() - 1; i++) {
    DoIndent();
    Print(op->stmts[i]);
    os() << ";\n";
  }
  if (op->stmts.size() >= 1) {
    DoIndent();
    Print(op->stmts.back());
    os() << ";";
  }

  DecIndent();
  os() << "\n";
  DoIndent();
  os() << "}";
}
void CodeGenC::Visit(const ir::Call *op) {
  if (op->name == runtime::intrinsic::buffer_malloc) {
    PrintCall_buffer_malloc(op);
  } else if (op->name == runtime::intrinsic::pod_values_to_array_repr) {
    PrintCall_pod_values_to_array(op);
  } else if (op->is_intrinsic_call()) {
    os() << op->name << "(";
    PrintCallArgs(op);
    os() << ")";
  } else if (op->is_cinn_call()) {  // call CINN LoweredFunc
    os() << op->name << "(";
    PrintCallArgs(op);
    os() << ")";
  } else if (op->is_extern_call()) {
    const auto &fn_name = ExternFunctionEmitterRegistry::Global().Lookup(
        ExternFuncID{backend_C, op->name.c_str()});
    if (!fn_name.empty()) {
      ExternFunctionLLVMEmitter emitter(fn_name);
      emitter.BindCodeGen(this);
      emitter.Emit(op);
    } else {
      CHECK(!op->read_args.empty() || !op->write_args.empty());
      os() << op->name << "(";
      PrintCallArgs(op);
      os() << ")";
    }
  } else {
    CINN_NOT_IMPLEMENTED
  }
}
void CodeGenC::PrintCallArgs(const ir::Call *op) {
  if (!op->read_args.empty()) {
    for (int i = 0; i < op->read_args.size() - 1; i++) {
      Print(op->read_args[i]);
      os() << ", ";
    }
    Print(op->read_args.back());
  }
  if (!op->write_args.empty()) {
    if (!op->read_args.empty()) os() << ", ";

    for (int i = 0; i < op->write_args.size() - 1; i++) {
      Print(op->write_args[i]);
      os() << ", ";
    }
    Print(op->write_args.back());
  }
}

void CodeGenC::PrintCall_buffer_malloc(const ir::Call *op) {
  CHECK_EQ(op->read_args.size(), 2UL);
  os() << op->name << "(";
  PrintCastExpr("void*", op->read_args[0]);
  os() << ", ";
  os() << op->read_args[1];
  os() << ")";
}

void CodeGenC::PrintCall_cinn_pod_value_to_(const ir::Call *op) {
  CHECK_EQ(op->read_args.size(), 1UL);
  os() << op->name << "(";
  os() << "&(";
  Print(op->read_args[0]);
  os() << ")";
  os() << ")";
}

void CodeGenC::PrintCall_get_address(const ir::Call *op) {
  CHECK_EQ(op->read_args.size(), 1UL);
  CHECK(op->write_args.empty());
  auto *read_var = op->read_args.front().as_var();
  auto *read_buf = op->read_args.front().as_buffer();
  CHECK(read_var || read_buf) << "Only Var or Buffer can get address";

  if (read_var) {
    if (read_var->type().lanes() <= 1) os() << "&";
    os() << read_var->name;
  } else if (read_buf) {
    if (read_buf->type().lanes() <= 1) os() << "&";
    os() << read_buf->name;
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

void CodeGenC::PrintCall_pod_values_to_array(const ir::Call *op) {
  CHECK(!op->read_args.empty());
  CHECK_EQ(op->write_args.size(), 1UL);
  auto output_var = op->write_args.front().as_var_ref();
  CHECK(output_var.defined());

  std::vector<std::string> arg_names;
  for (auto &arg : op->read_args) {
    auto arg_var = arg.as_var();
    CHECK(arg_var);
    arg_names.push_back(arg_var->name);
  }

  os() << "cinn_pod_value_t " << output_var->name << "[] = ";
  os() << "{ ";

  os() << utils::Join(arg_names, ", ");

  os() << " }";
}

void CodeGenC::Visit(const ir::_Module_ *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_Var_ *op) { os() << op->name; }

void CodeGenC::Visit(const ir::Load *op) {
  Expr dense_strided_ramp = detail::StridedRampBase(op->index(), 1);
  if (dense_strided_ramp.defined()) {  // Loading a continuous Ramp address.
    CHECK(op->type().is_vector());
    PrintStackVecType(op->type().ElementOf(), op->index().type().lanes());
    os() << "::"
         << "Load(";
    os() << op->tensor.As<ir::_Tensor_>()->name;
    os() << ",";
    Print(dense_strided_ramp);
    os() << ")";
  } else if (op->index().type().is_vector()) {
    // gather
    CHECK(op->type().is_vector());
    PrintStackVecType(op->type().ElementOf(), op->index().type().lanes());
    os() << "::Load(";
    os() << op->tensor.As<ir::_Tensor_>()->name;
    os() << ",";
    Print(op->index());
    os() << ")";
  } else if (op->is_addr_tensor()) {
    auto *tensor = op->tensor.As<ir::_Tensor_>();
    os() << tensor->name << "[";
    Print(op->index());
    os() << "]";
  } else {
    IrPrinter::Visit(op);
  }
}

void CodeGenC::Visit(const ir::Store *op) {
  CHECK(op->is_addr_tensor());

  auto *tensor = op->tensor.As<ir::_Tensor_>();
  CHECK(tensor);
  os() << tensor->name << "[";
  Print(op->index());
  os() << "]";
  os() << " = ";
  Print(op->value);
}
void CodeGenC::Visit(const ir::Alloc *op) {
  os() << runtime::intrinsic::buffer_malloc;
  os() << "(";
  os() << "(void*)(0), ";

  auto *buffer = op->destination.As<ir::_Buffer_>();
  os() << buffer->name;
  os() << ")";
}

void CodeGenC::Visit(const ir::Free *op) {
  os() << runtime::intrinsic::buffer_free;
  os() << "(";
  os() << "(void*)(0), ";

  auto *buffer = op->destination.As<ir::_Buffer_>();
  os() << buffer->name;
  os() << ")";
}

void CodeGenC::Visit(const ir::_Buffer_ *op) { os() << op->name; }
void CodeGenC::Visit(const ir::_Tensor_ *op) { os() << op->buffer->name; }
void CodeGenC::Visit(const ir::Let *op) {
  bool is_vec = false;
  CHECK(op->type().valid());
  if (op->body.defined() && op->body.As<ir::Broadcast>()) {
    // broadcast's type is hard to print, so use c++11 auto instead.
    os() << "auto";
    is_vec = true;
  } else {
    os() << GetTypeRepr(op->type());
  }

  os() << " ";
  Print(op->symbol);

  // native C array.
  if (op->type().lanes() > 1 && !is_vec) {
    os() << "[" << op->type().lanes() << "]";
  }

  if (op->body.defined()) {
    os() << " = ";
    Print(op->body);
  }
}

void CodeGenC::Visit(const ir::Reduce *op) {
  LOG(FATAL) << "Reduce IR is just for internal representation, should not be "
                "used for CodeGen.";
}

void CodeGenC::Visit(const ir::Ramp *op) {
  os() << "StackVec<" << op->lanes << "," << GetTypeRepr(op->type().ElementOf())
       << ">::Ramp(";
  Print(op->base);
  os() << ", ";
  Print(op->stride);
  os() << ", ";
  os() << op->lanes;
  os() << ")";
}

void CodeGenC::Visit(const ir::Broadcast *op) {
  os() << "StackVec<" << op->lanes << "," << GetTypeRepr(op->type().ElementOf())
       << ">::Broadcast(";
  Print(op->value);
  os() << ", ";
  os() << op->lanes << ")";
}

void CodeGenC::Visit(const ir::FracOp *op) { ir::IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Sum *op) { ir::IrPrinter::Visit(op); }
void CodeGenC::Visit(const ir::Product *op) { ir::IrPrinter::Visit(op); }

void CodeGenC::PrintCastExpr(const Type &type, Expr e) {
  os() << "((" << GetTypeRepr(type) << ")";
  os() << "(";
  Print(e);
  os() << "))";
}
void CodeGenC::PrintCastExpr(const std::string &type, Expr e) {
  os() << "(" << type << ")";
  os() << "(";
  Print(e);
  os() << ")";
}

void CodeGenC::PrintShape(const std::vector<Expr> &shape,
                          char leftb,
                          char rightb) {
  os() << leftb << " ";

  for (int i = 0; i < shape.size() - 1; i++) {
    Print(shape[i]);
    os() << ", ";
  }
  if (shape.size() > 1) Print(shape.back());

  os() << " " << rightb;
}

void CodeGenC::Visit(const ir::_LoweredFunc_ *op) {
  PrintFunctionDeclaration(op);
  os() << "\n";

  DoIndent();

  CHECK_EQ(op->alloc_output_buffer_exprs.size(),
           op->dealloc_output_buffer_exprs.size())
      << "the count of allocation and deallocaton expressions is not match";

  std::vector<Expr> new_body;

  std::vector<Expr> create_temp_buffers = op->PrepareCreateTempBufferExprs();
  std::vector<Expr> alloca_temp_buffers = op->PrepareAllocTempBufferExprs();
  std::vector<Expr> dealloca_temp_buffers = op->PrepareDeallocTempBufferExprs();
#define APPEND_TO_NEW_BODY(field__) \
  new_body.insert(                  \
      std::end(new_body), std::begin(op->field__), std::end(op->field__));
  APPEND_TO_NEW_BODY(argument_prepare_exprs)
  new_body.insert(std::end(new_body),
                  std::begin(create_temp_buffers),
                  std::end(create_temp_buffers));
  APPEND_TO_NEW_BODY(alloc_output_buffer_exprs)
  new_body.insert(std::end(new_body),
                  std::begin(alloca_temp_buffers),
                  std::end(alloca_temp_buffers));
  APPEND_TO_NEW_BODY(buffer_data_cast_exprs)
  new_body.push_back(op->body);
  new_body.insert(std::end(new_body),
                  std::begin(dealloca_temp_buffers),
                  std::end(dealloca_temp_buffers));
  APPEND_TO_NEW_BODY(dealloc_output_buffer_exprs)

  Expr func_body = ir::Block::Make(new_body);

  optim::RemoveNestedBlock(&func_body);

  Print(func_body);
}
void CodeGenC::PrintIncludes() {
  os() << "#include <cinn_runtime.h>\n";
  os() << "#include <stdio.h>\n";
  os() << "\n";
}

void CodeGenC::PrintFileGuardOpen(const std::string &name) {
  os() << utils::StringFormat("#ifndef _%s_CINN_H_\n", Uppercase(name).c_str());
  os() << utils::StringFormat("#define _%s_CINN_H_\n", Uppercase(name).c_str());
  os() << "\n";
}
void CodeGenC::PrintFileGuardClose(const std::string &module_name) {
  os() << utils::StringFormat("#endif  // _%s_CINN_H_\n",
                              Uppercase(module_name).c_str());
}

void CodeGenC::PrintBufferCreation(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    // Ignore the buffer in other devices.
    if (!buffer->is_on_host()) continue;
    DoIndent();
    auto buffer_ptr_type =
        Type()
            .set_customized_type(common::customized_type::kbuffer_t)
            .set_cpp_handle();
    Var variable = ir::_Var_::Make(buffer->name, buffer_ptr_type);
    auto expr = ir::intrinsics::BufferCreate::Make(buffer);
    expr = ir::Let::Make(variable, expr);
    Print(expr);
    os() << ";\n";
  }
}

void CodeGenC::PrintBufferDestroy(const std::vector<ir::Buffer> &buffers) {
  for (auto &buffer : buffers) {
    DoIndent();
    Print(buffer.DestroyExpr());
    os() << ";\n";
  }
}

void CodeGenC::GenerateHeaderFile(const ir::Module &module) {
  PrintFileGuardOpen(module.name());
  PrintIncludes();

  for (auto &func : module.functions()) {
    PrintFunctionDeclaration(func.As<ir::_LoweredFunc_>());
    os() << ";\n";
    os() << "\n\n";
  }

  PrintFileGuardClose(module.name());
}

void CodeGenC::PrintFuncArg(const ir::Argument &arg) {
  if (arg.is_buffer()) {
    if (arg.is_input()) {
      os() << "const struct cinn_buffer_t *";
    } else {
      os() << "struct cinn_buffer_t *";
    }
  } else if (arg.is_var()) {
    os() << GetTypeRepr(arg.type()) << " ";
    os() << arg.name();
  } else {
    CINN_NOT_IMPLEMENTED
  }
  os() << arg.name();
}

void CodeGenC::PrintRuntimeType(const cinn_type_t &type) {
  if (type == cinn_bool_t()) {
    os() << "cinn_bool_t()";
  } else if (type == cinn_int8_t()) {
    os() << "cinn_int8_t()";
  } else if (type == cinn_int16_t()) {
    os() << "cinn_int16_t()";
  } else if (type == cinn_int32_t()) {
    os() << "cinn_int32_t()";
  } else if (type == cinn_int64_t()) {
    os() << "cinn_int64_t()";
  } else if (type == cinn_uint8_t()) {
    os() << "cinn_uint8_t()";
  } else if (type == cinn_uint16_t()) {
    os() << "cinn_uint16_t()";
  } else if (type == cinn_uint32_t()) {
    os() << "cinn_uint32_t()";
  } else if (type == cinn_uint64_t()) {
    os() << "cinn_uint64_t()";
  } else if (type == cinn_bfloat16_t()) {
    os() << "cinn_bfloat16_t()";
  } else if (type == cinn_float16_t()) {
    os() << "cinn_float16_t()";
  } else if (type == cinn_float32_t()) {
    os() << "cinn_float32_t()";
  } else if (type == cinn_float64_t()) {
    os() << "cinn_float64_t()";
  } else {
    LOG(FATAL) << "Unknown type is not supported to print";
  }
}

void CodeGenC::PrintStackVecType(Type type, int lanes) {
  os() << "StackedVec<" << GetTypeRepr(type) << "," << lanes << ">";
}

void CodeGenC::Visit(const ir::PrimitiveNode *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::_BufferRange_ *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::ScheduleBlock *op) { CINN_NOT_IMPLEMENTED }
void CodeGenC::Visit(const ir::ScheduleBlockRealize *op) {
  CINN_NOT_IMPLEMENTED
}

void CodeGenC::Visit(const ir::IntrinsicOp *op) {
  switch (op->getKind()) {
#define __(x)                                     \
  case ir::IntrinsicKind::k##x:                   \
    Visit(llvm::dyn_cast<ir::intrinsics::x>(op)); \
    break;

    INTRINSIC_KIND_FOR_EACH(__)
#undef __
  }
}

void CodeGenC::Visit(const ir::intrinsics::BufferGetDataHandle *op) {
  os() << op->buffer.as_buffer()->name;
  os() << "->";
  os() << "memory";
}

void CodeGenC::Visit(const ir::intrinsics::BufferGetDataConstHandle *op) {
  os() << op->buffer.as_buffer()->name;
  os() << "->";
  os() << "memory";
}

void CodeGenC::Visit(const ir::intrinsics::PodValueToX *op) {
  auto to_type = op->GetOutputType(0);
  if (to_type == type_of<float>()) {
    os() << runtime::intrinsic::pod_value_to_float;
  } else if (to_type == type_of<double>()) {
    os() << runtime::intrinsic::pod_value_to_double;
  } else if (to_type == type_of<float16>()) {
    os() << runtime::intrinsic::pod_value_to_float16;
  } else if (to_type == type_of<bool>()) {
    os() << runtime::intrinsic::pod_value_to_bool;
  } else if (to_type == type_of<int8_t>()) {
    os() << runtime::intrinsic::pod_value_to_int8;
  } else if (to_type == type_of<int16_t>()) {
    os() << runtime::intrinsic::pod_value_to_int16;
  } else if (to_type == type_of<int32_t>()) {
    os() << runtime::intrinsic::pod_value_to_int32;
  } else if (to_type == type_of<int64_t>()) {
    os() << runtime::intrinsic::pod_value_to_int64;
  } else if (to_type == type_of<uint8_t>()) {
    os() << runtime::intrinsic::pod_value_to_uint8;
  } else if (to_type == type_of<uint16_t>()) {
    os() << runtime::intrinsic::pod_value_to_uint16;
  } else if (to_type == type_of<uint32_t>()) {
    os() << runtime::intrinsic::pod_value_to_uint32;
  } else if (to_type == type_of<uint64_t>()) {
    os() << runtime::intrinsic::pod_value_to_uint64;
  } else if (to_type == type_of<void *>()) {
    os() << runtime::intrinsic::pod_value_to_void_p;
  } else if (to_type == type_of<cinn_buffer_t *>()) {
    os() << runtime::intrinsic::pod_value_to_buffer_p;
  } else {
    LOG(FATAL) << "Not supported type: " << to_type;
  }

  os() << "(";
  Print(op->pod_value_ptr);
  os() << ")";
}

void CodeGenC::Visit(const ir::intrinsics::BufferCreate *op) {
  const ir::_Buffer_ *buffer_arg = op->buffer.as_buffer();
  CHECK(buffer_arg);

  os() << runtime::intrinsic::buffer_create;
  os() << "(";
  PrintCastExpr("cinn_device_kind_t", Expr(buffer_arg->target.runtime_arch()));
  os() << "/*target*/, ";
  PrintRuntimeType(runtime::ToRuntimeType(buffer_arg->dtype.ElementOf()));
  os() << ", ";
  PrintShape(buffer_arg->shape);
  if (buffer_arg->data_alignment > 0) {
    os() << ", " << buffer_arg->data_alignment << "/*align*/";
  }
  os() << ")";
}

void CodeGenC::Visit(const ir::intrinsics::GetAddr *op) {
  if (op->data.as_buffer()) {
    os() << "&" << op->data.as_buffer()->name;
  } else if (op->data.as_var()) {
    os() << "&" << op->data.as_var()->name;
  } else {
    os() << "&(";
    Print(op->data);
    os() << ")";
  }
}

void CodeGenC::Visit(const ir::intrinsics::ArgsConstruct *op) {
  os() << runtime::intrinsic::args_construct_repr << "(";
  os() << op->var->name << ", ";
  os() << op->args.size() << ", ";
  for (int i = 0; i < op->args.size() - 1; i++) {
    Print(op->args[i]);
    os() << ", ";
  }
  if (!op->args.empty()) {
    Print(op->args.back());
  }
  os() << ")";
}

void CodeGenC::Visit(const ir::intrinsics::BuiltinIntrin *op) {
  os() << op->name << "(";
  if (!op->args.empty()) {
    for (int i = 0; i < op->args.size() - 1; i++) {
      Print(op->args[i]);
      os() << ", ";
    }
    Print(op->args.back());
  }
  os() << ")";
}

std::string ReadWholeFile(const std::string &path) {
  CHECK(!path.empty());
  std::ifstream file(path);
  CHECK(file.is_open()) << "Failed to open file: " << path;
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

void CodeGenC::PrintBuiltinCodes() {
  CHECK(!FLAGS_cinn_x86_builtin_code_root.empty())
      << "The flag cinn_x86_builtin_code_root should be set first";

  const std::string x86_code_file = "_x86_builtin_source.cc";

  auto source =
      ReadWholeFile(FLAGS_cinn_x86_builtin_code_root + "/" + x86_code_file);

  os() << source << "\n";
}

namespace detail {

Expr StridedRampBase(Expr e, int stride) {
  auto *ramp_n = e.As<ir::Ramp>();
  if (ramp_n) {
    auto *iv = ramp_n->stride.As<ir::IntImm>();
    if (iv && iv->value == stride) return ramp_n->base;
  }
  return Expr();
}

}  // namespace detail

}  // namespace backends

}  // namespace cinn
