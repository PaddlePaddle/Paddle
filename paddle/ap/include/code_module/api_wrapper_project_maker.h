// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <fstream>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/code_module/func_declare.h"
#include "paddle/ap/include/code_module/project.h"

namespace ap::code_module {

struct ApiWrapperProjectMaker {
  adt::Result<Project> Make(const std::vector<FuncDeclare>& func_declares) {
    ADT_LET_CONST_REF(nested_files, MakeNestedFiles(func_declares));
    ADT_LET_CONST_REF(compile_cmd, GetCompileCmd());
    ADT_LET_CONST_REF(so_relative_path, GetSoRelativePath());
    axpr::AttrMap<axpr::SerializableValue> others;
    return Project{nested_files, compile_cmd, so_relative_path, others};
  }

  adt::Result<Directory<File>> MakeNestedFiles(
      const std::vector<FuncDeclare>& func_declares) {
    Directory<File> directory;
    ADT_LET_CONST_REF(file_content,
                      GenerateApiWrapperCFileContent(func_declares));
    directory.dentry2file->Set("api_wrapper.c", FileContent{file_content});
    return directory;
  }

  adt::Result<std::string> GenerateApiWrapperCFileContent(
      const std::vector<FuncDeclare>& func_declares) {
    std::ostringstream ss;
    ss << "#include <stdint.h>" << std::endl << std::endl;
    for (const auto& func_declare : func_declares) {
      ADT_RETURN_IF_ERR(GenerateCCode4FuncDeclare(&ss, func_declare));
      ss << std::endl;
    }
    return ss.str();
  }

  adt::Result<adt::Ok> GenerateCCode4FuncDeclare(
      std::ostringstream* ss, const FuncDeclare& func_declare) {
    (*ss) << "void " << func_declare->func_id
          << "(void* ret, void* f, void** args) {" << std::endl;
    ADT_LET_CONST_REF(func_ptr_var, DeclareFuncPtrType(func_declare, "func"));
    (*ss) << "  " << func_ptr_var << " = f"
          << ";\n";
    ADT_LET_CONST_REF(func_call_str,
                      GenerateFuncCall(func_declare, "func", "args"));
    if (IsVoidRet(func_declare)) {
      (*ss) << "  " << func_call_str << ";\n";
    } else {
      ADT_LET_CONST_REF(ret_type, GenCode4ArgType(func_declare->ret_type));
      (*ss) << "  *(" << ret_type << "*)ret = " << func_call_str << ";\n";
    }
    (*ss) << "}\n" << std::endl;
    return adt::Ok{};
  }

  bool IsVoidRet(const FuncDeclare& func_declare) {
    return func_declare->ret_type.Match(
        [&](const axpr::DataType& data_type) -> bool {
          return data_type.template Has<axpr::CppDataType<adt::Undefined>>();
        },
        [&](const axpr::PointerType& pointer_type) -> bool { return false; });
  }

  adt::Result<std::string> GenerateFuncCall(const FuncDeclare& func_declare,
                                            const std::string& func_var_name,
                                            const std::string& args_var_name) {
    std::ostringstream ss;
    ss << func_var_name << "(";
    for (size_t i = 0; i < func_declare->arg_types->size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      const auto& arg_type = func_declare->arg_types->at(i);
      ADT_LET_CONST_REF(arg_type_str, GenCode4ArgType(arg_type));
      ss << "*(" << arg_type_str << "*)" << args_var_name << "[" << i << "]";
    }
    ss << ")";
    return ss.str();
  }

  adt::Result<std::string> DeclareFuncPtrType(const FuncDeclare& func_declare,
                                              const std::string& func_name) {
    std::ostringstream ss;
    ADT_LET_CONST_REF(ret_type, GenCode4ArgType(func_declare->ret_type));
    ss << ret_type << "(*" << func_name << ")(";
    int i = 0;
    for (const auto& arg_type : *func_declare->arg_types) {
      if (i++ > 0) {
        ss << ", ";
      }
      ADT_LET_CONST_REF(arg_type_str, GenCode4ArgType(arg_type));
      ss << arg_type_str;
    }
    ss << ")";
    return ss.str();
  }

  adt::Result<std::string> GenCode4ArgType(const ArgType& arg_type) {
    return arg_type.Match(
        [&](const axpr::DataType& data_type) -> adt::Result<std::string> {
          return GenCode4DataType(data_type);
        },
        [&](const axpr::PointerType& pointer_type) -> adt::Result<std::string> {
          return GenCode4PointerType(pointer_type);
        });
  }

  adt::Result<std::string> GenCode4DataType(const axpr::DataType& data_type) {
    using RetT = adt::Result<std::string>;
    return data_type.Match(
        [&](axpr::CppDataType<bool>) -> RetT { return "bool"; },
        [&](axpr::CppDataType<int8_t>) -> RetT { return "int8_t"; },
        [&](axpr::CppDataType<uint8_t>) -> RetT { return "uint8_t"; },
        [&](axpr::CppDataType<int16_t>) -> RetT { return "int16_t"; },
        [&](axpr::CppDataType<uint16_t>) -> RetT { return "uint16_t"; },
        [&](axpr::CppDataType<int32_t>) -> RetT { return "int32_t"; },
        [&](axpr::CppDataType<uint32_t>) -> RetT { return "uint32_t"; },
        [&](axpr::CppDataType<int64_t>) -> RetT { return "int64_t"; },
        [&](axpr::CppDataType<uint64_t>) -> RetT { return "uint64_t"; },
        [&](axpr::CppDataType<float>) -> RetT { return "float"; },
        [&](axpr::CppDataType<double>) -> RetT { return "double"; },
        [&](axpr::CppDataType<axpr::bfloat16>) -> RetT {
          return adt::errors::TypeError{
              "bfloat16 are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::float8_e4m3fn>) -> RetT {
          return adt::errors::TypeError{
              "float8_e4m3fn are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::float8_e5m2>) -> RetT {
          return adt::errors::TypeError{
              "float8_e5m2 are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::float16>) -> RetT {
          return adt::errors::TypeError{
              "float16 are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::complex64>) -> RetT {
          return adt::errors::TypeError{
              "complex64 are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::complex128>) -> RetT {
          return adt::errors::TypeError{
              "complex128 are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<axpr::pstring>) -> RetT {
          return adt::errors::TypeError{
              "pstring are not allowed being used by so function"};
        },
        [&](axpr::CppDataType<adt::Undefined>) -> RetT { return "void"; });
  }

  adt::Result<std::string> GenCode4PointerType(
      const axpr::PointerType& pointer_type) {
    using RetT = adt::Result<std::string>;
    return pointer_type.Match(
        [&](axpr::CppPointerType<bool*>) -> RetT { return "bool*"; },
        [&](axpr::CppPointerType<int8_t*>) -> RetT { return "int8_t*"; },
        [&](axpr::CppPointerType<uint8_t*>) -> RetT { return "uint8_t*"; },
        [&](axpr::CppPointerType<int16_t*>) -> RetT { return "int16_t*"; },
        [&](axpr::CppPointerType<uint16_t*>) -> RetT { return "uint16_t*"; },
        [&](axpr::CppPointerType<int32_t*>) -> RetT { return "int32_t*"; },
        [&](axpr::CppPointerType<uint32_t*>) -> RetT { return "uint32_t*"; },
        [&](axpr::CppPointerType<int64_t*>) -> RetT { return "int64_t*"; },
        [&](axpr::CppPointerType<uint64_t*>) -> RetT { return "uint64_t*"; },
        [&](axpr::CppPointerType<float*>) -> RetT { return "float*"; },
        [&](axpr::CppPointerType<double*>) -> RetT { return "double*"; },
        [&](axpr::CppPointerType<axpr::bfloat16*>) -> RetT { return "void*"; },
        [&](axpr::CppPointerType<axpr::float8_e4m3fn*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<axpr::float8_e5m2*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<axpr::float16*>) -> RetT { return "void*"; },
        [&](axpr::CppPointerType<axpr::complex64*>) -> RetT { return "void*"; },
        [&](axpr::CppPointerType<axpr::complex128*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<axpr::pstring*>) -> RetT { return "void*"; },
        [&](axpr::CppPointerType<void*>) -> RetT { return "void*"; },
        [&](axpr::CppPointerType<const bool*>) -> RetT { return "bool*"; },
        [&](axpr::CppPointerType<const int8_t*>) -> RetT { return "int8_t*"; },
        [&](axpr::CppPointerType<const uint8_t*>) -> RetT {
          return "uint8_t*";
        },
        [&](axpr::CppPointerType<const int16_t*>) -> RetT {
          return "int16_t*";
        },
        [&](axpr::CppPointerType<const uint16_t*>) -> RetT {
          return "uint16_t*";
        },
        [&](axpr::CppPointerType<const int32_t*>) -> RetT {
          return "int32_t*";
        },
        [&](axpr::CppPointerType<const uint32_t*>) -> RetT {
          return "uint32_t*";
        },
        [&](axpr::CppPointerType<const int64_t*>) -> RetT {
          return "int64_t*";
        },
        [&](axpr::CppPointerType<const uint64_t*>) -> RetT {
          return "uint64_t*";
        },
        [&](axpr::CppPointerType<const float*>) -> RetT { return "float*"; },
        [&](axpr::CppPointerType<const double*>) -> RetT { return "double*"; },
        [&](axpr::CppPointerType<const axpr::bfloat16*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::float8_e4m3fn*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::float8_e5m2*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::float16*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::complex64*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::complex128*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const axpr::pstring*>) -> RetT {
          return "void*";
        },
        [&](axpr::CppPointerType<const void*>) -> RetT { return "void*"; });
  }

  adt::Result<std::string> GetCompileCmd() {
    return "gcc  -fPIC -shared api_wrapper.c -o api_wrapper.so";
  }

  adt::Result<std::string> GetSoRelativePath() { return "api_wrapper.so"; }
};

}  // namespace ap::code_module
