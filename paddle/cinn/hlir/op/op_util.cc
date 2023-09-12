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

#include "paddle/cinn/hlir/op/op_util.h"

#include <string>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"

namespace cinn {
namespace hlir {

CINNSchedule GetElementwiseScheduleFunc(
    const std::vector<std::vector<int>>& output_shapes,
    const Target& target,
    bool vectorizable) {
  return CINNSchedule([=](lang::Args args, lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input argument of ElementwiseSchedule is "
                            "empty! Please check.\n";
    common::CINNValuePack arg_pack = args[0];
    CHECK_GT(arg_pack.size(), 0U)
        << "arg_pack.size() must contains at least one element.";
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    CHECK(!vec_ast.empty());
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    pe::IRElementwiseSchedule(ir_sch, output_shapes.front(), target);
    std::vector<common::CINNValue> res{
        common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = common::CINNValuePack{res};
  });
}

CINNSchedule GetInjectiveScheduleFunc(
    const std::vector<std::vector<int>>& output_shapes,
    const Target& target,
    bool vectorizable) {
  return CINNSchedule([=](lang::Args args, lang::RetValue* ret) {
    CHECK(!args.empty()) << "The input argument of InjectiveSchedule is "
                            "empty! Please check.\n";
    common::CINNValuePack arg_pack = args[0];
    std::vector<Expr> vec_ast;
    for (int i = 0; i < arg_pack.size(); i++) {
      if (arg_pack[i].is_expr()) {
        Expr temp = arg_pack[i];
        vec_ast.emplace_back(temp);
      }
    }
    CHECK(!vec_ast.empty());
    ir::ModuleExpr mod_expr(vec_ast);
    ir::IRSchedule ir_sch(mod_expr);
    ir_sch.MergeExprs();
    pe::IRInjectiveSchedule(ir_sch, output_shapes.front(), target);
    /*if (target.arch == Target::Arch::NVGPU) {
      pe::IRInjectiveSchedule(ir_sch, output_shapes.front(), target);
    } else if (target.arch == Target::Arch::X86) {
      pe::IRScheduleInjectiveCPU(ir_sch, output_shapes.front(), target,
    vectorizable);
    }*/
    std::vector<common::CINNValue> res{
        common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
    *ret = common::CINNValuePack{res};
  });
}

std::string GetExternFuncName(const common::Target& target,
                              const common::Type& type,
                              const std::string& func_name,
                              const bool need_cinn,
                              const bool need_target,
                              const bool need_type) {
  std::string func_proto_name;
  if (need_cinn) {
    func_proto_name.append("cinn_");
  }
  if (need_target) {
    if (target.arch == common::Target::Arch::NVGPU) {
      func_proto_name.append("nvgpu_");
    } else if (target.arch == common::Target::Arch::X86) {
      func_proto_name.append("host_");
    } else {
      LOG(FATAL) << func_name
                 << " only supports X86 and NVGPU! Please Check.\n";
    }
  }
  func_proto_name.append(func_name);
  if (!need_type) {
    return func_proto_name;
  }
  func_proto_name.append("_");
  if (type.is_bool()) {
    func_proto_name.append("bool");
  } else if (type.is_float(8)) {
    func_proto_name.append("fp8");
  } else if (type.is_float16()) {
    func_proto_name.append("fp16");
  } else if (type.is_bfloat16()) {
    func_proto_name.append("bf16");
  } else if (type.is_float(32)) {
    func_proto_name.append("fp32");
  } else if (type.is_float(64)) {
    func_proto_name.append("fp64");
  } else if (type.is_int(8)) {
    func_proto_name.append("int8");
  } else if (type.is_int(16)) {
    func_proto_name.append("int16");
  } else if (type.is_int(32)) {
    func_proto_name.append("int32");
  } else if (type.is_int(64)) {
    func_proto_name.append("int64");
  } else if (type.is_uint(8)) {
    func_proto_name.append("uint8");
  } else if (type.is_uint(16)) {
    func_proto_name.append("uint16");
  } else if (type.is_uint(32)) {
    func_proto_name.append("uint32");
  } else if (type.is_uint(64)) {
    func_proto_name.append("uint64");
  } else {
    LOG(FATAL) << "Can not find type: " << type
               << " for extern function. Please Check.\n";
  }
  return func_proto_name;
}

}  // namespace hlir
}  // namespace cinn
