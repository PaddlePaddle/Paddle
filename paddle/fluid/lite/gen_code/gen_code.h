// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/framework.pb.h"
#include "paddle/fluid/lite/core/program.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/model_parser/cpp/op_desc.h"
#include "paddle/fluid/lite/model_parser/desc_apis.h"
#include "paddle/fluid/lite/utils/string.h"

namespace paddle {
namespace lite {
namespace gencode {

struct TensorRepr {
  TensorRepr() = default;
  TensorRepr(PrecisionType dtype, const std::vector<int64_t> &ddim,
             void *raw_data, size_t num_bytes)
      : dtype(dtype), ddim(ddim), raw_data(raw_data), num_bytes(num_bytes) {}

  PrecisionType dtype;
  lite::DDim ddim;
  const void *raw_data;
  size_t num_bytes{};
};

class Module {
  std::vector<cpp::OpDesc> ops;
  std::vector<TensorRepr> weights;
  std::vector<std::string> tmp_vars_;
  std::stringstream stream_;
  std::set<std::string> kernel_kinds_;
  std::set<std::string> op_kinds_;

  int line_indent_{};
  const int indent_unit_{2};

 public:
  void NewOp(const cpp::OpDesc &desc) { ops.push_back(desc); }
  void NewWeight(const TensorRepr &x) { weights.push_back(x); }
  void NewTmpVar(const std::string &x) { tmp_vars_.push_back(x); }

  std::stringstream &stream() { return stream_; }

  void AddHeaderIncludeGenCode();

  void AddNamespaceBegin() {
    Line("namespace paddle {");
    Line("namespace gencode{");
    Line("");
  }

  void AddNamespaceEnd() {
    Line("");
    Line("}  // namespace gencode");
    Line("}  // namespace paddle");
  }

  void AddInitFuncBegin() {
    Line("void PaddlePredictor::Init() {");
    Line("");
    IncIndent();
  }

  void AddInitFuncEnd() {
    DecIndent();
    Line("");
    Line("}");
  }

  void AddScopeDecl() {
    Line("lite::Scope* scope = static_cast<lite::Scope*>(raw_scope_);");

    // clang-format off
    Line("lite::Scope* exec_scope = static_cast<lite::Scope*>(raw_exe_scope_);");  // NOLINT
    // clang-format on

    // Create feed and fetch in exec_scope.
    Line(string_format("exec_scope->Var(%s);", Repr("feed").c_str()));
    Line(string_format("exec_scope->Var(%s);", Repr("fetch").c_str()));
  }

  void AddValidPlaceDecl() {
    // clang-format off
    Line("std::vector<lite::Place> valid_places({lite::Place({TARGET(kX86), PRECISION(kFloat), DATALAYOUT(kNCHW)}), lite::Place({TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny)})});");  // NOLINT
    // clang-format on
  }

  void AddMemberCast() {
    Line("// Cast the raw members");
    // clang-format off
    Line(string_format("auto& ops = *static_cast<std::vector<std::shared_ptr<lite::OpLite>>*>(raw_ops_);"));  // NOLINT
    Line(string_format("auto& kernels = *static_cast<std::vector<std::unique_ptr<lite::KernelBase>>*>(raw_kernels_);"));  // NOLINT
    // clang-format on
    Line("");
  }

  void AddWeight(const std::string &name, const TensorRepr &tensor);

  void AddTmpVar(const std::string &x) {
    Line(string_format("// Create temporary variable: %s", x.c_str()));
    Line(string_format("exec_scope->Var(%s);", Repr(x).c_str()));
    Line("");
  }

  void AddOp(const cpp::OpDesc &op);

  void AddOpDescHelper(const std::string &op_id, const cpp::OpDesc &desc);

  void AddOpCompileDeps() {
    Line("");
    Line("// Add Operator compile deps");
    for (auto &op_type : op_kinds_) {
      Line(string_format("USE_LITE_OP(%s)", op_type.c_str()));
    }
    Line("");
  }
  void AddKernelCompileDeps() {
    Line("// Add Kernel compile deps");

    std::string op_type, alias;
    Place place;
    for (auto &kernel_type : kernel_kinds_) {
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      Line(string_format("USE_LITE_KERNEL(%s, %s, %s, %s, %s)",  //
                         op_type.c_str(),                        //
                         TargetRepr(place.target).c_str(),
                         PrecisionRepr(place.precision).c_str(),
                         DataLayoutRepr(place.layout).c_str(), alias.c_str()));
    }
  }

 private:
  std::string WeightUniqueName() const {
    return "w_" + std::to_string(weight_counter_++);
  }
  std::string TmpVarUniqueName() const {
    return "tmp_" + std::to_string(tmp_var_counter_++);
  }
  std::string OpUniqueName() const {
    return "op_" + std::to_string(op_counter_++);
  }
  std::string KernelUniqueName() const {
    return "kernel_" + std::to_string(kernel_counter_++);
  }

  std::string DataRepr(const std::string &raw_data, PrecisionType dtype);

  void IncIndent() { line_indent_++; }
  void DecIndent() { line_indent_--; }

  void Line(const std::string &x) {
    std::string indent_str(line_indent_ * indent_unit_, ' ');
    stream() << indent_str << x << "\n";
  }

 private:
  mutable int weight_counter_{};
  mutable int tmp_var_counter_{};
  mutable int op_counter_{};
  mutable int kernel_counter_{};
};

class ProgramCodeGenerator {
 public:
  ProgramCodeGenerator(const framework::proto::ProgramDesc &program,
                       const lite::Scope &exec_scope)
      : program_(program), exec_scope_(exec_scope) {
    LOG(INFO) << program.DebugString();
  }

  std::string GenCode() {
    Module m;
    m.AddHeaderIncludeGenCode();
    m.AddNamespaceBegin();
    m.AddInitFuncBegin();
    m.AddMemberCast();
    m.AddScopeDecl();
    m.AddValidPlaceDecl();

    AddWeights(&m);
    AddTmpVars(&m);
    AddOps(&m);

    m.AddInitFuncEnd();
    m.AddNamespaceEnd();

    m.AddOpCompileDeps();
    m.AddKernelCompileDeps();

    return m.stream().str();
  }

  void AddWeights(Module *m) {
    for (auto &var : program_.blocks(0).vars()) {
      if (var.persistable()) {
        auto name = var.name();
        if (name == "feed" || name == "fetch") continue;
        const auto &tensor = exec_scope_.FindVar(name)->Get<lite::Tensor>();
        TensorRepr repr;
        TensorToRepr(tensor, &repr);
        m->AddWeight(name, repr);
      }
    }
  }
  void AddTmpVars(Module *m) {
    for (auto &var : program_.blocks(0).vars()) {
      if (!var.persistable()) {
        m->AddTmpVar(var.name());
      }
    }
  }
  void AddOps(Module *m) {
    for (auto &op : program_.blocks(0).ops()) {
      pb::OpDesc pb_desc(op);
      cpp::OpDesc cpp_desc;
      TransformOpDescPbToCpp(pb_desc, &cpp_desc);
      m->AddOp(cpp_desc);
    }
  }

 private:
  void TensorToRepr(const lite::Tensor &tensor, TensorRepr *repr) {
    repr->ddim = tensor.dims();
    // TODO(Superjomn) support other types.
    repr->dtype = PRECISION(kFloat);
    repr->raw_data = tensor.data<float>();
    repr->num_bytes = repr->ddim.production() * sizeof(float);
  }

 private:
  const framework::proto::ProgramDesc &program_;
  const lite::Scope &exec_scope_;
};

}  // namespace gencode
}  // namespace lite
}  // namespace paddle
