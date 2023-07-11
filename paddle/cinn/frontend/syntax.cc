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

#include "paddle/cinn/frontend/syntax.h"

#include <absl/types/variant.h>

#include <iomanip>
#include <memory>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "paddle/cinn/frontend/paddle/model_parser.h"
#include "paddle/cinn/frontend/paddle_model_to_program.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace frontend {
using hlir::framework::Scope;

void Instruction::PrepareOutputs() {
  auto* op_def = hlir::framework::OpRegistry::Global()->Find(get()->op_type);
  CHECK(op_def) << "No operator called [" << get()->op_type << "]";
  for (int i = 0; i < op_def->num_outputs; i++) {
    get()->outputs.push_back(Variable());
  }
}

Instruction::Instruction(absl::string_view op_type,
                         const std::vector<Variable>& inputs,
                         Program* parent)
    : common::Shared<_Instruction_>(common::make_shared<_Instruction_>()) {
  get()->op_type = std::string(op_type);
  get()->parent_program = parent;
  get()->inputs = inputs;
  PrepareOutputs();
}

Placeholder::operator Variable() const { return var_; }

Variable Program::conv2d(
    const Variable& a,
    const Variable& b,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::layout_transform(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("layout_transform");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::conv2d_NCHWc(
    const Variable& a,
    const Variable& b,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("conv2d_NCHWc");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::depthwise_conv2d(
    const Variable& a,
    const Variable& b,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("depthwise_conv2d");
  instr.SetInputs({a, b});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::pool2d(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("pool2d");
  instr.SetInputs({a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::batchnorm(
    const Variable& a,
    const Variable& scale,
    const Variable& bias,
    const Variable& mean,
    const Variable& variance,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("batch_norm");
  instr.SetInputs({a, scale, bias, mean, variance});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

template <typename PrimType>
Variable Program::primitive_const_scalar(PrimType value,
                                         const std::string& name) {
  Instruction instr("const_scalar");
  instr.SetInputs({});
  instr.SetAttr("value", value);
  AppendInstruction(instr);
  auto out = instr.GetOutput(0);
  out.set_id(name);
  auto out_type = type_of<PrimType>();
  CHECK(out_type.is_float() || out_type.is_int() || out_type.is_bool())
      << "no supported type: " << out_type;
  out->type = out_type;
  out.set_const(true);
  return out;
}

Variable Program::primitive_broadcast_to(
    const Variable& a,
    const std::vector<int>& out_shape,
    const std::vector<int>& broadcast_axes) {
  Instruction instr("broadcast_to");
  instr.SetInputs({a});
  instr.SetAttr("out_shape", out_shape);
  instr.SetAttr("broadcast_axes", broadcast_axes);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::fused_meta_batchnorm_inference(
    const Variable& a,
    const Variable& scale,
    const Variable& bias,
    const Variable& mean,
    const Variable& variance,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  float epsilon = 0.00001f;
  if (attr_store.find("epsilon") != attr_store.end()) {
    epsilon = absl::get<float>(attr_store.at("epsilon"));
  }
  auto eps_var =
      primitive_const_scalar<float>(epsilon, common::UniqName("epsilon"));
  CHECK(!scale->shape.empty()) << "scale's shape is empty.";
  auto broadcast_eps = primitive_broadcast_to(eps_var, scale->shape, {0});
  auto var_add_eps = add(variance, broadcast_eps);
  auto rsrqt_var = primitive_rsqrt(var_add_eps);
  auto new_scale = multiply(rsrqt_var, scale);
  auto neg_mean = primitive_negative(mean);
  auto new_shift = multiply(new_scale, neg_mean);
  auto shift_bias = add(new_shift, bias);
  CHECK(!a->shape.empty()) << "variable a's shape is empty.";
  auto broadcast_new_scale = primitive_broadcast_to(new_scale, a->shape, {1});
  auto broadcast_shift_bias = primitive_broadcast_to(shift_bias, a->shape, {1});
  auto temp_out = multiply(broadcast_new_scale, a);
  auto bn_out = add(temp_out, broadcast_shift_bias);

  return bn_out;
}

Variable Program::fused_batchnorm_inference(
    const Variable& a,
    const Variable& scale,
    const Variable& bias,
    const Variable& mean,
    const Variable& variance,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  float epsilon = 0.00001f;
  if (attr_store.find("epsilon") != attr_store.end()) {
    epsilon = absl::get<float>(attr_store.at("epsilon"));
  }
  auto eps_var =
      primitive_const_scalar<float>(epsilon, common::UniqName("epsilon"));
  CHECK(!scale->shape.empty()) << "scale's shape is empty.";
  auto var_add_eps = elementwise_add(variance, eps_var);
  auto rsrqt_var = primitive_rsqrt(var_add_eps);
  auto new_scale = elementwise_mul(rsrqt_var, scale);
  auto neg_mean = primitive_negative(mean);
  auto new_shift = elementwise_mul(new_scale, neg_mean);
  auto shift_bias = elementwise_add(new_shift, bias);
  auto temp_out = elementwise_mul(a, new_scale, 1);
  auto bn_out = elementwise_add(temp_out, shift_bias, 1);

  return bn_out;
}

Variable Program::scale(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("scale", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::softmax(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("softmax", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::sigmoid(const Variable& a) {
  Instruction instr("sigmoid", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::slice(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("slice", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::dropout_infer(
    const Variable& a,
    const absl::flat_hash_map<std::string, attr_t>& attr_store) {
  Instruction instr("dropout_infer", {a});
  for (auto& iter : attr_store) {
    instr.SetAttr(iter.first, iter.second);
  }
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Instruction& Program::operator[](size_t i) {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
}

const Instruction& Program::operator[](size_t i) const {
  CHECK_LT(i, instrs_.size());
  return instrs_[i];
}

std::ostream& operator<<(std::ostream& os, const Variable& x) {
  os << "Var(" << x->id << ": shape=[" << utils::Join(x->shape, ", ")
     << "], dtype=" << x->type;
  if (x->is_const) {
    os << ", CONST";
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Instruction& instr) {
  os << instr->debug_string();
  return os;
}

std::tuple<std::unique_ptr<Program>,
           absl::flat_hash_map<std::string, Variable>,
           absl::flat_hash_map<std::string, std::string>,
           absl::flat_hash_set<std::string>>
LoadPaddleProgram(const std::string& model_dir,
                  Scope* scope,
                  std::unordered_map<std::string, std::vector<int>>&
                      input_shape_map,  // NOLINT
                  bool is_combined,
                  const common::Target& target) {
  VLOG(1) << "Loading Paddle model from " << model_dir;
  PaddleModelToProgram paddle_to_program(scope, input_shape_map, target);
  return std::make_tuple(paddle_to_program(model_dir, is_combined),
                         paddle_to_program.var_map(),
                         paddle_to_program.var_model_to_program_map(),
                         paddle_to_program.fetch_names());
}

void Program::SetInputs(const std::vector<Variable>& xs) {
  CHECK(!xs.empty()) << "At least one input is needed for a program!";
  for (int i = 0; i < xs.size(); i++) {
    CHECK(!xs[i]->shape.empty())
        << "Found " << i << "-th input's shape is not set yet";
    CHECK(!xs[i]->type.is_unk())
        << "Found " << i << "-th input's type is not set yet";
    inputs_.push_back(xs[i]);
  }
}

void Program::Validate() const {
  // Existing some program don't have input, such as a program only has
  // `fill_constant` CHECK(!inputs_.empty()) << "Inputs of the program is not
  // set yet";
  CHECK(!instrs_.empty()) << "No instruction is added yet";
}

Variable Program::add(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_add", {a, b});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::multiply(const Variable& a, const Variable& b) {
  Instruction instr("elementwise_mul", {a, b});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

#define SYNTAX_PRIM_UNARY_IMPL(name__)                      \
  Variable Program::primitive_##name__(const Variable& a) { \
    Instruction instr(#name__, {a});                        \
    AppendInstruction(instr);                               \
    return instr.GetOutput(0);                              \
  }

SYNTAX_PRIM_UNARY_IMPL(exp);
SYNTAX_PRIM_UNARY_IMPL(erf);
SYNTAX_PRIM_UNARY_IMPL(sqrt);
SYNTAX_PRIM_UNARY_IMPL(log);
SYNTAX_PRIM_UNARY_IMPL(floor);
SYNTAX_PRIM_UNARY_IMPL(ceil);
SYNTAX_PRIM_UNARY_IMPL(round);
SYNTAX_PRIM_UNARY_IMPL(tanh);
SYNTAX_PRIM_UNARY_IMPL(log2);
SYNTAX_PRIM_UNARY_IMPL(log10);
SYNTAX_PRIM_UNARY_IMPL(trunc);
SYNTAX_PRIM_UNARY_IMPL(cos);
SYNTAX_PRIM_UNARY_IMPL(sin);
SYNTAX_PRIM_UNARY_IMPL(cosh);
SYNTAX_PRIM_UNARY_IMPL(tan);
SYNTAX_PRIM_UNARY_IMPL(sinh);
SYNTAX_PRIM_UNARY_IMPL(acos);
SYNTAX_PRIM_UNARY_IMPL(acosh);
SYNTAX_PRIM_UNARY_IMPL(asin);
SYNTAX_PRIM_UNARY_IMPL(asinh);
SYNTAX_PRIM_UNARY_IMPL(atan);
SYNTAX_PRIM_UNARY_IMPL(atanh);

SYNTAX_PRIM_UNARY_IMPL(isnan);
SYNTAX_PRIM_UNARY_IMPL(isfinite);
SYNTAX_PRIM_UNARY_IMPL(isinf);
SYNTAX_PRIM_UNARY_IMPL(bitwise_not);

SYNTAX_PRIM_UNARY_IMPL(negative);
SYNTAX_PRIM_UNARY_IMPL(identity);
SYNTAX_PRIM_UNARY_IMPL(logical_not);
SYNTAX_PRIM_UNARY_IMPL(sign);
SYNTAX_PRIM_UNARY_IMPL(abs);
SYNTAX_PRIM_UNARY_IMPL(rsqrt);

#define SYNTAX_PRIM_BINARY_IMPL(name__)                                        \
  Variable Program::primitive_##name__(const Variable& a, const Variable& b) { \
    Instruction instr(#name__, {a, b});                                        \
    AppendInstruction(instr);                                                  \
    return instr.GetOutput(0);                                                 \
  }

SYNTAX_PRIM_BINARY_IMPL(subtract)
SYNTAX_PRIM_BINARY_IMPL(divide)
SYNTAX_PRIM_BINARY_IMPL(floor_divide)
SYNTAX_PRIM_BINARY_IMPL(mod)
SYNTAX_PRIM_BINARY_IMPL(floor_mod)
SYNTAX_PRIM_BINARY_IMPL(max)
SYNTAX_PRIM_BINARY_IMPL(min)
SYNTAX_PRIM_BINARY_IMPL(power)
SYNTAX_PRIM_BINARY_IMPL(logical_and)
SYNTAX_PRIM_BINARY_IMPL(logical_or)
SYNTAX_PRIM_BINARY_IMPL(logical_xor)
SYNTAX_PRIM_BINARY_IMPL(greater)
SYNTAX_PRIM_BINARY_IMPL(less)
SYNTAX_PRIM_BINARY_IMPL(equal)
SYNTAX_PRIM_BINARY_IMPL(not_equal)
SYNTAX_PRIM_BINARY_IMPL(greater_equal)
SYNTAX_PRIM_BINARY_IMPL(less_equal)

SYNTAX_PRIM_BINARY_IMPL(bitwise_or)
SYNTAX_PRIM_BINARY_IMPL(bitwise_xor)
SYNTAX_PRIM_BINARY_IMPL(bitwise_and)
SYNTAX_PRIM_BINARY_IMPL(left_shift)
SYNTAX_PRIM_BINARY_IMPL(right_shift)

Variable Program::elementwise_add(const Variable& a,
                                  const Variable& b,
                                  int axis) {
  Instruction instr("elementwise_add", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_mul(const Variable& a,
                                  const Variable& b,
                                  int axis) {
  Instruction instr("elementwise_mul", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_div(const Variable& a,
                                  const Variable& b,
                                  int axis) {
  Instruction instr("divide", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::elementwise_sub(const Variable& a,
                                  const Variable& b,
                                  int axis) {
  Instruction instr("subtract", {a, b});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

#define SYNTAX_PRIM_REDUCE_IMPL(name__)                                \
  Variable Program::reduce_##name__(                                   \
      const Variable& a, const std::vector<int>& dim, bool keep_dim) { \
    Instruction instr("reduce_" #name__, {a});                         \
    instr.SetAttr("dim", dim);                                         \
    instr.SetAttr("keep_dim", keep_dim);                               \
    AppendInstruction(instr);                                          \
    return instr.GetOutput(0);                                         \
  }

SYNTAX_PRIM_REDUCE_IMPL(sum)
SYNTAX_PRIM_REDUCE_IMPL(prod)
SYNTAX_PRIM_REDUCE_IMPL(min)
SYNTAX_PRIM_REDUCE_IMPL(max)

Variable Program::assign(const Variable& a) {
  Instruction instr("identity", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu(const Variable& a) {
  Instruction instr("relu", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::relu6(const Variable& a) {
  Instruction instr("relu6", {a});
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::mul(const Variable& a,
                      const Variable& b,
                      int x_num_col_dims,
                      int y_num_col_dims) {
  Instruction instr("mul", {a, b});
  instr.SetAttr("x_num_col_dims", x_num_col_dims);
  instr.SetAttr("y_num_col_dims", y_num_col_dims);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::matmul(const Variable& a,
                         const Variable& b,
                         bool trans_a,
                         bool trans_b,
                         float alpha) {
  Instruction instr("matmul", {a, b});
  instr.SetAttr("trans_a", trans_a);
  instr.SetAttr("trans_b", trans_b);
  instr.SetAttr("alpha", alpha);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::reshape(const Variable& a, const std::vector<int>& shape) {
  Instruction instr("reshape", {a});
  instr.SetAttr("shape", shape);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::concat(const std::vector<Variable>& input_vars, int axis) {
  Instruction instr("concat", input_vars);
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

Variable Program::transpose(const Variable& a, const std::vector<int>& axis) {
  Instruction instr("transpose", {a});
  instr.SetAttr("axis", axis);
  AppendInstruction(instr);
  return instr.GetOutput(0);
}

std::string _Instruction_::debug_string() const {
  struct Visit {
    std::stringstream& s_;
    explicit Visit(std::stringstream& s) : s_(s) {}
    void operator()(int x) { s_ << x; }
    void operator()(int64_t x) { s_ << x; }
    void operator()(float x) { s_ << x; }
    void operator()(double x) { s_ << x; }
    void operator()(bool x) { s_ << (x ? "true" : "false"); }
    void operator()(const std::string& x) { s_ << x; }
    void operator()(const std::vector<int>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
    void operator()(const std::vector<int64_t>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
    void operator()(const std::vector<float>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
    void operator()(const std::vector<double>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
    void operator()(const std::vector<bool>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
    void operator()(const std::vector<std::string>& x) {
      s_ << "[" + utils::Join(x, ",") + "]";
    }
  };

  std::stringstream ss;
  std::vector<std::string> input_names, output_names;
  std::transform(inputs.begin(),
                 inputs.end(),
                 std::back_inserter(input_names),
                 [](const Variable& x) { return x->id; });
  std::transform(outputs.begin(),
                 outputs.end(),
                 std::back_inserter(output_names),
                 [](const Variable& x) { return x->id; });

  ss << utils::Join(output_names, ", ");
  ss << " = ";
  ss << op_type;
  ss << "(";
  ss << utils::Join(input_names, ", ");
  if (!attrs.empty() && !input_names.empty()) ss << ", ";

  std::map<std::string, std::string> attr_str_map;
  for (const auto& attr : attrs) {
    std::stringstream iss;
    absl::visit(Visit{iss}, attr.second);
    attr_str_map[attr.first] = iss.str();
  }
  bool is_first = true;
  for (const auto& attr : attr_str_map) {
    if (is_first) {
      is_first = false;
    } else {
      ss << ", ";
    }
    ss << attr.first << "=" << attr.second;
  }
  ss << ")";

  return ss.str();
}

struct HashVariable {
  bool operator()(const Variable& lhs, const Variable& rhs) const {
    return lhs->id == rhs->id && lhs->shape == rhs->shape &&
           lhs->type == rhs->type;
  }

  std::size_t operator()(const Variable& var) const {
    return std::hash<std::string>()(var->id +
                                    cinn::utils::Join(var->shape, ", ") +
                                    cinn::common::Type2Str(var->type));
  }
};

std::ostream& operator<<(std::ostream& os, const Program& program) {
  os << "Program {\n";

  std::unordered_set<Variable, HashVariable, HashVariable> var_set;
  for (int i = 0; i < program.size(); i++) {
    var_set.insert(program[i]->inputs.cbegin(), program[i]->inputs.cend());
    var_set.insert(program[i]->outputs.cbegin(), program[i]->outputs.cend());
  }

  for (const auto& var : var_set) {
    os << var << "\n";
  }
  os << "\n";

  for (int i = 0; i < program.size(); i++) {
    os << program[i] << "\n";
  }
  os << "}\n";
  return os;
}

}  // namespace frontend
}  // namespace cinn
