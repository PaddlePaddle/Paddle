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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/frontend/computation.h"
#include "paddle/cinn/frontend/decomposer/use_decomposer.h"
#include "paddle/cinn/frontend/decomposer_registry.h"
#include "paddle/cinn/frontend/interpreter.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/paddle_model_convertor.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/tensor.h"
#include "paddle/cinn/hlir/framework/visualize_helper.h"
#include "paddle/cinn/hlir/op/use_ops.h"
#include "paddle/cinn/pybind/bind.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn::pybind {
using common::Type;
using frontend::Placeholder;
namespace py = pybind11;
using namespace cinn::frontend;  // NOLINT

// this function is a helper function, not threadsafe,
// used in this file only for py function register
static const char *SnakeName(const char *name) {
  static char buf[256];
  char *p = buf;
  const char *q = name;
  for (; *q; q++, p++) {
    if ((*q >= 'A') && (*q <= 'Z')) {
      if (p > buf) *p++ = '_';
      *p = *q - 'A' + 'a';
    } else {
      *p = *q;
    }
  }
  *p = 0;
  return buf;
}

#define EXPAND_CINN_SUPPORT_TYPE(EXPAND_MACRO) \
  EXPAND_MACRO(bool)                           \
  EXPAND_MACRO(int64_t)                        \
  EXPAND_MACRO(double)

void BindFrontend(pybind11::module *m) {
  py::class_<Variable>(*m, "Variable")  //
      .def(py::init<const std::string &>(), py::arg("id") = "")
      .def(py::init([](const Placeholder &p) { return new Variable(p); }))
      .def("__str__", [](Variable &self) { return self->id; })
      .def("__repr__", [](Variable &self) { return utils::GetStreamCnt(self); })
      .def("id", [](Variable &self) { return self->id; })
      .def("name", [](Variable &self) { return self->id; })
      .def("shape", [](Variable &self) { return self->shape; })
      .def("type", [](Variable &self) { return common::Type2Str(self->type); })
      .def("set_type",
           [](Variable &self, const Type &type) {
             self->type = type;
             return self;
           })
      .def("set_type",
           [](Variable &self, const std::string &type) {
             self->type = common::Str2Type(type);
             return self;
           })
      .def("set_shape", [](Variable &self, const std::vector<int> &shape) {
        self->shape = shape;
        return self;
      });

  py::class_<Placeholder>(*m, "Placeholder")  //
      .def(py::init<const common::Type &,
                    const std::vector<int> &,
                    absl::string_view>(),
           py::arg("type"),
           py::arg("shape"),
           py::arg("id") = "")
      .def("shape", &Placeholder::shape)
      .def("type",
           [](Placeholder &self) { return common::Type2Str(self.type()); })
      .def("id", &Placeholder::id)
      .def("name", &Placeholder::id)
      .def("__str__", [](const Placeholder &self) { return self.id(); });

  py::implicitly_convertible<Placeholder, Variable>();

  py::class_<Instruction>(*m, "Instruction")  //
      .def("set_attr",
           [](Instruction &self, const std::string &key, int x) {
             self.SetAttr(key, x);
           })
      .def("set_attr",
           [](Instruction &self, const std::string &key, float x) {
             self.SetAttr(key, x);
           })
      .def("set_attr",
           [](Instruction &self, const std::string &key, const std::string &x) {
             self.SetAttr(key, x);
           })
      .def("set_attr",
           [](Instruction &self,
              const std::string &key,
              const std::vector<int> &x) { self.SetAttr(key, x); })
      .def("set_attr",
           [](Instruction &self,
              const std::string &key,
              const std::vector<float> &x) { self.SetAttr(key, x); })
      .def("set_attr",
           [](Instruction &self,
              const std::string &key,
              const std::vector<std::string> &x) { self.SetAttr(key, x); })
      .def("get_attr_int32", &Instruction::GetAttrs<int>)
      .def("get_attr_fp32", &Instruction::GetAttrs<float>)
      .def("get_attr_str", &Instruction::GetAttrs<std::string>)
      .def("get_attr_int32s", &Instruction::GetAttrs<std::vector<int>>)
      .def("get_attr_fp32s", &Instruction::GetAttrs<std::vector<float>>)
      .def("get_attr_strs", &Instruction::GetAttrs<std::vector<std::string>>)
      .def("__str__",
           [](Instruction &self) { return utils::GetStreamCnt(self); })
      .def("get_op_type", [](Instruction &self) { return self->op_type; })
      .def("get_inputs", [](Instruction &self) { return self->inputs; })
      .def("get_outputs", [](Instruction &self) { return self->outputs; });

  m->def("get_default_program_pass",
         []() { return DefaultTrainingOptimizeOptions().program_passes; })
      .def("get_default_graph_pass",
           []() { return DefaultTrainingOptimizeOptions().graph_passes; })
      .def("get_default_opfusion_pass",
           []() { return DefaultOpFusionPasses(); });

  py::class_<Program>(*m, "Program")
      .def(py::init<>())
      .def("size", &Program::size)
      .def("__getitem__", [](Program &self, int idx) { return self[idx]; })
      .def("__str__", [](Program &self) { return utils::GetStreamCnt(self); })
      .def("get_inputs", &Program::GetInputs)
      .def("add", &Program::add)
      .def("mul", &Program::mul)
      .def("elementwise_add", &Program::elementwise_add)
      .def("relu", &Program::relu)
      .def("relu6", &Program::relu6)
      .def("sigmoid", &Program::sigmoid)
      .def("dropout_infer", &Program::dropout_infer)
      .def("scale", &Program::scale)
      .def("slice", &Program::slice)
      .def("conv2d", &Program::conv2d)
      .def("depthwise_conv2d", &Program::depthwise_conv2d)
      .def("batchnorm", &Program::batchnorm)
      .def("softmax", &Program::softmax)
      .def("pool2d", &Program::pool2d)
      .def("concat", &Program::concat)
      .def("reshape", &Program::reshape)
      .def(
          "build_and_get_output",
          [](Program &self,
             const common::Target &target,
             const std::vector<Variable> &tensor_inputs,
             const std::vector<py::array> &input_data,
             const std::vector<Variable> &tensor_outputs,
             const std::vector<std::string> &passes =
                 std::vector<std::string>{},
             std::shared_ptr<hlir::framework::Scope> scope = nullptr) {
            cinn::runtime::CurrentTarget::SetCurrentTarget(target);
            std::unordered_set<std::string> fetch_ids;
            for (const auto &out : tensor_outputs) {
              fetch_ids.insert(out->id);
            }
            // Acquire all 0D outputs from frontend::Program
            std::unordered_set<std::string> zero_dim_outputs;
            for (std::size_t i = 0; i < self.size(); i++) {
              for (auto &output : self[i].GetOutputs()) {
                if (output->shape.empty()) {
                  zero_dim_outputs.insert(output->id);
                }
              }
            }

            auto graph = Optimize(&self, fetch_ids, target, passes);

            scope = hlir::framework::BuildScope(target, graph, scope);
            hlir::framework::CompilationContext context(graph, scope, target);

            // Keep compile option same as paddle
            context.with_instantiate_variables = true;
            context.remove_unused_variables = false;
            context.fetch_var_ids = fetch_ids;
            hlir::framework::GraphCompiler gc(context);
            const auto &program = gc.Build();

            for (size_t i = 0; i < tensor_inputs.size(); i++) {
              auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
              auto dtype = tensor_inputs[i]->type;
              auto *data = in_tensor->mutable_data(target, dtype);
              CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                  << "The size of tensor [" << tensor_inputs[i]->id
                  << "] is different with the input data's size! Please check.";
              if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                CUDA_CALL(cudaMemcpy(data,
                                     input_data[i].data(),
                                     in_tensor->shape().numel() * dtype.bytes(),
                                     cudaMemcpyHostToDevice));
#else
     LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
              } else if (target.arch == Target::Arch::X86) {
                memcpy(data,
                       input_data[i].data(),
                       in_tensor->shape().numel() *
                           dtype.bytes());  // All random data
              } else {
                CINN_NOT_IMPLEMENTED
              }
            }
            program->Execute();

            std::vector<hlir::framework::Tensor> outputs;
            for (size_t i = 0; i < tensor_outputs.size(); i++) {
              outputs.push_back(scope->GetTensor(tensor_outputs[i]->id));
              outputs.back()->set_type(tensor_outputs[i]->type);
              // Change Tensor from 1D to 0D
              if (outputs.back()->shape().numel() == 1 &&
                  zero_dim_outputs.find(tensor_outputs[i]->id) !=
                      zero_dim_outputs.end()) {
                outputs.back()->Resize({});
              }
            }

            return outputs;
          },
          py::arg("target"),
          py::arg("feed_list"),
          py::arg("feed_datas"),
          py::arg("fetch_list"),
          py::arg("passes") = std::vector<std::string>{},
          py::arg("scope") = nullptr)
      .def("apply_pass",
           [](Program &self,
              const std::unordered_set<std::string> &fetch_ids,
              const common::Target &target,
              const std::vector<std::string> &passes = {}) {
             auto graph = Optimize(&self, fetch_ids, target, passes);
             return graph->fusion_groups.size();
           })

      /**
       * @brief Test the performance of a single-op program
       * @param self The program built with only one op
       * @param target The Target that controls the backends to execute on
       * @param tensor_inputs The vector that contains all input Variables. Must
       * be on CPU
       * @param input_data The vector that contains each input Variable's
       * data(stored as py::array)
       * @param tensor_out The output Variable.
       * @param repeat_ The number of executing time. Increase it to avoid
       * testing noise.
       * @param info The string to be print before testing. Usually it implyies
       * the kind of op and input variable's shape.
       *
       * @return The output tensor after executing the op.
       *
       * @note
       *  This function is for user to test single op performance on python.
       *  To learn more about how to test op's benchmark, see
       * '/python/tests/test_op_benchmark.py'
       *
       */
      .def(
          "test_benchmark",
          [](Program &self,
             const common::Target &target,
             const std::vector<Variable> &tensor_inputs,
             const std::vector<py::array> &input_data,
             const Variable &tensor_out,
             int repeat_,
             const std::string &info) {
            std::shared_ptr<hlir::framework::Graph> g(
                new hlir::framework::Graph(self, target));
            hlir::framework::ApplyPass(g.get(), "InferShape");
            std::shared_ptr<hlir::framework::Scope> scope =
                hlir::framework::BuildScope(target, g);
            hlir::framework::CompilationContext context(g, scope, target);
            hlir::framework::GraphCompiler gc(context);
            auto program = gc.Build();
            for (size_t i = 0; i < tensor_inputs.size(); i++) {
              auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
              auto *data = in_tensor->mutable_data<float>(target);
              CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                  << "The size of tensor [" << tensor_inputs[i]->id
                  << "] is different with the input data's size! Please check.";
              if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                     input_data[i].data(),
                                     in_tensor->shape().numel() * sizeof(float),
                                     cudaMemcpyHostToDevice));
#else
     LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
              } else if (target.arch == Target::Arch::X86) {
                for (size_t j = 0; j < in_tensor->shape().numel(); j++) {
                  data[j] = reinterpret_cast<const float *>(
                      input_data[i].data())[j];  // All random data
                }
              } else {
                CINN_NOT_IMPLEMENTED
              }
            }
            VLOG(3) << info;
            program->ExecuteTest(repeat_);
            auto out = scope->GetTensor(tensor_out->id);
            return out;
          })
      .def(
          "test_benchmark_with_code",
          [](Program &self,
             const common::Target &target,
             const std::vector<Variable> &tensor_inputs,
             const std::vector<py::array> &input_data,
             const Variable &tensor_out,
             int repeat_,
             const std::string &info,
             const std::string &code) {
            // std::shared_ptr<hlir::framework::Graph> g(new
            // hlir::framework::Graph(self, target));
            // hlir::framework::ApplyPass(g.get(), "InferShape");
            std::unordered_set<std::string> fetch_ids;
            auto graph = cinn::frontend::Optimize(&self, fetch_ids, target);
            std::shared_ptr<hlir::framework::Scope> scope =
                hlir::framework::BuildScope(target, graph);

            hlir::framework::CompilationContext context(graph, scope, target);
            hlir::framework::GraphCompiler gc(context);
            auto program = gc.Build(code);
            for (size_t i = 0; i < tensor_inputs.size(); i++) {
              auto in_tensor = scope->GetTensor(tensor_inputs[i]->id);
              auto *data = in_tensor->mutable_data<float>(target);
              CHECK_EQ(input_data[i].size(), in_tensor->shape().numel())
                  << "The size of tensor [" << tensor_inputs[i]->id
                  << "] is different with the input data's size! Please check.";
              if (target.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
                CUDA_CALL(cudaMemcpy(reinterpret_cast<void *>(data),
                                     input_data[i].data(),
                                     in_tensor->shape().numel() * sizeof(float),
                                     cudaMemcpyHostToDevice));
#else
     LOG(FATAL) <<"To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
              } else if (target.arch == Target::Arch::X86) {
                for (size_t j = 0; j < in_tensor->shape().numel(); j++) {
                  data[j] = reinterpret_cast<const float *>(
                      input_data[i].data())[j];  // All random data
                }
              } else {
                CINN_NOT_IMPLEMENTED
              }
            }
            VLOG(3) << info;
            program->ExecuteTest(repeat_);
            auto out = scope->GetTensor(tensor_out->id);
            return out;
          });

  py::class_<frontend::Interpreter>(*m, "Interpreter")
      .def(py::init<const std::vector<std::string> &,
                    const std::vector<hlir::framework::shape_t> &>(),
           py::arg("input_names"),
           py::arg("input_shapes"))  //
      .def("load_paddle_model",
           &frontend::Interpreter::LoadPaddleModel,
           py::arg("model_dir"),
           py::arg("target"),
           py::arg("params_combined"),
           py::arg("model_name") = "")
      .def("run", &frontend::Interpreter::Run)
      .def("get_tensor", &frontend::Interpreter::GetTensor)
      .def("get_program", &frontend::Interpreter::GetProgram)
      .def("get_scope", &frontend::Interpreter::GetScope);

  py::class_<NetBuilder, std::shared_ptr<NetBuilder>>(*m, "NetBuilder")
      .def(py::init<const std::string &>(), py::arg("name") = "")
  // clang-format off
#define PY_REGISTER_CONSTANT_OP(TYPE__)                                   \
     .def("constant",                                                     \
          static_cast<Variable (NetBuilder::*)(                           \
               const TYPE__&, const std::string &, const std::string &)>( \
               &NetBuilder::template Constant<TYPE__>),                   \
          py::arg("value"),                                               \
          py::arg("name") = "",                                           \
          py::arg("dtype") = "")
     EXPAND_CINN_SUPPORT_TYPE(PY_REGISTER_CONSTANT_OP)
#define EXPAND_ONE_VECTOR(TYPE) PY_REGISTER_CONSTANT_OP(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_ONE_VECTOR)
#define EXPAND_TWICE_VECTOR(TYPE) EXPAND_ONE_VECTOR(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_TWICE_VECTOR)
#define EXPAND_TRIPLE_VECTOR(TYPE) EXPAND_TWICE_VECTOR(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_TRIPLE_VECTOR)
#define EXPAND_QUARTIC_VECTOR(TYPE) EXPAND_TRIPLE_VECTOR(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_QUARTIC_VECTOR)
#define EXPAND_QUINTIC_VECTOR(TYPE) EXPAND_QUARTIC_VECTOR(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_QUINTIC_VECTOR)
#define EXPAND_SEXTIC_VECTOR(TYPE) EXPAND_QUINTIC_VECTOR(std::vector<TYPE>)
     EXPAND_CINN_SUPPORT_TYPE(EXPAND_SEXTIC_VECTOR)
#undef EXPAND_ONE_VECTOR
#undef EXPAND_TWICE_VECTOR
#undef EXPAND_TRIPLE_VECTOR
#undef EXPAND_QUARTIC_VECTOR
#undef EXPAND_QUINTIC_VECTOR
#undef EXPAND_SEXTIC_VECTOR
#undef PY_REGISTER_CONSTANT_OP
#define PY_REGISTER_FILLCONSTANT_OP(TYPE__)                                   \
     .def("fill_constant",                                                    \
           static_cast<Variable (NetBuilder::*)(                              \
               const std::vector<int> &, TYPE__,                              \
               const std::string &,                                           \
               const std::string &, bool)>(                                   \
               &NetBuilder::FillConstant<TYPE__>),                            \
           py::arg("shape"),                                                  \
           py::arg("value"),                                                  \
           py::arg("name") = "",                                              \
           py::arg("dtype"),                                                  \
           py::arg("force_cpu") = false)                                      \
     .def("fill_constant",                                                    \
          static_cast<Variable (NetBuilder::*)(                               \
               const std::vector<int> &, TYPE__, const std::string &, bool)>( \
               &NetBuilder::template FillConstant<TYPE__>),                   \
          py::arg("shape"),                                                   \
          py::arg("value"),                                                   \
          py::arg("name") = "",                                               \
          py::arg("force_cpu") = false)
          EXPAND_CINN_SUPPORT_TYPE(PY_REGISTER_FILLCONSTANT_OP)
#undef PY_REGISTER_FILLCONSTANT_OP
#define PY_REGISTER_UNARY_FUNC(func_name__) \
  .def(SnakeName(#func_name__), &NetBuilder::func_name__, py::arg("x"))
      NETBUILDER_UNARY_OP_FOREACH(PY_REGISTER_UNARY_FUNC)
#undef PY_REGISTER_UNARY_FUNC
#define PY_REGISTER_BINARY_FUNC(func_name__) \
  .def(SnakeName(#func_name__), &NetBuilder::func_name__, py::arg("x"), \
       py::arg("y"), py::arg("axis") = -1)
      NETBUILDER_BINARY_OP_FOREACH(PY_REGISTER_BINARY_FUNC)
#undef PY_REGISTER_BINARY_FUNC
#define PY_REGISTER_REDUCE_FUNC(func_name__) \
  .def(SnakeName(#func_name__),              \
       &NetBuilder::func_name__,             \
       py::arg("x"),                         \
       py::arg("axis") = std::vector<int>{}, \
       py::arg("keepdim") = false)
      NETBUILDER_REDUCE_OP_FOREACH(PY_REGISTER_REDUCE_FUNC)
#undef PY_REGISTER_REDUCE_FUNC
#define PY_REGISTER_REDUCE_CINN_FUNC(func_name__) \
  .def(SnakeName(#func_name__),              \
       &NetBuilder::func_name__,             \
       py::arg("x"),                         \
       py::arg("dim") = std::vector<int>{}, \
       py::arg("keep_dim") = false)
      NETBUILDER_REDUCE_OP_FOREACH(PY_REGISTER_REDUCE_CINN_FUNC)
#undef PY_REGISTER_REDUCE_CINN_FUNC
      // clang-format on
      .def(py::init<const std::string &>(), py::arg("name") = "")
      .def("create_input",
           static_cast<Placeholder (NetBuilder::*)(const common::Type &,
                                                   const std::vector<int> &,
                                                   const std::string &)>(
               &NetBuilder::CreateInput),
           py::arg("type"),
           py::arg("shape"),
           py::arg("id_hint"))
      .def(
          "create_input",
          [](NetBuilder &self,
             const std::string &type,
             const std::vector<int> &shape,
             const std::string &id) {
            return self.CreateInput(cinn::common::Str2Type(type), shape, id);
          },
          py::arg("type"),
          py::arg("shape"),
          py::arg("id_hint"))
      .def("create_input",
           static_cast<Placeholder (NetBuilder::*)(const Variable &)>(
               &NetBuilder::CreateInput))
      .def("build", &NetBuilder::Build, py::arg("in_reverse") = false)
      .def("name", &NetBuilder::name)
      .def("__str__", [](NetBuilder &self) { return self.name(); })
      .def("append_instruction",
           &NetBuilder::AppendInstruction,
           py::arg("instr"))
      .def("fill_constant",
           static_cast<Variable (NetBuilder::*)(const std::vector<int> &,
                                                const std::string &,
                                                const std::string &,
                                                const std::string &,
                                                bool)>(
               &NetBuilder::FillConstant),
           py::arg("shape"),
           py::arg("value"),
           py::arg("name") = "",
           py::arg("dtype"),
           py::arg("force_cpu") = false)
      .def("broadcast_to",
           static_cast<Variable (NetBuilder::*)(const Variable &,
                                                const std::vector<int> &)>(
               &NetBuilder::BroadcastTo),
           py::arg("x"),
           py::arg("out_shape"))
      .def("broadcast_to",
           static_cast<Variable (NetBuilder::*)(const Variable &,
                                                const std::vector<int> &,
                                                const std::vector<int> &)>(
               &NetBuilder::BroadcastTo),
           py::arg("x"),
           py::arg("out_shape"),
           py::arg("broadcast_axes"))
      .def("concat", &NetBuilder::Concat, py::arg("xs"), py::arg("axis") = 0)
      .def("reshape", &NetBuilder::Reshape, py::arg("x"), py::arg("shape"))
      .def("transpose", &NetBuilder::Transpose, py::arg("x"), py::arg("axis"))
      .def("top_k",
           &NetBuilder::TopK,
           py::arg("x"),
           py::arg("k"),
           py::arg("axis"),
           py::arg("largest"))
      .def("sort",
           &NetBuilder::Sort,
           py::arg("operand"),
           py::arg("axis"),
           py::arg("is_ascend"))
      .def("argsort",
           &NetBuilder::ArgSort,
           py::arg("operand"),
           py::arg("axis"),
           py::arg("is_ascend"))
      .def("slice",
           &NetBuilder::Slice,
           py::arg("x"),
           py::arg("axes"),
           py::arg("starts"),
           py::arg("ends"),
           py::arg("infer_flags") = std::vector<int>{},
           py::arg("strides") = std::vector<int>{},
           py::arg("decrease_axis") = std::vector<int>{})
      .def("reverse", &NetBuilder::Reverse, py::arg("x"), py::arg("axis"))
      .def("resize",
           &NetBuilder::Resize,
           py::arg("x"),
           py::arg("out_shape"),
           py::arg("mode") = "bilinear")
      .def("select",
           &NetBuilder::Select,
           py::arg("condition"),
           py::arg("true_value"),
           py::arg("false_value"))
      .def("split",
           &NetBuilder::Split,
           py::arg("x"),
           py::arg("num_or_sections"),
           py::arg("axis") = 0)
      .def("gather",
           &NetBuilder::Gather,
           py::arg("x"),
           py::arg("index"),
           py::arg("axis") = 0)
      .def("slice_assign",
           &NetBuilder::SliceAssign,
           py::arg("x"),
           py::arg("assign"),
           py::arg("axes"),
           py::arg("starts"),
           py::arg("ends"),
           py::arg("strides") = std::vector<int>{})
      .def("scatter_assign",
           &NetBuilder::ScatterAssign,
           py::arg("x"),
           py::arg("updates"),
           py::arg("index"),
           py::arg("axis") = 0)
      .def("scatter_add",
           &NetBuilder::ScatterAdd,
           py::arg("x"),
           py::arg("updates"),
           py::arg("index"),
           py::arg("axis") = 0)
      .def("isclose",
           &NetBuilder::IsClose,
           py::arg("x"),
           py::arg("y"),
           py::arg("rtol") = 1e-05f,
           py::arg("atol") = 1e-08f,
           py::arg("equal_nan") = false)
      .def("mul",
           &NetBuilder::Mul,
           py::arg("x"),
           py::arg("y"),
           py::arg("x_num_col_dims") = 1,
           py::arg("y_num_col_dims") = 1,
           py::arg("is_infer") = false)
      .def("elementwise_add_grad",
           &NetBuilder::ElementwiseAddGrad,
           py::arg("dout"),
           py::arg("x"),
           py::arg("y"),
           py::arg("axis") = -1)
      .def("relu6",
           &NetBuilder::Relu6,
           py::arg("a"),
           py::arg("threshold") = 6.0f)
      .def("gelu", &NetBuilder::Gelu, py::arg("x"))
      .def("squeeze",
           &NetBuilder::Squeeze,
           py::arg("a"),
           py::arg("axes") = std::vector<int>{})
      .def(
          "expand_dims", &NetBuilder::ExpandDims, py::arg("x"), py::arg("axes"))
      .def("argmax",
           &NetBuilder::Argmax,
           py::arg("x"),
           py::arg("axis"),
           py::arg("keep_dim") = false)
      .def("argmin",
           &NetBuilder::Argmin,
           py::arg("x"),
           py::arg("axis"),
           py::arg("keep_dim") = false)
      .def("lookup_table",
           &NetBuilder::LookupTable,
           py::arg("table"),
           py::arg("ids"),
           py::arg("padding_idx"))
      .def("one_hot",
           &NetBuilder::OneHot,
           py::arg("indices"),
           py::arg("on_value"),
           py::arg("off_value"),
           py::arg("depth"),
           py::arg("axis") = -1,
           py::arg("dtype") = "float32")
      .def("conv2d",
           &NetBuilder::Conv2d,
           py::arg("x"),
           py::arg("w"),
           py::arg("strides") = std::vector<int>{1, 1},
           py::arg("paddings") = std::vector<int>{0, 0},
           py::arg("dilations") = std::vector<int>{1, 1},
           py::arg("groups") = 1,
           py::arg("data_format") = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("depthwise_conv2d",
           &NetBuilder::DepthwiseConv2d,
           py::arg("x"),
           py::arg("w"),
           py::arg("strides") = std::vector<int>{1, 1},
           py::arg("paddings") = std::vector<int>{0, 0},
           py::arg("dilations") = std::vector<int>{1, 1},
           py::arg("groups") = 1,
           py::arg("data_format") = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("pool2d",
           &NetBuilder::Pool2d,
           py::arg("x"),
           py::arg("pooling_type"),
           py::arg("kernel_size"),
           py::arg("stride") = std::vector<int>{1, 1},
           py::arg("padding") = std::vector<int>{0, 0},
           py::arg("ceil_mode") = false,
           py::arg("exclusive") = true,
           py::arg("global_pooling") = false,
           py::arg("data_format") = "NCHW",
           py::arg("adaptive") = false,
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("pool2d_grad",
           &NetBuilder::Pool2dGrad,
           py::arg("x"),
           py::arg("y"),
           py::arg("dy"),
           py::arg("pooling_type"),
           py::arg("kernel_size"),
           py::arg("stride") = std::vector<int>{1, 1},
           py::arg("padding") = std::vector<int>{0, 0},
           py::arg("ceil_mode") = false,
           py::arg("exclusive") = true,
           py::arg("global_pooling") = false,
           py::arg("data_format") = "NCHW",
           py::arg("adaptive") = false,
           py::arg("padding_algorithm") = "EXPLICIT")
      .def("batchnorm",
           &NetBuilder::BatchNorm,
           py::arg("x"),
           py::arg("scale"),
           py::arg("bias"),
           py::arg("mean"),
           py::arg("variance"),
           py::arg("epsilon") = 1e-5f,
           py::arg("momentum") = 0.9f,
           py::arg("data_layout") = "NCHW",
           py::arg("is_test") = true)
      .def("batch_norm_grad",
           &NetBuilder::BatchNormGrad,
           py::arg("dy"),
           py::arg("x"),
           py::arg("scale"),
           py::arg("save_mean"),
           py::arg("save_variance"),
           py::arg("epsilon") = 1e-5,
           py::arg("data_layout") = "NCHW")
      .def("scale",
           &NetBuilder::Scale,
           py::arg("x"),
           py::arg("scale") = 1.0f,
           py::arg("bias") = 0.0f,
           py::arg("bias_after_scale") = true)
      .def("softmax",
           &NetBuilder::Softmax,
           py::arg("x"),
           py::arg("axes") = std::vector<int>{-1},
           py::arg("mode") = "fast",
           py::arg("data_format") = "AnyLayout")
      .def("dropout_infer",
           &NetBuilder::DropoutInfer,
           py::arg("x"),
           py::arg("dropout_prob") = 0.5f,
           py::arg("dropout_implementation") = "downgrade_in_infer")
      .def("relu_grad", &NetBuilder::ReluGrad, py::arg("dout"), py::arg("x"))
      .def("sum", &NetBuilder::Sum, py::arg("inputs"))
      .def("matmul",
           &NetBuilder::Matmul,
           py::arg("x"),
           py::arg("y"),
           py::arg("transpose_x") = false,
           py::arg("transpose_y") = false,
           py::arg("alpha") = 1.0f)
      .def("conv",
           &NetBuilder::Conv,
           py::arg("x"),
           py::arg("w"),
           py::arg("strides") = std::vector<int>{1, 1},
           py::arg("paddings") = std::vector<int>{0, 0},
           py::arg("dilations") = std::vector<int>{1, 1},
           py::arg("groups") = 1,
           py::arg("conv_type") = "forward",
           py::arg("data_format") = "NCHW",
           py::arg("padding_algorithm") = "EXPLICIT",
           py::arg("output_shape") = std::vector<int>{})
      .def("cast", &NetBuilder::Cast, py::arg("x"), py::arg("dtype"))
      .def("bitcast_convert",
           &NetBuilder::BitcastConvert,
           py::arg("x"),
           py::arg("dtype"))
      .def("arange",
           &NetBuilder::Arange,
           py::arg("start"),
           py::arg("stop"),
           py::arg("step"),
           py::arg("dtype"))
      .def("gather_nd", &NetBuilder::GatherNd, py::arg("x"), py::arg("index"))
      .def("cbrt", &NetBuilder::Cbrt, py::arg("x"))
      .def("clz", &NetBuilder::Clz, py::arg("x"))
      .def("popc", &NetBuilder::Popc, py::arg("x"))
      .def("reciprocal", &NetBuilder::Reciprocal, py::arg("x"))
      .def("gaussian_random",
           &NetBuilder::GaussianRandom,
           py::arg("shape"),
           py::arg("mean") = 0.0f,
           py::arg("std") = 1.0f,
           py::arg("seed") = 0,
           py::arg("dtype") = "float32")
      .def("uniform_random",
           &NetBuilder::UniformRandom,
           py::arg("shape"),
           py::arg("min") = -1.0f,
           py::arg("max") = 1.0f,
           py::arg("seed") = 0,
           py::arg("dtype") = "float32",
           py::arg("diag_num") = 0,
           py::arg("diag_step") = 0,
           py::arg("diag_val") = 1.0f)
      .def("randint",
           &NetBuilder::RandInt,
           py::arg("shape"),
           py::arg("min") = 0,
           py::arg("max") = 0,
           py::arg("seed") = 0,
           py::arg("dtype") = "int64")
      .def("repeat",
           &NetBuilder::Repeat,
           py::arg("x"),
           py::arg("repeats"),
           py::arg("axis"))
      .def("flip", &NetBuilder::Flip, py::arg("x"), py::arg("axis"))
      .def("cholesky",
           &NetBuilder::Cholesky,
           py::arg("x"),
           py::arg("upper") = false)
      .def("triangular_solve",
           &NetBuilder::TriangularSolve,
           py::arg("input1"),
           py::arg("input2"),
           py::arg("left_side") = true,
           py::arg("upper") = false,
           py::arg("transpose_a") = false,
           py::arg("unit_diagonal") = false);

  auto computation =
      py::class_<CinnComputation, std::shared_ptr<CinnComputation>>(
          *m, "Computation");
  py::class_<CinnComputation::CompileOptions>(computation, "CompileOptions")
      .def_readwrite("use_decomposer",
                     &CinnComputation::CompileOptions::use_decomposer)
      .def_readwrite("do_prerun", &CinnComputation::CompileOptions::do_prerun)
      .def_readwrite("use_default_passes",
                     &CinnComputation::CompileOptions::use_default_passes)
      .def_readwrite("passes", &CinnComputation::CompileOptions::passes);

  computation
      .def("default_compile_options", &CinnComputation::DefaultCompileOptions)
      // currently stream param is not exported to python, the default stream is
      // used always
      .def_static(
          "build_and_compile",
          [](const common::Target &target,
             NetBuilder &builder,
             const CinnComputation::CompileOptions &options) {
            return CinnComputation::BuildAndCompile(target, builder, options);
          },
          py::arg("target"),
          py::arg("builder"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def_static(
          "compile",
          [](const common::Target &target,
             Program &program,
             const CinnComputation::CompileOptions &options) {
            return CinnComputation::Compile(target, program, options);
          },
          py::arg("target"),
          py::arg("program"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def_static(
          "compile_paddle_model",
          [](const common::Target &target,
             const std::string &model_path,
             const std::vector<std::string> &input_names,
             const std::vector<hlir::framework::shape_t> &input_shapes,
             bool params_combined,
             const CinnComputation::CompileOptions &options) {
            return CinnComputation::CompilePaddleModel(target,
                                                       model_path,
                                                       input_names,
                                                       input_shapes,
                                                       params_combined,
                                                       options);
          },
          py::arg("target"),
          py::arg("model_path"),
          py::arg("input_names"),
          py::arg("input_shapes"),
          py::arg("params_combined"),
          py::arg("options") = CinnComputation::DefaultCompileOptions())
      .def("get_all_tensor_names", &CinnComputation::GetAllTensorNames)
      .def("get_tensor", &CinnComputation::GetTensor)
      .def("execute", [](CinnComputation &self) { self.Execute(); });

  py::class_<PaddleModelConvertor>(*m, "PaddleModelConvertor")
      .def(py::init<>())
      .def(py::init<const common::Target &,
                    std::shared_ptr<NetBuilder>,
                    std::shared_ptr<hlir::framework::Scope>>(),
           py::arg("target"),
           py::arg("builder") = nullptr,
           py::arg("scope") = nullptr)
      .def("__call__", &PaddleModelConvertor::operator())
      .def("load_model",
           &PaddleModelConvertor::LoadModel,
           py::arg("model_dir"),
           py::arg("is_combined") = false,
           py::arg("feed") =
               std::unordered_map<std::string, std::vector<int64_t>>())
      .def("create_input",
           &PaddleModelConvertor::CreateInput,
           py::arg("dtype"),
           py::arg("shape"),
           py::arg("name"))
      .def("append_op",
           static_cast<void (PaddleModelConvertor::*)(
               const std::string &,
               const std::map<std::string, std::vector<std::string>> &,
               const std::map<std::string, std::vector<std::string>> &,
               const std::map<std::string, cinn::utils::Attribute> &)>(
               &PaddleModelConvertor::RunOp),
           py::arg("type"),
           py::arg("inputs"),
           py::arg("outputs"),
           py::arg("attrs"))
      .def("get_fetch_list",
           &PaddleModelConvertor::GetFetchList,
           py::arg("fetch_list") = std::unordered_set<std::string>{})
      .def("get_cinn_name",
           [](PaddleModelConvertor &self, const std::string &paddle_name) {
             CHECK(self.var_model_to_program_map().count(paddle_name))
                 << "Cannot find variabel " << paddle_name
                 << " in CINN! Please check.";
             return self.var_model_to_program_map().at(paddle_name);
           });
}  // namespace frontend

#undef EXPAND_CINN_SUPPORT_TYPE

}  // namespace cinn::pybind
