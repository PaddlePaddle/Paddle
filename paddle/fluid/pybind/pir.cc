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

#include "paddle/fluid/pybind/pir.h"

#include <Python.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/common/enforce.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_attribute.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_dialect.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_tools.h"
#include "paddle/fluid/pir/dialect/distributed/ir/dist_type.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_pylayer_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include "paddle/fluid/pir/transforms/gpu/fused_bn_add_act_pass.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/fluid/pybind/control_flow_api.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/core/parser/ir_parser.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/core/visitors.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pir_to_py_code_converter.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#endif

using paddle::dialect::ApiBuilder;
using paddle::dialect::DenseTensorArrayType;
using paddle::dialect::DenseTensorType;
using paddle::dialect::DistDenseTensorType;
using paddle::dialect::DistTypeInterface;
using paddle::dialect::IfOp;
using paddle::dialect::PyLayerOp;
using paddle::dialect::SelectedRowsType;
using paddle::dialect::SparseCooTensorType;
using paddle::dialect::SparseCsrTensorType;
using paddle::dialect::WhileOp;
using pir::TuplePopOp;

using paddle::dialect::IntArrayAttribute;
using paddle::dialect::OperationDistAttribute;
using paddle::dialect::TensorDistAttribute;
using pir::ArrayAttribute;
using pir::Attribute;
using pir::Block;
using pir::BlockArgument;
using pir::BoolAttribute;
using pir::CloneOptions;
using pir::Int32Attribute;
using pir::IrContext;
using pir::IrMapping;
using pir::IrParser;
using pir::Operation;
using pir::OpOperand;
using pir::OpResult;
using pir::Pass;
using pir::PassManager;
using pir::Program;
using pir::StrAttribute;
using pir::Type;
using pir::Value;
using pir::VectorType;
using pybind11::return_value_policy;

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(pir_apply_shape_optimization_pass);

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_value_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

pir::Value FakeValue() {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return pir::Value(nullptr);
}

bool IsFakeValue(const pir::Value &value) {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return value.impl() == nullptr || !value.type();
}

inline int64_t GetProgramInt64Attr(const std::shared_ptr<Program> &program,
                                   const std::string &attr_name,
                                   int64_t default_value = 0) {
  auto op = program->module_op();
  if (op->HasAttribute(attr_name)) {
    auto val = op->attribute(attr_name).dyn_cast<pir::Int64Attribute>().data();
    return val;
  } else {
    return default_value;
  }
}

inline void SetProgramInt64Attr(std::shared_ptr<Program> program,
                                const std::string &attr_name,
                                int64_t value) {
  auto op = program->module_op();
  op->set_attribute(
      attr_name, pir::Int64Attribute::get(pir::IrContext::Instance(), value));
}

std::string GetValueInfo(Value v) {
  if (v.impl() == nullptr) {
    return "nullptr value";
  }
  std::stringstream ss;
  if (auto op_result = v.dyn_cast<OpResult>()) {
    ss << "define_op_name=" << op_result.owner()->name();
    ss << ", index=" << op_result.index();
  } else if (auto arg = v.dyn_cast<BlockArgument>()) {
    if (arg.is_kwarg()) {
      ss << "keyword block_arg, keyword = " << arg.keyword();
    } else {
      ss << "position block_arg, index = " << arg.index();
    }
  }
  if (!v.type()) {
    ss << ", dtype=<<NULL TYPE>>";
  } else {
    ss << ", dtype=" << v.type();
    if (v.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
      ss << ", place="
         << v.type()
                .dyn_cast<paddle::dialect::AllocatedDenseTensorType>()
                .place();
    }
  }
  auto stop_gradient = v.attribute<BoolAttribute>(kAttrStopGradients);
  if (stop_gradient && !stop_gradient.data()) {
    ss << ", stop_gradient=False";
  } else {
    ss << ", stop_gradient=True";
  }
  return ss.str();
}

namespace name_analysis {
Value GetOutputValueByName(const Program &program, const std::string &name) {
  auto &block = *program.block();
  StrAttribute name_attr = StrAttribute::get(IrContext::Instance(), name);
  Value value;
  for (auto &op : block) {
    if (op.isa<pir::ShadowOutputOp>()) {
      if (op.attribute("output_name") == name_attr) {
        if (value) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "More than one shadow ouput named with %s found.", name));
        }
        value = op.operand_source(0);
      }
    }
  }
  return value;
}

Value GetParameterValueByName(const Program &program, const std::string &name) {
  auto &block = *program.block();
  StrAttribute name_attr = StrAttribute::get(IrContext::Instance(), name);
  Value value;
  for (auto &op : block) {
    if (op.isa<pir::ParameterOp>()) {
      if (op.attribute("parameter_name") == name_attr) {
        if (value) {
          PADDLE_THROW(common::errors::PreconditionNotMet(
              "More than one parameter named with %s found.", name));
        }
        value = op.result(0);
      }
    }
  }
  return value;
}

void SetValueAllNamesWith(Value value, const std::string name) {
  pir::Operation *define_op = value.defining_op();
  if (define_op->isa<pir::ParameterOp>()) {
    define_op->set_attribute(
        "parameter_name", StrAttribute::get(pir::IrContext::Instance(), name));
  } else if (define_op->isa<paddle::dialect::DataOp>()) {
    define_op->set_attribute(
        "name", StrAttribute::get(pir::IrContext::Instance(), name));
  } else if (auto block_arg = value.dyn_cast<BlockArgument>()) {
    PADDLE_THROW(
        phi::errors::InvalidArgument("Can Not set name for BlockArgument! "));
  } else if (value.first_use()) {
    auto nextOp = value.first_use().owner();
    if (nextOp->isa<::pir::ShadowOutputOp>()) {
      nextOp->set_attribute(
          "output_name", StrAttribute::get(pir::IrContext::Instance(), name));
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Currently, we can only set name of Value which is "
          "shadowoutput "));
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only set name of Value that "
        "is persistable"));
  }
}

std::optional<std::string> GetValueInputName(Value value) {
  std::optional<std::string> name;
  if (auto block_arg = value.dyn_cast<BlockArgument>()) {
    if (block_arg.is_kwarg()) {
      name = block_arg.keyword();
    } else {
      name = "arg_" + std::to_string(block_arg.index());
    }
  } else if (auto param_op = value.defining_op<::pir::ParameterOp>()) {
    name = param_op.param_name();
  } else if (auto data_op = value.defining_op<paddle::dialect::DataOp>()) {
    name = data_op.attribute<StrAttribute>("name").AsString();
  } else if (auto constant_op = value.defining_op<::pir::ConstantTensorOp>()) {
    name = constant_op.tensor_name();
  }
  return name;
}

std::vector<std::string> GetValueOutputNames(Value value) {
  std::vector<std::string> names;
  for (auto iter = value.use_begin(); iter != value.use_end(); ++iter) {
    if (iter->owner()->isa<::pir::ShadowOutputOp>()) {
      names.push_back(
          iter->owner()->attribute<StrAttribute>("output_name").AsString());
    } else if (iter->owner()->isa<::pir::SetParameterOp>()) {
      names.push_back(
          iter->owner()->attribute<StrAttribute>("parameter_name").AsString());
    }
  }
  return names;
}

std::vector<std::string> GetValueAllNames(Value value) {
  std::vector<std::string> names;
  std::optional<std::string> input_name = GetValueInputName(value);
  if (input_name.has_value()) {
    names.push_back(input_name.value());
  }

  std::vector<std::string> output_name = GetValueOutputNames(value);
  for (auto &name : output_name) {
    names.push_back(name);
  }

  return names;
}

std::string GetValueFirstName(Value value) {
  auto name = TryGetValueFirstName(value);

  PADDLE_ENFORCE(name.has_value(),
                 phi::errors::InvalidArgument(
                     "Currently, we can only get name of Value from "
                     "DataOp/ParameterOp/BlockArgument/ConstantTensorOp/"
                     "SetParameterOp and ShadowOutputOp."));

  return name.value();
}

std::optional<std::string> TryGetValueFirstName(Value value) {
  std::optional<std::string> name;

  auto names = GetValueAllNames(value);
  if (!names.empty()) {
    return names[0];
  }

  return name;
}
}  // namespace name_analysis

phi::DataType GetTensorDtype(Type type) {
  if (!type) {
    PADDLE_THROW(phi::errors::InvalidArgument("The type of value is nullptr."));
  }
  if (auto dense_tensor_type = type.dyn_cast<DenseTensorType>()) {
    return dialect::TransToPhiDataType(dense_tensor_type.dtype());
  } else if (auto sparse_coo_tensor_type =
                 type.dyn_cast<SparseCooTensorType>()) {
    return dialect::TransToPhiDataType(sparse_coo_tensor_type.dtype());
  } else if (auto sparse_csr_tensor_type =
                 type.dyn_cast<SparseCsrTensorType>()) {
    return dialect::TransToPhiDataType(sparse_csr_tensor_type.dtype());
  } else if (auto select_rows = type.dyn_cast<SelectedRowsType>()) {
    return dialect::TransToPhiDataType(select_rows.dtype());
  } else if (auto dense_array = type.dyn_cast<DenseTensorArrayType>()) {
    return dialect::TransToPhiDataType(dense_array.dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get phi::DataType from DenseTensorType and "
        "SelectedRowsType, DenseTensorArrayType,SparseCooTensorType or "
        "SparseCsrTensorType."));
  }
}

phi::DataType GetValueDtype(Value value) {
  return GetTensorDtype(value.type());
}

py::object Clone(const Program &self, IrMapping *p_mapper = nullptr) {
  IrMapping mapper;
  if (p_mapper == nullptr) {
    p_mapper = &mapper;
  }
  auto src_obj = py::cast(self);
  auto new_obj = py::cast(self.Clone(*p_mapper));
  for (auto item : src_obj.attr("__dict__").cast<py::dict>()) {
    new_obj.attr(item.first.cast<std::string>().c_str()) = item.second;
  }
  return new_obj;
}

bool SomeInSet(const std::vector<pir::Value> &vec,
               const std::set<pir::Value> &set) {
  for (auto &v : vec) {
    if (set.find(v) != set.end()) {
      return true;
    }
  }
  return false;
}

pir::Value AppendDataOp(pir::Block *block,
                        const pir::Value &value,
                        const std::string &name,
                        const pir::Operation &origin_op) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_info = ctx->GetRegisteredOpInfo(paddle::dialect::DataOp::name());
  pir::AttributeMap attribute_map = {
      {"name", StrAttribute::get(ctx, name)},
      {"shape",
       paddle::dialect::IntArrayAttribute::get(
           ctx, phi::IntArray(phi::vectorize(GetValueDims(value))))},
      {"dtype",
       paddle::dialect::DataTypeAttribute::get(ctx, GetValueDtype(value))},
      {"place", paddle::dialect::PlaceAttribute::get(ctx, phi::Place())}};
  std::vector<pir::Type> output_types{value.type()};
  pir::Operation *operation =
      pir::Operation::Create({}, attribute_map, output_types, op_info);

  block->insert(origin_op, operation);
  return operation->result(0);
}
std::vector<pir::Value> GetRealOpInputs(pir::Operation *op) {
  if (op->isa<paddle::dialect::IfOp>() ||
      op->isa<paddle::dialect::PyLayerOp>()) {
    return pir::GetUsedExternalValue(*op);
  } else if (op->isa<paddle::dialect::WhileOp>()) {
    paddle::dialect::WhileOp whileop = op->dyn_cast<paddle::dialect::WhileOp>();
    auto value_vector = op->operands_source();
    auto value_vector2 = pir::GetUsedExternalValue(whileop.body());
    value_vector.insert(
        value_vector.end(), value_vector2.begin(), value_vector2.end());
    return value_vector;
  } else {
    return op->operands_source();
  }
}
/*
  Variables in input_vars will be the pruned program's inputs,
  and variables in output_vars will be the pruned program's outputs.
  Therefore, the pruning logic includes replacing the input of
  input_vars with the data op, and then preserving all connected
  ops starting from output_vars.

  Note: The returned program is the original program.
  If you do not want the original program to be modified,
  please pass in a cloned result.
*/
void PruneWithInput(const std::vector<pir::Value> &input_vars,
                    const std::vector<pir::Value> &output_vars,
                    Program *prog) {
  auto global_block = prog->block();
  std::vector<pir::Value> new_input_vars;
  if (!input_vars.empty()) {
    std::vector<pir::Value> new_input_vars;
    for (uint64_t idx = 0; idx < input_vars.size(); idx++) {
      auto input = input_vars[idx];
      auto origin_op = input.defining_op();
      std::string name = "input_" + std::to_string(idx);
      if (auto names = name_analysis::TryGetValueFirstName(input)) {
        name = names.value();
      }
      auto new_input = AppendDataOp(global_block, input, name, *origin_op);
      input.ReplaceAllUsesWith(new_input);
      new_input_vars.push_back(new_input);
    }
  }
  VLOG(6) << "program after add new feed op = " << *prog;
  auto total_ops_list = global_block->ops();
  std::vector<pir::Operation *> total_ops(total_ops_list.begin(),
                                          total_ops_list.end());
  std::vector<bool> intersection_op_flags(total_ops.size(), true);
  std::set<pir::Value> output_vars_set(output_vars.begin(), output_vars.end());
  for (uint32_t index = total_ops.size() - 1; index != (uint32_t)(-1);
       --index) {
    auto op = total_ops[index];
    auto op_results = op->results();
    if (SomeInSet(op_results, output_vars_set)) {
      for (auto &operand : GetRealOpInputs(op)) {
        output_vars_set.insert(operand);
      }
    } else {
      VLOG(6) << "delete op " << index << ", name is " << op->name();
      intersection_op_flags[index] = false;
    }
  }

  std::set<pir::Value> input_vars_set(new_input_vars.begin(),
                                      new_input_vars.end());
  std::vector<pir::Operation *> remove_ops;
  for (uint32_t index = total_ops.size() - 1; index != (uint32_t)(-1);
       --index) {
    auto op = total_ops[index];
    if (!intersection_op_flags[index]) {
      auto op_results = op->results();
      if (!input_vars_set.empty() && SomeInSet(op_results, input_vars_set)) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "The input_var create by: '{%s}' is not involved in the "
            "output_vars calculation"
            "Please remove it from input_vars.",
            op->name()));
      }
      global_block->erase(*op);
    }
  }
}

void BindProgram(py::module *m) {
  static int64_t global_prog_seed = 0;
  py::class_<Program, std::shared_ptr<Program>> program(
      *m, "Program", py::dynamic_attr(), R"DOC(
    Create Python Program. Program is an abstraction of model structure, divided into
    computational graphs and weights. The Program has a main block that stores the computational
    graphs.

    A set of Program usually contains startup program and main program.
    A startup program is set to contain some initial work, eg. initialize the ``Parameter``, and the main
    program will contain the network structure and vars for train.

    A set of Program can be used for test or train, in train program ,
    Paddle will contain all content to build a train network,  in test
    program Paddle will prune some content which is irrelevant to test, eg.
    backward ops and vars.

    **Notes**:
        **we have** :ref:`api_paddle_static_default_startup_program` **and** :ref:`api_paddle_static_default_main_program`
        **by default, a pair of them will shared the parameters. The** :ref:`api_paddle_static_default_startup_program` **only run once to initialize parameters,**
        :ref:`api_paddle_static_default_main_program` **run in every mini batch and adjust the weights.**

    Returns:
        Program: An empty Program.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> main_program = static.Program()
            >>> startup_program = static.Program()
            >>> with static.program_guard(main_program=main_program, startup_program=startup_program):
            ...    x = static.data(name="x", shape=[-1, 784], dtype='float32')
            ...    y = static.data(name="y", shape=[-1, 1], dtype='int32')
            ...    z = static.nn.fc(name="fc", x=x, size=10, activation="relu")

            >>> print("main program is: {}".format(main_program))
            main program is: { // block 0
                var x : LOD_TENSOR.shape(-1, 784).dtype(float32).stop_gradient(True)
                var y : LOD_TENSOR.shape(-1, 1).dtype(int32).stop_gradient(True)
                persist trainable param fc.w_0 : LOD_TENSOR.shape(784, 10).dtype(float32).stop_gradient(False)
                var fc.tmp_0 : LOD_TENSOR.shape(-1, 10).dtype(float32).stop_gradient(False)
                persist trainable param fc.b_0 : LOD_TENSOR.shape(10,).dtype(float32).stop_gradient(False)
                var fc.tmp_1 : LOD_TENSOR.shape(-1, 10).dtype(float32).stop_gradient(False)
                var fc.tmp_2 : LOD_TENSOR.shape(-1, 10).dtype(float32).stop_gradient(False)

                {Out=['fc.tmp_0']} = mul(inputs={X=['x'], Y=['fc.w_0']}, force_fp32_output = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], scale_out = 1.0, scale_x = 1.0, scale_y = [1.0], use_mkldnn = False, with_quant_attr = False, x_num_col_dims = 1, y_num_col_dims = 1)
                {Out=['fc.tmp_1']} = elementwise_add(inputs={X=['fc.tmp_0'], Y=['fc.b_0']}, Scale_out = 1.0, Scale_x = 1.0, Scale_y = 1.0, axis = 1, mkldnn_data_type = float32, op_device = , op_namescope = /, op_role = 0, op_role_var = [], use_mkldnn = False, use_quantizer = False, with_quant_attr = False, x_data_format = , y_data_format = )
                {Out=['fc.tmp_2']} = relu(inputs={X=['fc.tmp_1']}, op_device = , op_namescope = /, op_role = 0, op_role_var = [], use_cudnn = False, use_mkldnn = False, with_quant_attr = False)
            }

            >>> print("start up program is: {}".format(startup_program))
            start up program is: { // block 0
                persist trainable param fc.w_0 : LOD_TENSOR.shape(784, 10).dtype(float32).stop_gradient(False)
                persist trainable param fc.b_0 : LOD_TENSOR.shape(10,).dtype(float32).stop_gradient(False)

                {Out=['fc.w_0']} = uniform_random(inputs={ShapeTensor=[], ShapeTensorList=[]}, diag_num = 0, diag_step = 0, diag_val = 1.0, dtype = 5, max = 0.08692913502454758, min = -0.08692913502454758, op_device = , op_namescope = /, op_role = 0, op_role_var = [], seed = 0, shape = [784, 10], with_quant_attr = False)
                {Out=['fc.b_0']} = fill_constant(inputs={}, dtype = 5, force_cpu = False, op_device = , op_namescope = /, op_role = 0, op_role_var = [], place_type = -1, shape = [10], str_value = 0.0, use_mkldnn = False, value = 0.0, with_quant_attr = False)
            }
  )DOC");
  program
      .def(py::init([]() {
        auto prog = std::make_shared<Program>(pir::IrContext::Instance());
        SetProgramInt64Attr(prog, "random_seed", global_prog_seed);
        return prog;
      }))
      .def("__str__",
           [](const std::shared_ptr<Program> &self) {
             std::ostringstream print_stream;
             self->Print(print_stream);
             return print_stream.str();
           })
      .def("__repr__",
           [](const std::shared_ptr<Program> &self) {
             std::ostringstream print_stream;
             self->Print(print_stream);
             return print_stream.str();
           })
      .def("parameters_num",
           [](const std::shared_ptr<Program> &self) {
             return self->parameters_num();
           })
      .def("set_parameters_from",
           [](const std::shared_ptr<Program> &self,
              const std::shared_ptr<Program> &other) {
             self->set_parameters(other->parameters());
           })
      .def(
          "global_block",
          [](std::shared_ptr<Program> self) { return self->block(); },
          return_value_policy::reference)
      .def("clone", [](Program &self) { return Clone(self); })
      .def("clone",
           [](Program &self, IrMapping &ir_mapper) {
             return Clone(self, &ir_mapper);
           })
      .def(
          "copy_to_block",
          [](std::shared_ptr<Program> self,
             pir::IrMapping &mapper,
             Block *block) { return self->CopyToBlock(mapper, block); },
          return_value_policy::reference)
      .def(
          "list_vars",
          [](std::shared_ptr<Program> self) {
            std::vector<pir::Value> vars;
            for (auto op : self->block()->ops()) {
              for (auto var : op->results()) {
                vars.push_back(var);
              }
            }
            return vars;
          },
          return_value_policy::reference)
      .def(
          "global_block",
          [](const std::shared_ptr<Program> &self) { return self->block(); },
          return_value_policy::reference)
      .def_property(
          "random_seed",
          [](const std::shared_ptr<Program> &self) {
            return GetProgramInt64Attr(self, "random_seed", 0);
          },
          [](std::shared_ptr<Program> self, int64_t random_seed) {
            SetProgramInt64Attr(self, "random_seed", random_seed);
          })
      .def_property(
          "_seed",
          [](const std::shared_ptr<Program> &self) {
            return GetProgramInt64Attr(self, "random_seed", 0);
          },
          [](std::shared_ptr<Program> self, int64_t random_seed) {
            SetProgramInt64Attr(self, "random_seed", random_seed);
          })
      .def("global_seed",
           [](std::shared_ptr<Program> self, int64_t random_seed) {
             global_prog_seed = random_seed;
             SetProgramInt64Attr(self, "random_seed", random_seed);
           })
      .def_property_readonly(
          "num_blocks",
          [](const std::shared_ptr<Program> &self) {
            size_t num_blocks = 0;
            auto top_level_op = self->module_op();
            for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
              auto &region = top_level_op->region(i);
              num_blocks += region.size();
            }
            return num_blocks;
          })
      .def_property_readonly(
          "blocks",
          [](const std::shared_ptr<Program> &self) {
            // Note: We only return global block currently.
            py::list op_list;
            op_list.append(self->block());
            return op_list;
          },
          return_value_policy::reference)
      .def("get_output_value_by_name",
           [](Program &self, const std::string &name) {
             return name_analysis::GetOutputValueByName(self, name);
           })
      .def("get_parameter_value_by_name",
           [](Program &self, const std::string &name) {
             return name_analysis::GetParameterValueByName(self, name);
           })
      .def("num_ops", [](Program &self) { return self.num_ops(); })
      .def(
          "state_dict",
          [](std::shared_ptr<Program> self,
             const std::string &mode = "all",
             const framework::Scope &scope = framework::Scope()) {
            std::unordered_map<std::string, phi::DenseTensor> state_dict_all;
            std::unordered_map<std::string, phi::DenseTensor> state_dict_param;
            std::unordered_map<std::string, phi::DenseTensor> state_dict_opt;
            for (auto op : self->block()->ops()) {
              for (auto var : op->results()) {
                auto is_persistable =
                    var.attribute<BoolAttribute>(kAttrIsPersistable);
                if (is_persistable && is_persistable.data()) {
                  if (var.defining_op()->isa<::pir::ParameterOp>()) {
                    std::string var_name =
                        name_analysis::GetValueAllNames(var)[0];
                    auto tensor =
                        scope.FindVar(var_name)->GetMutable<phi::DenseTensor>();
                    state_dict_param[var_name] = *tensor;
                    state_dict_all[var_name] = *tensor;
                  } else if (var.defining_op()
                                 ->isa<paddle::dialect::DataOp>()) {
                    std::string var_name =
                        name_analysis::GetValueAllNames(var)[0];
                    auto tensor =
                        scope.FindVar(var_name)->GetMutable<phi::DenseTensor>();
                    state_dict_opt[var_name] = *tensor;
                    state_dict_all[var_name] = *tensor;
                  }
                }
              }
            }
            if (mode == "all") {
              return state_dict_all;
            } else if (mode == "param") {
              return state_dict_param;
            } else if (mode == "opt") {
              return state_dict_opt;
            } else {
              PADDLE_THROW(
                  phi::errors::InvalidArgument("The mode is not supported."));
            }
          })
      .def("set_state_dict",
           [](std::shared_ptr<Program> self,
              const std::unordered_map<std::string, phi::DenseTensor>
                  &state_dict,
              const framework::Scope &scope = framework::Scope()) {
             for (auto item : state_dict) {
               auto var = scope.FindVar(item.first);
               if (var == nullptr) {
                 PADDLE_THROW(phi::errors::NotFound(
                     "The variable %s is not found.", item.first));
               } else {
                 *var->GetMutable<phi::DenseTensor>() = item.second;
               }
             }
           })
      .def(
          "_prune",
          [](Program &self, std::vector<pir::Value> output_vars) {
            std::vector<pir::Value> input_vars;
            PruneWithInput(input_vars, output_vars, &self);
            return &self;
          },
          py::arg("targets"),
          "A description for the _prune method")
      .def(
          "_prune_with_input",
          [](Program &self,
             std::vector<pir::Value> input_vars,
             std::vector<pir::Value> output_vars) {
            PruneWithInput(input_vars, output_vars, &self);
            return &self;
          },
          py::arg("feeded_vars"),
          py::arg("targets"))
      .def("_sync_with_cpp", [](const std::shared_ptr<Program> &self) {
        // It's not need _sync_with_cpp in pir, but it's necessary in old static
        // graph. Add empyt function to avoid python call error.
      });
}

std::shared_ptr<Program> ParseProgram(const std::string &program_str) {
  std::stringstream ss(program_str);
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto program = IrParser(ctx, ss).ParseProgram();
  return program;
}

void BindIrParser(py::module *m) { m->def("parse_program", &ParseProgram); }

void RefreshOpStopgradients(Operation *op) {
  if (op->num_operands() == 0 || op->isa<pir::ParameterOp>() ||
      op->isa<paddle::dialect::UniformOp>()) {
    return;
  } else if (op->isa<pir::SliceOp>()) {
    op->dyn_cast<pir::SliceOp>().RefreshStopGradients();
  } else if (op->isa<pir::SplitOp>()) {
    op->dyn_cast<pir::SplitOp>().RefreshStopGradients();
  } else {
    RefreshStopGradientsDefaultly(op);
  }
}

void BindBlock(py::module *m) {
  py::class_<Block> block(*m, "Block", R"DOC(
    In IR, a Block has a list of Operation and can represent a sub computational graph.

    Notes:
        The constructor of Block should not be invoked directly. You can
        use `Program.block()` to get a block.
  )DOC");
  block.def("empty", &Block::empty)
      .def(
          "front",
          [](Block &self) { return &self.front(); },
          return_value_policy::reference)
      .def(
          "back",
          [](Block &self) { return &self.back(); },
          return_value_policy::reference)
      .def_property_readonly(
          "parent_op",
          [](Block &self) { return self.GetParentOp(); },
          return_value_policy::reference)
      .def_property_readonly(
          "program",
          [](Block &self) { return self.GetParentOp()->GetParentProgram(); },
          return_value_policy::reference)
      .def_property_readonly(
          "parent_block",
          [](Block &self) { return self.GetParentOp()->GetParent(); },
          return_value_policy::reference)
      .def_property_readonly("ops",
                             [](Block &self) -> py::list {
                               py::list op_list;
                               for (auto &op : self) {
                                 op_list.append(&op);
                               }
                               return op_list;
                             })
      .def("num_ops", [](Block &self) { return self.num_ops(); })
      .def(
          "__enter__",
          [](Block &self) -> Block & {
            ApiBuilder::Instance().PushInsertionPoint();
            ApiBuilder::Instance().SetInsertionPointToBlockEnd(&self);
            return self;
          },
          return_value_policy::reference)
      .def("__exit__",
           [](Block &self, py::object, py::object, py::object) {
             ApiBuilder::Instance().LoadInsertionPoint();
           })
      .def("__len__", [](Block &self) { return self.size(); })
      .def("args", &Block::args, return_value_policy::reference)
      .def("kwargs", &Block::kwargs, return_value_policy::reference)
      .def("add_kwarg", &Block::AddKwarg)
      .def("erase_kwarg", &Block::EraseKwarg)
      .def("remove_op",
           [](Block &self, const Operation &op) { self.erase(op); })
      .def(
          "move_op",
          [](Block &self, Operation *op, uint32_t offset) {
            Block::Iterator position = self.begin();
            std::advance(position, offset);
            op->MoveTo(&self, position);
          },
          R"DOC(
          Move an op to a specific position (block.begin() + offset).

          Args:
              op (pir.Operation): the operator to be moved.
              offset (uint32_t) : offset relative to the begin of the block

          Returns:
              None

        )DOC")
      .def("all_parameters",
           [](Block &self) -> py::list {
             py::list param_list;
             for (auto &op : self) {
               if (op.name() == "builtin.parameter" &&
                   op.HasAttribute(kAttrIsPersistable)) {
                 auto attrs = op.attribute(kAttrIsPersistable)
                                  .dyn_cast<pir::ArrayAttribute>()
                                  .AsVector();
                 for (uint32_t i = 0; i < attrs.size(); i++) {
                   bool is_persistable =
                       attrs[i].dyn_cast<pir::BoolAttribute>().data();
                   if (is_persistable) {
                     param_list.append(static_cast<pir::Value>(op.result(i)));
                   }
                 }
               }
             }
             return param_list;
           })
      .def("refresh_stopgradient",
           [](Block &self) {
             for (auto &op : self) {
               RefreshOpStopgradients(&op);
             }
           })
      .def("_sync_with_cpp", [](const Block &self) {
        // It's not need _sync_with_cpp in pir, but it's necessary in old static
        // graph. Add empyt function to avoid python call error.
      });
}

void BindIrMapping(py::module *m) {
  py::class_<IrMapping> ir_mapping(*m, "IrMapping");
  ir_mapping.def(py::init<>())
      .def("look_up",
           [](IrMapping &self, Value from) { return self.Lookup(from); })
      .def("add",
           [](IrMapping &self, Value from, Value to) {
             self.Add<Value>(from, to);
           })
      .def("size",
           [](IrMapping &self) { return self.GetMutableMap<Value>().size(); });
}

void BindCloneOptions(py::module *m) {
  py::class_<CloneOptions> clone_options(*m, "CloneOptions");
  clone_options.def(
      "__init__",
      [](CloneOptions &self,
         bool clone_regions,
         bool clone_operands,
         bool clone_successors) {
        new (&self)
            CloneOptions(clone_regions, clone_operands, clone_successors);
      },
      return_value_policy::reference);
}

void BindOperation(py::module *m) {
  py::class_<Operation> op(*m, "Operation", R"DOC(
    In IR, all the operation are represented by Operation, and Operation
    is regarded as a build in an instruction of a Block. Users can call
    python api to describe their neural network.

    Notes:
        The constructor of operator should not be invoked directly. Use
        python api, for example: paddle.mean for building mean operation.

  )DOC");
  op.def("name", &Operation::name)
      .def("get_parent_block",
           &Operation::GetParent,
           return_value_policy::reference)
      .def("num_operands", &Operation::num_operands)
      .def("num_results", &Operation::num_results)
      .def("num_regions", &Operation::num_regions)

      .def("operand", &Operation::operand)
      .def("result",
           [](Operation &self, uint32_t index) {
             return static_cast<pir::Value>(self.result(index));
           })
      .def("operand_source", &Operation::operand_source)
      .def("operands", &Operation::operands)
      .def("results",
           [](Operation &self) -> py::list {
             py::list value_list;
             for (uint32_t i = 0; i < self.num_results(); i++) {
               value_list.append(static_cast<pir::Value>(self.result(i)));
             }
             return value_list;
           })
      .def(
          "blocks",
          [](Operation &self) { return &self.blocks(); },
          return_value_policy::reference)
      .def("has_attr", &Operation::HasAttribute)
      .def("str_attr",
           [](Operation &self, const std::string &attr_name) -> py::object {
             auto str_attr = self.attribute<StrAttribute>(attr_name);
             if (str_attr) {
               return py::cast(str_attr.AsString());
             } else {
               return py::cast<py::none>(Py_None);
             }
           })
      .def("int_attr",
           [](Operation &self, const std::string &attr_name) -> py::object {
             auto int_attr = self.attribute<Int32Attribute>(attr_name);
             if (int_attr) {
               return py::cast(int_attr.data());
             } else {
               return py::cast<py::none>(Py_None);
             }
           })
      .def("set_bool_attr",
           [](Operation &self, std::string &attr_name, bool flag) {
             self.set_attribute(
                 attr_name,
                 pir::BoolAttribute::get(pir::IrContext::Instance(), flag));
           })
      .def("set_int_array_attr",
           [](Operation &self,
              std::string &attr_name,
              const std::vector<int64_t> &val) {
             auto attr = IntArrayAttribute::get(pir::IrContext::Instance(),
                                                phi::IntArray(val));
             self.set_attribute(attr_name, attr);
           })
      .def("attrs",
           [](Operation &self) -> py::dict {
             py::dict attrs_dict;
             for (auto &pair : self.attributes()) {
               // SymbolAttribute is only used in PIR, no need to pass to Python
               if (pair.second.isa<pir::shape::SymbolAttribute>()) continue;
               if (pair.first == kAttrOpDistAttr) {
                 attrs_dict[pair.first.c_str()] =
                     pair.second.dyn_cast<OperationDistAttribute>();
               } else {
                 attrs_dict[pair.first.c_str()] =
                     paddle::dialect::GetAttributeData(pair.second);
               }
             }
             return attrs_dict;
           })
      .def("set_execution_stream",
           [](Operation &self, const std::string &exe_stream) {
             self.set_attribute(
                 "execution_stream",
                 StrAttribute::get(pir::IrContext::Instance(), exe_stream));
           })
      .def("set_scheduling_priority",
           [](Operation &self, int64_t priority) {
             self.set_attribute("scheduling_priority",
                                pir::Int64Attribute::get(
                                    pir::IrContext::Instance(), priority));
           })
      .def("operands_source",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_operands(); i++) {
               op_list.append(self.operand_source(i));
             }
             return op_list;
           })
      .def("get_input_names",
           [](Operation &self) -> py::list {
             if (self.HasInterface<paddle::dialect::OpYamlInfoInterface>() ==
                 false) {
               PADDLE_THROW(phi::errors::InvalidArgument(
                   "Currently, we can only get input names of Operation that "
                   "has OpYamlInfoInterface"));
             }

             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_info : inputs_info) {
               op_list.append(input_info.name);
             }
             return op_list;
           })
      .def("get_attr_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto attrs_info = std::get<1>(yaml_interface.GetOpInfo());
             for (auto &attr_info : attrs_info) {
               op_list.append(attr_info.name);
             }
             return op_list;
           })
      .def("get_output_names",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto outputs_info = std::get<2>(yaml_interface.GetOpInfo());
             for (auto &output_info : outputs_info) {
               op_list.append(output_info.name);
             }
             return op_list;
           })
      .def("get_output_intermediate_status",
           [](Operation &self) -> py::list {
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto outputs_info = std::get<2>(yaml_interface.GetOpInfo());
             for (auto &output_info : outputs_info) {
               op_list.append(output_info.intermediate);
             }
             return op_list;
           })
      .def("get_input_grad_semantics",
           [](Operation &self) -> py::list {
             if (self.HasInterface<paddle::dialect::OpYamlInfoInterface>() ==
                 false) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Currently, we can only get input grad semantics of "
                   "Operation that "
                   "has OpYamlInfoInterface"));
             }
             py::list op_list;
             paddle::dialect::OpYamlInfoInterface yaml_interface =
                 self.dyn_cast<paddle::dialect::OpYamlInfoInterface>();
             auto inputs_grad_info = std::get<0>(yaml_interface.GetOpInfo());
             for (auto &input_grad_info : inputs_grad_info) {
               op_list.append(input_grad_info.with_grad_semantic);
             }
             return op_list;
           })
      .def("replace_all_uses_with",
           [](Operation &self, const std::vector<Value> &values) {
             self.ReplaceAllUsesWith(values);
           })
      .def("as_if_op",
           [](Operation &self) { return PyIfOp(self.dyn_cast<IfOp>()); })
      .def("as_pylayer_op",
           [](Operation &self) -> PyLayerOp {
             auto pylayer_op = self.dyn_cast<PyLayerOp>();
             if (!pylayer_op) {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "Can't cast non-pylayer_op type Operation to PyLayerOp."));
             }
             return pylayer_op;
           })
      .def("as_while_op",
           [](Operation &self) { return PyWhileOp(self.dyn_cast<WhileOp>()); })
      .def(
          "as_tuple_pop_op",
          [](Operation &self) -> TuplePopOp {
            auto tuple_pop_op = self.dyn_cast<TuplePopOp>();
            if (!tuple_pop_op) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "Can't cast non-tuple_pop_op type Operation to TuplePopOp."));
            }
            return tuple_pop_op;
          })
      .def("__repr__",

           [](Operation &self) {
             std::ostringstream print_stream;
             print_stream << "Operation(";
             self.Print(print_stream);
             print_stream << ")";
             return print_stream.str();
           })
      .def(
          "clone",
          [](Operation &self, IrMapping &ir_mapping, CloneOptions options) {
            auto op = self.Clone(ir_mapping, options);
            return ApiBuilder::Instance().GetBuilder()->Insert(op);
          },
          return_value_policy::reference)
      .def("erase", &Operation::Erase)
      .def("move_before",
           [](Operation &self, Operation &other) {
             self.MoveTo(other.GetParent(), Block::Iterator{other});
           })
      .def_property(
          "callstack",
          [](Operation &self) -> py::list {
            py::list callstack_list;
            if (!self.HasAttribute(paddle::framework::OpProtoAndCheckerMaker::
                                       OpCreationCallstackAttrName())) {
              return callstack_list;
            }
            pir::Attribute op_callstack = self.attribute<pir::Attribute>(
                paddle::framework::OpProtoAndCheckerMaker::
                    OpCreationCallstackAttrName());
            PADDLE_ENFORCE(op_callstack.isa<pir::ArrayAttribute>(),
                           phi::errors::PreconditionNotMet(
                               "The callstack of operation `%s` should be an "
                               "array attribute.",
                               self.name()));
            auto op_callstack_array_attr =
                op_callstack.dyn_cast<pir::ArrayAttribute>();
            for (size_t i = 0; i < op_callstack_array_attr.size(); ++i) {
              PADDLE_ENFORCE(
                  op_callstack_array_attr.at(i).isa<StrAttribute>(),
                  phi::errors::PreconditionNotMet(
                      "The callstack info of operation `%s` should be array of "
                      "string attribute.",
                      self.name()));
              callstack_list.append(op_callstack_array_attr.at(i)
                                        .dyn_cast<StrAttribute>()
                                        .AsString());
            }
            return callstack_list;
          },
          [](Operation &self,
             const std::vector<std::string> &callstack) -> void {
            std::vector<pir::Attribute> op_callstack_infos;
            for (auto str : callstack) {
              op_callstack_infos.push_back(
                  StrAttribute::get(pir::IrContext::Instance(), str));
            }

            self.set_attribute(
                paddle::framework::OpProtoAndCheckerMaker::
                    OpCreationCallstackAttrName(),
                pir::ArrayAttribute::get(pir::IrContext::Instance(),
                                         op_callstack_infos));
          })
      .def_property(
          "dist_attr",
          [](Operation &self) -> py::object {
            if (self.HasAttribute(kAttrOpDistAttr)) {
              return py::cast(
                  self.attribute<OperationDistAttribute>(kAttrOpDistAttr));
            } else {
              return py::cast<py::none>(Py_None);
            }
          },
          [](Operation &self, OperationDistAttribute op_dist_attr) {
            self.set_attribute(kAttrOpDistAttr, op_dist_attr);
          });
  py::class_<Operation::BlockContainer> block_container(
      *m, "Operation_BlockContainer", R"DOC(
    The Operation_BlockContainer only use to walk all blocks in the operation.
     )DOC");
  block_container.def(
      "__iter__",
      [](Operation::BlockContainer &self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>());
}

py::str Value2String(Value self) {
  std::ostringstream print_stream;
  print_stream << "Value(";
  print_stream << GetValueInfo(self);
  print_stream << ")";
  return print_stream.str();
}

const phi::DDim &GetTensorDims(Type type) {
  if (!type) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The type used to get dims is nullptr."));
  }
  if (auto dense_type = type.dyn_cast<DenseTensorType>()) {
    return dense_type.dims();
  } else if (auto select_rows_type = type.dyn_cast<SelectedRowsType>()) {
    return select_rows_type.dims();
  } else if (auto sparse_coo_tensor_type =
                 type.dyn_cast<SparseCooTensorType>()) {
    return sparse_coo_tensor_type.dims();
  } else if (auto sparse_csr_tensr_type =
                 type.dyn_cast<SparseCsrTensorType>()) {
    return sparse_csr_tensr_type.dims();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Currently, we can only get shape for dense and selsect rows type."));
  }
}
const phi::DDim &GetValueDims(Value value) {
  return GetTensorDims(value.type());
}

pir::Value apply(Value self, py::object func) {
  py::gil_scoped_acquire gil;
  auto stop_gradient = self.attribute<BoolAttribute>(kAttrStopGradients);
  if (stop_gradient && !stop_gradient.data()) {
    PADDLE_THROW(phi::errors::Unavailable(
        "Cannot apply function on a tensor that required gradient."));
  }
  PyObject *py_func = func.release().ptr();
  Py_INCREF(py_func);
  PyObject *res = nullptr;
  try {
    py::object obj = py::cast(self);
    PyObject *tmp_self = obj.release().ptr();
    Py_INCREF(tmp_self);
    res = PyObject_CallFunctionObjArgs(py_func, tmp_self, nullptr);
    Py_DECREF(tmp_self);
  } catch (std::exception &e) {
    PADDLE_THROW(phi::errors::Unavailable(
        "Apply function of Tensor raises an exception: %s.", e.what()));
  } catch (...) {
    PADDLE_THROW(phi::errors::Fatal(
        "Apply function of Tensor raises an unknown exception."));
  }
  if (res == Py_None) {
    return self;
  }
  auto out = CastPyArg2Value(res, "", 0, false);
  Py_DECREF(py_func);
  Py_DECREF(res);
  return out;
}

#define DEF_VALUE_BOOL_PROPERTY(name)                                         \
  def_property(                                                               \
      name,                                                                   \
      [](Value self) {                                                        \
        auto bool_data = self.attribute<BoolAttribute>(name);                 \
        return bool_data && bool_data.data();                                 \
      },                                                                      \
      [](Value self, bool bool_data) {                                        \
        self.set_attribute(                                                   \
            name, BoolAttribute::get(pir::IrContext::Instance(), bool_data)); \
      })

#define DEF_VALUE_STOP_GRADIENT_PROPERTY(name)                                \
  def_property(                                                               \
      name,                                                                   \
      [](Value self) {                                                        \
        auto bool_data = self.attribute<BoolAttribute>(name);                 \
        return !bool_data || bool_data.data();                                \
      },                                                                      \
      [](Value self, bool bool_data) {                                        \
        self.set_attribute(                                                   \
            name, BoolAttribute::get(pir::IrContext::Instance(), bool_data)); \
      })

#define DEF_VALUE_POINTER_PROPERTY(name)                                     \
  def_property(                                                              \
      name,                                                                  \
      [](Value self) -> py::object {                                         \
        auto prop_ptr = self.property(name);                                 \
        if (!prop_ptr) {                                                     \
          return py::cast<py::none>(Py_None);                                \
        }                                                                    \
        auto py_data = reinterpret_cast<PyObject *>(prop_ptr);               \
        py::object obj =                                                     \
            py::reinterpret_borrow<py::object>(py::handle(py_data));         \
        return obj;                                                          \
      },                                                                     \
      [](Value self, py::object obj) {                                       \
        pir::PropertiesDeleter deleter = [](void *python_obj) {              \
          Py_DECREF(python_obj);                                             \
        };                                                                   \
        PyObject *pointer_data = obj.release().ptr();                        \
        pir::Property value_property(reinterpret_cast<void *>(pointer_data), \
                                     deleter);                               \
        self.set_property(name, value_property);                             \
      })

void BindValue(py::module *m) {
  py::class_<Value> value(*m,
                          "Value",
                          R"DOC(
    Value class represents the SSA value in the IR system. It is a directed edge
    and a base class.

    Notes:
        The constructor of Value should not be invoked directly. Value can be automatically constructed
        when build network.

  )DOC");
  g_ir_value_pytype = reinterpret_cast<PyTypeObject *>(value.ptr());
  value.def(py::init<>())
      .def_property_readonly(
          "block",
          [](Value self) {
            if (auto op_result = self.dyn_cast<OpResult>()) {
              return op_result.owner()->GetParent();
            }
            return self.dyn_cast<BlockArgument>().owner();
          },
          return_value_policy::reference)
      .def_property_readonly(
          "id",
          [](Value self) {
            if (self.impl() == nullptr) {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "Currently, we can only get id of Value whose impl "
                  "is not nullptr"));
            } else {
              std::stringstream ss;
              ss << std::hex << self.impl();
              return ss.str();
            }
          })
      .def_property(
          "name",
          [](Value self) -> std::string {
            return name_analysis::GetValueFirstName(self);
          },
          [](Value self, const std::string &name) {
            name_analysis::SetValueAllNamesWith(self, name);
          })
      .def_property_readonly(
          "has_name",
          [](Value self) {
            return name_analysis::TryGetValueFirstName(self).has_value();
          })
      // Return all Maybe names of given Value, for example:
      // DataOp("var_1") -> %0 -> shadow_output("output_2")
      // Return ["var_1", "output_2"]
      .def_property_readonly("_names",
                             [](Value self) -> py::list {
                               std::vector<std::string> names =
                                   name_analysis::GetValueAllNames(self);
                               return py::cast(names);
                             })
      .def_property(
          "shape",
          [](Value self) { return phi::vectorize(GetValueDims(self)); },
          [](Value self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set shape when building static graph"));
          })
      .def_property(
          "_local_shape",
          [](Value self) {
            if (!self.type().isa<DistDenseTensorType>()) {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "_local_shape is only for distdense tensor."));
            }
            return phi::vectorize(
                self.type().dyn_cast<DistDenseTensorType>().local_ddim());
          },
          [](Value self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set _local_shape when building static graph"));
          })
      .def_property(
          "dtype",
          [](Value self) { return GetValueDtype(self); },
          [](Value self, phi::DataType dtype) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set dtype when building static graph"));
          })
      .def("initialized",
           [](Value self) {
             if (self.impl() == nullptr || self.type().storage() == nullptr) {
               return false;
             } else {
               return true;
             }
           })
      .DEF_VALUE_STOP_GRADIENT_PROPERTY("stop_gradient")
      .DEF_VALUE_BOOL_PROPERTY("trainable")
      .DEF_VALUE_BOOL_PROPERTY("persistable")
      .DEF_VALUE_BOOL_PROPERTY("need_clip")
      .DEF_VALUE_BOOL_PROPERTY("is_distributed")
      .DEF_VALUE_BOOL_PROPERTY("is_parameter")
      .DEF_VALUE_POINTER_PROPERTY("optimize_attr")
      .DEF_VALUE_POINTER_PROPERTY("regularizer")
      .DEF_VALUE_POINTER_PROPERTY("do_model_average")
      .def("all_used_ops",
           [](Value &self) -> py::list {
             py::list op_list;
             for (auto it = self.use_begin(); it != self.use_end(); ++it) {
               op_list.append(it.owner());
             }
             return op_list;
           })
      .def(
          "get_defining_op",
          [](Value self) -> pir::Operation * { return self.defining_op(); },
          return_value_policy::reference)
      .def("numel", [](Value self) { return phi::product(GetValueDims(self)); })
      .def("type", &Value::type)
      .def("index",
           [](Value self) -> uint32_t {
             if (auto op_result = self.dyn_cast<OpResult>()) {
               return op_result.index();
             }
             PADDLE_THROW(phi::errors::InvalidArgument(
                 "only support accesss index from op_result."));
           })
      .def("is_dense_tensor_type",
           [](Value self) { return self.type().isa<DenseTensorType>(); })
      .def("is_selected_row_type",
           [](Value self) { return self.type().isa<SelectedRowsType>(); })
      .def("is_sparse_coo_tensor_type",
           [](Value self) { return self.type().isa<SparseCooTensorType>(); })
      .def("is_sparse_csr_tensor_type",
           [](Value self) { return self.type().isa<SparseCsrTensorType>(); })
      .def("is_dense_tensor_array_type",
           [](Value self) { return self.type().isa<DenseTensorArrayType>(); })
      .def("is_dist_dense_tensor_type",
           [](Value self) { return self.type().isa<DistDenseTensorType>(); })
      .def("value_assign", [](Value &self, Value value) { self = value; })
      .def("replace_all_uses_with",
           [](Value self, Value value) { self.ReplaceAllUsesWith(value); })
      .def("replace_grad_users_with",
           [](Value self,
              Value value,
              std::unordered_set<Operation *> &grad_ops) {
             for (auto it = self.use_begin(); it != self.use_end();) {
               auto use_op = it.owner();
               if (grad_ops.find(use_op) != grad_ops.end()) {
                 (it++)->set_source(value);
               } else {
                 it++;
               }
             }
           })
      .def("set_type", [](Value self, Type type) { self.set_type(type); })
      .def("first_use", &Value::first_use, return_value_policy::reference)
      .def("has_one_use", &Value::HasOneUse)
      .def("use_empty", &Value::use_empty)
      .def("apply", &apply)
      .def("is_same", &Value::operator==)
      .def("hash", [](Value self) { return std::hash<pir::Value>{}(self); })
      .def("detach",
           [](Value self) {
             auto share_data_op =
                 ApiBuilder::Instance()
                     .GetBuilder()
                     ->Build<paddle::dialect::ShareData_Op>(self);
             auto out = share_data_op.out();
             out.set_attribute(
                 kAttrStopGradients,
                 BoolAttribute::get(pir::IrContext::Instance(), true));
             return out;
           })
      .def("__repr__", &Value2String)
      .def("is_combine",
           [](Value self) { return self.type().isa<pir::VectorType>(); })
      .def("is_dist",
           [](Value self) { return self.type().isa<DistTypeInterface>(); })
      // The function will calculate the new local shape based on the global
      // shape and the dist_attr argument.
      .def("update_dist_attr",
           [](Value &self, TensorDistAttribute dist_attr) {
             self.set_type(dialect::CvtToPirDistType(self.type(), dist_attr));
           })
      .def_property_readonly("process_mesh", [](Value &self) -> py::object {
        auto type = self.type();
        if (auto dist_type = type.dyn_cast<DistTypeInterface>()) {
          return py::cast(
              dist_type.tensor_dist_attr().process_mesh_attr().process_mesh());
        } else {
          return py::cast<py::none>(Py_None);
        }
      });
}

void BindOpOperand(py::module *m) {
  py::class_<OpOperand> op_operand(*m,
                                   "OpOperand",
                                   R"DOC(
    OpOperand class represents the op_operand (input) of operation.

    Notes:
        The constructor of OpOperand should not be invoked directly. OpOperand can be automatically constructed
        when build network.

  )DOC");
  op_operand.def("source", [](OpOperand &self) { return self.source(); })
      .def("set_source",
           [](OpOperand &self, Value *value) {
             value ? self.set_source(*value) : self.set_source(nullptr);
           })
      .def("owner", &OpOperand::owner, return_value_policy::reference)
      .def("index", &OpOperand::index);
}

bool GetValueBoolAttr(Value value, const std::string &attr_name) {
  auto bool_attr = value.attribute<BoolAttribute>(attr_name);
  return !bool_attr || bool_attr.data();
}

void BindType(py::module *m) {
  py::class_<Type> ir_type(*m, "Type");
  ir_type.def("__eq__", &Type::operator==)
      .def_property(
          "shape",
          [](Type self) { return phi::vectorize(GetTensorDims(self)); },
          [](Type self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set shape when building static graph"));
          })
      .def_property(
          "dtype",
          [](Type self) { return GetTensorDtype(self); },
          [](Type self, phi::DataType dtype) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set dtype when building static graph"));
          })
      .def_property(
          "_local_shape",
          [](Type self) {
            if (!self.isa<DistDenseTensorType>()) {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "_local_shape is only for distdense tensor."));
            }
            return phi::vectorize(
                self.dyn_cast<DistDenseTensorType>().local_ddim());
          },
          [](Type self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set _local_shape when building static graph"));
          })
      .def("as_vec_type",
           [](Type self) -> py::object {
             if (auto vec_type = self.dyn_cast<VectorType>()) {
               return py::cast(vec_type);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("as_dist_type",
           [](Type &self) -> py::object {
             if (auto dist_type = self.dyn_cast<DistTypeInterface>()) {
               return py::cast(dist_type);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("__str__", [](Type &self) {
        std::ostringstream print_stream;
        print_stream << self;
        return print_stream.str();
      });

  m->def("create_shaped_type",
         [](Type &type, const std::vector<int> &shape) -> Type {
           if (type.isa<DenseTensorType>()) {
             DenseTensorType src_type = type.dyn_cast<DenseTensorType>();
             DenseTensorType dst_type =
                 DenseTensorType::get(pir::IrContext::Instance(),
                                      src_type.dtype(),
                                      phi::make_ddim(shape),
                                      src_type.data_layout(),
                                      src_type.lod(),
                                      src_type.offset());
             return dst_type;
           } else if (type.isa<SelectedRowsType>()) {
             SelectedRowsType src_type = type.dyn_cast<SelectedRowsType>();
             SelectedRowsType dst_type =
                 SelectedRowsType::get(pir::IrContext::Instance(),
                                       src_type.dtype(),
                                       phi::make_ddim(shape),
                                       src_type.data_layout(),
                                       src_type.lod(),
                                       src_type.offset());
             return dst_type;
           } else {
             PADDLE_THROW(phi::errors::InvalidArgument(
                 "Currently, we can only set shape for dense tensor"));
           }
         });
}
void BindVectorType(py::module *m) {
  py::class_<VectorType, Type> vec_type(*m, "VectorType");
  vec_type.def("as_list", &VectorType::data);
  m->def("create_vec_type", [](std::vector<Type> &types) {
    return VectorType::get(pir::IrContext::Instance(), types);
  });
}
void BindAttribute(py::module *m) {
  py::class_<Attribute> ir_attr(*m, "Attribute", py::module_local());
  ir_attr.def("__eq__", &Attribute::operator==)
      .def("__str__",
           [](Attribute &self) {
             std::ostringstream print_stream;
             print_stream << self;
             return print_stream.str();
           })
      .def("as_tensor_dist_attr",
           [](Attribute &self) -> py::object {
             if (auto dist_attr = self.dyn_cast<TensorDistAttribute>()) {
               return py::cast(dist_attr);
             }
             return py::cast<py::none>(Py_None);
           })
      .def("as_array_attr", [](Attribute &self) -> py::object {
        if (auto array_attr = self.dyn_cast<ArrayAttribute>()) {
          return py::cast(array_attr);
        }
        return py::cast<py::none>(Py_None);
      });
  py::class_<ArrayAttribute, Attribute> array_attr(*m, "ArrayAttribute");
  array_attr.def("__len__", [](ArrayAttribute &self) { return self.size(); })
      .def("__getitem__",
           [](ArrayAttribute &self, int idx) { return self.at(idx); });
}

struct PyInsertionPoint {
  pir::InsertionPoint value;
};
void BindInsertionPoint(pybind11::module *m) {
  py::class_<PyInsertionPoint> ir_insertion_point(*m, "InsertionPoint", R"DOC(
    InsertionPoint class represents the insertion point in the Builder.)DOC");
  ir_insertion_point
      .def(
          "next",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->end()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the end and can't call "
                  "next()."));
            }
            return *(self.value.second++);
          },
          return_value_policy::reference)
      .def(
          "prev",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->begin()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the begin and can't call "
                  "prev()."));
            }
            return *(self.value.second--);
          },
          return_value_policy::reference)
      .def(
          "get_operation",
          [](PyInsertionPoint &self) -> Operation & {
            if (self.value.second == self.value.first->begin()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the begin."));
            } else if (self.value.second == self.value.first->end()) {
              PADDLE_THROW(common::errors::InvalidArgument(
                  "The insertion point is already at the end."));
            }
            return *(self.value.second);
          },
          return_value_policy::reference)
      .def(
          "block",
          [](PyInsertionPoint &self) { return self.value.first; },
          return_value_policy::reference);
}

std::list<Operation *>::const_iterator list_offset(const Block *block,
                                                   int start_idx) {
  auto it = block->begin();
  while (it != block->end() && start_idx--) ++it;
  return it;
}

template <typename F, typename S>
void range_block_do(const Block *block,
                    std::vector<int> range,
                    F fn,
                    S skip_fn) {
  for (auto it = list_offset(block, range[0]);
       it != list_offset(block, range[1]);
       ++it) {
    if (skip_fn(*it)) {
      continue;
    }
    fn(*it);
  }
}

template <typename F>
void range_block_do(const Block *block, std::vector<int> range, F fn) {
  range_block_do(block, range, fn, [](Operation *op) { return false; });
}

std::map<int, int> GetOpInplaceInfo(const pir::Operation *op) {
  std::map<int, int> inplace_info;
  if (!op->HasTrait<paddle::dialect::InplaceTrait>()) {
    return inplace_info;
  }
  pir::IrContext *ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string &inplace_name = yaml_parser.InplaceName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(inplace_name);
    }
    if (yaml_parser.HasView(value_name)) {
      const std::string &view_name = yaml_parser.ViewName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(view_name);
    }
  }

  return inplace_info;
}

std::pair<std::vector<pir::Value>, std::unordered_set<pir::Value>>
AnalysisMiddleVariable(const Program &program,
                       const std::vector<pir::Value> &forward_inputs,
                       const std::vector<int> &forward_range,
                       const std::vector<int> &backward_range) {
  std::vector<pir::Value> middle_values;

  std::unordered_set<pir::Value> backward_inputs;
  std::unordered_set<pir::Value> x_or_param(forward_inputs.begin(),
                                            forward_inputs.end());
  range_block_do(
      program.block(), backward_range, [&backward_inputs](Operation *op) {
        pir::Walk(op, [&](Operation *inner_op) {
          for (auto &t : inner_op->operands()) {
            backward_inputs.insert(t.source());
          }
        });
      });

  range_block_do(
      program.block(),
      forward_range,
      [&middle_values, &backward_inputs, &x_or_param](Operation *op) {
        pir::Walk(op, [&](Operation *inner_op) {
          for (auto &t : inner_op->results()) {
            auto v = Value(t.Value::impl());
            if (backward_inputs.count(v) && !x_or_param.count(v)) {
              middle_values.push_back(v);
            }
          }
        });
      });
  return std::make_pair(middle_values, backward_inputs);
}

void mapping_value(const std::vector<pir::Value> &origin,
                   const std::unordered_map<pir::Value, pir::Value> &value_map,
                   std::vector<pir::Value> &out) {  // NOLINT
  std::transform(origin.begin(),
                 origin.end(),
                 std::back_inserter(out),
                 [&value_map](const pir::Value &v) {
                   if (v.impl() == nullptr) return Value(nullptr);
                   if (!value_map.count(v)) {
                     VLOG(2) << "mapping value found v is not exist. may not "
                                "used by backward program.";
                     return Value(nullptr);
                   }
                   return value_map.at(v);
                 });
}

using SplitedProgram = std::vector<std::shared_ptr<Program>>;
using SplitedAttribute = std::map<std::string, std::vector<pir::Value>>;
using SplitedResult = std::pair<SplitedProgram, SplitedAttribute>;

static auto GetNoNeedBufferValue(const ::pir::Block *whole_block,
                                 std::vector<int> range) {
  // filter no need buffer values.
  std::unordered_set<::pir::Value> need_buffer_values;
  std::unordered_set<::pir::Value> no_need_buffer_values;
  range_block_do(
      whole_block, range, [&need_buffer_values](::pir::Operation *op) {
        // NOTE(SigureMo): We should process the CombineOp in it's users.
        if (op->isa<pir::CombineOp>()) {
          return;
        }
        if (op->HasInterface<paddle::dialect::OpYamlInfoInterface>() == false) {
          // not a OpYamlInfoInterface, can't have no_need_buffer.
          for (const auto &operand : op->operands_source()) {
            need_buffer_values.insert(operand);
          }
        } else {
          auto opinfo =
              op->dyn_cast<paddle::dialect::OpYamlInfoInterface>().GetOpInfo();
          int counter = 0;
          for (const auto &op_input_info : std::get<0>(opinfo)) {
            auto value = op->operand_source(counter);
            if (!op_input_info.no_need_buffer) {
              need_buffer_values.insert(value);
              if (!IsFakeValue(value) && value.defining_op() &&
                  value.defining_op()->isa<pir::CombineOp>()) {
                for (const auto &combine_value :
                     value.defining_op()->operands_source()) {
                  need_buffer_values.insert(combine_value);
                }
              }
            }
            counter += 1;
          }
        }
      });
  range_block_do(whole_block,
                 range,
                 [&need_buffer_values,
                  &no_need_buffer_values](const ::pir::Operation *op) {
                   for (const auto &operand : op->operands_source()) {
                     if (need_buffer_values.count(operand) == 0) {
                       no_need_buffer_values.insert(operand);
                     }
                   }
                 });
  return std::vector<::pir::Value>(no_need_buffer_values.begin(),
                                   no_need_buffer_values.end());
}

using ValueMap = std::pair<std::vector<pir::Value>, std::vector<pir::Value>>;
std::pair<std::shared_ptr<Program>, ValueMap> CloneProgram(
    const Program &program) {
  // Limitation of this function:
  // 1. don't support Parameters.
  pir::IrMapping mapper;
  auto cloned_program = program.Clone(mapper);
  std::vector<pir::Value> associated_array_key, associated_array_value;
  for (auto &pair : mapper.GetMap<pir::Value>()) {
    associated_array_key.push_back(pair.first);
    associated_array_value.push_back(pair.second);
  }
  return std::make_pair(
      cloned_program,
      std::make_pair(associated_array_key, associated_array_value));
}

void AppendShadowOutput(Program *program,
                        const pir::Value &value,
                        const std::string &name,
                        size_t start_point) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
  pir::AttributeMap attribute_map = {
      {"output_name", StrAttribute::get(ctx, name)},
  };
  pir::Operation *operation =
      pir::Operation::Create({value}, attribute_map, {}, op_info);
  auto position = program->block()->begin();
  std::advance(position, start_point);
  if (position == program->block()->end()) {
    program->block()->push_back(operation);
  } else {
    program->block()->insert(position, operation);
  }
}

int AppendShadowOutputs(Program *program,
                        const std::vector<pir::Value> &outputs,
                        int start_point,
                        std::string name_prefix) {
  int counter = 0;
  std::unordered_set<pir::Value> added_value;
  for (const auto &value : outputs) {
    if (!added_value.count(value) || IsFakeValue(value)) {
      std::string shadow_output_name = name_prefix + std::to_string(counter);
      if (auto names = name_analysis::GetValueOutputNames(value);
          !names.empty()) {
        shadow_output_name = names[0];
      }
      AppendShadowOutput(
          program, value, shadow_output_name, start_point + counter);
      counter += 1;
      added_value.insert(value);
    }
  }
  // return the inserted op.
  return counter;
}

std::unordered_map<::pir::Value, std::string> GetNameMap(
    const ::pir::Block *block) {
  std::unordered_map<::pir::Value, std::string> value2name;
  for (auto &kwarg : block->kwargs()) {
    value2name[kwarg.second] = kwarg.first;
  }
  for (auto &op : *block) {
    std::string name;
    if (op.name() == "pd_op.data") {
      name = op.attributes().at("name").dyn_cast<StrAttribute>().AsString();
      value2name[op.results()[0].Value::impl()] = name;
    } else if (op.name() == "builtin.set_parameter") {
      name = op.attributes()
                 .at("parameter_name")
                 .dyn_cast<StrAttribute>()
                 .AsString();
      value2name[op.operand(0).source()] = name;
    } else if (op.name() == "builtin.shadow_output") {
      name =
          op.attributes().at("output_name").dyn_cast<StrAttribute>().AsString();
      value2name[op.operand(0).source()] = name;
    } else if (op.name() == "builtin.parameter") {
      name = op.attributes()
                 .at("parameter_name")
                 .dyn_cast<StrAttribute>()
                 .AsString();
      value2name[op.result(0).Value::impl()] = name;
    } else if (op.name() == "builtin.constant") {
      if (op.isa<pir::ConstantTensorOp>()) {
        name = op.dyn_cast<pir::ConstantTensorOp>().tensor_name();
        value2name[op.result(0).Value::impl()] = name;
      }
    }
  }
  return value2name;
}

SplitedResult SplitForwardBackward(
    const Program &program,
    const std::vector<pir::Value> &forward_inputs,
    const std::vector<pir::Value> &forward_params,
    const std::vector<pir::Value> &forward_outputs,
    const std::vector<pir::Value> &forward_inputs_grads,
    const std::vector<pir::Value> &forward_params_grads,
    const std::vector<pir::Value> &forward_outputs_grads,
    const std::vector<int> &forward_range,
    const std::vector<int> &backward_range) {
  std::vector<pir::Value> forward_in_out_values;
  for (auto &v :
       std::vector({&forward_inputs, &forward_outputs, &forward_params})) {
    forward_in_out_values.insert(
        forward_in_out_values.end(), v->begin(), v->end());
  }

  std::vector<pir::Value> fx, fp, fm, fo, bx, bp, bm, bo_g, bx_g, bp_g, bo;
  std::vector<pir::Value> no_need_buffer_values;
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto forward_program = std::make_shared<Program>(ctx);
  auto backward_program = std::make_shared<Program>(ctx);
  std::vector<pir::Value> middle_values;
  std::unordered_set<pir::Value> backward_inputs;
  std::tie(middle_values, backward_inputs) = AnalysisMiddleVariable(
      program, forward_in_out_values, forward_range, backward_range);

  pir::Block &backward_block = *backward_program->block();
  bool has_backward = (backward_range[1] > backward_range[0]);

  // forward program construct.
  VLOG(4) << "start create forward program.";
  pir::IrMapping forward_mapper;
  auto clone_options = pir::CloneOptions::All();
  range_block_do(
      program.block(),
      forward_range,
      [&forward_mapper, &forward_program, &clone_options](Operation *op) {
        auto *cloned_op = op->Clone(forward_mapper, clone_options);
        forward_program->block()->push_back(cloned_op);
      },
      // Skip the ShadowOutputOp.
      /*skip_fn=*/[](Operation *op) { return op->isa<pir::ShadowOutputOp>(); });
  auto &forward_value_map = forward_mapper.GetMutableMap<pir::Value>();

  // backward program construct.
  // Step1. insert data op for inputs_values and middle_values
  pir::IrMapping backward_mapper;
  auto &backward_value_map = backward_mapper.GetMutableMap<pir::Value>();
  int counter = forward_outputs.size();

  auto create_output_fn_forward =
      [&ctx, &forward_value_map, &counter, &forward_program, &forward_params](
          const pir::Value &v) {
        if (v.impl() == nullptr) {
          return;
        }
        // Skip the value that already in forward_params.
        if (std::find(forward_params.begin(), forward_params.end(), v) !=
            forward_params.end()) {
          return;
        }
        std::string shadow_output_name =
            std::string("output_") + std::to_string(counter);
        if (auto names = name_analysis::TryGetValueFirstName(v)) {
          shadow_output_name = names.value();
        }
        auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
        pir::AttributeMap attribute_map = {
            {"output_name", StrAttribute::get(ctx, shadow_output_name)},
        };
        pir::Operation *operation = pir::Operation::Create(
            {forward_value_map[v]}, attribute_map, {}, op_info);
        forward_program->block()->push_back(operation);
        counter += 1;
      };

  auto create_output_fn_backward = [&ctx,
                                    &backward_value_map,
                                    &counter,
                                    &backward_program](const pir::Value &v) {
    if (v.impl() == nullptr) {
      return;
    }
    auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
    pir::AttributeMap attribute_map = {
        {"output_name",
         StrAttribute::get(ctx,
                           std::string("output_") + std::to_string(counter))},
    };
    pir::Operation *operation = pir::Operation::Create(
        {backward_value_map.at(v)}, attribute_map, {}, op_info);
    backward_program->block()->push_back(operation);
    counter += 1;
  };

  VLOG(4) << "start create forward outputs, inserting shadow_output ops.";
  std::for_each(
      middle_values.begin(), middle_values.end(), create_output_fn_forward);
  std::for_each(
      forward_outputs.begin(), forward_outputs.end(), create_output_fn_forward);

  pir::Block *forward_block = forward_program->block();
  const auto &forward_name_map = GetNameMap(forward_block);
  auto create_kwarg_fn = [&backward_block,
                          &backward_inputs,
                          &backward_value_map,
                          &forward_value_map,
                          &forward_name_map,
                          &counter](const pir::Value &v) {
    if (v && !backward_value_map.count(v) && (backward_inputs.count(v))) {
      auto forward_value = forward_value_map[v];
      std::string name = "input_" + std::to_string(counter++);
      if (forward_name_map.count(forward_value)) {
        name = forward_name_map.at(forward_value);
      }

      backward_value_map[v] = backward_block.AddKwarg(name, v.type());
    }
  };

  if (has_backward) {
    VLOG(4) << "start create backward inputs, creating keyword argument.";
    VLOG(4)
        << "Create keyword argument for backward program: fo, start with input_"
        << counter;
    std::for_each(
        forward_outputs.begin(), forward_outputs.end(), create_kwarg_fn);
    VLOG(4)
        << "Create keyword argument for backward program: fx, start with input_"
        << counter;
    std::for_each(
        forward_inputs.begin(), forward_inputs.end(), create_kwarg_fn);
    VLOG(4)
        << "Create keyword argument for backward program: fp, start with input_"
        << counter;
    std::for_each(
        forward_params.begin(), forward_params.end(), create_kwarg_fn);
    VLOG(4)
        << "Create keyword argument for backward program: fm, start with input_"
        << counter;
    std::for_each(middle_values.begin(), middle_values.end(), create_kwarg_fn);
    VLOG(4) << "Create keyword argument for backward program: fo_g, start with "
               "input_"
            << counter;
    std::for_each(forward_outputs_grads.begin(),
                  forward_outputs_grads.end(),
                  create_kwarg_fn);
    VLOG(4) << "Create keyword argument for backward program end. input_"
            << counter;
  }

  // Step2. copy backward ops .
  VLOG(4) << "start copy backward ops";
  range_block_do(
      program.block(),
      backward_range,
      [&backward_mapper, &backward_program, &clone_options](Operation *op) {
        auto *cloned_op = op->Clone(backward_mapper, clone_options);
        backward_program->block()->push_back(cloned_op);
      });
  VLOG(4) << "start create backward outputs, inserting shadow_output ops.";
  if (has_backward) {
    std::for_each(forward_inputs_grads.begin(),
                  forward_inputs_grads.end(),
                  create_output_fn_backward);
    std::for_each(forward_params_grads.begin(),
                  forward_params_grads.end(),
                  create_output_fn_backward);
  }

  VLOG(4) << "forward_value_map.size() is " << forward_value_map.size();
  VLOG(4) << "backward_value_map.size() is " << backward_value_map.size();
  if (FLAGS_print_ir) {
    std::ostringstream print_stream;
    print_stream << "ForwardProgram is :\n";
    forward_program->Print(print_stream);
    print_stream << "BackwardProgram is:\n";
    backward_program->Print(print_stream);
    std::cout << "Splited Program (fwd | bwd): \n"
              << print_stream.str() << std::endl;
  }

  // construct all attributes we needed.

  mapping_value(middle_values, forward_value_map, fm);    // write 'fm'
  mapping_value(middle_values, backward_value_map, bm);   // write 'bm'
  mapping_value(forward_inputs, forward_value_map, fx);   // write 'fx'
  mapping_value(forward_inputs, backward_value_map, bx);  // write 'bx'
  mapping_value(forward_params, forward_value_map, fp);   // write 'fp'
  mapping_value(forward_params, backward_value_map, bp);  // write 'bp'
  mapping_value(forward_outputs, forward_value_map, fo);  // write 'fo'
  mapping_value(
      forward_inputs_grads, backward_value_map, bx_g);  // write 'bx_g'
  mapping_value(
      forward_params_grads, backward_value_map, bp_g);  // write 'bp_g'
  mapping_value(
      forward_outputs_grads, backward_value_map, bo_g);    // write 'bo_g'
  mapping_value(forward_outputs, backward_value_map, bo);  // write 'bo'
  mapping_value(GetNoNeedBufferValue(program.block(), backward_range),
                forward_value_map,
                no_need_buffer_values);  // write 'no_need_buffers'

  std::map<std::string, std::vector<pir::Value>> attr = {
      {"fx", fx},
      {"fp", fp},
      {"fm", fm},
      {"fo", fo},
      {"bx", bx},
      {"bp", bp},
      {"bm", bm},
      {"bo_g", bo_g},
      {"bx_g", bx_g},
      {"bp_g", bp_g},
      {"no_need_buffers", no_need_buffer_values},
      {"bo", bo}};
  std::vector<std::shared_ptr<Program>> programs = {forward_program,
                                                    backward_program};
  return std::make_pair(programs, attr);
}

pir::Type CreateSelectedRowsTypeByDenseTensor(pir::Type dense_tensor_type) {
  if (dense_tensor_type.isa<DenseTensorType>()) {
    DenseTensorType type = dense_tensor_type.dyn_cast<DenseTensorType>();
    return SelectedRowsType::get(pir::IrContext::Instance(),
                                 type.dtype(),
                                 type.dims(),
                                 type.data_layout(),
                                 type.lod(),
                                 type.offset());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, input is not a dense tensor type."));
  }
}

pir::Type CreateDistDenseTensorTypeByDenseTensor(
    const pir::Type &gdense_tensor_type,
    const std::vector<int> &lshape,
    const phi::distributed::ProcessMesh &mesh,
    const std::vector<int64_t> &dims_mapping) {
  if (gdense_tensor_type.isa<DenseTensorType>()) {
    DenseTensorType type = gdense_tensor_type.dyn_cast<DenseTensorType>();
    paddle::flat_hash_map<int64_t, phi::ReduceType> partial_status;
    paddle::dialect::TensorDistAttribute tensor_dist_attr =
        paddle::dialect::TensorDistAttribute::get(
            pir::IrContext::Instance(), mesh, dims_mapping, partial_status);
    return DistDenseTensorType::get(pir::IrContext::Instance(),
                                    type,
                                    tensor_dist_attr,
                                    phi::make_ddim(lshape));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, input is not a dense tensor type are not supported."));
  }
}

static void inline CreateVariableIfNotExist(
    const std::vector<pir::Value> &var_list,
    framework::Scope *scope,
    const framework::Executor *exe = nullptr) {
  size_t len = var_list.size();

  for (size_t i = 0; i < len; ++i) {
    pir::Value value = var_list[i];
    std::string para_name = name_analysis::GetValueFirstName(value);
    auto var = scope->FindVar(para_name);
    if (var == nullptr) {
      PADDLE_ENFORCE_NOT_NULL(exe,
                              phi::errors::InvalidArgument(
                                  "Parameter not Initialized, "
                                  "Please set argument [executor] not None "
                                  "or run startup program first"));
      var = scope->Var(para_name);
      auto *tensor_temp = var->GetMutable<phi::DenseTensor>();
      tensor_temp->Resize(
          common::make_ddim(phi::vectorize(GetValueDims(value))));
      phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
      const phi::DeviceContext *dev_ctx = nullptr;
      dev_ctx = pool.Get(exe->GetPlace());
      dev_ctx->Alloc(tensor_temp, GetValueDtype(value));
    }
  }
  return;
}

void ResetShadowOutputName(pir::Operation *op, const std::string &name) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  if (op->isa<pir::ShadowOutputOp>()) {
    op->set_attribute("output_name", StrAttribute::get(ctx, name));
  }
}

void BindUtils(pybind11::module *m) {
  m->def("create_loaded_parameter", CreateVariableIfNotExist);
  m->def("clone_program", CloneProgram);
  m->def("get_op_inplace_info", GetOpInplaceInfo);
  m->def("reset_shadow_output_name", ResetShadowOutputName);
  m->def("split_program", SplitForwardBackward);
  m->def("append_shadow_outputs", AppendShadowOutputs);
  m->def("append_shadow_output", AppendShadowOutput);
  m->def("fake_value", FakeValue);
  m->def("is_fake_value", IsFakeValue);
  m->def("get_current_insertion_point", []() -> PyInsertionPoint {
    return {ApiBuilder::Instance().GetCurrentInsertionPoint()};
  });
  m->def("set_insertion_point", [](const PyInsertionPoint &insertion_point) {
    ApiBuilder::Instance().SetInsertionPoint(insertion_point.value);
  });
  m->def("set_insertion_point",
         [](Operation *op) { ApiBuilder::Instance().SetInsertionPoint(op); });
  m->def("set_insertion_point_after", [](Operation *op) {
    ApiBuilder::Instance().SetInsertionPointAfter(op);
  });
  m->def("set_insertion_point_to_block_end", [](Block *block) {
    ApiBuilder::Instance().SetInsertionPointToBlockEnd(block);
  });
  m->def("reset_insertion_point_to_start",
         []() { ApiBuilder::Instance().ResetInsertionPointToStart(); });
  m->def("reset_insertion_point_to_end",
         []() { ApiBuilder::Instance().ResetInsertionPointToEnd(); });
  m->def("register_paddle_dialect", []() {
    pir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  });
  m->def("register_dist_dialect", []() {
    pir::IrContext::Instance()
        ->GetOrRegisterDialect<paddle::dialect::DistDialect>();
  });
  m->def("create_selected_rows_type_by_dense_tensor",
         CreateSelectedRowsTypeByDenseTensor);
  m->def("create_dist_dense_tensor_type_by_dense_tensor",
         CreateDistDenseTensorTypeByDenseTensor);
  m->def(
      "translate_to_pir",
      [](const ::paddle::framework::ProgramDesc &legacy_program) {
        std::shared_ptr<Program> ret =
            paddle::TranslateLegacyProgramToProgram(legacy_program);
        return ret;
      },
      R"DOC(
        Convert Fluid Program to New IR Program.

        Args:

            legacy_program (ProgramDesc): The Fluid Program that will be converted.

        Returns:
            Program: The New IR Program

        Raises:
            PreconditionNotMet: If legacy_program has multi block will raise error.

        Examples:
            .. code-block:: python

                >>> import os
                >>> # Paddle will remove this flag in the next version
                >>> pir_flag = 'FLAGS_enable_pir_in_executor'
                >>> os.environ[pir_flag] = 'True'

                >>> import paddle
                >>> from paddle import pir
                >>> paddle.enable_static()

                >>> x = paddle.randn([4, 4])
                >>> main_program, start_program = (
                ...    paddle.static.Program(),
                ...    paddle.static.Program(),
                ...)

                >>> with paddle.static.program_guard(main_program, start_program):
                ...    x_s = paddle.static.data('x', [4, 4], x.dtype)
                ...    x_s.stop_gradient = False
                ...    y_s = paddle.matmul(x_s, x_s)
                ...    z_s = paddle.add(y_s, y_s)
                ...    k_s = paddle.tanh(z_s)
                >>> pir_program = pir.translate_to_pir(main_program.desc)

                >>> print(pir_program)
                {
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> builtin.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                }


      )DOC");
  m->def(
      "check_unregistered_ops",
      [](const framework::ProgramDesc &legacy_program) {
        pir::IrContext *ctx = pir::IrContext::Instance();
        return paddle::translator::CheckUnregisteredOperation(ctx,
                                                              legacy_program);
      },
      R"DOC(
      Check unregistered operators in paddle dialect.

      Args:
        legacy_program (ProgramDesc): The Fluid Program that need checked.
      Returns:
        list[str] : List of unregistered operators in paddle dialect, the name is expressed by origin op name.
    )DOC");
  m->def(
      "translate_to_pir_with_param_map",
      [](const framework::ProgramDesc &legacy_program) {
        auto ir_ctx = pir::IrContext::Instance();
        auto program = std::make_shared<pir::Program>(ir_ctx);
        translator::ProgramTranslator program_translator(&legacy_program,
                                                         program.get());
        program_translator.Translate();
        return std::make_pair(program, program_translator.VarDesc2Value());
      },
      R"DOC(
        Convert Fluid Program to New IR Program and get the mappings of VarDesc -> pir::Value.

        Args:

            legacy_program (ProgramDesc): The Fluid Program that will be converted.

        Returns:
            Program: The New IR Program
            dict[str, pir::Value]: Mapping between VarDesc(by name) and pir::Value.

        Raises:
            PreconditionNotMet: If legacy_program has multi block will raise error.

        Examples:
            .. code-block:: python

                >>> import os
                >>> # Paddle will remove this flag in the next version
                >>> pir_flag = 'FLAGS_enable_pir_in_executor'
                >>> os.environ[pir_flag] = 'True'

                >>> import paddle
                >>> from paddle import pir
                >>> paddle.enable_static()

                >>> x = paddle.randn([4, 4])
                >>> main_program, start_program = (
                ...     paddle.static.Program(),
                ...     paddle.static.Program(),
                ... )

                >>> with paddle.static.program_guard(main_program, start_program):
                ...     x_s = paddle.static.data('x', [4, 4], x.dtype)
                ...     x_s.stop_gradient = False
                ...     y_s = paddle.matmul(x_s, x_s)
                ...     z_s = paddle.add(y_s, y_s)
                ...     k_s = paddle.tanh(z_s)
                >>> pir_program, mappings = pir.translate_to_pir_with_param_map(main_program.desc)

                >>> print(pir_program)
                {
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> builtin.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>, builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (builtin.tensor<4x4xf32>) -> builtin.tensor<4x4xf32>
                }

                >>> print(mappings)
                {'matmul_v2_0.tmp_0': [Value(define_op_name=pd_op.matmul, index=0, dtype=builtin.tensor<4x4xf32>)], 'x': [Value(define_op_name=pd_op.data, index=0, dtype=builtin.tensor<4x4xf32>)], 'tanh_0.tmp_0': [Value(define_op_name=pd_op.tanh, index=0, dtype=builtin.tensor<4x4xf32>)], 'elementwise_add_0': [Value(define_op_name=pd_op.add, index=0, dtype=builtin.tensor<4x4xf32>)]}
    )DOC");
  m->def("clear_cinn_compilation_cache", []() {
#ifdef PADDLE_WITH_CINN
    pybind11::gil_scoped_release release;
    VLOG(4) << "clear CINN CompilationCache and free BackendResource.";
    cinn::hlir::framework::CompilationCache::Instance().Clear();
#endif
  });

  m->def("cinn_compilation_cache_size", []() {
#ifdef PADDLE_WITH_CINN
    pybind11::gil_scoped_release release;
    VLOG(4) << "clear CINN CompilationCache and free BackendResource.";
    return cinn::hlir::framework::CompilationCache::Instance().Size();
#endif
  });
}

namespace {

#ifdef PADDLE_WITH_CINN
std::shared_ptr<pir::PassManager> CreatePassManager() {
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  auto pass_manager = std::make_shared<pir::PassManager>(ctx);
  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }
  return pass_manager;
}
#endif

void ApplyCinnPass(Program &program) {  // NOLINT
#ifdef PADDLE_WITH_CINN
  cinn::dialect::ir::ApplyCinnPass(&program, CreatePassManager);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "Currently we only support CINN Pass for Pir under @to_static, please "
      "compile PaddlePaddle with CINN"));
#endif
}

void CheckInferSymbolicIfNeed(Program &program) {  // NOLINT
#ifdef PADDLE_WITH_CINN
  cinn::dialect::ir::CheckInferSymbolicIfNeed(&program, CreatePassManager);
#else
  // Do nothing.
#endif
}

}  // namespace

void InferSymbolicShapePass(
    std::shared_ptr<pir::PassManager> &pass_manager,  // NOLINT
    pir::Program &program) {                          // NOLINT
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  if (FLAGS_pir_apply_shape_optimization_pass) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
}

std::shared_ptr<Program> ApplyCommonSubexpressionEliminationPass(
    std::shared_ptr<Program> program) {
  pir::PassManager pm(pir::IrContext::Instance(), 2);
  pm.AddPass(pir::CreateCommonSubexpressionEliminationPass());
  pm.Run(program.get());
  if (FLAGS_print_ir) {
    std::cout
        << "IR After CommonSubexpressionEliminationPass -------------------"
        << std::endl;
    std::cout << *program << std::endl;
  }
  return program;
}

std::shared_ptr<Program> ApplyFusedBnAddActPass(
    std::shared_ptr<Program> program) {
  pir::PassManager pm(pir::IrContext::Instance(), 3);
  pm.AddPass(pir::CreateFusedBnAddActPass());
  pm.Run(program.get());
  if (FLAGS_print_ir) {
    std::cout << "IR After FusedBnAddActPass -------------------" << std::endl;
    std::cout << *program << std::endl;
  }
  return program;
}

void BindIrPass(pybind11::module *m) {
  m->def("apply_cinn_pass", ApplyCinnPass);
  m->def("check_infer_symbolic_if_need", CheckInferSymbolicIfNeed);
  m->def("infer_symbolic_shape_pass", InferSymbolicShapePass);
  m->def("apply_cse_pass", ApplyCommonSubexpressionEliminationPass);
  m->def("apply_bn_add_act_pass", ApplyFusedBnAddActPass);

  py::class_<Pass, std::shared_ptr<Pass>> pass(*m,
                                               "Pass",
                                               R"DOC(
    Pass class.

  )DOC");
  pass.def("name", &Pass::name)
      .def("opt_level",
           [](const Pass &self) { return self.pass_info().opt_level; })
      .def("dependents",
           [](const Pass &self) { return self.pass_info().dependents; });
}

void BindPassManager(pybind11::module *m) {
  py::class_<PassManager, std::shared_ptr<PassManager>> pass_manager(
      *m,
      "PassManager",
      R"DOC(
    A class that manages all passes.

  )DOC");
  pass_manager
      .def(py::init([](uint8_t opt_level) {
             return std::make_unique<PassManager>(pir::IrContext::Instance(),
                                                  opt_level);
           }),
           py::arg("opt_level") = 2)
      .def("add_pass",
           [](PassManager &self,
              const std::string &pass_name,
              const std::unordered_map<std::string, py::object> attrs = {}) {
             auto pass = pir::PassRegistry::Instance().Get(pass_name);
             for (const auto &attr : attrs) {
               if (py::isinstance<py::str>(attr.second)) {
                 pass->Set(attr.first,
                           new std::string(attr.second.cast<std::string>()));
               } else if (py::isinstance<py::bool_>(attr.second)) {
                 pass->Set(attr.first, new bool(attr.second.cast<bool>()));
               } else if (py::isinstance<py::int_>(attr.second)) {
                 pass->Set(attr.first, new int(attr.second.cast<int>()));
               } else if (py::isinstance<py::float_>(attr.second)) {
                 pass->Set(attr.first, new float(attr.second.cast<float>()));
               } else {
                 PADDLE_THROW(phi::errors::InvalidArgument(
                     "The pass attr is not supported this type."));
               }
             }
             self.AddPass(std::move(pass));
           })
      .def("passes",
           [](PassManager &self) {
             std::vector<std::string> pass_names;
             for (const auto &pass : self.passes()) {
               pass_names.emplace_back(pass->name());
             }
             return pass_names;
           })
      .def("run", [](PassManager &self, Program *p) { self.Run(p); })
      .def("empty", &PassManager::empty)
      .def("clear", &PassManager::clear)
      .def("enable_ir_printing",
           [](PassManager &self) { self.EnableIRPrinting(); })
      .def("enable_print_statistics",
           [](PassManager &self) { self.EnablePrintStatistics(); });
}

void BindShapeOrDataDimExprs(pybind11::module *m) {
  py::class_<symbol::ShapeOrDataDimExprs,
             std::shared_ptr<symbol::ShapeOrDataDimExprs>>
      shape_or_data_dim_exprs(*m, "ShapeOrDataDimExprs", R"DOC(
      A class that store the shape or data of value.
    )DOC");
  shape_or_data_dim_exprs
      .def("shape",
           &symbol::ShapeOrDataDimExprs::shape,
           return_value_policy::reference)
      .def("data",
           &symbol::ShapeOrDataDimExprs::data,
           return_value_policy::reference)
      .def("is_equal",
           [](symbol::ShapeOrDataDimExprs &self,
              std::vector<int64_t> expect_shape,
              std::vector<int64_t> expect_data = {}) -> bool {
             VLOG(3) << "Start compare shape and data.";

             const auto &compare_func =
                 [&](const std::vector<int64_t> &expect,
                     const std::vector<symbol::DimExpr> &actual) -> bool {
               if (actual.size() != expect.size()) {
                 LOG(ERROR) << "expect size " << expect.size()
                            << " is not equal to actual size " << actual.size()
                            << " .";
                 return false;
               } else if (actual.empty()) {
                 return true;
               }
               for (size_t i = 0; i < actual.size(); i++) {
                 if (!actual.at(i).isa<int64_t>()) {
                   LOG(ERROR)
                       << "expect[" << i << "]: " << expect.at(i) << " actual["
                       << i << "]: " << actual.at(i) << " .";
                   PADDLE_THROW(phi::errors::InvalidArgument(
                       "In OpTest, only supports cases where the type of "
                       "DimExpr "
                       "is int64_t."));
                   return false;
                 }
                 if (actual.at(i) != expect.at(i)) {
                   LOG(ERROR) << "expect[" << i << "]: " << expect.at(i)
                              << " is not equal to actual[" << i
                              << "]: " << actual.at(i) << " .";
                   return false;
                 }
               }
               return true;
             };

             // compare shape
             const std::vector<symbol::DimExpr> &actual_shape = self.shape();

             // TODO(gongshaotian): compare data
             return compare_func(expect_shape, actual_shape);
           });
}

void BindShapeConstraintIRAnalysis(pybind11::module *m) {
  m->def(
      "get_shape_constraint_ir_analysis",
      [](const pir::Program *program) -> pir::ShapeConstraintIRAnalysis & {
        return pir::ShapeAnalysisManager::Instance().Get(program);
      },
      return_value_policy::reference);
  m->def("all_ops_defined_symbol_infer",
         [](const pir::Program *program) -> bool {
           // check that all ops have defined the InferSymbolicShapeInterface
           bool flag = true;
           for (pir::Operation &op : *(program->block())) {
             pir::InferSymbolicShapeInterface infer_interface =
                 op.dyn_cast<pir::InferSymbolicShapeInterface>();
             if (!infer_interface) {
               LOG(ERROR) << "The op: " << op.name()
                          << " does not implement InferSymbolicShapeInterface.";
               flag = false;
             }
           }
           return flag;
         });

  py::class_<pir::ShapeConstraintIRAnalysis,
             std::shared_ptr<pir::ShapeConstraintIRAnalysis>>
      shape_constraint_ir_analysis(*m, "ShapeConstraintIRAnalysis", R"DOC(
      A class that store the shape information of all operators.
    )DOC");
  shape_constraint_ir_analysis
      .def("get_shape_or_data_for_var",
           &pir::ShapeConstraintIRAnalysis::GetShapeOrDataForValue,
           return_value_policy::reference)
      .def("set_shape_or_data_for_var",
           &pir::ShapeConstraintIRAnalysis::SetShapeOrDataForValue)
      .def("register_symbol_cstr_from_shape_analysis",
           &pir::ShapeConstraintIRAnalysis::
               RegisterSymbolConstraintFromShapeAnalysis);
}

void BindPir(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindProgram(&ir_module);
  BindBlock(&ir_module);
  BindValue(&ir_module);
  BindIrMapping(&ir_module);
  BindCloneOptions(&ir_module);
  BindOperation(&ir_module);
  BindOpOperand(&ir_module);
  BindType(&ir_module);
  BindVectorType(&ir_module);
  BindAttribute(&ir_module);
  BindInsertionPoint(&ir_module);
  BindUtils(&ir_module);
  BindIrPass(&ir_module);
  BindPassManager(&ir_module);
  BindControlFlowApi(&ir_module);
  BindShapeOrDataDimExprs(&ir_module);
  BindShapeConstraintIRAnalysis(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindOpsAPI(&ops_modules);
  BindIrParser(&ir_module);
}

}  // namespace pybind
}  // namespace paddle
