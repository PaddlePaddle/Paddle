// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pybind/control_flow_api.h"
#include "paddle/fluid/pybind/pybind_variant_caster.h"

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_0d_to_1d_pass.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_add_act_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_add_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/conv2d_bn_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/embedding_eltwise_layernorm_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fc_elementwise_layernorm_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fc_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fused_dot_product_attention_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fused_dropout_add_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fused_gemm_epilogue_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fused_linear_param_grad_add_pass.h"
#include "paddle/fluid/pir/transforms/fusion/fused_weight_only_linear_pass.h"
#include "paddle/fluid/pir/transforms/fusion/matmul_scale_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/multihead_matmul_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/silu_fuse_pass.h"
#include "paddle/fluid/pir/transforms/fusion/transpose_flatten_concat_fuse_pass.h"
#include "paddle/fluid/pir/transforms/identity_op_clean_pass.h"
#include "paddle/fluid/pir/transforms/inplace_pass.h"
#include "paddle/fluid/pir/transforms/map_op_to_another_pass.h"
#include "paddle/fluid/pir/transforms/replace_fetch_with_shadow_output_pass.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/attribute.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/parser/ir_parser.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/shape/ir/shape_attribute.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/common/flags.h"
#include "pybind11/stl.h"

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_dynamic_to_static_dim_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_static_dim_to_dynamic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/divide_group_op_to_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/move_generate_shape_ops_to_prologue_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/simplify_dim_expr_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/insert_broadcast_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/remove_unchanged_reshape_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#endif

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/transforms/onednn/batch_norm_act_fuse_pass.h"
#endif

namespace py = pybind11;
using paddle::dialect::ApiBuilder;
using paddle::dialect::DenseTensorArrayType;
using paddle::dialect::DenseTensorType;
using paddle::dialect::IfOp;
using paddle::dialect::PyLayerOp;
using paddle::dialect::SelectedRowsType;
using paddle::dialect::WhileOp;

using pir::Attribute;
using pir::Block;
using pir::BlockArgument;
using pir::BoolAttribute;
using pir::IrParser;
using pir::Operation;
using pir::OpOperand;
using pir::OpResult;
using pir::Pass;
using pir::PassManager;
using pir::Program;
using pir::Type;
using pir::Value;
using pybind11::return_value_policy;

USE_PIR_PASS(dead_code_elimination_pass);
USE_PIR_PASS(multihead_matmul_fuse_pass);
USE_PIR_PASS(transpose_flatten_concat_fuse_pass);
USE_PIR_PASS(fused_gemm_epilogue_pass);
USE_PIR_PASS(fused_dropout_add_pass);
USE_PIR_PASS(fused_weight_only_linear_pass);
USE_PIR_PASS(fused_linear_param_grad_add_pass);
USE_PIR_PASS(inplace_pass);
USE_PIR_PASS(replace_fetch_with_shadow_output_pass);
USE_PIR_PASS(identity_op_clean_pass);
USE_PIR_PASS(map_op_to_another_pass);
USE_PIR_PASS(matmul_scale_fuse_pass);
USE_PIR_PASS(fc_fuse_pass);
USE_PIR_PASS(silu_fuse_pass);
USE_PIR_PASS(fc_elementwise_layernorm_fuse_pass);
USE_PIR_PASS(conv2d_bn_fuse_pass);
USE_PIR_PASS(conv2d_add_fuse_pass);
USE_PIR_PASS(conv2d_add_act_fuse_pass);
USE_PIR_PASS(embedding_eltwise_layernorm_fuse_pass);
USE_PIR_PASS(fused_dot_product_attention_pass);

#ifdef PADDLE_WITH_DNNL
USE_PIR_PASS(batch_norm_act_fuse_pass);
#endif

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(pir_apply_shape_optimization_pass);
COMMON_DECLARE_bool(check_infer_symbolic);

namespace paddle {
namespace pybind {

PyTypeObject *g_ir_value_pytype = nullptr;

void BindOpsAPI(pybind11::module *module);

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

void BindProgram(py::module *m) {
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
        return std::make_unique<Program>(pir::IrContext::Instance());
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
      .def(
          "clone",
          [](std::shared_ptr<Program> self) {
            pir::IrMapping mapper;
            return self->Clone(mapper);
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
      .def("num_ops", [](Program &self) { return self.num_ops(); });
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
      .def(
          "remove_op",
          [](Block &self, Operation *op) {
            auto op_iter = std::find(self.begin(), self.end(), *op);
            self.erase(op_iter);
          },
          R"DOC(
        Remove the specific position operator.

        Args:
            index(int): the position that the operator to insert.

        Returns:
            None

      )DOC")
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
      .def("refresh_stopgradient", [](Block &self) {
        for (auto &op : self) {
          RefreshOpStopgradients(&op);
        }
      });
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
      .def("operand", &Operation::operand)
      .def("result",
           [](Operation &self, uint32_t index) {
             return static_cast<pir::Value>(self.result(index));
           })
      .def("operand_source", &Operation::operand_source)
      .def("operands", &Operation::operands)
      .def("results",
           [](Operation &self) -> py::list {
             py::list op_list;
             for (uint32_t i = 0; i < self.num_results(); i++) {
               op_list.append(static_cast<pir::Value>(self.result(i)));
             }
             return op_list;
           })
      .def(
          "blocks",
          [](Operation &self) { return &self.blocks(); },
          return_value_policy::reference)
      .def("has_attr", &Operation::HasAttribute)
      .def("attrs",
           [](Operation &self) -> py::dict {
             py::dict attrs_dict;
             for (auto &pair : self.attributes()) {
               // SymbolAttribute is only used in PIR, no need to pass to Python
               if (pair.second.isa<pir::shape::SymbolAttribute>()) continue;
               attrs_dict[pair.first.c_str()] =
                   paddle::dialect::GetAttributeData(pair.second);
             }
             return attrs_dict;
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
               PADDLE_THROW(phi::errors::InvalidArgument(
                   "Can't cast non-pylayer_op type Operation to PyLayerOp."));
             }
             return pylayer_op;
           })
      .def("as_while_op",
           [](Operation &self) { return PyWhileOp(self.dyn_cast<WhileOp>()); })
      .def("__repr__", [](Operation &self) {
        std::ostringstream print_stream;
        print_stream << "Operation(";
        self.Print(print_stream);
        print_stream << ")";
        return print_stream.str();
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

phi::DataType GetValueDtype(Value value) {
  if (!value.type()) {
    PADDLE_THROW(phi::errors::InvalidArgument("The type of value is nullptr."));
  }
  if (value.type().isa<DenseTensorType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<DenseTensorType>().dtype());
  } else if (value.type().isa<SelectedRowsType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<SelectedRowsType>().dtype());
  } else if (value.type().isa<DenseTensorArrayType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<DenseTensorArrayType>().dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get phi::DataType from DenseTensorType and "
        "SelectedRowsType."));
  }
}

const phi::DDim &GetValueDims(Value value) {
  if (!value.type()) {
    PADDLE_THROW(phi::errors::InvalidArgument("The type of value is nullptr."));
  }
  if (value.type().isa<DenseTensorType>()) {
    return value.type().dyn_cast<DenseTensorType>().dims();
  } else if (value.type().isa<SelectedRowsType>()) {
    return value.type().dyn_cast<SelectedRowsType>().dims();
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get shape for dense "
        "tensor."));
  }
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
  auto out = CastPyArg2Value(res, "", 0);
  Py_DECREF(py_func);
  Py_DECREF(res);
  return out;
}

void BindValue(py::module *m) {
  py::class_<Value> value(*m,
                          "Value",
                          R"DOC(
    Value class represents the SSA value in the IR system. It is a directed edge
    and a base class.

    Notes:
        The constructor of Value should not be invoked directly. Value can be automatically constructed
        when build network.

  )DOC",
                          pybind11::dynamic_attr());
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
      .def_property_readonly(
          "name",
          [](Value self) {
            if (auto param_op = self.defining_op<::pir::ParameterOp>()) {
              return param_op.param_name();
            } else if (auto data_op =
                           self.defining_op<paddle::dialect::DataOp>()) {
              return data_op.attribute<pir::StrAttribute>("name").AsString();
            } else {
              PADDLE_THROW(phi::errors::InvalidArgument(
                  "Currently, we can only get name of Value that "
                  "is persistable"));
            }
          })
      .def_property(
          "shape",
          [](Value self) { return phi::vectorize(GetValueDims(self)); },
          [](Value self, const std::vector<int> &shape) {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "can't set shape when building static graph"));
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
      .def_property(
          "stop_gradient",
          [](Value self) {
            auto stop_gradient =
                self.attribute<BoolAttribute>(kAttrStopGradients);
            return !stop_gradient || stop_gradient.data();
          },
          [](Value self, bool stop_gradient) {
            self.set_attribute(
                kAttrStopGradients,
                BoolAttribute::get(pir::IrContext::Instance(), stop_gradient));
          })
      .def_property(
          "persistable",
          [](Value self) {
            auto persistable =
                self.attribute<BoolAttribute>(kAttrIsPersistable);
            return !persistable || persistable.data();
          },
          [](Value self, bool persistable) {
            self.set_attribute(
                kAttrIsPersistable,
                BoolAttribute::get(pir::IrContext::Instance(), persistable));
          })
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
      .def("is_dense_tensor_type",
           [](Value self) { return self.type().isa<DenseTensorType>(); })
      .def("is_selected_row_type",
           [](Value self) { return self.type().isa<SelectedRowsType>(); })
      .def("is_dense_tensor_array_type",
           [](Value self) { return self.type().isa<DenseTensorArrayType>(); })
      .def("replace_all_uses_with",
           [](Value self, Value value) { self.ReplaceAllUsesWith(value); })
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
                     ->Build<paddle::dialect::ShareDataOp>(self);
             auto out = share_data_op.out();
             out.set_attribute(
                 kAttrStopGradients,
                 BoolAttribute::get(pir::IrContext::Instance(), true));
             return out;
           })
      .def("__repr__", &Value2String);
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
      .def(
          "set_source",
          [](OpOperand &self, const Value &result) { self.set_source(result); })
      .def("owner", &OpOperand::owner, return_value_policy::reference)
      .def("index", &OpOperand::index);
}

bool GetValueBoolAttr(Value value, const std::string &attr_name) {
  auto bool_attr = value.attribute<BoolAttribute>(attr_name);
  return !bool_attr || bool_attr.data();
}

void BindType(py::module *m) {
  py::class_<Type> ir_type(*m, "Type");
  ir_type.def("__eq__", [](Type &self, Type &other) { return self == other; })
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

void BindAttribute(py::module *m) {
  py::class_<Attribute> ir_attr(*m, "Attribute", py::module_local());
  ir_attr.def("__str__", [](Attribute &self) {
    std::ostringstream print_stream;
    print_stream << self;
    return print_stream.str();
  });
}

struct PyInsertionPoint {
  pir::InsertionPoint value;
};
void BindInsertionPoint(pybind11::module *m) {
  py::class_<PyInsertionPoint> ir_insertion_point(*m, "InsertionPoint", R"DOC(
    InsertionPoint class represents the insertion point in the Builder.)DOC");
}

std::list<Operation *>::const_iterator list_offset(const Block *block,
                                                   int start_idx) {
  auto it = block->begin();
  while (it != block->end() && start_idx--) ++it;
  return it;
}

template <class F>
void range_block_do(const Block *block, std::vector<int> range, F fn) {
  for (auto it = list_offset(block, range[0]);
       it != list_offset(block, range[1]);
       ++it) {
    fn(*it);
  }
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
        for (auto &t : op->operands()) {
          backward_inputs.insert(t.source());
        }
      });

  range_block_do(
      program.block(),
      forward_range,
      [&middle_values, &backward_inputs, &x_or_param](Operation *op) {
        for (auto &t : op->results()) {
          auto v = Value(t.Value::impl());
          if (backward_inputs.count(v) && !x_or_param.count(v))
            middle_values.push_back(v);
        }
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

pir::Value FakeValue() {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return pir::Value(nullptr);
}

bool IsFakeValue(const pir::Value &value) {
  // create a fake value to simplify `ForwardBackwardSplit`.
  return value.impl() == nullptr || !value.type();
}

static auto GetNoNeedBufferValue(const ::pir::Block *whole_block,
                                 std::vector<int> range) {
  // filter no need buffer values.
  std::unordered_set<::pir::Value> need_buffer_values;
  std::unordered_set<::pir::Value> no_need_buffer_values;
  range_block_do(
      whole_block, range, [&need_buffer_values](::pir::Operation *op) {
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
            if (!op_input_info.no_need_buffer) {
              need_buffer_values.insert(op->operand_source(counter));
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

void AppendShadowOutput(Program *forward_program,
                        const pir::Value &value,
                        const std::string &name,
                        size_t start_point) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
  pir::AttributeMap attribute_map = {
      {"output_name", pir::StrAttribute::get(ctx, name)},
  };
  pir::Operation *operation =
      pir::Operation::Create({value}, attribute_map, {}, op_info);
  auto position = forward_program->block()->begin();
  std::advance(position, start_point);
  if (position == forward_program->block()->end()) {
    forward_program->block()->push_back(operation);
  } else {
    forward_program->block()->insert(position, operation);
  }
}

int AppendShadowOutputs(Program *forward_program,
                        const std::vector<pir::Value> &outputs,
                        int start_point,
                        std::string name_prefix) {
  int counter = 0;
  std::unordered_set<pir::Value> added_value;

  for (const auto &value : outputs) {
    if (!added_value.count(value) || IsFakeValue(value)) {
      std::string shadow_output_name = name_prefix + std::to_string(counter);
      AppendShadowOutput(
          forward_program, value, shadow_output_name, start_point + counter);
      counter += 1;
      added_value.insert(value);
    }
  }
  // return the inserted op.
  return counter;
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
      });
  auto &forward_value_map = forward_mapper.GetMutableMap<pir::Value>();

  // backward program construc.
  // Step1. insert data op for inputs_values and middle_values
  pir::IrMapping backward_mapper;
  auto &backward_value_map = backward_mapper.GetMutableMap<pir::Value>();
  int counter = 0;
  auto create_kwarg_fn = [&backward_block,
                          &backward_inputs,
                          &backward_value_map,
                          &counter](const pir::Value &v) {
    if (v && backward_inputs.count(v)) {
      backward_value_map[v] = backward_block.AddKwarg(
          "input_" + std::to_string(counter++), v.type());
    }
  };

  auto create_output_fn_forward = [&ctx,
                                   &forward_value_map,
                                   &counter,
                                   &forward_program](const pir::Value &v) {
    if (v.impl() == nullptr) {
      return;
    }
    // NOTE(Aurelius84): we should skip insert ShadowOutputOp repeatly by
    // calling SplitForwardBackward multi-times.
    std::string shadow_output_name =
        std::string("output_") + std::to_string(counter);
    std::unordered_set<pir::Value> inserted_value;
    for (auto it = forward_program->block()->rbegin();
         it != forward_program->block()->rend();
         ++it) {
      if (it->isa<pir::ShadowOutputOp>()) {
        auto out_name =
            it->attribute<pir::StrAttribute>("output_name").AsString();
        if (out_name == shadow_output_name) {
          VLOG(4) << out_name
                  << " has been inserted ShadowOutputOp, skip it now.";
          return;
        }

        inserted_value.insert(it->operand_source(0));
      }
    }

    if (inserted_value.count(forward_value_map[v])) {
      return;
    }
    auto op_info = ctx->GetRegisteredOpInfo(pir::ShadowOutputOp::name());
    pir::AttributeMap attribute_map = {
        {"output_name", pir::StrAttribute::get(ctx, shadow_output_name)},
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
         pir::StrAttribute::get(
             ctx, std::string("output_") + std::to_string(counter))},
    };
    pir::Operation *operation = pir::Operation::Create(
        {backward_value_map.at(v)}, attribute_map, {}, op_info);
    backward_program->block()->push_back(operation);
    counter += 1;
  };

  // counter = 0;
  if (has_backward) {
    VLOG(4) << "start create backward inputs, inserting pd.data ops.";
    VLOG(4) << "Create pd.data for backward program: fo, start with input_"
            << counter;
    std::for_each(
        forward_outputs.begin(), forward_outputs.end(), create_kwarg_fn);
    VLOG(4) << "Create pd.data for backward program: fx, start with input_"
            << counter;
    std::for_each(
        forward_inputs.begin(), forward_inputs.end(), create_kwarg_fn);
    VLOG(4) << "Create pd.data for backward program: fp, start with input_"
            << counter;
    std::for_each(
        forward_params.begin(), forward_params.end(), create_kwarg_fn);
    VLOG(4) << "Create pd.data for backward program: fm, start with input_"
            << counter;
    std::for_each(middle_values.begin(), middle_values.end(), create_kwarg_fn);
    VLOG(4) << "Create pd.data for backward program: fo_g, start with input_"
            << counter;
    std::for_each(forward_outputs_grads.begin(),
                  forward_outputs_grads.end(),
                  create_kwarg_fn);
    VLOG(4) << "Create pd.data for backward program end. input_" << counter;
  }

  // counter = 0;
  VLOG(4) << "start create forward outputs, inserting set_parameter ops.";
  std::for_each(
      middle_values.begin(), middle_values.end(), create_output_fn_forward);
  std::for_each(
      forward_outputs.begin(), forward_outputs.end(), create_output_fn_forward);

  // Step2. copy backward ops .
  VLOG(4) << "start copy backward ops";
  range_block_do(
      program.block(),
      backward_range,
      [&backward_mapper, &backward_program, &clone_options](Operation *op) {
        auto *cloned_op = op->Clone(backward_mapper, clone_options);
        backward_program->block()->push_back(cloned_op);
      });
  // counter = 0;
  VLOG(4) << "start create backward outputs, inserting set_parameter ops.";
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

void ResetShadowOutputName(pir::Operation *op, const std::string &name) {
  pir::IrContext *ctx = pir::IrContext::Instance();
  if (op->isa<pir::ShadowOutputOp>()) {
    op->set_attribute("output_name", pir::StrAttribute::get(ctx, name));
  }
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
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
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
  }
  return inplace_info;
}

void BindUtils(pybind11::module *m) {
  m->def("clone_program", CloneProgram);
  m->def("get_op_inplace_info", GetOpInplaceInfo);
  m->def("reset_shadow_output_name", ResetShadowOutputName);
  m->def("split_program", SplitForwardBackward);
  m->def("append_shadow_outputs", AppendShadowOutputs);
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
  m->def("create_selected_rows_type_by_dense_tensor",
         CreateSelectedRowsTypeByDenseTensor);
  m->def(
      "translate_to_pir",
      [](const ::paddle::framework::ProgramDesc &legacy_program) {
        std::shared_ptr<Program> ret =
            std::move(paddle::TranslateLegacyProgramToProgram(legacy_program));
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
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> pd_op.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (pd_op.tensor<4x4xf32>, pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (pd_op.tensor<4x4xf32>, pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
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
                 (%0) = "pd_op.data" () {dtype:(pd_op.DataType)float32,is_persistable:[false],name:"x",place:(pd_op.Place)Place(undefined:0),shape:(pd_op.IntArray)[4,4],stop_gradient:[false]} : () -> pd_op.tensor<4x4xf32>
                 (%1) = "pd_op.matmul" (%0, %0) {is_persistable:[false],stop_gradient:[false],transpose_x:false,transpose_y:false} : (pd_op.tensor<4x4xf32>, pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
                 (%2) = "pd_op.add" (%1, %1) {is_persistable:[false],stop_gradient:[false]} : (pd_op.tensor<4x4xf32>, pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
                 (%3) = "pd_op.tanh" (%2) {is_persistable:[false],stop_gradient:[false]} : (pd_op.tensor<4x4xf32>) -> pd_op.tensor<4x4xf32>
                }

                >>> print(mappings)
                {'matmul_v2_0.tmp_0': [Value(define_op_name=pd_op.matmul, index=0, dtype=pd_op.tensor<4x4xf32>)], 'x': [Value(define_op_name=pd_op.data, index=0, dtype=pd_op.tensor<4x4xf32>)], 'tanh_0.tmp_0': [Value(define_op_name=pd_op.tanh, index=0, dtype=pd_op.tensor<4x4xf32>)], 'elementwise_add_0': [Value(define_op_name=pd_op.add, index=0, dtype=pd_op.tensor<4x4xf32>)]}
    )DOC");

  m->def("clear_pir_compiler_manager", []() {
#ifdef PADDLE_WITH_CINN
    pybind11::gil_scoped_release release;
    VLOG(4) << "clear PirCompilerManager and free PirCompiler resources.";
    cinn::hlir::framework::PirCompilerManager::Instance().clear();
#endif
  });
}

static bool HasDynamicShape(const Program &program) {
  for (const auto &op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          return true;
        }
      }
    }
  }
  return false;
}

void AddCinnPass(std::shared_ptr<PassManager> &pass_manager,  // NOLINT
                 Program &program) {                          // NOLINT
#ifdef PADDLE_WITH_CINN
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();

  bool has_dynamic_shape = HasDynamicShape(program);

  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }

  pass_manager->AddPass(cinn::dialect::ir::CreateConvert0DTo1DPass());
  if (!has_dynamic_shape && FLAGS_check_infer_symbolic) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateCheckInferSymbolicPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateRemoveUnchangedReshapePass());
  pass_manager->AddPass(
      cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  if (has_dynamic_shape) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateSimplifyDimExprPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateInsertBroadcastPass());
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(
        cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }

  pass_manager->AddPass(pir::CreateBuildCinnPass());

  pass_manager->AddPass(
      cinn::dialect::ir::CreateMoveGenerateShapeOpsToProloguePass());
  pass_manager->AddPass(cinn::dialect::ir::CreateDivideGroupOpToFusionOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateDynamicReshapeOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateReplaceDynamicExpandOpPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  bool force_static_shape = false;
  if (auto pass = cinn::dialect::ir::CreateConvertDynamicToStaticDimPass()) {
    pass_manager->AddPass(std::move(pass.value()));
    force_static_shape = true;
  }
  if (auto pass = cinn::dialect::ir::CreateConvertStaticDimToDynamicPass()) {
    pass_manager->AddPass(std::move(pass.value()));
  }

  if (has_dynamic_shape && !force_static_shape) {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateLowerCinnDyShapeFusionOpPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pass_manager->AddPass(
      cinn::dialect::ir::CreateSplitGenerateShapeIntoShapeOpsPass());
#else
  PADDLE_THROW(platform::errors::Unimplemented(
      "Currently we only support CINN Pass for Pir under @to_static, please "
      "compile PaddlePaddle with CINN"));
#endif
}

void InferSymbolicShapePass(
    std::shared_ptr<PassManager> &pass_manager,  // NOLINT
    Program &program) {                          // NOLINT
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  if (HasDynamicShape(program) && FLAGS_pir_apply_shape_optimization_pass) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  }
}

void BindIrPass(pybind11::module *m) {
  m->def("add_cinn_pass", AddCinnPass);
  m->def("infer_symbolic_shape_pass", InferSymbolicShapePass);

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
           [](PassManager &self, const std::string &pass_name) {
             self.AddPass(
                 std::move(pir::PassRegistry::Instance().Get(pass_name)));
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
           [](PassManager &self) { self.EnableIRPrinting(); });
}

void BindPir(pybind11::module *module) {
  auto ir_module = module->def_submodule("pir");
  BindProgram(&ir_module);
  BindBlock(&ir_module);
  BindOperation(&ir_module);
  BindValue(&ir_module);
  BindOpOperand(&ir_module);
  BindType(&ir_module);
  BindAttribute(&ir_module);
  BindInsertionPoint(&ir_module);
  BindUtils(&ir_module);
  BindIrPass(&ir_module);
  BindPassManager(&ir_module);
  BindControlFlowApi(&ir_module);
  auto ops_modules = ir_module.def_submodule("ops");
  BindOpsAPI(&ops_modules);
  BindIrParser(&ir_module);
}

}  // namespace pybind
}  // namespace paddle
