/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/jit.h"
#include "glog/logging.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/jit/function.h"
#include "paddle/fluid/jit/function_schema.h"
#include "paddle/fluid/jit/layer.h"
#include "paddle/fluid/jit/serializer.h"
#include "paddle/fluid/pybind/sot/eval_frame.h"
#include "paddle/fluid/pybind/sot/eval_frame_tools.h"
#include "paddle/fluid/pybind/sot/frame_proxy.h"
#include "paddle/fluid/pybind/sot/guards.h"
#include "paddle/fluid/pybind/sot/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/utils/pybind.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;

namespace paddle::pybind {

PyTypeObject *g_jit_function_pytype = nullptr;
using Variable = paddle::framework::Variable;

void BindJit(pybind11::module *m) {
  py::class_<jit::Layer>(*m, "Layer", R"DOC(Layer Class.)DOC")
      .def("function_names", &jit::Layer::FunctionNames)
      .def("function", &jit::Layer::Function)
      .def("function_info", &jit::Layer::FunctionInfo);

  py::class_<jit::Function, std::shared_ptr<jit::Function>> function(
      *m, "Function", R"DOC(Function Class.)DOC");
  g_jit_function_pytype = reinterpret_cast<PyTypeObject *>(function.ptr());

  py::class_<jit::BaseFunctionInfo, std::shared_ptr<jit::BaseFunctionInfo>>(
      *m, "FunctionInfo", R"DOC(BaseFunctionInfo Class.)DOC")
      .def("name", &jit::BaseFunctionInfo::FunctionName)
      .def("input_names", &jit::BaseFunctionInfo::InputArgNames)
      .def("output_names", &jit::BaseFunctionInfo::OutputArgNames);

  m->def("Load", [](const std::string &path, const phi::CPUPlace &cpu_place) {
    return paddle::jit::Load(path, cpu_place);
  });

  m->def("Load", [](const std::string &path, const phi::GPUPlace &cuda_place) {
    return paddle::jit::Load(path, cuda_place);
  });
}

void BindGuard(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  py::class_<GuardBase, std::shared_ptr<GuardBase>>(
      *m, "GuardBase", R"DOC(GuardBase Class.)DOC")
      .def("check", &GuardBase::check_pybind);
  py::class_<LambdaGuard, GuardBase, std::shared_ptr<LambdaGuard>>(
      *m, "LambdaGuard", R"DOC(LambdaGuard Class.)DOC")
      .def(py::init<const py::function &>(), py::arg("guard_check_fn"));
  py::class_<GuardGroup, GuardBase, std::shared_ptr<GuardGroup>>(
      *m, "GuardGroup", R"DOC(GuardGroup Class.)DOC")
      .def(py::init<const std::vector<std::shared_ptr<GuardBase>> &>(),
           py::arg("guards"));
  py::class_<TypeMatchGuard, GuardBase, std::shared_ptr<TypeMatchGuard>>(
      *m, "TypeMatchGuard", R"DOC(TypeMatchGuard Class.)DOC")
      .def(py::init<const py::type &>(), py::arg("py_type"));
  py::class_<IdMatchGuard, GuardBase, std::shared_ptr<IdMatchGuard>>(
      *m, "IdMatchGuard", R"DOC(IdMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("py_obj"));
  py::class_<LengthMatchGuard, GuardBase, std::shared_ptr<LengthMatchGuard>>(
      *m, "LengthMatchGuard", R"DOC(LengthMatchGuard Class.)DOC")
      .def(py::init<const Py_ssize_t &>(), py::arg("length"));
  py::class_<ValueMatchGuard, GuardBase, std::shared_ptr<ValueMatchGuard>>(
      *m, "ValueMatchGuard", R"DOC(ValueMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("py_value"));
  py::class_<DtypeMatchGuard, GuardBase, std::shared_ptr<DtypeMatchGuard>>(
      *m, "DtypeMatchGuard", R"DOC(DtypeMatchGuard Class.)DOC")
      .def(py::init<const paddle::framework::proto::VarType &>(),
           py::arg("dtype"))
      .def(py::init<const phi::DataType &>(), py::arg("dtype"));
  py::class_<AttributeMatchGuard,
             GuardBase,
             std::shared_ptr<AttributeMatchGuard>>(
      *m, "AttributeMatchGuard", R"DOC(AttributeMatchGuard Class.)DOC")
      .def(py::init<const py::object &, const std::string &>(),
           py::arg("obj"),
           py::arg("attr_name"));
  py::class_<ShapeMatchGuard, GuardBase, std::shared_ptr<ShapeMatchGuard>>(
      *m, "ShapeMatchGuard", R"DOC(ShapeMatchGuard Class.)DOC")
      .def(py::init<const std::vector<py::object> &>(), py::arg("shape"));
  py::class_<LayerMatchGuard, GuardBase, std::shared_ptr<LayerMatchGuard>>(
      *m, "LayerMatchGuard", R"DOC(LayerMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("layer_obj"));
  py::class_<InstanceCheckGuard,
             GuardBase,
             std::shared_ptr<InstanceCheckGuard>>(
      *m, "InstanceCheckGuard", R"DOC(InstanceCheckGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("isinstance_obj"));
  py::class_<NumPyDtypeMatchGuard,
             GuardBase,
             std::shared_ptr<NumPyDtypeMatchGuard>>(
      *m, "NumPyDtypeMatchGuard", R"DOC(NumPyDtypeMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("dtype"));
  py::class_<NumPyArrayValueMatchGuard,
             GuardBase,
             std::shared_ptr<NumPyArrayValueMatchGuard>>(
      *m,
      "NumPyArrayValueMatchGuard",
      R"DOC(NumPyArrayValueMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("array"));
  py::class_<WeakRefMatchGuard, GuardBase, std::shared_ptr<WeakRefMatchGuard>>(
      *m, "WeakRefMatchGuard", R"DOC(WeakRefMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("func"));
  py::class_<DummyGuard, GuardBase, std::shared_ptr<DummyGuard>>(
      *m, "DummyGuard", R"DOC(DummyGuard Class.)DOC")
      .def(py::init<>());
  py::class_<TensorDistMetaMatchGuard,
             GuardBase,
             std::shared_ptr<TensorDistMetaMatchGuard>>(
      *m,
      "TensorDistMetaMatchGuard",
      R"DOC(TensorDistMetaMatchGuard Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("dist_info"));

  m->def(
      "merge_guard",
      [](const std::vector<std::shared_ptr<GuardBase>> &py_guards) {
        return GuardGroup(py_guards);
      },
      py::arg("py_guards"));
#endif
}

void BindGuardTree(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  py::class_<GuardTree, std::shared_ptr<GuardTree>>(
      *m, "GuardTree", R"DOC(GuardTree Class.)DOC")
      .def(py::init<const std::vector<
               std::vector<std::shared_ptr<GuardNodeBase>>> &>(),
           py::arg("guard_chain_list"))
      .def_property_readonly(
          "guard_nodes", [](GuardTree &self) { return self.get_guard_nodes(); })
      .def(
          "lookup",
          [](GuardTree &self, py::object frame) {
            return self.lookup(reinterpret_cast<FrameProxy *>(frame.ptr()));
          },
          py::arg("frame"))
      .def(
          "add_guard_chain",
          [](GuardTree &self,
             const std::vector<std::shared_ptr<GuardNodeBase>> &guard_chain) {
            self.add_guard_chain(guard_chain);
          },
          py::arg("guard_chain"))
      .def("stringify", &GuardTree::stringify);

  py::class_<GuardNodeBase, std::shared_ptr<GuardNodeBase>>(
      *m, "GuardNodeBase", R"DOC(GuardNodeBase Class.)DOC")
      .def_property(
          "return_cache_index",
          [](GuardNodeBase &self) { return self.return_cache_index; },
          [](GuardNodeBase &self, int index) {
            self.return_cache_index = index;
          })
      .def_property(
          "next_guard_nodes",
          [](GuardNodeBase &self) { return self.next_guard_nodes; },
          [](GuardNodeBase &self,
             const std::vector<std::shared_ptr<GuardNodeBase>>
                 &next_guard_nodes) {
            self.next_guard_nodes = next_guard_nodes;
          })
      .def(
          "lookup",
          [](GuardNodeBase &self, py::object frame) {
            return self.lookup(reinterpret_cast<FrameProxy *>(frame.ptr()));
          },
          py::arg("frame"))
      .def("stringify", &GuardNodeBase::stringify, py::arg("indent") = 0);

  py::class_<ExprNodeBase, std::shared_ptr<ExprNodeBase>>(
      *m, "ExprNodeBase", R"DOC(ExprNodeBase Class.)DOC")
      .def(
          "eval",
          [](ExprNodeBase &self, py::object frame) {
            return self.eval(reinterpret_cast<FrameProxy *>(frame.ptr()));
          },
          py::arg("frame"))
      .def("stringify", &ExprNodeBase::stringify, py::arg("indent") = 0);

  py::class_<GuardNode, GuardNodeBase, std::shared_ptr<GuardNode>>(
      *m, "GuardNode", R"DOC(GuardNode Class.)DOC")
      .def(py::init<const std::shared_ptr<GuardBase> &,
                    const std::vector<std::shared_ptr<ExprNodeBase>> &,
                    const std::vector<std::shared_ptr<GuardNodeBase>> &,
                    const std::optional<int> &>(),
           py::arg("guard"),
           py::arg("exprs"),
           py::arg("next_guard_nodes") = py::list(),
           py::arg("return_cache_index") = py::none());

  py::class_<ExprGuardNode, GuardNodeBase, std::shared_ptr<ExprGuardNode>>(
      *m, "ExprGuardNode", R"DOC(ExprGuardNode Class.)DOC")
      .def(py::init<const std::shared_ptr<ExprNodeBase> &,
                    const std::vector<std::shared_ptr<GuardNodeBase>> &,
                    const std::optional<int> &>(),
           py::arg("expr"),
           py::arg("next_guard_nodes") = py::list(),
           py::arg("return_cache_index") = py::none());

  py::class_<ConstantExprNode, ExprNodeBase, std::shared_ptr<ConstantExprNode>>(
      *m, "ConstantExprNode", R"DOC(ConstantExprNode Class.)DOC")
      .def(py::init<const py::object &>(), py::arg("value_ptr"));

  py::class_<ExternVarExprNode,
             ExprNodeBase,
             std::shared_ptr<ExternVarExprNode>>(
      *m, "ExternVarExprNode", R"DOC(ExternVarExprNode Class.)DOC")
      .def(py::init<const std::string &, const py::object &>(),
           py::arg("var_name"),
           py::arg("value_ptr"));

  py::class_<LocalVarExprNode, ExprNodeBase, std::shared_ptr<LocalVarExprNode>>(
      *m, "LocalVarExprNode", R"DOC(LocalVarExprNode Class.)DOC")
      .def(py::init<const std::string &>(), py::arg("var_name"));

  py::class_<GlobalVarExprNode,
             ExprNodeBase,
             std::shared_ptr<GlobalVarExprNode>>(
      *m, "GlobalVarExprNode", R"DOC(GlobalVarExprNode Class.)DOC")
      .def(py::init<const std::string &>(), py::arg("var_name"));

  py::class_<AttributeExprNode,
             ExprNodeBase,
             std::shared_ptr<AttributeExprNode>>(
      *m, "AttributeExprNode", R"DOC(AttributeExprNode Class.)DOC")
      .def(py::init<std::shared_ptr<ExprNodeBase>, const std::string &>(),
           py::arg("var_expr"),
           py::arg("attr_name"));

  py::class_<ItemExprNode, ExprNodeBase, std::shared_ptr<ItemExprNode>>(
      *m, "ItemExprNode", R"DOC(ItemExprNode Class.)DOC")
      .def(py::init<std::shared_ptr<ExprNodeBase>,
                    std::shared_ptr<ExprNodeBase>>(),
           py::arg("var_expr"),
           py::arg("key_expr"));

  py::class_<BinaryExprNode, ExprNodeBase, std::shared_ptr<BinaryExprNode>>(
      *m, "BinaryExprNode", R"DOC(BinaryExprNode Class.)DOC")
      .def(py::init<std::shared_ptr<ExprNodeBase>,
                    std::shared_ptr<ExprNodeBase>,
                    const std::string &>(),
           py::arg("lhs"),
           py::arg("rhs"),
           py::arg("op"));

  py::class_<UnaryExprNode, ExprNodeBase, std::shared_ptr<UnaryExprNode>>(
      *m, "UnaryExprNode", R"DOC(UnaryExprNode Class.)DOC")
      .def(py::init<std::shared_ptr<ExprNodeBase>, const std::string &>(),
           py::arg("expr"),
           py::arg("op"));
#endif
}

void BindSot(pybind11::module *m) {
#if SOT_IS_SUPPORTED
  PyInit__eval_frame();
#if PY_3_11_PLUS
  PyInit__frame_proxy();
#endif
  m->def(
      "set_eval_frame",
      [](const py::object &py_func) {
        VLOG(5) << "start call set_eval_frame_py.";
        auto ret = set_eval_frame_py(py_func.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("callback"));

  m->def("has_custom_getattro", [](py::object obj) {
    PyObject *py_obj = obj.ptr();

    if (!PyType_Check(py_obj)) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The input object should be a type object, but got %s.",
          py::str(py_obj).cast<std::string>()));
    }
    PyTypeObject *type = reinterpret_cast<PyTypeObject *>(py_obj);

    return type->tp_getattro != PyObject_GenericGetAttr;
  });

  m->def(
      "sot_setup_codes_with_graph",
      [](const py::object &py_codes) {
        auto ret = setup_codes_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "sot_set_with_graph",
      [](const py::object &py_codes) {
        auto ret = set_with_graph(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_no_skip_codes",
      [](const py::object &py_codes) {
        auto ret = no_skip_codes(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));

  m->def(
      "eval_frame_skip_file_prefix",
      [](const py::object &py_codes) {
        auto ret = skip_file_prefix(py_codes.ptr());
        auto obj = py::reinterpret_borrow<py::object>(ret);
        return obj;
      },
      py::arg("py_codes"));
  BindGuard(m);
  BindGuardTree(m);
#endif
}

}  // namespace paddle::pybind
