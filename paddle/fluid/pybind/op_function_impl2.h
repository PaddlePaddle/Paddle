// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/tracer.h"
namespace py = pybind11;
namespace paddle {
namespace pybind {

void imperative_concat(
    const std::vector<std::shared_ptr<imperative::VarBase>>& x, py::args args) {
  Py_ssize_t args_size = args.size();
  std::cout << args_size << std::endl;
  // auto tracer = imperative::GetCurrentTracer();
  // imperative::NameVarBaseMap outs_ = {
  //     {"Out",
  //      {std::shared_ptr<imperative::VarBase>(
  //          new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  // imperative::NameVarBaseMap ins_ = {{"X", x}};
  // framework::AttributeMap attrs_ = {{"axis", axis}};

  // tracer->TraceOp("concat", std::move(ins_), std::move(outs_),
  //                 std::move(attrs_));
  // return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_elementwise_mul(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& y,
    const framework::Attribute& axis, const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Y", {y}}};
  framework::AttributeMap attrs_ = {{"axis", axis}, {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("elementwise_mul", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_elementwise_add(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& y,
    const framework::Attribute& axis, const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Y", {y}}};
  framework::AttributeMap attrs_ = {{"axis", axis}, {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("elementwise_add", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_elementwise_max(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& y,
    const framework::Attribute& axis, const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Y", {y}}};
  framework::AttributeMap attrs_ = {{"axis", axis}, {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("elementwise_max", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_elementwise_div(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& y,
    const framework::Attribute& axis, const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Y", {y}}};
  framework::AttributeMap attrs_ = {{"axis", axis}, {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("elementwise_div", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_sigmoid(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("sigmoid", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_tanh(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("tanh", std::move(ins_), std::move(outs_), std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_square(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("square", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_sqrt(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("sqrt", std::move(ins_), std::move(outs_), std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_relu(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("relu", std::move(ins_), std::move(outs_), std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_mean(
    const std::shared_ptr<imperative::VarBase>& x) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("mean", std::move(ins_), std::move(outs_), std::move(attrs_));
  return outs_["Out"][0];
}

inline std::vector<std::shared_ptr<imperative::VarBase>> imperative_split(
    const std::shared_ptr<imperative::VarBase>& x,
    // const std::shared_ptr<imperative::VarBase>& axis_tensor,
    const framework::Attribute& axis, const framework::Attribute& attr_num,
    // const framework::Attribute& sections,
    const unsigned& num) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_;
  for (size_t i = 0; i < num; i++) {
    auto var_base_name = tracer->GenerateUniqueName();
    outs_["Out"].emplace_back(new imperative::VarBase(var_base_name));
  }

  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {{"axis", axis}, {"num", attr_num}};

  tracer->TraceOp("split", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"];
}  // namespace pybind

inline std::shared_ptr<imperative::VarBase> imperative_matmul(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& y,
    const framework::Attribute& transpose_x,
    const framework::Attribute& transpose_y,
    const framework::Attribute& alpha) {
  auto tracer = imperative::GetCurrentTracer();
  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Y", {y}}};
  framework::AttributeMap attrs_ = {{"transpose_x", transpose_x},
                                    {"transpose_y", transpose_y},
                                    {"alpha", alpha}};
  tracer->TraceOp("matmul", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_slice(
    const std::shared_ptr<imperative::VarBase>& input,
    const framework::Attribute& axes, const framework::Attribute& starts,
    const framework::Attribute& ends,
    const framework::Attribute& decrease_axis) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};

  imperative::NameVarBaseMap ins_ = {{"Input", {input}}};
  framework::AttributeMap attrs_ = {{"axes", axes},
                                    {"starts", starts},
                                    {"ends", ends},
                                    {"decrease_axis", decrease_axis}};

  tracer->TraceOp("slice", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}  // namespace pybind

inline std::shared_ptr<imperative::VarBase> imperative_reduce_sum(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& dim, const framework::Attribute& keep_dim,
    const framework::Attribute& reduce_all) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {
      {"dim", dim}, {"keep_dim", keep_dim}, {"reduce_all", reduce_all}};

  tracer->TraceOp("reduce_sum", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_reduce_mean(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& dim, const framework::Attribute& keep_dim,
    const framework::Attribute& reduce_all) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {
      {"dim", dim}, {"keep_dim", keep_dim}, {"reduce_all", reduce_all}};

  tracer->TraceOp("reduce_mean", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_reshape2(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& shape) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}},
      {"XShape",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {{"shape", shape}};

  tracer->TraceOp("reshape2", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_sgd(
    const std::shared_ptr<imperative::VarBase>& param,
    const std::shared_ptr<imperative::VarBase>& grad,
    const std::shared_ptr<imperative::VarBase>& learning_rate,
    const std::shared_ptr<imperative::VarBase>& param_out) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {{"ParamOut", {param_out}}};
  imperative::NameVarBaseMap ins_ = {
      {"Param", {param}}, {"Grad", {grad}}, {"LearningRate", {learning_rate}}};
  framework::AttributeMap attrs_ = {};

  tracer->TraceOp("sgd", std::move(ins_), std::move(outs_), std::move(attrs_));
  return outs_["ParamOut"][0];
}

inline std::tuple<std::shared_ptr<imperative::VarBase>,
                  std::shared_ptr<imperative::VarBase>>
imperative_softmax_with_cross_entropy(
    const std::shared_ptr<imperative::VarBase>& logits,
    const std::shared_ptr<imperative::VarBase>& label,
    const framework::Attribute& soft_label,
    const framework::Attribute& ignore_index,
    const framework::Attribute& numeric_stable_mode,
    const framework::Attribute& axis) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Softmax",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}},
      {"Loss",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};

  imperative::NameVarBaseMap ins_ = {{"Logits", {logits}}, {"Label", {label}}};
  framework::AttributeMap attrs_ = {
      {"soft_label", soft_label},
      {"ignore_index", ignore_index},
      {"numeric_stable_mode", numeric_stable_mode},
      {"axis", axis}};

  tracer->TraceOp("softmax_with_cross_entropy", std::move(ins_),
                  std::move(outs_), std::move(attrs_));
  return std::make_tuple(outs_["Softmax"][0], outs_["Loss"][0]);
}

inline std::shared_ptr<imperative::VarBase> imperative_cross_entropy2(
    const std::shared_ptr<imperative::VarBase>& x,
    const std::shared_ptr<imperative::VarBase>& label,
    const framework::Attribute& ignore_index) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Y",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}},
      {"XShape",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}},
      {"MatchX",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};

  imperative::NameVarBaseMap ins_ = {{"X", {x}}, {"Label", {label}}};
  framework::AttributeMap attrs_ = {{"ignore_index", ignore_index}};

  tracer->TraceOp("cross_entropy2", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Y"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_softmax(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& use_cudnn,
    const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};

  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {{"use_cudnn", use_cudnn},
                                    {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("softmax", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_transpose2(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& axis) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}},
      {"XShape",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {{"axis", axis}};

  tracer->TraceOp("transpose2", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_conv2d(
    const std::shared_ptr<imperative::VarBase>& input,
    const std::shared_ptr<imperative::VarBase>& filter,
    const framework::Attribute& strides, const framework::Attribute& paddings,
    const framework::Attribute& dilations, const framework::Attribute& groups,
    const framework::Attribute& use_cudnn,
    const framework::Attribute& use_mkldnn) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Output",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"Input", {input}}, {"Filter", {filter}}};
  framework::AttributeMap attrs_ = {
      {"strides", strides},     {"paddings", paddings},
      {"dilations", dilations}, {"groups", groups},
      {"use_cudnn", use_cudnn}, {"use_mkldnn", use_mkldnn}};

  tracer->TraceOp("conv2d", ins_, outs_, attrs_);
  return outs_["Output"][0];
}

inline std::shared_ptr<imperative::VarBase> imperative_pool2d(
    const std::shared_ptr<imperative::VarBase>& x,
    const framework::Attribute& pooling_type, const framework::Attribute& ksize,
    const framework::Attribute& global_pooling,
    const framework::Attribute& strides, const framework::Attribute& paddings,
    const framework::Attribute& use_cudnn,
    const framework::Attribute& ceil_mode,
    const framework::Attribute& use_mkldnn,
    const framework::Attribute& exclusive) {
  auto tracer = imperative::GetCurrentTracer();

  imperative::NameVarBaseMap outs_ = {
      {"Out",
       {std::shared_ptr<imperative::VarBase>(
           new imperative::VarBase(tracer->GenerateUniqueName()))}}};
  imperative::NameVarBaseMap ins_ = {{"X", {x}}};
  framework::AttributeMap attrs_ = {{"pooling_type", pooling_type},
                                    {"ksize", ksize},
                                    {"global_pooling", global_pooling},
                                    {"strides", strides},
                                    {"paddings", paddings},
                                    {"use_cudnn", use_cudnn},
                                    {"use_mkldnn", use_mkldnn},
                                    {"ceil_mode", ceil_mode},
                                    {"exclusive", exclusive}};

  tracer->TraceOp("pool2d", std::move(ins_), std::move(outs_),
                  std::move(attrs_));
  return outs_["Out"][0];
}

inline void BindOpFunctions2(pybind11::module* module) {
  auto m = module->def_submodule("ops");
  m.def("concat", &imperative_concat, py::call_guard<py::gil_scoped_release>());
  m.def("elementwise_mul", &imperative_elementwise_mul,
        py::call_guard<py::gil_scoped_release>());
  m.def("elementwise_add", &imperative_elementwise_add,
        py::call_guard<py::gil_scoped_release>());
  m.def("elementwise_max", &imperative_elementwise_max,
        py::call_guard<py::gil_scoped_release>());
  m.def("elementwise_div", &imperative_elementwise_div,
        py::call_guard<py::gil_scoped_release>());
  m.def("sigmoid", &imperative_sigmoid,
        py::call_guard<py::gil_scoped_release>());
  m.def("tanh", &imperative_tanh, py::call_guard<py::gil_scoped_release>());
  m.def("square", &imperative_square, py::call_guard<py::gil_scoped_release>());
  m.def("sqrt", &imperative_sqrt, py::call_guard<py::gil_scoped_release>());
  m.def("relu", &imperative_relu, py::call_guard<py::gil_scoped_release>());
  m.def("mean", &imperative_mean, py::call_guard<py::gil_scoped_release>());
  m.def("split", &imperative_split, py::call_guard<py::gil_scoped_release>());
  m.def("matmul", &imperative_matmul, py::call_guard<py::gil_scoped_release>());
  m.def("slice", &imperative_slice, py::call_guard<py::gil_scoped_release>());
  m.def("reduce_sum", &imperative_reduce_sum,
        py::call_guard<py::gil_scoped_release>());
  m.def("reduce_mean", &imperative_reduce_mean,
        py::call_guard<py::gil_scoped_release>());
  m.def("sgd", &imperative_sgd, py::call_guard<py::gil_scoped_release>());
  m.def("softmax_with_cross_entropy", &imperative_softmax_with_cross_entropy,
        py::call_guard<py::gil_scoped_release>());
  m.def("cross_entropy2", &imperative_cross_entropy2,
        py::call_guard<py::gil_scoped_release>());
  m.def("softmax", &imperative_softmax,
        py::call_guard<py::gil_scoped_release>());
  m.def("transpose2", &imperative_transpose2,
        py::call_guard<py::gil_scoped_release>());
  m.def("reshape2", &imperative_reshape2,
        py::call_guard<py::gil_scoped_release>());
  m.def("conv2d", &imperative_conv2d, py::call_guard<py::gil_scoped_release>());
  m.def("pool2d", &imperative_pool2d, py::call_guard<py::gil_scoped_release>());
}

}  // namespace pybind
}  // namespace paddle
