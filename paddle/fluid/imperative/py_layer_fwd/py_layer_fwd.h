// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstdint>
#include <list>
// #include <map>
#include <memory>
// #include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/operators/py_layer_op.h"

namespace paddle {
namespace imperative {

namespace py = ::pybind11;
// copy from tracer.cc
void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& pair : outs) {
    for (const auto& var : pair.second) {
      // NOTE(zhiqiu): this happends when None output are passed from python
      // side. For example, fake_quantize_dequantize_moving_average_abs_max may
      // pass None OutAccum in eval mode.
      // It can be refined by generate several different pybind interface for
      // one operator with different function signature.
      if (var == nullptr) {
        VLOG(4) << pair.first << " is NULL";
        continue;
      }
      VLOG(6) << "Set output: " << var->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      var->InnerSetOverridedStopGradient(generate_grad);
    }
  }
}

// copy from tracer.cc
bool RequiredGrad(const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverridedStopGradient());
        return true;
      }
    }
  }
  return false;
}
static void ClearNoNeedBufferInputs(OpBase* op) {
  auto& inferer = op->Info().NoNeedBufferVarsInferer();
  if (!inferer) return;
  auto* ins = op->GetMutableInsMap();
  const auto& no_need_buffer_slots =
      inferer(*ins, op->GetOutsMap(), op->Attrs());
  if (no_need_buffer_slots.empty()) return;

  for (auto& slot : no_need_buffer_slots) {
    auto iter = ins->find(slot);
    if (iter == ins->end()) continue;
    VLOG(2) << "Clear data buffer of " << slot << " in " << op->Type();

    PADDLE_ENFORCE_EQ(
        iter->second.IsGrad(), false,
        platform::errors::InvalidArgument(
            "Only forward variable buffers can be clear, this may be a bug"));

    for (auto& each_var : *(iter->second.MutableVarList())) {
      if (!each_var) continue;

      auto& var = each_var->Var();
      PADDLE_ENFORCE_EQ(var.IsType<framework::LoDTensor>(), true,
                        platform::errors::PermissionDenied(
                            "NoNeedBufferVars only support LoDTensor"));
      auto new_var = new VariableWrapper(each_var->Name());
      auto* new_tensor =
          new_var->MutableVar()->GetMutable<framework::LoDTensor>();
      auto& old_tensor = var.Get<framework::LoDTensor>();
      new_tensor->Resize(old_tensor.dims());
      new_tensor->set_lod(old_tensor.lod());
      each_var.reset(new_var);
    }
  }
}

std::shared_ptr<GradOpNode> CreateGradOpNode(
    const std::string& type, const NameVarBaseMap& ins,
    const NameVarBaseMap& outs, const framework::AttributeMap& attrs,
    const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map,
    const std::shared_ptr<PyLayerContext>& PyLayerContext) {
  operators::test();
  operators::PyLayerGradOpMaker<paddle::imperative::OpBase> maker(
      type, ins, outs, attrs, inplace_map);

  maker.GetMutablePyLayerContext() = PyLayerContext;
  auto grad_node = maker();
  if (grad_node && !grad_node->empty()) {
    for (auto& grad_op : *grad_node) {
      grad_op.SetId(OpBase::GenerateUniqueId());
      grad_op.SetPlace(place);
      ClearNoNeedBufferInputs(&grad_op);
    }
    return grad_node;
  } else {
    return nullptr;
  }
}

py::object CusPyFunc_apply(const platform::Place& place, const py::object& cls,
                           const py::args args, const py::kwargs kwargs) {
  // TODO(weixin): to support saving for backward

  auto bk_function = cls.attr("_backward_function");
  auto contex = bk_function();
  auto forward = cls.attr("forward");

  auto result_forward = forward(contex, *args, **kwargs);

  std::shared_ptr<PyLayerContext> py_layer_ctx =
      std::make_shared<PyLayerContext>(contex);
  // make inputs to varbase
  std::vector<std::shared_ptr<imperative::VarBase>> input_vars;
  // process args,`input_vars` only collect `imperative::VarBase`
  if (!args.empty()) {
    for (auto ptr = args.begin(); ptr != args.end(); ptr++) {
      try {
        if (Py_None != ptr->ptr()) {
          auto a = ptr->cast<std::shared_ptr<VarBase>>();
          // only push varbase
          input_vars.push_back(a);
        }
      } catch (py::cast_error& err) {
      } catch (...) {
        PADDLE_THROW(platform::errors::Fatal(
            "PyLayer raises an unknown exception in rumtime."));
      }
    }
  }
  // process kwargs, only collect `imperative::VarBase`
  if (!kwargs.empty()) {
    for (auto ptr = kwargs.begin(); ptr != kwargs.end(); ptr++) {
      try {
        if (Py_None != ptr->second.ptr()) {
          auto a = ptr->second.cast<std::shared_ptr<VarBase>>();
          // only push varbase
          input_vars.push_back(a);
        }
      } catch (py::cast_error&) {
        // error msg
      } catch (...) {
        PADDLE_THROW(platform::errors::Fatal(
            "PyLayer raises an unknown exception in rumtime."));
      }
    }
  }
  NameVarBaseMap ins = {{"X", input_vars}};

  // process outputs:result_forward
  std::vector<std::shared_ptr<imperative::VarBase>> output_vars;
  // if tuple/list
  if (PyTuple_Check(result_forward.ptr()) ||
      PyList_Check(result_forward.ptr())) {
    auto tuple_result = result_forward.cast<py::tuple>();
    for (size_t i = 0; i < tuple_result.size(); i++) {
      try {
        if (Py_None != tuple_result[i].ptr()) {
          auto temp_out =
              tuple_result[i].cast<std::shared_ptr<imperative::VarBase>>();
          output_vars.push_back(temp_out);
        } else {
          PADDLE_THROW(platform::errors::Unimplemented(
              "The output of forward can not be `None`."));
        }
      } catch (py::cast_error&) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The output of forward should be `Tensor`."));
      } catch (...) {
        // TODO(weixin): to support returning None.
        PADDLE_THROW(platform::errors::Fatal(
            "PyLayer raises an unknown exception in apply."));
      }
    }
  } else {  // varbase
    try {
      if (Py_None != result_forward.ptr()) {
        auto temp_out =
            result_forward.cast<std::shared_ptr<imperative::VarBase>>();
        output_vars.push_back(temp_out);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The output of forward can not be `None`."));
      }
    } catch (py::cast_error&) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The output of forward should be `Tensor`."));
    } catch (...) {
      PADDLE_THROW(platform::errors::Fatal(
          "PyLayer raises an unknown exception in apply."));
    }
  }

  NameVarBaseMap outs = {{"Out", output_vars}};

  if (RequiredGrad(ins, outs)) {
    CreateGradOpNode("py_layer", ins, outs, {{}}, place, {}, py_layer_ctx);
  } else {
    VLOG(3) << "No Grad to track for Op: py_layer";
  }

  return result_forward;
}

}  // namespace imperative
}  // namespace paddle
