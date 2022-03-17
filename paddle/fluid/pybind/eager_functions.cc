/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
// disable numpy compile error
#include <Python.h>

#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "paddle/fluid/eager/accumulation/accumulation_node.h"
#include "paddle/fluid/eager/api/all.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/backward.h"
#include "paddle/fluid/eager/custom_operator/custom_operator_node.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/custom_operator.h"
#include "paddle/fluid/framework/op_meta_info_helper.h"
#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/phi/api/ext/op_meta_info.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/dense_tensor.h"
namespace paddle {
namespace pybind {

namespace py = ::pybind11;

extern PyTypeObject* p_tensor_type;
extern PyTypeObject* g_multidevicefeedreader_pytype;
extern PyTypeObject* g_orderedmultidevicefeedreader_pytype;

size_t PyArray_Size_(PyObject* numpy_data) {
  size_t res = 1;
  auto dims = pybind11::detail::array_proxy(numpy_data)->dimensions;
  auto nd = pybind11::detail::array_proxy(numpy_data)->nd;
  while (nd--) {
    res *= (*dims++);
  }
  return res;
}

class EagerNumpyAllocation : public phi::Allocation {
 public:
  explicit EagerNumpyAllocation(PyObject* numpy_data, phi::DataType dtype)
      : Allocation(
            static_cast<void*>(pybind11::detail::array_proxy(numpy_data)->data),
            framework::DataTypeSize(dtype) * PyArray_Size_(numpy_data),
            paddle::platform::CPUPlace()),
        arr_(numpy_data) {
    PADDLE_ENFORCE_NOT_NULL(arr_, platform::errors::InvalidArgument(
                                      "The underlying PyObject pointer of "
                                      "numpy array cannot be nullptr"));
    PADDLE_ENFORCE_NE(
        arr_, Py_None,
        platform::errors::PreconditionNotMet(
            "The underlying PyObject pointer of numpy array cannot be None"));
    Py_INCREF(arr_);
  }
  ~EagerNumpyAllocation() override {
    py::gil_scoped_acquire gil;
    Py_DECREF(arr_);
  }

 private:
  PyObject* arr_;
};

static PyObject* eager_api_set_expected_place(PyObject* self, PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 0), 0);
  egr::Controller::Instance().SetExpectedPlace(place);

  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_get_expected_place(PyObject* self, PyObject* args,
                                              PyObject* kwargs) {
  EAGER_TRY
  return ToPyObject(egr::Controller::Instance().GetExpectedPlace());
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_scale(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  EAGER_TRY
  // TODO(jiabin): Sync Tensor and Variable here when we support
  paddle::experimental::Tensor ret = egr::scale(
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor,
      CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 1), 1),
      CastPyArg2AttrFloat(PyTuple_GET_ITEM(args, 2), 2),
      CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3),
      CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4));
  return ToPyObject(ret);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_backward(PyObject* self, PyObject* args,
                                        PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  egr::Backward(tensors, grad_tensors,
                CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2));
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_run_partial_grad(PyObject* self, PyObject* args,
                                            PyObject* kwargs) {
  EAGER_TRY
  auto tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 0), 0);
  auto inputs = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 1), 1);
  auto grad_tensors = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 2), 2);
  auto retain_graph = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);
  auto create_graph = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 4), 4);
  auto only_inputs = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 5), 5);
  auto allow_unused = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 6), 6);
  auto no_grad_vars = CastPyArg2VectorOfTensor(PyTuple_GET_ITEM(args, 7), 7);

  std::vector<paddle::experimental::Tensor> result =
      egr::Grad(tensors, inputs, grad_tensors, retain_graph, create_graph,
                only_inputs, allow_unused, no_grad_vars);
  VLOG(1) << " in eager_api_run_partial_grad, after runing egr::Grad";
  return ToPyObject(result, true /* return_py_none_if_not_initialize */);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_tensor_copy(PyObject* self, PyObject* args,
                                       PyObject* kwargs) {
  EAGER_TRY
  paddle::experimental::Tensor& src =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 0))->tensor;
  paddle::experimental::Tensor& dst =
      reinterpret_cast<TensorObject*>(PyTuple_GET_ITEM(args, 1))->tensor;
  auto place = CastPyArg2Place(PyTuple_GET_ITEM(args, 2), 2);
  bool blocking = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 3), 3);

  dst = src.copy_to(phi::TransToPhiBackend(place), blocking);
  egr::EagerUtils::autograd_meta(&dst)->SetStopGradient(
      egr::EagerUtils::autograd_meta(&(src))->StopGradient());
  egr::EagerUtils::autograd_meta(&dst)->SetPersistable(
      egr::EagerUtils::autograd_meta(&(src))->Persistable());
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static PyObject* eager_api_read_next_tensor_list(PyObject* self, PyObject* args,
                                                 PyObject* kwargs) {
  EAGER_TRY
  auto tensor_base_list =
      CastPyArg2VectorOfTensorBase(PyTuple_GET_ITEM(args, 0), 0);
  std::vector<paddle::experimental::Tensor> tensor_list;
  tensor_list.reserve(tensor_base_list.size());
  auto func = [](framework::Tensor& tensor_base) {
    paddle::experimental::Tensor tensor(
        egr::Controller::Instance().GenerateUniqueName());
    auto autograd_meta = egr::EagerUtils::autograd_meta(&tensor);
    autograd_meta->SetPersistable(false);
    autograd_meta->SetStopGradient(true);
    tensor.set_impl(std::make_shared<phi::DenseTensor>(tensor_base));
    return tensor;
  };
  for (auto& tensor_base : tensor_base_list) {
    tensor_list.emplace_back(func(tensor_base));
  }
  return ToPyObject(tensor_list);
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

static void ConstructFwdAndBwdMap(
    const std::vector<paddle::OpMetaInfo>& vec_map,
    const std::string& op_type) {
  auto& in_out_map = egr::Controller::Instance().GetCustomEdgesSlotMap();
  if (in_out_map.find(op_type) != in_out_map.end()) {
    VLOG(7) << "Find Exist CustomEdgesSlotMap Skip >>>> ";
    return;
  } else {
    VLOG(7) << "Construct CustomEdgesSlotMap ";
    auto inputs_names =
        paddle::framework::OpMetaInfoHelper::GetInputs(vec_map[0]);
    auto outputs_names =
        paddle::framework::OpMetaInfoHelper::GetOutputs(vec_map[0]);
    auto attrs_names =
        paddle::framework::OpMetaInfoHelper::GetAttrs(vec_map[0]);
    auto grad_outputs_names =
        paddle::framework::OpMetaInfoHelper::GetOutputs(vec_map[1]);
    auto grad_inputs_names =
        paddle::framework::OpMetaInfoHelper::GetInputs(vec_map[1]);
    auto grad_attrs_names =
        paddle::framework::OpMetaInfoHelper::GetAttrs(vec_map[1]);
    std::vector<std::unordered_map<int, int>> res(5);
    in_out_map.insert({op_type, res});
    // Prepare pos map for grad_outputs
    VLOG(7) << "Prepare pos map for grad_outputs";
    PADDLE_ENFORCE_LE(
        grad_outputs_names.size(), inputs_names.size(),
        paddle::platform::errors::InvalidArgument(
            "Grad outputs num should be less equal than forward inputs num."));
    for (size_t i = 0; i < grad_outputs_names.size(); i++) {
      size_t end = grad_outputs_names[i].find("@GRAD");
      PADDLE_ENFORCE_NE(
          end, std::string::npos,
          paddle::platform::errors::NotFound(
              "All Grad outputs should be grad and we got %s is not grad var, "
              "please check your op and change to fit the rule.",
              grad_outputs_names[i]));
      for (size_t j = 0; j < inputs_names.size(); j++) {
        if (grad_outputs_names[i].substr(0, end) == inputs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                  << " inputs: " << inputs_names[j] << " related to No." << i
                  << " grad_outputs: " << grad_outputs_names[i];
          in_out_map[op_type][0][j] = i;
        }
      }
    }
    // Prepare pos map for grad_inputs
    for (size_t i = 0; i < grad_inputs_names.size(); i++) {
      size_t end = grad_inputs_names[i].find("@GRAD");
      if (end != std::string::npos) {
        for (size_t j = 0; j < outputs_names.size(); j++) {
          if (grad_inputs_names[i].substr(0, end) == outputs_names[j]) {
            VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                    << " outputs: " << outputs_names[j] << " related to No."
                    << i << " grad_inputs's grad: " << grad_inputs_names[i];
            in_out_map[op_type][1][j] = i;
          }
        }
      } else {
        if (std::find(outputs_names.begin(), outputs_names.end(),
                      grad_inputs_names[i]) != outputs_names.end()) {
          for (size_t j = 0; j < outputs_names.size(); j++) {
            if (grad_inputs_names[i] == outputs_names[j]) {
              VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                      << " outputs: " << outputs_names[j] << " related to No."
                      << i
                      << " grad_inputs fwd outputs: " << grad_inputs_names[i];
              in_out_map[op_type][2][j] = i;
            }
          }
        } else {
          for (size_t j = 0; j < inputs_names.size(); j++) {
            if (grad_inputs_names[i] == inputs_names[j]) {
              VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                      << " inputs: " << inputs_names[j] << " related to No."
                      << i
                      << " grad_inputs fwd inputs: " << grad_inputs_names[i];
              in_out_map[op_type][3][j] = i;
            }
          }
        }
      }
    }

    // Prepare pos map for grad attrs_
    for (size_t i = 0; i < grad_attrs_names.size(); i++) {
      auto end = std::find(attrs_names.begin(), attrs_names.end(),
                           grad_attrs_names[i]);
      PADDLE_ENFORCE_NE(end, attrs_names.end(),
                        paddle::platform::errors::NotFound(
                            "All Grad attrs should be one of forward attrs and "
                            "we got %s is not one of them, please check your "
                            "op and change to fit the rule.",
                            grad_attrs_names[i]));
      for (size_t j = 0; j < attrs_names.size(); j++) {
        if (grad_attrs_names[i] == attrs_names[j]) {
          VLOG(7) << " ==== Custom Operator: " << op_type << "'s No." << j
                  << " attrs: " << attrs_names[j] << " related to No." << i
                  << " grad_attrs: " << grad_attrs_names[i];
          in_out_map[op_type][4][j] = i;
        }
      }
    }
  }
}

static std::vector<paddle::any> CastAttrsToTragetType(
    const std::vector<paddle::any>& src,
    const std::vector<std::string>& attrs_names) {
  std::vector<paddle::any> res;
  PADDLE_ENFORCE_EQ(src.size(), attrs_names.size(),
                    paddle::platform::errors::InvalidArgument(
                        "We Expected same size of attrs and attrs_name list, "
                        "if u got this error indicate your custom op setting "
                        "%s attrs, but you just give %s",
                        attrs_names.size(), src.size()));
  for (size_t i = 0; i < src.size(); i++) {
    size_t end = attrs_names[i].find(": ");
    std::string type_name =
        attrs_names[i].substr(end + 2, attrs_names.size() - end - 2);
    if (type_name == "int") {
      if (src[i].type() == typeid(bool)) {
        res.emplace_back(static_cast<int>(paddle::any_cast<bool>(src[i])));
      } else if (src[i].type() == typeid(int)) {
        res.emplace_back(src[i]);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Your No. %s attrs should only can be bool or int32, other type is "
            "forbidden for now but we got %s. Check your code first please",
            i, src[i].type().name()));
      }
    } else if (type_name == "int64_t") {
      if (src[i].type() == typeid(bool)) {
        res.emplace_back(static_cast<int64_t>(paddle::any_cast<bool>(src[i])));
      } else if (src[i].type() == typeid(int)) {
        res.emplace_back(static_cast<int64_t>(paddle::any_cast<int>(src[i])));
      } else if (src[i].type() == typeid(int64_t)) {
        res.emplace_back(src[i]);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Your No. %s attrs should only can be bool or int32 or int64_t, "
            "other type is forbidden for now but we got %s. Check your code "
            "first please",
            i, src[i].type().name()));
      }
    } else {
      res.emplace_back(src[i]);
    }
  }
  return res;
}

static PyObject* eager_api_run_costum_op(PyObject* self, PyObject* args,
                                         PyObject* kwargs) {
  EAGER_TRY
  paddle::CustomOpKernelContext ctx =
      CastPyArg2CustomOpKernelContext(PyTuple_GET_ITEM(args, 0), 0);
  std::string op_type = CastPyArg2AttrString(PyTuple_GET_ITEM(args, 1), 1);
  bool trace_backward = CastPyArg2AttrBoolean(PyTuple_GET_ITEM(args, 2), 2);
  VLOG(7) << "Get things for python for Custom Op: " << op_type
          << ", trace_backward is: " << trace_backward;
  auto meta_info_map = egr::Controller::Instance().GetOpMetaInfoMap();
  PADDLE_ENFORCE_NE(meta_info_map.find(op_type), meta_info_map.end(),
                    paddle::platform::errors::NotFound(
                        "Can't find %s in Eager OpMetaInfoMap which should be "
                        "created by LoadOpMetaInfoAndRegisterOp, please make "
                        "sure you registered your op first and try again. ",
                        op_type));
  VLOG(7) << "Run Kernel of Custom Op: " << op_type;
  std::vector<paddle::any> res_attrs = CastAttrsToTragetType(
      ctx.Attrs(), paddle::framework::OpMetaInfoHelper::GetAttrs(
                       meta_info_map.at(op_type)[0]));
  ctx.EmplaceBackAttrs(res_attrs);
  const auto& vec_map = meta_info_map.at(op_type);
  (*paddle::framework::OpMetaInfoHelper::GetKernelFn(vec_map[0]))(&ctx);

  VLOG(7) << "Get AutogradMeta for inputs and outputs for Custom Op";
  std::vector<std::vector<egr::AutogradMeta*>> ins_auto_grad_metas;
  std::vector<std::vector<egr::AutogradMeta*>> outs_auto_grad_metas;
  VLOG(7) << "We got slot num of ins is: " << ctx.InputRange().size();
  ins_auto_grad_metas.resize(ctx.InputRange().size());
  VLOG(7) << "We got slot num of outs is: " << ctx.OutputRange().size();
  outs_auto_grad_metas.resize(ctx.OutputRange().size());

  for (size_t i = 0; i < ctx.InputRange().size(); i++) {
    ins_auto_grad_metas[i] =
        egr::EagerUtils::nullable_autograd_meta(ctx.InputsBetween(
            ctx.InputRangeAt(i).first, ctx.InputRangeAt(i).second));
  }
  for (size_t i = 0; i < ctx.OutputRange().size(); i++) {
    outs_auto_grad_metas[i] =
        egr::EagerUtils::unsafe_autograd_meta(ctx.OutputsBetweeen(
            ctx.OutputRangeAt(i).first, ctx.OutputRangeAt(i).second));
  }
  bool require_any_grad = false;
  for (size_t i = 0; i < ins_auto_grad_metas.size(); i++) {
    require_any_grad =
        require_any_grad || egr::EagerUtils::ComputeRequireGrad(
                                trace_backward, &(ins_auto_grad_metas[i]));
  }
  if (require_any_grad) {
    VLOG(6) << " Construct Grad for Custom Op: " << op_type;
    ConstructFwdAndBwdMap(vec_map, op_type);
    for (size_t i = 0; i < outs_auto_grad_metas.size(); i++) {
      egr::EagerUtils::PassStopGradient(false, &(outs_auto_grad_metas[i]));
    }
    auto grad_node = std::make_shared<egr::RunCustomOpNode>(
        outs_auto_grad_metas.size(), ins_auto_grad_metas.size(), op_type);
    auto slot_map =
        egr::Controller::Instance().GetCustomEdgesSlotMap().at(op_type);
    // Prepare Grad outputs
    size_t no_grad_cnt = 0;
    for (size_t i = 0; i < ins_auto_grad_metas.size(); i++) {
      const std::vector<paddle::experimental::Tensor>& in_tensors =
          ctx.InputsBetween(ctx.InputRangeAt(i).first,
                            ctx.InputRangeAt(i).second);

      if (slot_map[0].find(i) != slot_map[0].end()) {
        grad_node->SetGradOutMeta(in_tensors, slot_map[0][i]);
        grad_node->AddEdges(&ins_auto_grad_metas[i], slot_map[0][i]);
      } else {
        grad_node->SetGradOutMeta(in_tensors,
                                  ins_auto_grad_metas.size() - 1 - no_grad_cnt);
        grad_node->AddEdges(&ins_auto_grad_metas[i],
                            ins_auto_grad_metas.size() - 1 - no_grad_cnt);
        no_grad_cnt++;
      }
    }
    // Prepare Grad inputs with grad of fwd outputs
    for (size_t i = 0; i < outs_auto_grad_metas.size(); i++) {
      const std::vector<paddle::experimental::Tensor>& out_tensors =
          ctx.OutputsBetweeen(ctx.OutputRangeAt(i).first,
                              ctx.OutputRangeAt(i).second);

      egr::EagerUtils::SetOutRankWithSlot(&(outs_auto_grad_metas[i]), i);
      egr::EagerUtils::SetHistory(&(outs_auto_grad_metas[i]), grad_node);
      grad_node->SetGradInMeta(out_tensors, i);
      egr::EagerUtils::CheckAndRetainGrad(out_tensors);
    }

    // Prepare Grad inputs with fwd outputs
    for (auto it = slot_map[2].begin(); it != slot_map[2].end(); it++) {
      VLOG(7) << "Prepare fwd_outs: " << it->first
              << " to grad_inputs: " << it->second;
      grad_node->fwd_outs[it->second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.OutputsBetweeen(ctx.OutputRangeAt(it->first).first,
                                  ctx.OutputRangeAt(it->first).second));
    }

    // Prepare Grad inputs with fwd inputs
    for (auto it = slot_map[3].begin(); it != slot_map[3].end(); it++) {
      VLOG(7) << "Prepare fwd_ins: " << it->first
              << " to grad_inputs: " << it->second;
      grad_node->fwd_ins[it->second] =
          egr::RunCustomOpNode::ConstructTensorWrapper(
              ctx.InputsBetween(ctx.InputRangeAt(it->first).first,
                                ctx.InputRangeAt(it->first).second));
    }

    auto attrs_names = paddle::framework::OpMetaInfoHelper::GetAttrs(
        meta_info_map.at(op_type)[1]);
    std::vector<paddle::any> attrs(attrs_names.size());
    // Prepare attrs for Grad node
    for (auto it = slot_map[4].begin(); it != slot_map[4].end(); it++) {
      VLOG(7) << "Prepare fwd attrs: " << it->first
              << " to grad_attrs: " << it->second;
      attrs[it->second] = res_attrs[it->first];
    }
    grad_node->SetAttrs(attrs);
  }
  Py_INCREF(Py_None);
  return Py_None;
  EAGER_CATCH_AND_THROW_RETURN_NULL
}

PyMethodDef variable_functions[] = {
    // TODO(jiabin): Remove scale when we have final state tests
    {"scale", (PyCFunction)(void (*)(void))eager_api_scale,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_set_expected_place",
     (PyCFunction)(void (*)(void))eager_api_set_expected_place,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_get_expected_place",
     (PyCFunction)(void (*)(void))eager_api_get_expected_place,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"run_backward", (PyCFunction)(void (*)(void))eager_api_run_backward,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"run_partial_grad",
     (PyCFunction)(void (*)(void))eager_api_run_partial_grad,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"_run_custom_op", (PyCFunction)(void (*)(void))eager_api_run_costum_op,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"tensor_copy", (PyCFunction)(void (*)(void))eager_api_tensor_copy,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"read_next_tensor_list",
     (PyCFunction)(void (*)(void))eager_api_read_next_tensor_list,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}};

void BindFunctions(PyObject* module) {
  if (PyModule_AddFunctions(module, variable_functions) < 0) {
    PADDLE_THROW(platform::errors::Fatal(
        "Init Paddle erroe in BindFunctions(PyModule_AddFunctions)."));
    return;
  }
}

}  // namespace pybind
}  // namespace paddle
