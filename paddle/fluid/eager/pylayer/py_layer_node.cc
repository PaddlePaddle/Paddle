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

#include "paddle/fluid/eager/pylayer/py_layer_node.h"
#include "paddle/fluid/eager/eager_tensor.h"

#include "paddle/phi/api/all.h"
#include "paddle/phi/core/dense_tensor.h"

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/pybind/eager.h"
#include "paddle/fluid/pybind/eager_utils.h"

#include "glog/logging.h"
#pragma GCC diagnostic ignored "-Wattributes"
#include "pybind11/pytypes.h"

namespace egr {
std::vector<std::vector<paddle::experimental::Tensor>> GradNodePyLayer::
operator()(
    std::vector<std::vector<paddle::experimental::Tensor>>& grads,  // NOLINT
    bool create_graph) {
  VLOG(3) << "Running Eager Backward Node: " << name();

  std::vector<std::vector<paddle::experimental::Tensor>> hooked_grads =
      GradNodePyLayer::ApplyGradientHooks(grads);

  paddle::pybind::PyLayerObject* ctx =
      reinterpret_cast<paddle::pybind::PyLayerObject*>(ctx_);

  PADDLE_ENFORCE_EQ(ctx->forward_output_tensor_is_duplicable.size(),
                    grads.size(),
                    paddle::platform::errors::InvalidArgument(
                        "%s's grad input size(%s) mast be equal with it's "
                        "forward's output size(%s).",
                        name(), grads.size(),
                        ctx->forward_output_tensor_is_duplicable.size()));

  auto backward_args = PyTuple_New(grads.size());
  for (size_t i = 0; i < grads.size(); i++) {
    if (ctx->forward_output_tensor_is_duplicable[i]) {
      PyObject* pylist = PyList_New((Py_ssize_t)grads[i].size());
      for (size_t j = 0; j < grads[i].size(); j++) {
        if (ctx->materialize_grads && !grads[i][j].initialized()) {
          paddle::experimental::Tensor tensor_tmp;
          auto dense_tensor = std::make_shared<phi::DenseTensor>();
          dense_tensor->set_meta(forward_outputs_meta_[i][j]);
          tensor_tmp.set_impl(dense_tensor);
          PyList_SET_ITEM(
              pylist, static_cast<Py_ssize_t>(i),
              paddle::pybind::ToPyObject(paddle::experimental::zeros_like(
                  tensor_tmp, tensor_tmp.dtype(),
                  forward_outputs_place_[i][j])));
        } else {
          PyList_SET_ITEM(pylist, static_cast<Py_ssize_t>(i),
                          paddle::pybind::ToPyObject(grads[i][0], true));
        }
      }
      PyTuple_SET_ITEM(backward_args, i, pylist);
    } else {
      if (ctx->materialize_grads && !grads[i][0].initialized()) {
        paddle::experimental::Tensor tensor_tmp;
        auto dense_tensor = std::make_shared<phi::DenseTensor>();
        dense_tensor->set_meta(forward_outputs_meta_[i][0]);
        tensor_tmp.set_impl(dense_tensor);
        PyTuple_SET_ITEM(
            backward_args, i,
            paddle::pybind::ToPyObject(paddle::experimental::zeros_like(
                tensor_tmp, tensor_tmp.dtype(), forward_outputs_place_[i][0])));
      } else {
        PyTuple_SET_ITEM(backward_args, i,
                         paddle::pybind::ToPyObject(grads[i][0], true));
      }
    }
  }

  VLOG(6) << "PyLayer backward args is ready, begin call user's backward "
             "function...";

  auto backward_fn =
      PyObject_GetAttrString(reinterpret_cast<PyObject*>(ctx), "backward");
  if (!backward_fn) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "Get backward function faild."));
  }
  auto outputs = PyObject_CallObject(backward_fn, backward_args);
  if (!outputs) {
    PADDLE_THROW(paddle::platform::errors::External(
        pybind11::detail::error_string().c_str()));
  }

  outputs_ = outputs;

  VLOG(6) << "PyLayer backward function finish...";

  PyObject* outputs_tuple = nullptr;
  if (PyTuple_Check(outputs)) {
    outputs_tuple = outputs;
  } else {
    outputs_tuple = PyTuple_New(1);
    Py_INCREF(outputs);
    PyTuple_SET_ITEM(outputs_tuple, 0, outputs);
  }

  size_t outputs_size = PyTuple_GET_SIZE(outputs_tuple);

  if (outputs_size > ctx->forward_input_tensor_is_duplicable.size()) {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The number of outputs of `PyLayer.backward` should be %d, but "
        "received %d.",
        ctx->forward_input_tensor_is_duplicable.size(), outputs_size));
  }

  std::vector<std::vector<paddle::experimental::Tensor>> grad_out;
  grad_out.reserve(ctx->forward_input_tensor_is_duplicable.size());
  for (size_t i = 0; i < ctx->forward_input_tensor_is_duplicable.size(); i++) {
    if (i < outputs_size) {
      PyObject* obj = PyTuple_GET_ITEM(outputs_tuple, i);
      if (this->OutputMeta()[i][0].IsStopGradient()) {
        PADDLE_ENFORCE_EQ(
            obj, Py_None,
            paddle::platform::errors::InvalidArgument(
                "%s's backward function should return None at %d position, "
                "because it's forward Tensor's stopgradient is true.",
                name(), i));
        grad_out.push_back({});
      } else {
        if (ctx->forward_input_tensor_is_duplicable[i]) {
          grad_out.push_back(paddle::pybind::GetTensorListFromPyObject(obj));
        } else {
          grad_out.push_back({paddle::pybind::GetTensorFromPyObject(obj)});
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(
          this->OutputMeta()[i][0].IsStopGradient(), true,
          paddle::platform::errors::InvalidArgument(
              "%s's backward function should not return empyt at %d position.",
              name(), i));
      grad_out.push_back({});
    }
  }

  return grad_out;
}
}  // namespace egr
