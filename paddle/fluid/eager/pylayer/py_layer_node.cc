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
    const std::vector<std::vector<paddle::experimental::Tensor>>& grads) {
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
      PyTuple_SET_ITEM(backward_args, i, paddle::pybind::ToPyObject(grads[i]));
    } else {
      PyTuple_SET_ITEM(backward_args, i,
                       paddle::pybind::ToPyObject(grads[i][0]));
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
      if (this->OutputMeta()[i].IsStopGradient(0)) {
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
          this->OutputMeta()[i].IsStopGradient(0), true,
          paddle::platform::errors::InvalidArgument(
              "%s's backward function should not return empyt at %d position.",
              name(), i));
      grad_out.push_back({});
    }
  }

  return grad_out;
}

void GradNodePyLayer::RegisterReduceHook(
    std::shared_ptr<TensorVoidHook>&& hook) {
  reduce_hooks_.emplace_back(std::move(hook));
}

void GradNodePyLayer::ApplyReduceHooks() {
  for (auto& hook : reduce_hooks_) {
    (*hook)();
  }
}
}  // namespace egr
