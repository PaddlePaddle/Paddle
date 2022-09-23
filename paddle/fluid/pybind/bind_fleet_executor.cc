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

#include "paddle/fluid/pybind/bind_fleet_executor.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/dist_model.h"
#include "paddle/fluid/distributed/fleet_executor/dist_model_tensor_wrapper.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;

// Note: Since float16 is not a builtin type in C++, we register
// paddle::platform::float16 as numpy.float16.
// Ref: https://github.com/pybind/pybind11/issues/1776
template <>
struct npy_format_descriptor<paddle::platform::float16> {
  static py::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16_);
    return reinterpret_borrow<py::dtype>(ptr);
  }
  static std::string format() {
    // Note: "e" represents float16.
    // Details at:
    // https://docs.python.org/3/library/struct.html#format-characters.
    return "e";
  }
  static constexpr auto name = _("float16");
};

}  // namespace detail
}  // namespace pybind11

namespace paddle {
namespace pybind {

using paddle::distributed::DistModel;
using paddle::distributed::DistModelConfig;
using paddle::distributed::DistModelDataBuf;
using paddle::distributed::DistModelDataType;
using paddle::distributed::DistModelTensor;
using paddle::distributed::FleetExecutor;
using paddle::distributed::TaskNode;
using paddle::framework::OpDesc;
using paddle::framework::ProgramDesc;

template <typename T>
DistModelDataBuf DistModelDataBufCreate(
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {
  // accept numpy array directly
  DistModelDataBuf buf(data.size() * sizeof(T));
  std::copy_n(static_cast<const T*>(data.data()),
              data.size(),
              static_cast<T*>(buf.data()));
  return buf;
}

template <typename T>
void DistModelDataBufReset(
    DistModelDataBuf& buf,                                             // NOLINT
    py::array_t<T, py::array::c_style | py::array::forcecast> data) {  // NOLINT
  // reset the data with numpy array directly
  buf.Resize(data.size() * sizeof(T));
  std::copy_n(static_cast<const T*>(data.data()),
              data.size(),
              static_cast<T*>(buf.data()));
}

template <typename T>
DistModelTensor DistModelTensorCreate(
    py::array_t<T, py::array::c_style | py::array::forcecast> data,
    const std::string name,
    const std::vector<std::vector<size_t>>& lod,
    bool copy) {
  DistModelTensor tensor;

  if (copy) {
    DistModelDataBuf buf(data.size() * sizeof(T));
    std::copy_n(static_cast<const T*>(data.data()),
                data.size(),
                static_cast<T*>(buf.data()));
    tensor.data = std::move(buf);
  } else {
    tensor.data =
        DistModelDataBuf(data.mutable_data(), data.size() * sizeof(T));
  }

  tensor.dtype = paddle::distributed::DistModelGetDtype<T>();
  tensor.name = name;
  tensor.lod = lod;
  tensor.shape.resize(data.ndim());
  std::copy_n(data.shape(), data.ndim(), tensor.shape.begin());

  return tensor;
}

py::dtype DistModelTypeToNumpyDType(DistModelDataType dtype) {
  py::dtype dt;
  switch (dtype) {
    case DistModelDataType::INT32:
      dt = py::dtype::of<int32_t>();
      break;
    case DistModelDataType::INT64:
      dt = py::dtype::of<int64_t>();
      break;
    case DistModelDataType::FLOAT32:
      dt = py::dtype::of<float>();
      break;
    case DistModelDataType::INT8:
      dt = py::dtype::of<int8_t>();
      break;
    case DistModelDataType::FLOAT16:
      dt = py::dtype::of<paddle::platform::float16>();
      break;
    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type. Now only supports INT32, INT64, INT8, "
          "FLOAT16 and FLOAT32."));
  }

  return dt;
}

py::array DistModelTensorGetData(DistModelTensor& tensor) {  // NOLINT
  py::dtype dt = DistModelTypeToNumpyDType(tensor.dtype);
  return py::array(std::move(dt), {tensor.shape}, tensor.data.data());
}

void BindFleetExecutor(py::module* m) {
  py::class_<FleetExecutor>(*m, "FleetExecutor")
      .def(py::init<const std::string&>())
      .def("init", &FleetExecutor::Init)
      .def(
          "run", &FleetExecutor::Run, py::call_guard<py::gil_scoped_release>());

  py::class_<TaskNode>(*m, "TaskNode")
      .def(py::init<framework::ProgramDesc*,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t>())
      .def(py::init<framework::ProgramDesc*, int64_t, int64_t, int64_t>())
      .def(py::init<int32_t,
                    const std::vector<framework::OpDesc*>&,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t>())
      .def("task_id", &TaskNode::task_id)
      .def("add_upstream_task", &TaskNode::AddUpstreamTask)
      .def("add_downstream_task", &TaskNode::AddDownstreamTask)
      .def("set_run_pre_steps", &TaskNode::SetRunPerSteps)
      .def("set_run_at_offset", &TaskNode::SetRunAtOffset)
      .def("set_type", &TaskNode::SetType)
      .def("role", &TaskNode::role)
      .def("init", [](TaskNode& self) { self.Init(); })
      .def("set_program", &TaskNode::SetProgram);

  py::class_<DistModelConfig>(*m, "DistModelConfig")
      .def(py::init<>())
      .def_readwrite("model_dir", &DistModelConfig::model_dir)
      .def_readwrite("program_desc", &DistModelConfig::program_desc)
      .def_readwrite("scope", &DistModelConfig::scope)
      .def_readwrite("place", &DistModelConfig::place)
      .def_readwrite("device_id", &DistModelConfig::device_id)
      .def_readwrite("trainer_endpoints", &DistModelConfig::trainer_endpoints)
      .def_readwrite("current_endpoint", &DistModelConfig::current_endpoint)
      .def_readwrite("nranks", &DistModelConfig::nranks)
      .def_readwrite("local_rank", &DistModelConfig::local_rank)
      .def_readwrite("ring_id_to_ranks", &DistModelConfig::ring_id_to_ranks_)
      .def_readwrite("rank_to_ring_ids", &DistModelConfig::rank_to_ring_ids_)
      .def_readwrite("enable_timer", &DistModelConfig::enable_timer);

  py::class_<DistModel>(*m, "DistModel")
      .def(py::init<const DistModelConfig&>())
      .def("init", &DistModel::Init)
      .def("run",
           [](DistModel& self, const std::vector<DistModelTensor>& inputs) {
             std::vector<DistModelTensor> outputs;
             self.Run(inputs, &outputs);
             return outputs;
           });

  py::class_<DistModelDataBuf>(*m, "DistModelDataBuf")
      .def(py::init<size_t>())
      .def(py::init([](std::vector<float>& data) {
        auto buf = DistModelDataBuf(data.size() * sizeof(float));
        std::memcpy(buf.data(), static_cast<void*>(data.data()), buf.length());
        return buf;
      }))
      .def(py::init(&DistModelDataBufCreate<int32_t>))
      .def(py::init(&DistModelDataBufCreate<int64_t>))
      .def(py::init(&DistModelDataBufCreate<float>))
      .def(py::init(&DistModelDataBufCreate<paddle::platform::float16>))
      .def("reset",
           [](DistModelDataBuf& self, std::vector<float>& data) {
             self.Resize(data.size() * sizeof(float));
             std::memcpy(self.data(), data.data(), self.length());
           })
      .def("reset", &DistModelDataBufReset<int32_t>)
      .def("reset", &DistModelDataBufReset<int64_t>)
      .def("reset", &DistModelDataBufReset<float>)
      .def("reset", &DistModelDataBufReset<paddle::platform::float16>)
      .def("length", &DistModelDataBuf::length)
      .def("tolist",
           [](DistModelDataBuf& self, const std::string& dtype) -> py::list {
             py::list l;
             if (dtype == "int32") {
               auto* data = static_cast<int32_t*>(self.data());
               auto size = self.length() / sizeof(int32_t);
               l = py::cast(std::vector<int32_t>(data, data + size));
             } else if (dtype == "int64") {
               auto* data = static_cast<int64_t*>(self.data());
               auto size = self.length() / sizeof(int64_t);
               l = py::cast(std::vector<int64_t>(data, data + size));
             } else if (dtype == "float32") {
               auto* data = static_cast<float*>(self.data());
               auto size = self.length() / sizeof(float);
               l = py::cast(std::vector<float>(data, data + size));
             } else if (dtype == "float16") {
               auto* data =
                   static_cast<paddle::platform::float16*>(self.data());
               auto size = self.length() / sizeof(paddle::platform::float16);
               l = py::cast(
                   std::vector<paddle::platform::float16>(data, data + size));
             } else {
               PADDLE_THROW(platform::errors::Unimplemented(
                   "Unsupported data type. Now only supports INT32, INT64, "
                   "FLOAT16 and FLOAT32."));
             }
             return l;
           });

  py::class_<DistModelTensor>(*m, "DistModelTensor")
      .def(py::init<>())
      .def(py::init(&DistModelTensorCreate<int32_t>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&DistModelTensorCreate<int64_t>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&DistModelTensorCreate<float>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def(py::init(&DistModelTensorCreate<paddle::platform::float16>),
           py::arg("data"),
           py::arg("name") = "",
           py::arg("lod") = std::vector<std::vector<size_t>>(),
           py::arg("copy") = true)
      .def_readwrite("name", &DistModelTensor::name)
      .def_readwrite("shape", &DistModelTensor::shape)
      .def_readwrite("data", &DistModelTensor::data)
      .def_readwrite("dtype", &DistModelTensor::dtype)
      .def_readwrite("lod", &DistModelTensor::lod)
      .def("as_ndarray", &DistModelTensorGetData);

  py::enum_<DistModelDataType>(*m, "DistModelDataType")
      .value("FLOAT32", DistModelDataType::FLOAT32)
      .value("INT64", DistModelDataType::INT64)
      .value("INT32", DistModelDataType::INT32)
      .value("FLOAT16", DistModelDataType::FLOAT16);
}
}  // namespace pybind
}  // namespace paddle
