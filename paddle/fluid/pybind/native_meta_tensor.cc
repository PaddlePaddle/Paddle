// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/ext/native_meta_tensor.h"
#include "paddle/fluid/pybind/native_meta_tensor.h"
#include "paddle/utils/pybind.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace paddle::pybind {

void BindNativeMetaTensor(py::module* m) {
  py::class_<phi::NativeMetaTensor>(*m, "NativeMetaTensor")
      .def(py::init<>())
      .def(py::init<const phi::NativeMetaTensor&>())
      .def(py::init([](const py::object& dtype, const py::object& shape) {
             phi::DataType dt = phi::DataType::FLOAT32;
             if (!dtype.is_none()) {
               dt = dtype.cast<phi::DataType>();
             }
             std::vector<int64_t> dims;
             if (py::isinstance<py::list>(shape) ||
                 py::isinstance<py::tuple>(shape)) {
               dims = shape.cast<std::vector<int64_t>>();
             } else {
               PADDLE_THROW(common::errors::InvalidArgument(
                   "The shape argument must be a list or tuple of integers "
                   "or None, but got %s.",
                   py::str(shape)));
             }
             return phi::NativeMetaTensor(dt, phi::make_ddim(dims));
           }),
           py::arg("dtype") = py::none(),
           py::arg("shape") = py::list())
      .def(
          "copy",
          [](const phi::NativeMetaTensor& self) {
            return phi::NativeMetaTensor(self);
          },
          "Create a deep copy of this tensor")
      .def(
          "set_shape",
          [](phi::NativeMetaTensor& self, const std::vector<int64_t>& dims) {
            phi::DDim ddim = phi::make_ddim(dims);
            self.set_dims(ddim);
          },
          "Set tensor dimensions")
      .def(
          "set_dtype",
          [](phi::NativeMetaTensor& self, const std::string& dtype_str) {
            self.set_dtype(phi::StringToDataType(dtype_str));
          },
          "Set tensor data type from string")
      .def(
          "set_dtype",
          [](phi::NativeMetaTensor& self, const phi::DataType& dtype) {
            self.set_dtype(dtype);
          },
          "Set tensor data type from DataType object")
      .def_property_readonly(
          "dtype",
          [](const phi::NativeMetaTensor& self) -> phi::DataType {
            return self.dtype();
          },
          "Get tensor data type")
      .def_property_readonly(
          "shape",
          [](const phi::NativeMetaTensor& self) -> std::vector<int64_t> {
            const phi::DDim& dims = self.dims();
            return common::vectorize<int64_t>(dims);
          },
          "Get tensor shape")
      .def("__eq__",
           [](const phi::NativeMetaTensor& self,
              const phi::NativeMetaTensor& other) {
             return self.dtype() == other.dtype() &&
                    self.dims() == other.dims();
           })
      .def("__repr__", [](const phi::NativeMetaTensor& self) {
        const phi::DDim& dims = self.dims();
        std::ostringstream shape_ss;
        shape_ss << "[";
        for (int i = 0; i < dims.size(); ++i) {
          if (i > 0) {
            shape_ss << ", ";
          }
          shape_ss << dims[i];
        }
        shape_ss << "]";
        std::string dtype_str = phi::DataTypeToString(self.dtype());
        return "NativeMetaTensor(shape=" + shape_ss.str() +
               ", dtype=" + dtype_str + ")";
      });
}
}  // namespace paddle::pybind
