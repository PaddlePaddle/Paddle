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
#pragma once

#include "paddle/fluid/pir/dialect/operator/ir/ir_meta_tensor.h"
#include "paddle/fluid/pybind/ir_meta_tensor.h"
#include "paddle/phi/core/tensor_base.h"
// #include "paddle/fluid/pybind/ir_tensor.h"
// #include "paddle/utils/pybind.h"
// #include "pybind11/functional.h"
// #include "pybind11/pybind11.h"
// #include "pybind11/stl.h"

namespace paddle::pybind {

using IrMetaTensor = paddle::dialect::IrMetaTensor;

void BindIrMetaTensor(py::module* m) {
  py::class_<IrMetaTensor, MetaTensor>(*m, "IrMetaTensor")
      .def(py::init(
               [](const TensorBase& tensor, const bool strided_kernel_used) {
                 return IrMetaTensor(tensor, strided_kernel_used);
               }),
           py::arg("tensor"),
           py::arg("strided_kernel_used") = false)
      .def(
          "set_dim",
          [](IrMetaTensor& self, const std::vector<int64_t>& dims) {
            phi::DDim ddim = phi::make_ddim(dims);
            self.set_dims(ddim);
          },
          "Set tensor dimensions")
      .def(
          "set_dtype",
          [](IrMetaTensor& self, const std::string& dtype_str) {
            self.set_dtype(phi::StringToDataType(dtype_str));
          },
          "Set tensor data type from string")
      .def(
          "set_dtype",
          [](IrMetaTensor& self, const phi::DataType& dtype) {
            self.set_dtype(dtype);
          },
          "Set tensor data type from DataType object")
      .def_property_readonly(
          "dtype",
          [](const IrMetaTensor& self) -> phi::DataType {
            return self.dtype();
          },
          "Get tensor data type")
      .def_property_readonly(
          "shape",
          [](const IrMetaTensor& self) -> std::vector<int64_t> {
            const phi::DDim& dims = self.dims();
            std::vector<int64_t> shape;
            for (int i = 0; i < dims.size(); ++i) {
              shape.push_back(dims[i]);
            }
            return shape;
          },
          "Get tensor shape");
}
}  // namespace paddle::pybind
