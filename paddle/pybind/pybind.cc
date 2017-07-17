/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <Python.h>
#include <paddle/framework/op_registry.h>
#include <paddle/framework/scope.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>

namespace py = pybind11;
namespace pd = paddle::framework;

USE_OP(add_two);

struct PlaceDebugString : public boost::static_visitor<std::string> {
  std::string operator()(const paddle::platform::GPUPlace& place) const {
    return "GPU(" + std::to_string(place.device) + ")";
  }

  std::string operator()(const paddle::platform::CPUPlace& place) const {
    return "CPU";
  }
};

template <typename T>
struct TensorToPyBuffer {
  pd::Tensor& self_;
  explicit TensorToPyBuffer(pd::Tensor& self) : self_(self) {}

  bool CanCast() const { return std::type_index(typeid(T)) == self_.type(); }

  py::buffer_info Cast() const {
    auto dim_vec = pd::vectorize(self_.dims());
    std::vector<size_t> dims_outside;
    std::vector<size_t> strides;
    dims_outside.resize(dim_vec.size());
    strides.resize(dim_vec.size());

    size_t prod = 1;
    for (size_t i = dim_vec.size(); i != 0; --i) {
      dims_outside[i - 1] = (size_t)dim_vec[i - 1];
      strides[i - 1] = sizeof(float) * prod;
      prod *= dims_outside[i - 1];
    }

    return py::buffer_info(self_.mutable_data<T>(self_.place()),
                           sizeof(T),
                           py::format_descriptor<T>::format(),
                           (size_t)pd::arity(self_.dims()),
                           dims_outside,
                           strides);
  }
};

template <bool less, size_t I, typename... ARGS>
struct CastToPyBufferImpl;

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<false, I, ARGS...> {
  py::buffer_info operator()(pd::Tensor& tensor) {
    PADDLE_THROW("This type of tensor cannot be expose to Python");
    return py::buffer_info();
  }
};

template <size_t I, typename... ARGS>
struct CastToPyBufferImpl<true, I, ARGS...> {
  using CUR_TYPE = typename std::tuple_element<I, std::tuple<ARGS...>>::type;
  py::buffer_info operator()(pd::Tensor& tensor) {
    TensorToPyBuffer<CUR_TYPE> cast_object(tensor);
    if (cast_object.CanCast()) {
      return cast_object.Cast();
    } else {
      constexpr bool less = I + 1 < std::tuple_size<std::tuple<ARGS...>>::value;
      return CastToPyBufferImpl<less, I + 1, ARGS...>()(tensor);
    }
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i + 1 != vec.size()) {
      os << ", ";
    }
  }
  return os;
}

py::buffer_info CastToPyBuffer(pd::Tensor& tensor) {
  auto buffer_info = CastToPyBufferImpl<true, 0, float, int>()(tensor);
  return buffer_info;
}

template <typename T>
void PyTensorSet(
    pd::Tensor& self,
    py::array_t<T, py::array::c_style | py::array::forcecast> array) {
  std::vector<int> dims;
  dims.reserve(array.ndim());
  for (size_t i = 0; i < array.ndim(); ++i) {
    dims.push_back((int)array.shape()[i]);
  }

  self.set_dims(pd::make_ddim(dims));
  auto* dst = self.mutable_data<T>(paddle::platform::CPUPlace());
  std::memcpy(dst, array.data(), sizeof(T) * array.size());
}

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of Paddle Paddle");

  py::class_<paddle::platform::Place>(
      m, "Place", R"DOC(Device Place Class.)DOC")
      .def("__str__",
           [](const paddle::platform::Place& self) {
             return boost::apply_visitor(PlaceDebugString(), self);
           })
      .def("is_gpu",
           [](const paddle::platform::Place& self) {
             return paddle::platform::is_gpu_place(self);
           })
      .def("is_cpu", [](const paddle::platform::Place& self) {
        return paddle::platform::is_cpu_place(self);
      });

  py::class_<pd::Tensor>(m, "Tensor", py::buffer_protocol())
      .def("get_place", &pd::Tensor::place)
      .def_buffer([](pd::Tensor& self) -> py::buffer_info {
        PADDLE_ENFORCE(paddle::platform::is_cpu_place(self.place()),
                       "Only CPU tensor can cast to numpy array");
        return CastToPyBuffer(self);
      })
      .def("get_dims",
           [](const pd::Tensor& self) { return pd::vectorize(self.dims()); })
      .def("set_dims",
           [](pd::Tensor& self, const std::vector<int>& dim) {
             self.set_dims(pd::make_ddim(dim));
           })
      .def("alloc_float",
           [](pd::Tensor& self) {
             self.mutable_data<float>(paddle::platform::CPUPlace());
           })
      .def("alloc_int",
           [](pd::Tensor& self) {
             self.mutable_data<int>(paddle::platform::CPUPlace());
           })
      .def("set", PyTensorSet<float>)
      .def("set", PyTensorSet<int>);

  py::class_<pd::Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const pd::Variable& var) { return var.IsType<int>(); })
      .def("set_int",
           [](pd::Variable& var, int val) -> void {
             *var.GetMutable<int>() = val;
           })
      .def("get_int",
           [](const pd::Variable& var) -> int { return var.Get<int>(); })
      .def("get_tensor",
           [](pd::Variable& self) -> pd::Tensor* {
             return self.GetMutable<pd::Tensor>();
           },
           py::return_value_policy::reference);

  py::class_<pd::Scope, std::shared_ptr<pd::Scope>>(m, "Scope")
      .def(py::init<const std::shared_ptr<pd::Scope>&>())
      .def("get_var",
           &pd::Scope::GetVariable,
           py::return_value_policy::reference)
      .def("create_var",
           &pd::Scope::CreateVariable,
           py::return_value_policy::reference);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<std::string> {
    auto& protos = pd::OpRegistry::protos();
    std::vector<std::string> ret_values;
    for (auto it = protos.begin(); it != protos.end(); ++it) {
      PADDLE_ENFORCE(it->second.IsInitialized(),
                     "OpProto must all be initialized");
      ret_values.emplace_back();
      PADDLE_ENFORCE(it->second.SerializeToString(&ret_values.back()),
                     "Serialize OpProto Error. This could be a bug of Paddle.");
    }
    return ret_values;
  });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", pd::OperatorBase::EMPTY_VAR_NAME)
      .def("temp", pd::OperatorBase::TMP_VAR_NAME);

  py::class_<pd::OperatorBase, pd::OperatorPtr>(m, "Operator")
      .def("__str__", &pd::OperatorBase::DebugString)
      .def_static("create", [](const std::string& protobin) {
        pd::OpDesc desc;
        PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                       "Cannot parse user input to OpDesc");
        PADDLE_ENFORCE(desc.IsInitialized(),
                       "User OpDesc is not initialized, reason %s",
                       desc.InitializationErrorString());
        return pd::OpRegistry::CreateOp(desc);
      });

  return m.ptr();
}
