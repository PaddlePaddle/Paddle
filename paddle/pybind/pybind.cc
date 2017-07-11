#include <paddle/framework/scope.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace pd = paddle::framework;

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of Paddle Paddle");

  py::class_<pd::Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const pd::Variable& var) { return var.IsType<int>(); })
      .def("set_int",
           [](pd::Variable& var, int val) -> void {
             *var.GetMutable<int>() = val;
           })
      .def("get_int",
           [](const pd::Variable& var) -> int { return var.Get<int>(); });

  py::class_<pd::Scope, std::shared_ptr<pd::Scope>>(m, "Scope")
      .def(py::init<const std::shared_ptr<pd::Scope>&>())
      .def("get_var",
           &pd::Scope::GetVariable,
           py::return_value_policy::reference)
      .def("create_var",
           &pd::Scope::CreateVariable,
           py::return_value_policy::reference);

  return m.ptr();
}