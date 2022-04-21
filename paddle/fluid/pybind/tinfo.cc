//
// Created by linxu on 2022/4/21.
//

#include "tinfo.h"

void BindFInfoVarDsec(pybind11::module *m){
  pybind11::class_<pd::VarDesc> finfo_var_desc(*m, "VarDesc", "");
  finfo_var_desc.def(pybind11::init<const std::string &>())
      .def("bits", &pd::Tinfo::Bits)
      .def("eps", &pd::Tinfo::Eps)
      .def("min", &pd::Tinfo::Min)
      .def("max", &pd::Tinfo::Max)
      .def("tiny", &pd::Tinfo::Tiny)
      .def("resolution", &pd::Tinfo::Resolution)
}

void BindFInfoVarDsec(pybind11::module *m){
  pybind11::class_<pd::VarDesc> finfo_var_desc(*m, "VarDesc", "");
  finfo_var_desc.def(pybind11::init<const std::string &>())
      .def("bits", &pd::Tinfo::Bits)
      .def("min", &pd::Tinfo::Min)
      .def("max", &pd::Tinfo::Max)
}