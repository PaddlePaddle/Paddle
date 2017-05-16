#pragma once
#include "details/Register.h"
#include "paddle/topology/Attribute.h"
#include "paddle/topology/Tensor.h"
namespace paddle {
namespace function {

Function createFunction(const topology::Function& conf);

#define BEGIN_REGISTER_FUNCTION(name, __func__, __attr_type__)                 \
  static void __init_##name##_function__(                                      \
      paddle::topology::meta::FunctionMetaPtr meta);                           \
  static paddle::InitFunction __init_##name##__([] {                           \
    auto func = paddle::topology::meta::FunctionMeta::registerFuncMeta(#name); \
    function::details::FunctionRegister reg(func);                             \
    reg.reg<__attr_type__, DEVICE_TYPE_CPU>(__func__<DEVICE_TYPE_CPU>);        \
    reg.reg<__attr_type__, DEVICE_TYPE_GPU>(__func__<DEVICE_TYPE_GPU>);        \
    __attr_type__::registerFunctionAttribute(func);                            \
    __init_##name##_function__(func);                                          \
  });                                                                          \
  static void __init_##name##_function__(                                      \
      paddle::topology::meta::FunctionMetaPtr func) {                          \
  (*func)

#define END_REGISTER_FUNCTION() }
}  // namespace function
}  // namespace paddle
