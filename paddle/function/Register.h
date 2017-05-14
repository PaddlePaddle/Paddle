#pragma once
#include "details/Register.h"
#include "paddle/topology/Attribute.h"
namespace paddle {
namespace function {

Function createFunction(const topology::Function& conf);

#define BEGIN_REGISTER_FUNCTION(name, __func__, __attr_type__) \
  static paddle::InitFunction __init_##name##__([] {\
    paddle::topology::meta::FunctionMeta::registerFuncMeta(\
      #name, [](paddle::topology::meta::FunctionMetaPtr& func) {\
  do {\
  function::details::FunctionRegister reg(func);\
  reg.reg<__attr_type__, DEVICE_TYPE_CPU>(__func__<DEVICE_TYPE_CPU>);\
  reg.reg<__attr_type__, DEVICE_TYPE_GPU>(__func__<DEVICE_TYPE_GPU>);\
} while(0);\
  __attr_type__::registerFunctionAttribute(func);\
  (*func)
#define END_REGISTER_FUNCTION() \
  return paddle::Error();       \
  }).check();                   \
  });
}  // namespace function
}  // namespace paddle
