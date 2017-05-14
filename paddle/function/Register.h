#pragma once
#include "details/Register.h"
#include "paddle/topology/Attribute.h"
namespace paddle {
namespace function {

Function createFunction(const topology::Function& conf);

#define BEGIN_REGISTER_FUNCTION(name, __func__) \
  static paddle::InitFunction __init_##name##__([] {\
    paddle::topology::meta::FunctionMeta::registerFuncMeta(\
      #name, [](paddle::topology::meta::FunctionMetaPtr& func) {\
  do {\
  function::details::FunctionRegister reg(func);\
  reg.addCPUFunction(__func__<DEVICE_TYPE_CPU>);\
  reg.addGPUFunction(__func__<DEVICE_TYPE_GPU>);\
} while(0);\
  func->addAttribute<bool>("useGPU", "is this function use gpu or not")\
  .mustSet();
#define END_REGISTER_FUNCTION() \
  return paddle::Error();       \
  }).check();                   \
  });
}  // namespace function
}  // namespace paddle
