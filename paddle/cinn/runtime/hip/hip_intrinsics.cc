#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/runtime/custom_function.h"

#include "paddle/cinn/runtime/hip/hip_module.h"
using cinn::runtime::hip::cinn_call_hip_kernel;
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
using cinn::backends::GlobalSymbolRegistry;
// todo
//#include "paddle/cinn/runtime/hip/hip_backend_api.h"
//using cinn::runtime::hip::HIPBackendAPI;

CINN_REGISTER_HELPER(cinn_hip_host_api) {
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_hip_kernel,
                              cinn::common::DefaultHostTarget())
      .SetRetType<void>()
      .AddInputType<void *>()  // kernel_fn
      .AddInputType<void *>()  // args
      .AddInputType<int>()     // num_args
      .AddInputType<int>()     // grid_x
      .AddInputType<int>()     // grid_y
      .AddInputType<int>()     // grid_z
      .AddInputType<int>()     // block_x
      .AddInputType<int>()     // block_y
      .AddInputType<int>()     // block_z
      .AddInputType<void *>()  // stream
      .End();
  //todo
  //GlobalSymbolRegistry::Global().RegisterFn("backend_api.hip", reinterpret_cast<void*>(HIPBackendAPI::Global()));
  return true;
}