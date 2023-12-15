#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/runtime/custom_function.h"

#include "paddle/cinn/runtime/sycl/sycl_module.h"

CINN_REGISTER_HELPER(cinn_sycl_host_api) {
  using cinn::runtime::Sycl::cinn_call_sycl_kernel;
  REGISTER_EXTERN_FUNC_HELPER(cinn_call_sycl_kernel,
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

  return true;
}