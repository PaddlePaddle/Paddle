#include <paddle/cinn/runtime/sycl/sycl_module.h>
#include <paddle/cinn/runtime/sycl/sycl_backend_api.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <glog/raw_logging.h>

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/cinn/utils/profiler.h"

namespace cinn {
namespace runtime {
namespace Sycl {

SYCLModule::SYCLModule(const std::string& source_code, std::string& shared_library, Kind kind)
    : source_code_(source_code), shared_library_(shared_library), kind_(kind) {
  CHECK(!shared_library.empty());
}

SYCLModule::~SYCLModule() {
  std::cout << "destructor for SYCLModule" << std::endl;
  if (so_handler_ != nullptr) {
    //dlclose(so_handler_);
  }
}

void* SYCLModule::GetFunction(const std::string& func_name) {
  if (so_handler_ == nullptr) {
    so_handler_ = dlopen(shared_library_.c_str(), RTLD_NOW | RTLD_GLOBAL);
  }
  VLOG(5) << "getting function " << func_name;
  CHECK(so_handler_ != nullptr) << "ERROR:" << dlerror();
  void (*kernel_func)(sycl::queue& Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) =
      (void (*)(sycl::queue& Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args))dlsym(
          so_handler_, func_name.c_str());
  CHECK(kernel_func != nullptr) << "ERROR:" << dlerror() << ":dlsym\n";
  return reinterpret_cast<void*>(kernel_func);
}

void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream) {
  VLOG(3) << "cinn_call_sycl_kernel, grid_dim={" << grid_x << ", " << grid_y << ", " << grid_z << "}, block_dim={"
          << block_x << ", " << block_y << ", " << block_z << "}, num_args=" << num_args << ", stream=" << stream;

  std::vector<void*> kernel_args;
  {
    cinn::utils::RecordEvent record_run("prepare_args", cinn::utils::EventType::kInstruction);
    kernel_args.reserve(num_args);
    cinn_pod_value_t* args = static_cast<cinn_pod_value_t*>(v_args);
    for (int idx = 0; idx < num_args; ++idx) {
      if (args[idx].type_code() == ::cinn_type_code<cinn_buffer_t*>()) {
        kernel_args.emplace_back(&((cinn_buffer_t*)(args[idx]))->memory);
      } else {
        kernel_args.emplace_back((args[idx].data_addr()));
      }
    }
  }

  {
    cinn::utils::RecordEvent record_run("syclLaunchKernel", cinn::utils::EventType::kInstruction);
    void (*kernel_func)(sycl::queue& Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args) =
        (void (*)(sycl::queue& Q, sycl::range<3> k0_dimGrid, sycl::range<3> k0_dimBlock, void** void_args))(kernel_fn);
    sycl::queue* Queue = SYCLBackendAPI::Global()->get_now_queue();
    sycl::range<3> Grid(grid_z, grid_y, grid_x);
    sycl::range<3> Block(block_z, block_y, block_x);
    // need malloc_shared
    // std::cout << "kernel args :" << (float* )(*(void **)(kernel_args[0]))[0] << std::endl;
    kernel_func(*Queue, Grid, Block, kernel_args.data());
  }
}

}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn