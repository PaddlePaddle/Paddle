#pragma once

#include <mutex>  // NOLINT
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

namespace cinn {
namespace runtime {
namespace Sycl {
/**
 * The SYCL module, helps to compile SYCL codes and fetch symbols.
 * Currently, it is a wrapper of NVRTC.
 */
class SYCLModule {
 public:
  enum class Kind {
    so = 0,
  };
  SYCLModule(const std::string& source_code, std::string& shared_library, Kind kind);
  void* GetFunction(const std::string& func_name);
  ~SYCLModule();

 private:
  //! sycl source code
  std::string source_code_;

  std::string shared_library_;
  // handler of the shared library
  void* so_handler_ = nullptr;
  //! Kind of the input.
  Kind kind_;
  std::mutex mutex_;
};

/**
 * Call a SYCL compiled kernel.
 *
 * @param kernel_fn the func pointer.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream);

}  // namespace Sycl
}  // namespace runtime
}  // namespace cinn