// hipcc hipcc_test.cc -o hipcc_test
#include<iostream>
#include <hip/hip_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>

template<class BidirectionalIterator>
inline std::string format_range(const BidirectionalIterator begin, const BidirectionalIterator end)
{
    std::stringstream sstream;
    sstream << "[ ";
    for(auto it = begin; it != end; ++it)
    {
        sstream << *it;
        if(it != std::prev(end))
        {
            sstream << ", ";
        }
    }
    sstream << " ]";
    return sstream.str();
}

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(-1);                                                     \
        }                                                                                   \
    }

int main(){
  // hipcc compile command
  std::string options = "hipcc -O3 --genco";
  // arch
  // Get device properties from the first device available.
  hipDeviceProp_t        props;
  constexpr unsigned int device_id = 0;
  hipGetDeviceProperties(&props, device_id);
  if(props.gcnArchName[0])
  {
      options += std::string(" --offload-arch=") + props.gcnArchName;
  }
  // include path
  options += " -I /workspace/Paddle/paddle/cinn/backends/hip/hipcc_test";
  options += " -o kernel.hsaco";
  options += " kernel.cc";
  std::cout << options << std::endl;
  system(options.c_str());

  // Now we launch the kernel on the device.

  // Total number of float elements in each device vector.
  constexpr unsigned int size = 4096;

  // Total number of bytes to allocate for each device vector.
  constexpr size_t size_bytes = size * sizeof(float);

  // Number of threads per kernel block.
  constexpr unsigned int block_size = 128;

  // Number of blocks per kernel grid, calculated as ceil(size/block_size).
  constexpr unsigned int grid_size = (size + block_size - 1) / block_size;

  // Constant value 'a' to be used in the expression 'a*x+y'.
  constexpr float a = 5.1f;

  // Allocate x vector in host and fill it with increasing sequence 1, 2, 3, 4, ... .
  std::vector<float> x(size);
  std::iota(x.begin(), x.end(), 1.f);

  // Allocate y vector in host and fill it with increasing sequence 2, 4, 6, 8, ... .
  std::vector<float> y(x);
  std::for_each(y.begin(), y.end(), [](float& f) { f = 2 * f; });

  // Allocate vectors in device and copy from host to device memory.
  float* d_x{};
  float* d_y{};
  HIP_CHECK(hipMalloc(&d_x, size_bytes));
  HIP_CHECK(hipMalloc(&d_y, size_bytes));
  HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));

  // Load the HIP module corresponding to the compiled binary into the current context.
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, "kernel.hsaco"));

  // Extract SAXPY kernel from module into a function object.
  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy_kernel"));

  // Create and fill array with kernel arguments.
  size_t offset    = 0;
  char   args[256] = {};

  *(reinterpret_cast<float*>(&args[offset])) = a;
  offset += sizeof(a);
  offset += 4; // aligning fix for CUDA executions
  *(reinterpret_cast<float**>(&args[offset])) = d_x;
  offset += sizeof(d_x);
  *(reinterpret_cast<float**>(&args[offset])) = d_y;
  offset += sizeof(d_y);
  *(reinterpret_cast<unsigned int*>(&args[offset])) = size;
  offset += sizeof(size);

  // Create array with kernel arguments and its size.
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &offset,
                    HIP_LAUNCH_PARAM_END};

  std::cout << "Calculating y[i] = a * x[i] + y[i] over " << size << " elements." << std::endl;

  // Launch the kernel on the NULL stream and with the above configuration.
  HIP_CHECK(hipModuleLaunchKernel(kernel,
                                  grid_size,
                                  1,
                                  1,
                                  block_size,
                                  1,
                                  1,
                                  0,
                                  nullptr,
                                  nullptr,
                                  (void**)&config));

  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError())

  // Copy results from device to host.
  HIP_CHECK(hipMemcpy(y.data(), d_y, size_bytes, hipMemcpyDeviceToHost));

  // Free device memory.
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));

  // Unload module.
  HIP_CHECK(hipModuleUnload(module));

  // Print the first few elements of the results for validation.
  constexpr size_t elements_to_print = 10;
  std::cout << "First " << elements_to_print << " elements of the results: "
            << format_range(y.begin(), y.begin() + elements_to_print) << std::endl;

  return 0;
}