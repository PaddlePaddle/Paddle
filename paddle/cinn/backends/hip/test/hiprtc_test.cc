// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// hipcc hiprtc_test.cc -o hiprtc_test
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "glog/logging.h"

template <class BidirectionalIterator>
inline std::string format_range(const BidirectionalIterator begin,
                                const BidirectionalIterator end) {
  std::stringstream sstream;
  sstream << "[ ";
  for (auto it = begin; it != end; ++it) {
    sstream << *it;
    if (it != std::prev(end)) {
      sstream << ", ";
    }
  }
  sstream << " ]";
  return sstream.str();
}

#define HIP_CHECK(condition)                                              \
  {                                                                       \
    const hipError_t error = condition;                                   \
    if (error != hipSuccess) {                                            \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error) \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;  \
      std::exit(-1);                                                      \
    }                                                                     \
  }

#define HIPRTC_CALL(func)                                            \
  {                                                                  \
    auto status = func;                                              \
    if (status != HIPRTC_SUCCESS) {                                  \
      std::cerr << "NVRTC Error : " << hiprtcGetErrorString(status); \
    }                                                                \
  }

bool use_bit_code = false;  // error when use_bit_code = true

int main() {
  hiprtcProgram prog;
  // Vector containing example header names.
  std::vector<const char*> header_names;
  header_names.push_back("test_header.h");

  // Vector containing example names to be included in the program.
  std::vector<const char*> header_sources;
  header_sources.push_back(
      "#ifndef HIPRTC_TEST_HEADER_H\n#define HIPRTC_TEST_HEADER_H\ntypedef "
      "float real;\n#endif //HIPRTC_TEST_HEADER_H\n");
  header_sources.push_back(
      "#ifndef HIPRTC_TEST_HEADER1_H\n#define HIPRTC_TEST_HEADER1_H\ntypedef "
      "float* "
      "realptr;\n#endif //HIPRTC_TEST_HEADER1_H\n");

  std::string kernel_code =
      R"ROC(
    #include "test_header.h"
extern "C"
__global__ void saxpy_kernel(const real a, const realptr d_x, realptr d_y, const unsigned int size)
{
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_idx < size)
    {
        d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
    }
}
)ROC";

  // Create program.
  hiprtcCreateProgram(&prog,
                      kernel_code.c_str(),
                      "kernel.cu",
                      header_sources.size(),
                      header_sources.data(),
                      header_names.data());

  // Get device properties from the first device available.
  hipDeviceProp_t props;
  constexpr unsigned int device_id = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device_id));

  std::vector<const char*> options;

  std::string arch_option;
  if (props.gcnArchName[0]) {
    arch_option = std::string("--gpu-architecture=") + props.gcnArchName;
    options.push_back(arch_option.c_str());
  }
  if (use_bit_code) {
    options.push_back("-fgpu-rdc");
  }

  // Compile program in runtime. Parameters are the program, number of options
  // and array with options.
  const hiprtcResult compile_result{
      hiprtcCompileProgram(prog, options.size(), options.data())};

  // Get the size of the log (possibly) generated during the compilation.
  size_t log_size;
  hiprtcGetProgramLogSize(prog, &log_size);

  // If the compilation generated a log, print it.
  if (log_size) {
    std::string log(log_size, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    VLOG(6) << log << std::endl;
  }

  // If the compilation failed, say so and exit.
  if (compile_result != HIPRTC_SUCCESS) {
    VLOG(6) << "Error: compilation failed." << std::endl;
    return EXIT_FAILURE;
  }
  size_t bi_size;
  std::string data;
  if (use_bit_code) {
    HIPRTC_CALL(hiprtcGetBitcodeSize(prog, &bi_size));
    data.resize(bi_size);
    HIPRTC_CALL(hiprtcGetBitcode(prog, &data[0]));
  } else {
    // Get the size (in number of characters) of the binary compiled from the
    // program.
    hiprtcGetCodeSize(prog, &bi_size);
    // Store compiled binary as a vector of characters.
    data.resize(bi_size);
    hiprtcGetCode(prog, &data[0]);
  }
  // Destroy program object.
  hiprtcDestroyProgram(&prog);

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

  // Allocate x vector in host and fill it with increasing sequence 1, 2, 3, 4,
  // ... .
  std::vector<float> x(size);
  std::iota(x.begin(), x.end(), 1.f);

  // Allocate y vector in host and fill it with increasing sequence 2, 4, 6, 8,
  // ... .
  std::vector<float> y(x);
  std::for_each(y.begin(), y.end(), [](float& f) { f = 2 * f; });

  // Allocate vectors in device and copy from host to device memory.
  float* d_x{};
  float* d_y{};
  HIP_CHECK(hipMalloc(&d_x, size_bytes));
  HIP_CHECK(hipMalloc(&d_y, size_bytes));
  HIP_CHECK(hipMemcpy(d_x, x.data(), size_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_y, y.data(), size_bytes, hipMemcpyHostToDevice));

  // Compilation with parameters
  const size_t jit_num_options = 5;
  std::vector<hipJitOption> jit_options(jit_num_options);
  std::vector<void*> jit_opt_vals(jit_num_options);

  // set up size of compilation log buffer
  jit_options[0] = hipJitOptionErrorLogBufferSizeBytes;
  size_t log_buffer_size = 1024;
  jit_opt_vals[0] = reinterpret_cast<void*>(log_buffer_size);

  // set up pointer to the compilation log buffer
  jit_options[1] = hipJitOptionErrorLogBuffer;
  std::vector<char> log_buffer(log_buffer_size, '\0');
  jit_opt_vals[1] = log_buffer.data();

  int value = 1;
  // Specifies whether to create debug information in output (-g)
  jit_options[2] = hipJitOptionGenerateDebugInfo;
  jit_opt_vals[2] = reinterpret_cast<void*>(value);

  // Generate verbose log messages
  jit_options[3] = hipJitOptionLogVerbose;
  jit_opt_vals[3] = reinterpret_cast<void*>(value);

  // Generate line number information (-lineinfo)
  jit_options[4] = hipJitOptionGenerateLineInfo;
  jit_opt_vals[4] = reinterpret_cast<void*>(value);
  // Load the HIP module corresponding to the compiled binary into the current
  // context.
  hipModule_t module;
  HIP_CHECK(hipModuleLoadDataEx(&module,
                                data.c_str(),
                                jit_num_options,
                                jit_options.data(),
                                jit_opt_vals.data()));

  // Extract SAXPY kernel from module into a function object.
  hipFunction_t kernel;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy_kernel"));

  // Create and fill array with kernel arguments.
  size_t offset = 0;
  char args[256] = {};

  *(reinterpret_cast<float*>(&args[offset])) = a;
  offset += sizeof(a);
  offset += 4;  // aligning fix for CUDA executions
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

  VLOG(6) << "Calculating y[i] = a * x[i] + y[i] over " << size << " elements."
          << std::endl;

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
                                  static_vast<void**>(&config));

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
  VLOG(6) << "First " << elements_to_print << " elements of the results: "
            << format_range(y.begin(), y.begin() + elements_to_print)
            << std::endl;

  return 0;
}
