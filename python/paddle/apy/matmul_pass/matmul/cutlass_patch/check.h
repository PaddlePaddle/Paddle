#pragma once

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CHECK_CUDA(func)                                                      \
  {                                                                           \
    cudaError_t err = func;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "CUDA error(" << err << "), " << cudaGetErrorString(err)   \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }