#pragma once

#define CHECK_HIP(func)                                                       \
  {                                                                           \
    hipError_t err = func;                                                    \
    if (err != hipSuccess) {                                                  \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "HIP error(" << err << "), " << hipGetErrorString(err)     \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }
