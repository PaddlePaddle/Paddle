#pragma once
#include <cuda_fp16.h>
#include "conv2d_all.h"
#include <vector>




namespace phi {
  namespace fusion {


// This two functions calculate diff of cutlass output and baseline output
// We recommand use conv2d_diff_gpu bacause gpu is more fast than cpu
// return value is the max diff between cutlass and baseline
float conv2d_diff_cpu(COMMON_CONV_PARAMS, const half* residual, std::string op_name);
float conv2d_diff_gpu(COMMON_CONV_PARAMS, const half* residual, std::string op_name);

  } // namespace fusion
}// namespace phi

