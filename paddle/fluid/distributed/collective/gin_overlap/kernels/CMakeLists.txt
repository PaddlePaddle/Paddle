# Device kernels of the NCCL one-sided primitives of ProcessGroupNCCL. Whether
# this directory is built is decided by the gate in the parent CMakeLists.txt.
# Code is generated only for sm_90 and newer: the gencode flags for 8.x and
# older are removed.
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# NCCL headers (nccl.h / nccl_device.h) come from the include path set by
# cmake/nccl.cmake. The kernels only use the device (inline) part of the NCCL
# device API; host-side NCCL calls go through phi::dynload.
cc_library(
  gin_overlap_comm_kernels
  SRCS comm.cu
  DEPS phi common)
