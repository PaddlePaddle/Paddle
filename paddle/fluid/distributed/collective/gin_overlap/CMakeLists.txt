# gin_overlap: primitives for sharding-stage1 gradient communication. The
# reduce_scatter of each gradient bucket is done in stages with NCCL GIN
# one-sided puts (described in the gin_engine.h file header).
#
# Whether this directory is built is decided by the gate in the parent
# CMakeLists.txt. Code is generated only for sm_90 and newer: the gencode
# flags for 8.x and older are removed.
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# - gin_engine.cc：host 侧编排。
# - paddle_comm.cc / paddle_tensor.cc：取 ProcessGroup 的 ncclComm、把 window
#   切片包成 paddle::Tensor。
set(GIN_OVERLAP_SRCS
    gin_engine.cc
    paddle_comm.cc
    paddle_tensor.cc
    kernels/reduce.cu
    kernels/put.cu
    kernels/signal.cu
    kernels/scale.cu)

# NCCL 头文件（nccl.h / nccl_device.h）来自 cmake/nccl.cmake 设置的 include 路径。
# 不链接 libnccl 和 libcuda：host 侧的 NCCL / 驱动调用走 phi::dynload（见
# gin_dynload.h）。process_group_nccl：paddle_comm.cc 要用
# ProcessGroupNCCL::GetOrCreateCommContext。
cc_library(
  gin_overlap
  SRCS ${GIN_OVERLAP_SRCS}
  DEPS phi common process_group_nccl)

# Paddle compiles with -Werror; the warning classes below are off for the C++
# sources of this target (not passed to nvcc).
target_compile_options(
  gin_overlap
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-but-set-variable>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>)
