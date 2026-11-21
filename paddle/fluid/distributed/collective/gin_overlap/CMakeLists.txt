# gin_overlap: primitives for sharding-stage1 gradient communication. The
# reduce_scatter of each gradient bucket is done in stages with NCCL GIN
# one-sided puts (described in the gin_engine.h file header).
#
# Whether this directory is built is decided by the gate in the parent
# CMakeLists.txt. Code is generated only for sm_90 and newer: the gencode
# flags for 8.x and older are removed.
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# - gin_engine.cc: host-side orchestration.
# - paddle_comm.cc / paddle_tensor.cc: get the ncclComm of a ProcessGroup, and
#   wrap window slices as paddle::Tensor.
# - kernels/: device kernels. release.cc: the release layer (host code).
set(GIN_OVERLAP_SRCS
    gin_engine.cc
    paddle_comm.cc
    paddle_tensor.cc
    kernels/reduce.cu
    kernels/put.cu
    kernels/signal.cu
    kernels/scale.cu
    release.cc)

# NCCL headers (nccl.h / nccl_device.h) come from the include path set by
# cmake/nccl.cmake. libnccl and libcuda are not linked: host-side NCCL /
# driver calls go through phi::dynload (see gin_dynload.h).
# process_group_nccl: paddle_comm.cc uses
# ProcessGroupNCCL::GetOrCreateCommContext.
cc_library(
  gin_overlap
  SRCS ${GIN_OVERLAP_SRCS}
  DEPS phi common process_group_nccl)

# Paddle compiles with -Werror; the warning classes below are off for this
# target.
target_compile_options(
  gin_overlap PRIVATE -Wno-unused-variable -Wno-unused-but-set-variable
                      -Wno-sign-compare -Wno-reorder)
