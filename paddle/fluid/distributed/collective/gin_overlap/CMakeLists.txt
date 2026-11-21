# gin_overlap: primitives for sharding-stage1 gradient communication. The
# reduce_scatter of each gradient bucket is done in stages with NCCL GIN
# one-sided puts (described in the gin_engine.h file header).
#
# Whether this directory is built is decided by the gate in the parent
# CMakeLists.txt. Code is generated only for sm_90 and newer: the gencode
# flags for 8.x and older are removed.
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(GIN_OVERLAP_SRCS kernels/reduce.cu kernels/put.cu kernels/signal.cu
                     kernels/scale.cu)

cc_library(
  gin_overlap
  SRCS ${GIN_OVERLAP_SRCS}
  DEPS phi common)

# Paddle compiles with -Werror; the warning classes below are off for the C++
# sources of this target (not passed to nvcc).
target_compile_options(
  gin_overlap
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-but-set-variable>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
          $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>)
