# gin_overlap —— sharding stage1 梯度通信的原语：用 NCCL GIN 单边写分阶段完成
# 每个梯度桶的 reduce_scatter（说明见 gin_engine.h 文件头）。
#
# 是否编译这个目录由上一级 CMakeLists.txt 的门控决定。这里只为 sm_90 及以上
# 生成代码：去掉 8.x 及以下的 gencode。
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# - gin_engine.cc：host 侧编排。
# - paddle_comm.cc / paddle_tensor.cc：取 ProcessGroup 的 ncclComm、把 window
#   切片包成 paddle::Tensor。
# - kernels/：设备侧 kernel，以及放行层 release.cu（含尾部线程，是 host 代码）。
set(GIN_OVERLAP_SRCS
    gin_engine.cc
    paddle_comm.cc
    paddle_tensor.cc
    kernels/reduce.cu
    kernels/put.cu
    kernels/signal.cu
    kernels/scale.cu
    kernels/release.cu)

# NCCL 头文件（nccl.h / nccl_device.h）来自 cmake/nccl.cmake 设置的 include 路径。
# 不链接 libnccl 和 libcuda：host 侧的 NCCL / 驱动调用走 phi::dynload（见
# gin_dynload.h）。process_group_nccl：paddle_comm.cc 要用
# ProcessGroupNCCL::GetOrCreateCommContext。
cc_library(
  gin_overlap
  SRCS ${GIN_OVERLAP_SRCS}
  DEPS phi common process_group_nccl)

# Paddle 的编译选项带 -Werror；下面几类告警在这个目标上关掉。
target_compile_options(
  gin_overlap PRIVATE -Wno-unused-variable -Wno-unused-but-set-variable
                      -Wno-sign-compare -Wno-reorder)
