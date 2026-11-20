# gin_overlap —— sharding stage1 的梯度通信：每个梯度桶本来一次 NCCL
# reduce_scatter，这里改成用 NCCL 的 GIN 单边写（put）分几步做，好处是每一步都能
# 塞进反向里 DeepEP 不占网 / 不占 SM 的空档。放行时机（哪个阶段挑哪个窗口）不在
# 这里，在 PaddleFleet 的 gin_overlap/scheduler.py；这边只有通信原语。
#
# 编哪些 cubin：和 DeepEP 一样只对 sm_90 及以上有意义（gin_overlap 就是为了和
# DeepEP 并排跑），而 NCCL 的设备 API 本身没有 arch 下限。要不要进这个目录由上
# 一级的 arch / NCCL 版本门控决定（../CMakeLists.txt）；进来之后把 8.x 及以下的
# gencode 去掉，免得为用不到的卡多编一份device code —— 同 deep_ep/CMakeLists.txt
# 里那一行 REGEX REPLACE。
string(REGEX REPLACE "-gencode arch=compute_[0-8][0-9],code=sm_[0-8][0-9]" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# gin_engine.cc 是 host 侧的编排；paddle_comm.cc / paddle_tensor.cc 是和 Paddle
# 对接的那两小块（取 ProcessGroup 的 ncclComm、把 window 切片包成
# paddle::Tensor）；kernels/ 下是设备侧的那几个 kernel 和放行层（release.cu 里
# 还有尾部线程，是 host 代码，但和放行判断在同一个文件里）。
set(GIN_OVERLAP_SRCS
    gin_engine.cc
    paddle_comm.cc
    paddle_tensor.cc
    kernels/reduce.cu
    kernels/put.cu
    kernels/signal.cu
    kernels/scale.cu
    kernels/release.cu)

# NCCL 的头（nccl.h / nccl_device.h）由 cmake/nccl.cmake 的
# include_directories(BEFORE ${NCCL_INCLUDE_DIR}) 提供；libnccl 和 libcuda 都不
# 链，host 侧的 NCCL / 驱动调用走 phi::dynload（见 gin_dynload.h）。
# process_group_nccl：paddle_comm.cc 要 ProcessGroupNCCL::GetOrCreateCommContext。
cc_library(
  gin_overlap
  SRCS ${GIN_OVERLAP_SRCS}
  DEPS phi common process_group_nccl)

# 这套代码在 PaddleFleet 里是按 -w 编的，Paddle 的 CUDA flags 带 -Wall -Wextra
# -Werror（cmake/flags.cmake 里 LINUX 那一行），所以照 deep_ep 的做法把几类纯风
# 格告警降级，别让它们变成编译失败。
target_compile_options(
  gin_overlap PRIVATE -Wno-unused-variable -Wno-unused-but-set-variable
                      -Wno-sign-compare -Wno-reorder)
