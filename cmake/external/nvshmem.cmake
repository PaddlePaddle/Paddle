# 假设预编译的 nvshmem 库和头文件位于以下路径
set(PREBUILT_NVSHMEM_HOME /root/paddlejob/workspace/env_run/lidong/nvshmem)
set(NVSHMEM_INCLUDE_DIR "${PREBUILT_NVSHMEM_HOME}/include")
set(NVSHMEM_LIB "${PREBUILT_NVSHMEM_HOME}/lib/libnvshmem.a")
set(NVSHMEM_INSTALL_DIR ${PREBUILT_NVSHMEM_HOME})
set(NVSHMEM_BOOTSTRAP_UID_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_uid.so)
set(NVSHMEM_BOOTSTRAP_MPI_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_mpi.so)
set(NVSHMEM_BOOTSTRAP_PMI_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi.so)
set(NVSHMEM_BOOTSTRAP_PMI2_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_bootstrap_pmi2.so)
set(NVSHMEM_TRANSPORT_IBRC_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibrc.so.3)
set(NVSHMEM_TRANSPORT_IBGDA_LIB
    ${NVSHMEM_INSTALL_DIR}/lib/nvshmem_transport_ibgda.so.3)

# 包含头文件目录
include_directories(${NVSHMEM_INCLUDE_DIR})

# 导入预编译的库
add_library(nvshmem STATIC IMPORTED GLOBAL)
set_property(TARGET nvshmem PROPERTY IMPORTED_LOCATION ${NVSHMEM_LIB})

# 如果需要，可以添加其他依赖项（但在这个场景中，我们假设没有其他依赖项需要构建）
# 注意：extern_nvshmem 目标不再需要，因为它与源代码构建相关

# 添加定义以指示 PaddlePaddle 使用 NVSHMEM
add_definitions(-DPADDLE_WITH_NVSHMEM)

message(STATUS "Using loaded nvshmem")