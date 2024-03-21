include(ExternalProject)

set(CCCL_PATH
    "${THIRD_PARTY_PATH}/cccl"
    CACHE STRING "A path setting for external_cccl path.")
set(CCCL_PREFIX_DIR ${CCCL_PATH})
set(CCCL_SOURCE_DIR ${PADDLE_SOURCE_DIR}/third_party/cccl)

# The latest commit has bugs in windows, so we set a fix commit.
set(CCCL_TAG 1f6e4bcae0fbf1bbed87f88544d8d2161c490fc1)
execute_process(COMMAND git --git-dir=${CCCL_SOURCE_DIR}/.git
                        --work-tree=${CCCL_SOURCE_DIR} checkout ${CCCL_TAG})

set(CCCL_INCLUDE_DIR ${CCCL_SOURCE_DIR})
message("CCCL_INCLUDE_DIR is ${CCCL_INCLUDE_DIR}")
include_directories(${CCCL_INCLUDE_DIR})

file(TO_NATIVE_PATH ${PADDLE_SOURCE_DIR}/patches/cccl/util_device.cuh.patch
     native_src)
set(CCCL_PATCH_COMMAND git checkout -- . && git checkout ${CCCL_TAG} && patch
                       -p1 -Nd ${CCCL_SOURCE_DIR} < ${native_src})

ExternalProject_Add(
  extern_cccl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  SOURCE_DIR ${CCCL_SOURCE_DIR}
  PREFIX ${CCCL_PREFIX_DIR}
  UPDATE_COMMAND ""
  PATCH_COMMAND ${CCCL_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

# update include dir and set cccl first for using
include_directories(BEFORE "${CCCL_SOURCE_DIR}/cub")
include_directories(BEFORE "${CCCL_SOURCE_DIR}/libcudacxx/include")
include_directories(BEFORE "${CCCL_SOURCE_DIR}/thrust")
