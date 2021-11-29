# NPU

add_subdirectory(dynload)

if(WITH_ASCEND)
  cc_library(ascend_npu_info SRCS ascend_npu_info.cc DEPS gflags glog enforce atlas_acl)
endif()

if(WITH_ASCEND_CL)
  cc_library(npu_info SRCS npu_info.cc DEPS gflags glog enforce monitor ascendcl acl_op_compiler)
  cc_library(npu_resource_pool SRCS npu_resource_pool.cc DEPS npu_info)
  cc_library(npu_stream SRCS npu_stream.cc DEPS enforce boost stream_callback_manager)
  cc_library(npu_collective_helper SRCS npu_collective_helper.cc DEPS npu_stream npu_info)
  cc_library(npu_op_runner SRCS npu_op_runner.cc DEPS operator npu_info)
endif()
