if(WITH_PSLIB)
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope pslib_brpc pslib)
else()
    cc_library(fleet_wrapper SRCS fleet_wrapper.cc DEPS framework_proto variable_helper scope)
endif(WITH_PSLIB)

if(WITH_NCCL)
    cc_library(nccl_wrapper SRCS nccl_wrapper.cc DEPS framework_proto variable_helper scope)
endif()
if(WITH_BOX_PS)
    nv_library(box_wrapper SRCS box_wrapper.cc box_wrapper.cu DEPS framework_proto lod_tensor box_ps)
else()
    cc_library(box_wrapper SRCS box_wrapper.cc DEPS framework_proto lod_tensor)
endif(WITH_BOX_PS)

set(gloo_library_SRCS gloo_wrapper.cc)

set_source_files_properties(
        ${gloo_library_SRCS}
        PROPERTIES
        COMPILE_FLAGS  "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-sign-compare")

if(WITH_GLOO)
    cc_library(gloo_wrapper SRCS ${gloo_library_SRCS} DEPS framework_proto variable_helper scope gloo)
else()
    cc_library(gloo_wrapper SRCS ${gloo_library_SRCS} DEPS framework_proto variable_helper scope)
endif(WITH_GLOO)

cc_library(heter_wrapper SRCS heter_wrapper.cc DEPS framework_proto device_context)

cc_test(test_fleet SRCS test_fleet.cc DEPS fleet_wrapper gloo_wrapper fs shell)
