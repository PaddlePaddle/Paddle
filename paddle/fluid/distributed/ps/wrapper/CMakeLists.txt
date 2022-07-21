get_property(RPC_DEPS GLOBAL PROPERTY RPC_DEPS)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses")
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

set_source_files_properties(fleet.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(fleet
        SRCS fleet.cc
        DEPS framework_proto ps_framework_proto ps_service variable_helper scope 
        ps_gpu_wrapper op_registry fs shell ${RPC_DEPS})

target_link_libraries(fleet z)
