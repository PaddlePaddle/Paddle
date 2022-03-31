add_subdirectory(collective)
add_subdirectory(store)
if(NOT WITH_PSCORE)
    if(WITH_HETERPS)
        add_subdirectory(ps)
    endif()
    add_subdirectory(fleet_executor)
    return()
endif()

proto_library(ps_framework_proto SRCS ps.proto)

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-error=unused-value -Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=sign-compare -Wno-error=unused-variable -Wno-error=return-type -Wno-error=unused-but-set-variable -Wno-error=type-limits -Wno-error=unknown-pragmas -Wno-error=parentheses -Wno-error=unused-result")

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS
            "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

add_subdirectory(common)
add_subdirectory(ps)
add_subdirectory(test)
add_subdirectory(index_dataset)
add_subdirectory(fleet_executor)
