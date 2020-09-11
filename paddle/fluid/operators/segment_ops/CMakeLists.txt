include(operators)
register_operators()

function(math_library TARGET)
    # math_library is a function to create math library.
    # The interface is the same as cc_library.
    # But it handle split GPU/CPU code and link some common library.
    set(cc_srcs)
    set(cu_srcs)
    set(hip_srcs)
    set(math_common_deps device_context framework_proto enforce)
    if (WITH_GPU)
        list(APPEND math_common_deps cub)
    endif()
    set(multiValueArgs DEPS)
    cmake_parse_arguments(math_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
        list(APPEND cc_srcs ${TARGET}.cc)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
        list(APPEND cu_srcs ${TARGET}.cu)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu.cc)
        list(APPEND cu_srcs ${TARGET}.cu.cc)
    endif()
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.hip.cu)
        list(APPEND hip_srcs ${TARGET}.hip.cu)
    endif()

    list(LENGTH cc_srcs cc_srcs_len)
    if (WITH_GPU)
        nv_library(${TARGET} SRCS ${cc_srcs} ${cu_srcs} DEPS ${math_library_DEPS} ${math_common_deps})
    elseif (WITH_AMD_GPU)
        hip_library(${TARGET} SRCS ${cc_srcs} ${hip_srcs} DEPS ${math_library_DEPS} ${math_common_deps})
    elseif(${cc_srcs_len} GREATER 0)
        cc_library(${TARGET} SRCS ${cc_srcs} DEPS ${math_library_DEPS} ${math_common_deps})
    endif()
endfunction()

# please add new math_library in alphabetical order
math_library(segment_pooling)
