file(GLOB GENERAL_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
string(REPLACE ".cc" "" GENERAL_OPS "${GENERAL_OPS}")
function(op_library TARGET)
    # op_library is a function to create op library. The interface is same as
    # cc_library. But it handle split GPU/CPU code and link some common library
    # for ops.
    set(OP_LIBRARY ${TARGET} ${OP_LIBRARY} PARENT_SCOPE)
    set(cc_srcs)
    set(cu_srcs)
    set(op_common_deps operator op_registry)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(op_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    list(LENGTH op_library_SRCS op_library_SRCS_len)
    if (${op_library_SRCS_len} EQUAL 0)
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
            list(APPEND cc_srcs ${TARGET}.cc)
        endif()
        if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
            list(APPEND cu_srcs ${TARGET}.cu)
        endif()
    else()
        foreach(src ${op_library_SRCS})
            if (${src} MATCHES ".*\\.cu$")
                list(APPEND cu_srcs ${src})
            elseif(${src} MATCHES ".*\\.cc$")
                list(APPEND cc_srcs ${src})
            else()
                message(FATAL_ERROR "${TARGET} Source file ${src} should only be .cc or .cu")
            endif()
        endforeach()
    endif()

    list(LENGTH cc_srcs cc_srcs_len)
    if (${cc_srcs_len} EQUAL 0)
        message(FATAL_ERROR "The op library ${TARGET} should contains at least one .cc file")
    endif()

    if (WITH_GPU)
        nv_library(${TARGET} SRCS ${cc_srcs} ${cu_srcs} DEPS ${op_library_DEPS}
                ${op_common_deps})
    else()
        cc_library(${TARGET} SRCS ${cc_srcs} DEPS ${op_library_DEPS}
                ${op_common_deps})
    endif()
endfunction()

add_subdirectory(math)

set(DEPS_OPS
    identity_op
    minus_op
    mul_op
    recurrent_op
    scale_op)
op_library(identity_op DEPS scale_op)
op_library(minus_op DEPS scale_op)
op_library(mul_op DEPS math_function)
op_library(recurrent_op SRCS recurrent_op.cc rnn/recurrent_op_utils.cc 
  DEPS framework_proto tensor operator net_op)
op_library(scale_op DEPS net_op)

list(REMOVE_ITEM GENERAL_OPS ${DEPS_OPS})
foreach(src ${GENERAL_OPS})
    op_library(${src})
endforeach()

set(GLOB_OP_LIB ${OP_LIBRARY} CACHE INTERNAL "Global OP library")

cc_test(gather_test SRCS gather_test.cc DEPS tensor)
cc_test(net_op_test SRCS net_op_test.cc DEPS net_op)
cc_test(scatter_test SRCS scatter_test.cc DEPS tensor)
