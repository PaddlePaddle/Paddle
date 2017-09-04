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

    foreach(src ${op_library_SRCS})
        if (${src} MATCHES ".*\\.cu$")
            list(APPEND cu_srcs ${src})
        elseif(${src} MATCHES ".*\\.cc$")
            list(APPEND cc_srcs ${src})
        else()
            message(FATAL_ERROR "${TARGET} Source file ${src} should only be .cc or .cu")
        endif()
    endforeach()

    list(LENGTH cc_srcs cc_srcs_len)
    if (${cc_srcs_len} EQUAL 0)
        message(FATAL_ERROR "The op library ${TARGET} should contains at least one .cc file")
    endif()

    list(LENGTH cu_srcs cu_srcs_len)
    list(LENGTH op_library_DEPS dep_len)
    if (${cu_srcs_len} EQUAL 0 AND ${dep_len} EQUAL 0)
        message(WARNING "The op library ${TARGET} not support GPU!")
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

list(REMOVE_ITEM GENERAL_OPS
     net_op
     minus_op
     mul_op
     recurrent_op
     scale_op)

op_library(net_op SRCS net_op.cc)
op_library(minus_op SRCS minus_op.cc minus_op.cu DEPS scale_op)
op_library(mul_op SRCS mul_op.cc mul_op.cu DEPS math_function)
op_library(recurrent_op SRCS recurrent_op.cc rnn/recurrent_op_utils.cc 
  DEPS framework_proto tensor operator net_op)
op_library(scale_op SRCS scale_op.cc scale_op.cu DEPS net_op)

foreach(src ${GENERAL_OPS})
    op_library(${src} SRCS ${src}.cc ${src}.cu)
endforeach()

set(GLOB_OP_LIB ${OP_LIBRARY} CACHE INTERNAL "Global OP library")

cc_test(gather_test SRCS gather_test.cc DEPS tensor)
cc_test(net_op_test SRCS net_op_test.cc DEPS net_op)
cc_test(scatter_test SRCS scatter_test.cc DEPS tensor)