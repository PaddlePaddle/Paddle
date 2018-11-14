include(operators)

set(DISTRIBUTE_DEPS "")
if(WITH_GRPC)
    set(DISTRIBUTE_DEPS sendrecvop_grpc grpc++_unsecure grpc_unsecure gpr cares zlib protobuf node)
else()
    set(DISTRIBUTE_DEPS sendrecvop_brpc brpc leveldb snappystream snappy protobuf ssl crypto zlib node)
    if(WITH_BRPC_RDMA)
        find_library(IBVERBS_LIBRARY NAMES ibverbs)
        ADD_LIBRARY(ibverbs SHARED IMPORTED GLOBAL)
        SET_PROPERTY(TARGET ibverbs PROPERTY IMPORTED_LOCATION ${IBVERBS_LIBRARY})


        find_library(RDMACM_LIBRARY NAMES rdmacm)
        ADD_LIBRARY(rdmacm SHARED IMPORTED GLOBAL)
        SET_PROPERTY(TARGET rdmacm PROPERTY IMPORTED_LOCATION ${RDMACM_LIBRARY})

        set(DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} ibverbs rdmacm)
    endif()
endif()

set(DISTRIBUTE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

function(register_operators)
    file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
    list(REMOVE_DUPLICATES OPS)

    foreach(src ${OPS})
        set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    endforeach()
    
endfunction()

# foreach(dist_op "prefetch_op" "checkpoint_notify_op" "listen_and_serv_op" "send_op" "recv_op" "send_barrier_op" "fetch_barrier_op")
#     op_library(${dist_op} DEPS ${DISTRIBUTE_DEPS})
#     set_source_files_properties(${dist_op}.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
# endforeach()

register_operators()

if(WITH_GPU AND NOT WIN32)
    set(DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} nccl_common)
    #set_source_files_properties(send_recv_op_test.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    #cc_test(test_send_recv SRCS send_recv_op_test.cc DEPS prefetch_op send_op
    #        listen_and_serv_op sum_op executor SERIAL)
    # set_source_files_properties(test_send_nccl_id.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
    # cc_test(test_send_nccl_id SRCS test_send_nccl_id.cc DEPS listen_and_serv_op ${DISTRIBUTE_DEPS} executor SERIAL)
    # if(WITH_GRPC)
    #     op_library(gen_nccl_id_op DEPS nccl_common sendrecvop_grpc)
    # else()
    #     op_library(gen_nccl_id_op DEPS nccl_common sendrecvop_brpc)
    # endif()
    # set_source_files_properties(gen_nccl_id_op.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()
#     set(DEPS_OPS ${DEPS_OPS} gen_nccl_id_op)
# endif() # WITH_GPU AND NOT WIN32

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${DISTRIBUTE_DEPS} PARENT_SCOPE)
set(GLOB_DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} CACHE INTERNAL "distributed dependency")
