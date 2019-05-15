include(operators)

set(DISTRIBUTE_DEPS "")
if(WITH_GRPC)
    set(DISTRIBUTE_DEPS sendrecvop_rpc parameter_send parameter_recv communicator async_sparse_param_update_recorder grpc++_unsecure grpc_unsecure gpr cares zlib protobuf node)
else()
    set(DISTRIBUTE_DEPS sendrecvop_rpc parameter_send parameter_recv communicator async_sparse_param_update_recorder brpc leveldb snappystream snappy protobuf ssl crypto zlib node)
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


file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endforeach()

register_operators(EXCLUDES gen_nccl_id_op DEPS ${DISTRIBUTE_DEPS})

if(WITH_GPU AND NOT WIN32)
    set(DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} nccl_common)
    op_library(gen_nccl_id_op DEPS ${DISTRIBUTE_DEPS} nccl_common)
endif()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${DISTRIBUTE_DEPS} PARENT_SCOPE)
set(GLOB_DISTRIBUTE_DEPS ${DISTRIBUTE_DEPS} CACHE INTERNAL "distributed dependency")
