include(operators)

set(COLLECTIVE_DEPS "")
if(WITH_GRPC)
    set(COLLECTIVE_DEPS sendrecvop_rpc parameter_send parameter_recv communicator async_sparse_param_update_recorder grpc++_unsecure grpc_unsecure gpr zlib protobuf node)
else()
    set(COLLECTIVE_DEPS sendrecvop_rpc parameter_send parameter_recv communicator async_sparse_param_update_recorder brpc leveldb protobuf ssl crypto zlib node)
    if(WITH_BRPC_RDMA)
        find_library(IBVERBS_LIBRARY NAMES ibverbs)
        ADD_LIBRARY(ibverbs SHARED IMPORTED GLOBAL)
        SET_PROPERTY(TARGET ibverbs PROPERTY IMPORTED_LOCATION ${IBVERBS_LIBRARY})


        find_library(RDMACM_LIBRARY NAMES rdmacm)
        ADD_LIBRARY(rdmacm SHARED IMPORTED GLOBAL)
        SET_PROPERTY(TARGET rdmacm PROPERTY IMPORTED_LOCATION ${RDMACM_LIBRARY})

        set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} ibverbs rdmacm)
    endif()
endif()

set(COLLECTIVE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${COLLECTIVE_COMPILE_FLAGS})
endforeach()

register_operators(EXCLUDES c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})

if(WITH_NCCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} nccl_common collective_helper)
    op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS} nccl_common)
endif()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${COLLECTIVE_DEPS} PARENT_SCOPE)
set(GLOB_COLLECTIVE_DEPS ${COLLECTIVE_DEPS} CACHE INTERNAL "collective dependency")
