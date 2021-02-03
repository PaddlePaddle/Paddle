include(operators)

set(COLLECTIVE_DEPS "")

set(COLLECTIVE_COMPILE_FLAGS "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor")

file(GLOB OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
    set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS ${COLLECTIVE_COMPILE_FLAGS})
endforeach()

register_operators(EXCLUDES c_gen_nccl_id_op gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})

if(WITH_NCCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} nccl_common collective_helper)
    op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
    op_library(gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

if(WITH_BKCL)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} collective_helper)
endif()

if(WITH_GLOO)
    set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} gloo_wrapper)
endif()

set(OPERATOR_DEPS ${OPERATOR_DEPS} ${COLLECTIVE_DEPS} PARENT_SCOPE)
set(GLOB_COLLECTIVE_DEPS ${COLLECTIVE_DEPS} CACHE INTERNAL "collective dependency")
