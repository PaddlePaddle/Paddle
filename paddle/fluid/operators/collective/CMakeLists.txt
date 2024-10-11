include(operators)

set(COLLECTIVE_DEPS "")

set(COLLECTIVE_COMPILE_FLAGS
    "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor"
)

file(
  GLOB OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*_op.cc")
list(REMOVE_DUPLICATES OPS)

foreach(src ${OPS})
  set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS
                                                ${COLLECTIVE_COMPILE_FLAGS})
endforeach()

if(WITH_GLOO)
  set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} gloo_wrapper phi common)
endif()

register_operators(
  EXCLUDES
  c_gen_bkcl_id_op
  gen_bkcl_id_op
  c_gen_nccl_id_op
  gen_nccl_id_op
  DEPS
  ${COLLECTIVE_DEPS})

if(WITH_NCCL OR WITH_RCCL)
  set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} nccl_common phi)
  op_library(c_gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
  op_library(gen_nccl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

if(WITH_XPU_BKCL)
  set(COLLECTIVE_DEPS ${COLLECTIVE_DEPS} phi)
  op_library(c_gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
  op_library(gen_bkcl_id_op DEPS ${COLLECTIVE_DEPS})
endif()

set(OPERATOR_DEPS
    ${OPERATOR_DEPS} ${COLLECTIVE_DEPS}
    PARENT_SCOPE)
set(GLOB_COLLECTIVE_DEPS
    ${COLLECTIVE_DEPS}
    CACHE INTERNAL "collective dependency")
