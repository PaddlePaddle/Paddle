# paddle experimental common components
add_subdirectory(common)

# pten (low level) api headers: include
# pten (high level) api
add_subdirectory(api)
# pten core components
add_subdirectory(core)
# pten components of specific backends
add_subdirectory(backends)
# pten kernels for diff device
add_subdirectory(kernels)
# pten infermeta
add_subdirectory(infermeta)
# pten operator definitions
add_subdirectory(ops)
# pten tests
add_subdirectory(tests)

# make an unity target for compile deps
set(PTEN_DEPS convert_utils dense_tensor pten_context kernel_factory kernel_context arg_map_context infermeta lod_utils)
get_property(pten_kernels GLOBAL PROPERTY PTEN_KERNELS)
# keep this message for debug, remove it later if needless
message(STATUS "All standard pten kernels: ${pten_kernels}")
set(PTEN_DEPS ${PTEN_DEPS} ${pten_kernels})

cc_library(pten DEPS ${PTEN_DEPS})
