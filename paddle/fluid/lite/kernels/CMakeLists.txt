message(STATUS "add lite kernels")

set(lite_kernel_deps type_system kernel_lite op_lite op_registry_lite context_lite ${tensor_lite} CACHE INTERNAL "" FORCE)

add_subdirectory(host)
add_subdirectory(arm)
add_subdirectory(cuda)
add_subdirectory(x86)
add_subdirectory(opencl)
 
