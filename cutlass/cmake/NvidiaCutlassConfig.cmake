get_filename_component(NvidiaCutlass_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

include(CMakeFindDependencyMacro)

if(NOT TARGET nvidia::cutlass::CUTLASS)
    include("${NvidiaCutlass_CMAKE_DIR}/NvidiaCutlassTargets.cmake")
endif()
