set(MAJEL_CXX_FILES place.cpp)
set(MAJEL_CUDA_FILES "")

if(CUDA_FOUND)
    cuda_add_library(majel ${MAJEL_CUDA_FILES} ${MAJEL_CXX_FILES})
else()
    add_library(majel ${MAJEL_CXX_FILES})
endif()

file(GLOB_RECURSE CHECK_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp" "*.h")
add_style_check_target(majel ${CHECK_FILES})

if(WITH_TESTING)
    add_subdirectory(test)
endif()
