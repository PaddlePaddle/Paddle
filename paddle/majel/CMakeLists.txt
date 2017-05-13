cmake_minimum_required(VERSION 3.0)

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    # find #include <majel/xx.h>
    get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    include_directories(${PARENT_DIR})

    # find cmake directory modules
    get_filename_component(PARENT_DIR ${PARENT_DIR} DIRECTORY)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

    # enable c++11
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # enable gtest
    set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    set(WITH_TESTING ON)
    include(external/gtest)

    # enable external variant
    include(external/variant)
else()    
    message("-- Found gtest (include: ${GTEST_INCLUDE_DIR}, library: ${GTEST_LIBRARIES})")
    message("-- Found variant (include: ${VARIANT_INCLUDE_DIR})")
endif()

########################### Build Majel #############################
set(MAJEL_CXX_FILES place.cc)
set(MAJEL_CUDA_FILES "")

if(CUDA_FOUND)
    cuda_add_library(majel ${MAJEL_CUDA_FILES} ${MAJEL_CXX_FILES})
else()
    add_library(majel ${MAJEL_CXX_FILES})
endif()

add_dependencies(majel ${external_project_dependencies})
#####################################################################

add_subdirectory(test)
