IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

include_directories(include)

FIND_PACKAGE(CUDA 6.5)
FIND_PACKAGE(Torch)

MESSAGE(STATUS "cuda found ${CUDA_FOUND}")
MESSAGE(STATUS "Torch found ${Torch_DIR}")

option(WITH_GPU     "compile warp-ctc with CUDA."     ${CUDA_FOUND})
option(WITH_TORCH   "compile warp-ctc with Torch."    ${Torch_FOUND})
option(WITH_OMP     "compile warp-ctc with OpenMP."   ON)
option(BUILD_TESTS  "build warp-ctc unit tests."      ON)
option(BUILD_SHARED "build warp-ctc shared library."  ON)

if(BUILD_SHARED)
    set(WARPCTC_SHARED "SHARED")
else(BUILD_SHARED)
    set(WARPCTC_SHARED "STATIC")
endif(BUILD_SHARED)

if(WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX lib)
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
else(WIN32)
    # Set c++ flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
endif(WIN32)

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DAPPLE)
endif()

if(WITH_OMP AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
    add_definitions(-DCTC_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION VERSION_GREATER "7.6")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

IF(NOT APPLE AND NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    if(WITH_OMP)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
    endif()
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()

# windows treat symbolic file as a real file, which is different with unix
# We create a hidden file and compile it instead of origin source file.
function(windows_symbolic TARGET)
    set(oneValueArgs "")
    set(multiValueArgs SRCS PATH DEPS)
    cmake_parse_arguments(windows_symbolic "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(final_path ${CMAKE_CURRENT_SOURCE_DIR}/${windows_symbolic_PATH})
    foreach(src ${windows_symbolic_SRCS})
        get_filename_component(src ${src} NAME_WE)
        if (NOT EXISTS ${final_path}/${src}.cpp OR NOT EXISTS ${final_path}/${src}.cu)
            message(FATAL " ${final_path}/${src}.cc and ${final_path}/${src}.cu must exsits, and ${final_path}/${src}.cu must be symbolic file.")
        endif()

        # only copy the xx.cu to .xx.cu when the content are modified
        set(copy_flag 1)
        if (EXISTS ${final_path}/.${src}.cu)
            file(READ ${final_path}/${src}.cpp SOURCE_STR)
            file(READ ${final_path}/.${src}.cu TARGET_STR)
            if (SOURCE_STR STREQUAL TARGET_STR)
                set(copy_flag 0)
            endif()
        endif()
        if (copy_flag)
            add_custom_command(OUTPUT ${final_path}/.${src}.cu
                    COMMAND ${CMAKE_COMMAND} -E remove ${final_path}/.${src}.cu
                    COMMAND ${CMAKE_COMMAND} -E copy "${final_path}/${src}.cpp" "${final_path}/.${src}.cu"
                    COMMENT "create hidden file of ${src}.cu")
        endif(copy_flag)
        add_custom_target(${TARGET} ALL DEPENDS ${final_path}/.${src}.cu)
    endforeach()
endfunction()

IF (WITH_GPU)

    MESSAGE(STATUS "Building shared library with GPU support")
    MESSAGE(STATUS "NVCC_ARCH_FLAGS" ${CUDA_NVCC_FLAGS})

    if (WIN32)
        SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler \"/wd 4068 /wd 4244 /wd 4267 /wd 4305 /wd 4819\"")
        windows_symbolic(ctc_entrypoint SRCS ctc_entrypoint.cu PATH src)
        CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/.ctc_entrypoint.cu src/reduce.cu)
    else()
        CUDA_ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cu src/reduce.cu)
    endif(WIN32)

    IF (!WITH_TORCH)
        TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})
    ENDIF()

    if(BUILD_TESTS)
        add_executable(test_cpu tests/test_cpu.cpp )
        TARGET_LINK_LIBRARIES(test_cpu warpctc)
        SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

        cuda_add_executable(test_gpu tests/test_gpu.cu)
        TARGET_LINK_LIBRARIES(test_gpu warpctc ${CUDA_curand_LIBRARY})
    endif(BUILD_TESTS)

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

    IF (WITH_TORCH)
        MESSAGE(STATUS "Building Torch Bindings with GPU support")
        INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
        INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH ${Torch_INSTALL_INCLUDE}/THC)

        TARGET_LINK_LIBRARIES(warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY})
        INSTALL(TARGETS warpctc
                RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
                LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
                ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

        SET(src torch_binding/binding.cpp torch_binding/utils.c)
        SET(luasrc torch_binding/init.lua)

        ADD_TORCH_PACKAGE(warp_ctc "${src}" "${luasrc}")
        IF (APPLE)
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY})
        ELSE()
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY} gomp)
        ENDIF()
    ENDIF()

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE AND NOT WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc ${WARPCTC_SHARED} src/ctc_entrypoint.cpp)

    if(BUILD_TESTS)
        add_executable(test_cpu tests/test_cpu.cpp )
        TARGET_LINK_LIBRARIES(test_cpu warpctc)
        SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
    endif(BUILD_TESTS)

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

    IF (WITH_TORCH)
        MESSAGE(STATUS "Building Torch Bindings with no GPU support")
        add_definitions(-DTORCH_NOGPU)
        INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH)

        TARGET_LINK_LIBRARIES(warpctc luajit luaT TH)

        INSTALL(TARGETS warpctc
                RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
                LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
                ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

        SET(src torch_binding/binding.cpp torch_binding/utils.c)
        SET(luasrc torch_binding/init.lua)

        ADD_TORCH_PACKAGE(warp_ctc "${src}" "${luasrc}")
        IF (APPLE)
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT TH)
        ELSE()
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT TH gomp)
        ENDIF()
    ENDIF()

ENDIF()
