if(NOT WITH_GPU)
    return()
endif()


if (WITH_NV_JETSON)
  add_definitions(-DWITH_NV_JETSON)
  set(paddle_known_gpu_archs "53 62 72")
  set(paddle_known_gpu_archs7 "53")
  set(paddle_known_gpu_archs8 "53 62")
  set(paddle_known_gpu_archs9 "53 62")
  set(paddle_known_gpu_archs10 "53 62 72")
else()
  set(paddle_known_gpu_archs "30 35 50 52 60 61 70")
  set(paddle_known_gpu_archs7 "30 35 50 52")
  set(paddle_known_gpu_archs8 "30 35 50 52 60 61")
  set(paddle_known_gpu_archs9 "30 35 50 52 60 61 70")
  set(paddle_known_gpu_archs10 "30 35 50 52 60 61 70 75")
endif()

######################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${cufile} ""
      "#include \"stdio.h\"\n"
      "#include \"cuda.h\"\n"
      "#include \"cuda_runtime.h\"\n"
      "int main() {\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device) {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}"
                    "--run" "${cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE nvcc_res OUTPUT_VARIABLE nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(nvcc_res EQUAL 0)
      # only keep the last line of nvcc_out
      STRING(REGEX REPLACE ";" "\\\\;" nvcc_out "${nvcc_out}")
      STRING(REGEX REPLACE "\n" ";" nvcc_out "${nvcc_out}")
      list(GET nvcc_out -1 nvcc_out)
      string(REPLACE "2.1" "2.1(2.0)" nvcc_out "${nvcc_out}")
      set(CUDA_gpu_detect_output ${nvcc_out} CACHE INTERNAL "Returned GPU architetures from detect_installed_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${paddle_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


########################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   select_nvcc_arch_flags(out_variable)
function(select_nvcc_arch_flags out_variable)
  # List of arch names
  set(archs_names "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "All" "Manual")
  set(archs_name_default "Auto")
  list(APPEND archs_names "Auto")

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " archs_names "${archs_names}")
    message(FATAL_ERROR "Only ${archs_names} architectures names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${paddle_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    if (NOT ${CUDA_VERSION} LESS 10.0)
      add_definitions("-DSUPPORTS_CUDA_FP16")
    endif()
    set(cuda_arch_bin "70")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    if (NOT ${CUDA_VERSION} LESS 10.0)
      add_definitions("-DSUPPORTS_CUDA_FP16")
    endif()
    set(cuda_arch_bin "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(cuda_arch_bin ${paddle_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    message(STATUS "WARNING: This is just a warning for publishing release.
      You are building GPU version without supporting different architectures.
      So the wheel package may fail on other GPU architectures.
      You can add -DCUDA_ARCH_NAME=All in cmake command
      to get a full wheel package to resolve this warning.
      While, this version will still work on local GPU architecture.")
    detect_installed_gpus(cuda_arch_bin)
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" cuda_arch_bin "${cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   cuda_arch_ptx "${cuda_arch_ptx}")
  list(REMOVE_DUPLICATES cuda_arch_bin)
  list(REMOVE_DUPLICATES cuda_arch_ptx)

  set(nvcc_flags "")
  set(nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(arch ${cuda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND nvcc_flags -gencode arch=compute_${arch},code=sm_${arch})
      list(APPEND nvcc_archs_readable sm_${arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(arch ${cuda_arch_ptx})
    list(APPEND nvcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND nvcc_archs_readable compute_${arch})
  endforeach()

  string(REPLACE ";" " " nvcc_archs_readable "${nvcc_archs_readable}")
  set(${out_variable}          ${nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${nvcc_archs_readable} PARENT_SCOPE)
endfunction()

message(STATUS "CUDA detected: " ${CUDA_VERSION})
if (${CUDA_VERSION} LESS 7.0)
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs})
elseif (${CUDA_VERSION} LESS 8.0) # CUDA 7.x
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs7})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
elseif (${CUDA_VERSION} LESS 9.0) # CUDA 8.x
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs8})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
  # CUDA 8 may complain that sm_20 is no longer supported. Suppress the
  # warning for now.
  list(APPEND CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets")
elseif (${CUDA_VERSION} LESS 10.0) # CUDA 9.x
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs9})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
elseif (${CUDA_VERSION} LESS 11.0) # CUDA 10.x
  set(paddle_known_gpu_archs ${paddle_known_gpu_archs10})
  list(APPEND CUDA_NVCC_FLAGS "-D_MWAITXINTRIN_H_INCLUDED")
  list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__")
endif()
add_definitions("-DPADDLE_CUDA_BINVER=\"${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}\"")

include_directories(${CUDA_INCLUDE_DIRS})
if(NOT WITH_DSO)
    if(WIN32)
      set_property(GLOBAL PROPERTY CUDA_MODULES ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusolver_LIBRARY})
    endif(WIN32)
endif(NOT WITH_DSO)

# setting nvcc arch flags
select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")

# Set C++11 support
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Release/Debug flags set by cmake. Such as -O3 -g -DNDEBUG etc.
# So, don't set these flags here.
if (NOT WIN32) # windows msvc2015 support c++11 natively. 
  # -std=c++11 -fPIC not recoginize by msvc, -Xcompiler will be added by cmake.
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
endif(NOT WIN32)

# in cuda9, suppress cuda warning on eigen 
list(APPEND CUDA_NVCC_FLAGS "-w")
# Set :expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

if (NOT WIN32)
  if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
      list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_DEBUG})
  elseif(CMAKE_BUILD_TYPE  STREQUAL "Release")
      list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELEASE})
  elseif(CMAKE_BUILD_TYPE  STREQUAL "RelWithDebInfo")
      list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  elseif(CMAKE_BUILD_TYPE  STREQUAL "MinSizeRel")
      # nvcc 9 does not support -Os. Use Release flags instead
      list(APPEND CUDA_NVCC_FLAGS  ${CMAKE_CXX_FLAGS_RELEASE})
  endif()
else(NOT WIN32)
  list(APPEND CUDA_NVCC_FLAGS  "-Xcompiler \"/wd 4244 /wd 4267 /wd 4819\"")
  list(APPEND CUDA_NVCC_FLAGS  "--compiler-options;/bigobj")
  if(MSVC_STATIC_CRT)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MT$<$<CONFIG:Debug>:d>")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MD$<$<CONFIG:Debug>:d>")
  endif()
  if(CMAKE_BUILD_TYPE  STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS  "-g -G")
    # match the cl's _ITERATOR_DEBUG_LEVEL
    list(APPEND CUDA_NVCC_FLAGS  "-D_DEBUG")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND CUDA_NVCC_FLAGS "-O3 -DNDEBUG")
  else()
  message(FATAL "Windows only support Release or Debug build now. Please set visual studio build type to Release/Debug, x64 build.")
endif()
endif(NOT WIN32)

mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_SDK_ROOT_DIR CUDA_SEPARABLE_COMPILATION)
