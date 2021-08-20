#===============================================================================
# Copyright 2021 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#===================================================================
# CMake Config file for Intel(R) oneAPI Math Kernel Library (oneMKL)
#===================================================================

#===============================================================================
# Input parameters
#=================
#-------------
# Main options
#-------------
# MKL_ROOT: oneMKL root directory (May be required for non-standard install locations. Optional otherwise.)
#    Default: use location from MKLROOT environment variable or <Full path to this file>/../../../ if MKLROOT is not defined
# MKL_ARCH
#    Values:  ia32 intel64
#    Default: intel64
# MKL_LINK
#    Values:  static, dynamic, sdl
#    Default: dynamic
#       Exceptions:- DPC++ doesn't support sdl
# MKL_THREADING
#    Values:  sequential,
#             intel_thread (Intel OpenMP),
#             gnu_thread (GNU OpenMP),
#             pgi_thread (PGI OpenMP),
#             tbb_thread
#    Default: intel_thread
#       Exceptions:- DPC++ defaults to tbb, PGI compiler on Windows defaults to pgi_thread
# MKL_INTERFACE (for MKL_ARCH=intel64 only)
#    Values:  lp64, ilp64
#       GNU or INTEL interface will be selected based on Compiler.
#    Default: ilp64
# MKL_MPI
#    Values:  intelmpi, mpich, openmpi, msmpi, mshpc
#    Default: intelmpi
#-----------------------------------
# Special options (OFF by default)
#-----------------------------------
# ENABLE_BLAS95:      Enables BLAS Fortran95 API
# ENABLE_LAPACK95:    Enables LAPACK Fortran95 API
# ENABLE_BLACS:       Enables cluster BLAS library
# ENABLE_CDFT:        Enables cluster DFT library
# ENABLE_CPARDISO:    Enables cluster PARDISO functionality
# ENABLE_SCALAPACK:   Enables cluster LAPACK library
# ENABLE_OMP_OFFLOAD: Enables OpenMP Offload functionality
#
#==================
# Output parameters
#==================
# MKL_ROOT
#     oneMKL root directory.
# MKL_INCLUDE
#     Use of target_include_directories() is recommended.
#     INTERFACE_INCLUDE_DIRECTORIES property is set on mkl_core and mkl_rt libraries.
#     Alternatively, this variable can be used directly (not recommended as per Modern CMake)
# MKL_ENV
#     Provides all environment variables based on input parameters.
#     Currently useful for mkl_rt linking and BLACS on Windows.
#     Must be set as an ENVIRONMENT property.
# Example:
#     add_test(NAME mytest COMMAND myexe)
#     if(MKL_ENV)
#       set_tests_properties(mytest PROPERTIES ENVIRONMENT "${MKL_ENV}")
#     endif()
#
# MKL::<library name>
#     IMPORTED targets to link MKL libraries individually or when using a custom link-line.
#     mkl_core and mkl_rt have INTERFACE_* properties set to them.
#     Please refer to Intel(R) oneMKL Link Line Advisor for help with linking.
#
# Below INTERFACE targets provide full link-lines for direct use.
# Example:
#     target_link_options(<my_linkable_target> PUBLIC $<LINK_ONLY:MKL::MKL>)
#
# MKL::MKL
#     Link line for C and Fortran API
# MKL::MKL_DPCPP
#     Link line for DPC++ API
#
# Note: For Device API, library linking is not required.
#       Compile options can be added from the INTERFACE_COMPILE_OPTIONS property on MKL::MKL_DPCPP
#       Include directories can be added from the INTERFACE_INCLUDE_DIRECTORIES property on MKL::MKL_DPCPP
#
# Note: Output parameters' and targets' availability can change
# based on Input parameters and application project languages.
#===============================================================================

include_guard()
include(FindPackageHandleStandardArgs)

if(NOT MKL_LIBRARIES)

# Set CMake policies for well-defined behavior across CMake versions
cmake_policy(SET CMP0011 NEW)
cmake_policy(SET CMP0057 NEW)

# Project Languages
get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
list(APPEND MKL_LANGS C CXX Fortran)
foreach(lang ${languages})
  if(${lang} IN_LIST MKL_LANGS)
    list(APPEND CURR_LANGS ${lang})
  endif()
endforeach()
list(REMOVE_DUPLICATES CURR_LANGS)

option(ENABLE_BLAS95      "Enables BLAS Fortran95 API"            OFF)
option(ENABLE_LAPACK95    "Enables LAPACK Fortran95 API"          OFF)
option(ENABLE_BLACS       "Enables cluster BLAS library"          OFF)
option(ENABLE_CDFT        "Enables cluster DFT library"           OFF)
option(ENABLE_CPARDISO    "Enables cluster PARDISO functionality" OFF)
option(ENABLE_SCALAPACK   "Enables cluster LAPACK library"        OFF)
option(ENABLE_OMP_OFFLOAD "Enables OpenMP Offload functionality"  OFF)

# Use MPI if any of these are enabled
if(ENABLE_BLACS OR ENABLE_CDFT OR ENABLE_SCALAPACK OR ENABLE_CPARDISO)
  set(USE_MPI ON)
endif()

# Check Parameters
function(define_param TARGET_PARAM DEFAULT_PARAM SUPPORTED_LIST)
  if(NOT DEFINED ${TARGET_PARAM} AND NOT DEFINED ${DEFAULT_PARAM})
    message(STATUS "${TARGET_PARAM}: Undefined")
  elseif(NOT DEFINED ${TARGET_PARAM} AND DEFINED ${DEFAULT_PARAM})
    set(${TARGET_PARAM} "${${DEFAULT_PARAM}}" CACHE STRING "Choose ${TARGET_PARAM} options are: ${${SUPPORTED_LIST}}")
    foreach(opt ${${DEFAULT_PARAM}})
      set(STR_LIST "${STR_LIST} ${opt}")
    endforeach()
    message(STATUS "${TARGET_PARAM}: None, set to `${STR_LIST}` by default")
  elseif(${SUPPORTED_LIST})
    set(ITEM_FOUND 1)
    foreach(opt ${${TARGET_PARAM}})
      if(NOT ${opt} IN_LIST ${SUPPORTED_LIST})
        set(ITEM_FOUND 0)
      endif()
    endforeach()
    if(ITEM_FOUND EQUAL 0)
      foreach(opt ${${SUPPORTED_LIST}})
        set(STR_LIST "${STR_LIST} ${opt}")
      endforeach()
      message(FATAL_ERROR "Invalid ${TARGET_PARAM} `${${TARGET_PARAM}}`, options are: ${STR_LIST}")
    else()
      message(STATUS "${TARGET_PARAM}: ${${TARGET_PARAM}}")
    endif()
  else()
    message(STATUS "${TARGET_PARAM}: ${${TARGET_PARAM}}")
  endif()
endfunction()

#================
# Compiler checks
#================

# Determine Compiler Family
if(CMAKE_CXX_COMPILER MATCHES ".*dpcpp[.exe]*")
  set(DPCPP_COMPILER ON)
endif()
if(CMAKE_C_COMPILER_ID STREQUAL "PGI" OR CMAKE_Fortran_COMPILER_ID STREQUAL "PGI")
  set(PGI_COMPILER ON)
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(INTEL_COMPILER ON)
else()
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(GNU_C_COMPILER ON)
  endif()
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(GNU_Fortran_COMPILER ON)
  endif()
endif()

if(USE_MPI AND (CMAKE_C_COMPILER MATCHES ".*/mpi*[.bat]*" OR CMAKE_Fortran_COMPILER MATCHES ".*/mpi*[.bat]*"))
  set(USE_MPI_SCRIPT ON)
endif()

#================

#================
# System-specific
#================

# Extensions
if(UNIX)
  set(LIB_PREFIX "lib")
  set(LIB_EXT ".a")
  set(DLL_EXT ".so")
  if(APPLE)
    set(DLL_EXT ".dylib")
  endif()
  set(LINK_PREFIX "-l")
  set(LINK_SUFFIX "")
else()
  set(LIB_PREFIX "")
  set(LIB_EXT ".lib")
  set(DLL_EXT "_dll.lib")
  set(LINK_PREFIX "")
  set(LINK_SUFFIX ".lib")
endif()

# Set target system architecture
set(DEFAULT_MKL_ARCH intel64)
if(DPCPP_COMPILER OR PGI_COMPILER OR ENABLE_OMP_OFFLOAD OR USE_MPI)
  set(MKL_ARCH_LIST intel64)
else()
  set(MKL_ARCH_LIST ia32 intel64)
endif()
define_param(MKL_ARCH DEFAULT_MKL_ARCH MKL_ARCH_LIST)

#================

#==========
# Setup MKL
#==========

# Set MKL_ROOT directory
if(NOT DEFINED MKL_ROOT)
  if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT})
  else()
    get_filename_component(MKL_CMAKE_PATH "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
    get_filename_component(MKL_ROOT "${MKL_CMAKE_PATH}/../../../" ABSOLUTE)
    message(STATUS "MKL_ROOT ${MKL_ROOT}")
  endif()
endif()

# Define MKL_LINK
set(DEFAULT_MKL_LINK dynamic)
if(DPCPP_COMPILER OR USE_MPI)
  set(MKL_LINK_LIST static dynamic)
else()
  set(MKL_LINK_LIST static dynamic sdl)
endif()
define_param(MKL_LINK DEFAULT_MKL_LINK MKL_LINK_LIST)

# Define MKL_INTERFACE
if(MKL_ARCH STREQUAL "intel64")
  set(IFACE_TYPE intel)
  if(GNU_Fortran_COMPILER)
    set(IFACE_TYPE gf)
  endif()
  if(DPCPP_COMPILER)
    if(MKL_INTERFACE)
      set(MKL_INTERFACE_FULL intel_${MKL_INTERFACE})
    endif()
    set(DEFAULT_MKL_INTERFACE intel_ilp64)
    set(MKL_INTERFACE_LIST intel_ilp64)
  else()
    if(MKL_INTERFACE)
      set(MKL_INTERFACE_FULL ${IFACE_TYPE}_${MKL_INTERFACE})
    endif()
    set(DEFAULT_MKL_INTERFACE ${IFACE_TYPE}_ilp64)
    set(MKL_INTERFACE_LIST ${IFACE_TYPE}_ilp64 ${IFACE_TYPE}_lp64)
  endif()
  define_param(MKL_INTERFACE_FULL DEFAULT_MKL_INTERFACE MKL_INTERFACE_LIST)
else()
  if(WIN32)
    set(MKL_INTERFACE_FULL intel_c)
  elseif(NOT APPLE)
    if(GNU_Fortran_COMPILER)
      set(MKL_INTERFACE_FULL gf)
    else()
      set(MKL_INTERFACE_FULL intel)
    endif()
  else()
    message(FATAL_ERROR "OSX does not support MKL_ARCH ia32.")
  endif()
endif()
if(MKL_INTERFACE_FULL MATCHES ".*ilp64.*")
  set(MKL_INTERFACE "ilp64")
else()
  set(MKL_INTERFACE "lp64")
endif()

# Define MKL headers
find_path(MKL_H mkl.h
  HINTS ${MKL_ROOT}
  PATH_SUFFIXES include)
list(APPEND MKL_INCLUDE ${MKL_H})

# Add pre-built F95 Interface Modules
if(INTEL_COMPILER AND (ENABLE_BLAS95 OR ENABLE_LAPACK95))
  if(MKL_ARCH STREQUAL "intel64")
    list(APPEND MKL_INCLUDE "${MKL_ROOT}/include/${MKL_ARCH}/${MKL_INTERFACE}")
  else()
    list(APPEND MKL_INCLUDE "${MKL_ROOT}/include/${MKL_ARCH}")
  endif()
endif()

# Define MKL_THREADING
# All APIs support sequential threading
set(MKL_THREADING_LIST "sequential" "intel_thread" "tbb_thread")
set(DEFAULT_MKL_THREADING intel_thread)
# DPC++ API supports TBB threading, but not OpenMP threading
if(DPCPP_COMPILER)
  set(DEFAULT_MKL_THREADING tbb_thread)
  list(REMOVE_ITEM MKL_THREADING_LIST intel_thread)
# C, Fortran API
elseif(PGI_COMPILER)
  # PGI compiler supports PGI OpenMP threading, additionally
  list(APPEND MKL_THREADING_LIST pgi_thread)
  # PGI compiler does not support TBB threading
  list(REMOVE_ITEM MKL_THREADING_LIST tbb_thread)
  if(WIN32)
    # PGI 19.10 and 20.1 on Windows, do not support Intel OpenMP threading
    list(REMOVE_ITEM MKL_THREADING_LIST intel_thread)
    set(DEFAULT_MKL_THREADING pgi_thread)
  endif()
elseif(GNU_C_COMPILER OR GNU_Fortran_COMPILER)
  list(APPEND MKL_THREADING_LIST gnu_thread)
else()
  # Intel and Microsoft compilers
  # Nothing to do, only for completeness
endif()
define_param(MKL_THREADING DEFAULT_MKL_THREADING MKL_THREADING_LIST)

# Define MKL_MPI
set(DEFAULT_MKL_MPI intelmpi)
if(UNIX)
  if(APPLE)
    # Override defaults for OSX
    set(DEFAULT_MKL_MPI mpich)
    set(MKL_MPI_LIST mpich)
  else()
    set(MKL_MPI_LIST intelmpi openmpi mpich mpich2)
  endif()
else()
  # Windows
  set(MKL_MPI_LIST intelmpi mshpc msmpi)
endif()
define_param(MKL_MPI DEFAULT_MKL_MPI MKL_MPI_LIST)
# MSMPI is now called MSHPC. MSMPI option exists for backward compatibility.
if(MKL_MPI STREQUAL "mshpc")
  set(MKL_MPI msmpi)
endif()
find_package_handle_standard_args(MKL REQUIRED_VARS MKL_MPI)
# Fix for Intel MPI 2021
if(USE_MPI AND MKL_MPI STREQUAL "intelmpi")
  if(UNIX AND NOT APPLE AND $ENV{I_MPI_ROOT} MATCHES "2021.")
    set(MPI_C_ADDITIONAL_INCLUDE_DIRS $ENV{I_MPI_ROOT}/include)
    set(MPI_Fortran_ADDITIONAL_INCLUDE_DIRS $ENV{I_MPI_ROOT}/include)
  endif()
endif()

  # Force mshpc to be the first in mpi find package
  if(MKL_MPI STREQUAL "msmpi" OR MKL_MPI STREQUAL "mshpc")
    set(MPI_GUESS_LIBRARY_NAME MSMPI)
    # temporary workaround to skip internal fortran mpi testing
    set(MPI_Fortran_WORKS 1)
  endif()

# Checkpoint - Verify if required options are defined
find_package_handle_standard_args(MKL REQUIRED_VARS MKL_ROOT MKL_ARCH MKL_INCLUDE MKL_LINK MKL_THREADING MKL_INTERFACE_FULL)

# Provides a list of IMPORTED targets for the project
if(NOT DEFINED MKL_IMPORTED_TARGETS)
  set(MKL_IMPORTED_TARGETS "")
endif()

# Clear temporary variables
set(MKL_C_COPT "")
set(MKL_F_COPT "")
set(MKL_SDL_COPT "")
set(MKL_DPCPP_COPT "")
set(MKL_DPCPP_LOPT "")
set(MKL_OFFLOAD_COPT "")
set(MKL_OFFLOAD_LOPT "")

set(MKL_SUPP_LINK "")    # Other link options. Usually at the end of the link-line.
set(MKL_LINK_LINE)       # For MPI only
set(MKL_ENV_PATH "")     # Temporary variable to work with PATH
set(MKL_ENV "")          # Exported environment variables

# Modify PATH variable to make it CMake-friendly
set(OLD_PATH $ENV{PATH})
string(REPLACE ";" "\;" OLD_PATH "${OLD_PATH}")

# Compiler options
if(GNU_C_COMPILER OR GNU_Fortran_COMPILER)
  if(MKL_ARCH STREQUAL "ia32")
    list(APPEND MKL_C_COPT -m32)
    list(APPEND MKL_F_COPT -m32)
  else()
    list(APPEND MKL_C_COPT -m64)
    list(APPEND MKL_F_COPT -m64)
  endif()
endif()

# Additonal compiler & linker options
if(DPCPP_COMPILER OR ENABLE_OMP_OFFLOAD)
  list(APPEND MKL_DPCPP_COPT -fsycl-unnamed-lambda)
  if(MKL_LINK STREQUAL "static")
    list(APPEND MKL_DPCPP_LOPT "-fsycl-device-code-split=per_kernel")
    if(NOT "Fortran" IN_LIST CURR_LANGS)
      list(APPEND MKL_OFFLOAD_LOPT "-fsycl-device-code-split=per_kernel")
    endif()
  endif()
endif()

# For OpenMP Offload
if(ENABLE_OMP_OFFLOAD)
  if(WIN32)
    list(APPEND MKL_OFFLOAD_COPT -Qopenmp -Qopenmp-targets:spir64 -MD)
    list(APPEND MKL_OFFLOAD_LOPT -Qopenmp -Qopenmp-targets:spir64 -fsycl)
    set(SKIP_LIBPATH ON)
  else()
    list(APPEND MKL_OFFLOAD_COPT -fiopenmp -fopenmp-targets=spir64)
    list(APPEND MKL_OFFLOAD_LOPT -fiopenmp -fopenmp-targets=spir64 -fsycl)
    if(APPLE)
      list(APPEND MKL_SUPP_LINK -lc++)
    else()
      list(APPEND MKL_SUPP_LINK -lstdc++)
    endif()
  endif()
endif()

# For selected Interface
if(MKL_INTERFACE_FULL)
  if(MKL_ARCH STREQUAL "ia32")
    if(GNU_Fortran_COMPILER)
      set(MKL_SDL_IFACE_ENV "GNU")
    endif()
  else()
    if(GNU_Fortran_COMPILER)
      set(MKL_SDL_IFACE_ENV "GNU,${MKL_INTERFACE}")
    else()
      set(MKL_SDL_IFACE_ENV "${MKL_INTERFACE}")
    endif()
    if(MKL_INTERFACE STREQUAL "ilp64")
      if("Fortran" IN_LIST CURR_LANGS)
        if(INTEL_COMPILER)
          if(WIN32)
            list(APPEND MKL_F_COPT "-4I8")
          else()
            list(APPEND MKL_F_COPT "-i8")
          endif()
        elseif(GNU_Fortran_COMPILER)
          list(APPEND MKL_F_COPT "-fdefault-integer-8")
        elseif(PGI_COMPILER)
          list(APPEND MKL_F_COPT "-i8")
        endif()
      endif()
      list(PREPEND MKL_C_COPT "-DMKL_ILP64")
      list(PREPEND MKL_SDL_COPT "-DMKL_ILP64")
      list(PREPEND MKL_OFFLOAD_COPT "-DMKL_ILP64")
    else()
      # lp64
    endif()
  endif()
  if(MKL_SDL_IFACE_ENV)
    string(TOUPPER ${MKL_SDL_IFACE_ENV} MKL_SDL_IFACE_ENV)
  endif()
endif() # MKL_INTERFACE_FULL

# All MKL Libraries
set(MKL_SYCL          mkl_sycl)
set(MKL_IFACE_LIB     mkl_${MKL_INTERFACE_FULL})
set(MKL_CORE          mkl_core)
set(MKL_THREAD        mkl_${MKL_THREADING})
set(MKL_SDL           mkl_rt)
if(MKL_ARCH STREQUAL "ia32")
  set(MKL_BLAS95      mkl_blas95)
  set(MKL_LAPACK95    mkl_lapack95)
else()
  set(MKL_BLAS95      mkl_blas95_${MKL_INTERFACE})
  set(MKL_LAPACK95    mkl_lapack95_${MKL_INTERFACE})
endif()
# BLACS
set(MKL_BLACS mkl_blacs_${MKL_MPI}_${MKL_INTERFACE})
if(UNIX AND NOT APPLE AND MKL_MPI MATCHES "mpich")
  # MPICH is compatible with INTELMPI Wrappers on Linux
  set(MKL_BLACS mkl_blacs_intelmpi_${MKL_INTERFACE})
endif()
if(WIN32)
  if(MKL_MPI STREQUAL "msmpi")
    if("Fortran" IN_LIST CURR_LANGS)
      list(APPEND MKL_SUPP_LINK "msmpifec.lib")
    endif()
    # MSMPI and MSHPC are supported with the same BLACS library
    set(MKL_BLACS mkl_blacs_msmpi_${MKL_INTERFACE})
    if(NOT MKL_LINK STREQUAL "static")
      set(MKL_BLACS mkl_blacs_${MKL_INTERFACE})
      set(MKL_BLACS_ENV MSMPI)
    endif()
  elseif(MKL_MPI STREQUAL "intelmpi" AND NOT MKL_LINK STREQUAL "static")
    set(MKL_BLACS mkl_blacs_${MKL_INTERFACE})
    set(MKL_BLACS_ENV INTELMPI)
  endif()
endif()
# CDFT & SCALAPACK
set(MKL_CDFT      mkl_cdft_core)
set(MKL_SCALAPACK mkl_scalapack_${MKL_INTERFACE})


if (UNIX)
  if(NOT APPLE)
    if(MKL_LINK STREQUAL "static")
      set(START_GROUP "-Wl,--start-group")
      set(END_GROUP "-Wl,--end-group")
      if(DPCPP_COMPILER OR ENABLE_OMP_OFFLOAD)
        set(EXPORT_DYNAMIC "-Wl,-export-dynamic")
      endif()
    elseif(MKL_LINK STREQUAL "dynamic")
      set(MKL_RPATH "-Wl,-rpath=$<TARGET_FILE_DIR:MKL::${MKL_CORE}>")
      if((GNU_Fortran_COMPILER OR PGI_COMPILER) AND "Fortran" IN_LIST CURR_LANGS)
        set(NO_AS_NEEDED -Wl,--no-as-needed)
      endif()
    else()
      set(MKL_RPATH "-Wl,-rpath=$<TARGET_FILE_DIR:MKL::${MKL_SDL}>")
    endif()
  endif()
endif()

# Create a list of requested libraries, based on input options (MKL_LIBRARIES)
# Create full link-line in MKL_LINK_LINE
list(APPEND MKL_LINK_LINE $<IF:$<BOOL:${ENABLE_OMP_OFFLOAD}>,${MKL_OFFLOAD_LOPT},>
    $<IF:$<BOOL:${DPCPP_COMPILER}>,${MKL_DPCPP_LOPT},> ${EXPORT_DYNAMIC} ${NO_AS_NEEDED} ${MKL_RPATH})
if(ENABLE_BLAS95)
  list(APPEND MKL_LIBRARIES ${MKL_BLAS95})
  list(APPEND MKL_LINK_LINE MKL::${MKL_BLAS95})
endif()
if(ENABLE_LAPACK95)
  list(APPEND MKL_LIBRARIES ${MKL_LAPACK95})
  list(APPEND MKL_LINK_LINE MKL::${MKL_LAPACK95})
endif()
if(ENABLE_SCALAPACK)
  list(APPEND MKL_LIBRARIES ${MKL_SCALAPACK})
  list(APPEND MKL_LINK_LINE MKL::${MKL_SCALAPACK})
endif()
if(DPCPP_COMPILER OR ENABLE_OMP_OFFLOAD)
  list(APPEND MKL_LIBRARIES ${MKL_SYCL})
  list(APPEND MKL_LINK_LINE MKL::${MKL_SYCL})
endif()
list(APPEND MKL_LINK_LINE ${START_GROUP})
if(ENABLE_CDFT)
  list(APPEND MKL_LIBRARIES ${MKL_CDFT})
  list(APPEND MKL_LINK_LINE MKL::${MKL_CDFT})
endif()
if(MKL_LINK STREQUAL "sdl")
  list(APPEND MKL_LIBRARIES ${MKL_SDL})
  list(APPEND MKL_LINK_LINE MKL::${MKL_SDL})
else()
  list(APPEND MKL_LIBRARIES ${MKL_IFACE_LIB} ${MKL_CORE} ${MKL_THREAD})
  list(APPEND MKL_LINK_LINE MKL::${MKL_IFACE_LIB} MKL::${MKL_CORE} MKL::${MKL_THREAD})
endif()
if(USE_MPI)
  list(APPEND MKL_LIBRARIES ${MKL_BLACS})
  list(APPEND MKL_LINK_LINE MKL::${MKL_BLACS})
endif()
list(APPEND MKL_LINK_LINE ${END_GROUP})

# Find all requested libraries
foreach(lib ${MKL_LIBRARIES})
  unset(${lib}_file CACHE)
  if(MKL_LINK STREQUAL "static" AND NOT ${lib} STREQUAL ${MKL_SDL})
    find_library(${lib}_file ${LIB_PREFIX}${lib}${LIB_EXT}
                  PATHS ${MKL_ROOT}
                  PATH_SUFFIXES "lib" "lib/${MKL_ARCH}")
    add_library(MKL::${lib} STATIC IMPORTED)
  else()
    find_library(${lib}_file NAMES ${LIB_PREFIX}${lib}${DLL_EXT} ${lib}
                  PATHS ${MKL_ROOT}
                  PATH_SUFFIXES "lib" "lib/${MKL_ARCH}")
    add_library(MKL::${lib} SHARED IMPORTED)
  endif()
  find_package_handle_standard_args(MKL REQUIRED_VARS ${lib}_file)
  # CMP0111, implemented in CMake 3.20+ requires a shared library target on Windows
  # to be defined with IMPLIB and LOCATION property.
  # It also requires a static library target to be defined with LOCATION property.
  # Setting the policy to OLD usage, using cmake_policy() does not work as of 3.20.0, hence the if-else below.
  if(WIN32 AND NOT MKL_LINK STREQUAL "static")
    set_target_properties(MKL::${lib} PROPERTIES IMPORTED_IMPLIB "${${lib}_file}")
    # Find corresponding DLL
    set(MKL_DLL_GLOB ${lib}.*.dll)
    file(GLOB MKL_DLL_FILE "${MKL_ROOT}/redist/${MKL_ARCH}/${MKL_DLL_GLOB}"
        "${MKL_ROOT}/../redist/${MKL_ARCH}/${MKL_DLL_GLOB}"
        "${MKL_ROOT}/../redist/${MKL_ARCH}/mkl/${MKL_DLL_GLOB}")
    if(${MKL_DLL_FILE})
      set_target_properties(MKL::${lib} PROPERTIES IMPORTED_LOCATION "${MKL_DLL_FILE}")
    else()
      # MKL interface library on Windows does not have a corresponding DLL.
      set_target_properties(MKL::${lib} PROPERTIES IMPORTED_LOCATION "${${lib}_file}")
    endif()
  else()
    set_target_properties(MKL::${lib} PROPERTIES IMPORTED_LOCATION "${${lib}_file}")
  endif()
  list(APPEND MKL_IMPORTED_TARGETS MKL::${lib})
endforeach()

# Threading selection
if(MKL_THREADING)
  if(MKL_THREADING STREQUAL "tbb_thread")
    find_package(TBB REQUIRED CONFIG COMPONENTS tbb)
    set(MKL_THREAD_LIB $<TARGET_LINKER_FILE:TBB::tbb>)
    set(MKL_SDL_THREAD_ENV "TBB")
    get_property(TBB_LIB TARGET TBB::tbb PROPERTY IMPORTED_LOCATION_RELEASE)
    get_filename_component(TBB_LIB_DIR ${TBB_LIB} DIRECTORY)
    if(UNIX)
      if(CMAKE_SKIP_BUILD_RPATH)
        set(TBB_LINK "-L${TBB_LIB_DIR} -ltbb")
      else()
        set(TBB_LINK "-Wl,-rpath,${TBB_LIB_DIR} -L${TBB_LIB_DIR} -ltbb")
      endif()
      list(APPEND MKL_SUPP_LINK ${TBB_LINK})
      if(APPLE)
        list(APPEND MKL_SUPP_LINK -lc++)
      else()
        list(APPEND MKL_SUPP_LINK -lstdc++)
      endif()
    endif()
    if(WIN32 OR APPLE)
      set(MKL_ENV_PATH ${TBB_LIB_DIR})
    endif()
  elseif(MKL_THREADING MATCHES ".*_thread.*")
    if(MKL_THREADING STREQUAL "pgi_thread")
      list(APPEND MKL_SUPP_LINK -mp -pgf90libs)
      set(MKL_SDL_THREAD_ENV "PGI")
    elseif(MKL_THREADING STREQUAL "gnu_thread")
      list(APPEND MKL_SUPP_LINK -lgomp)
      set(MKL_SDL_THREAD_ENV "GNU")
    else()
      # intel_thread
      if(UNIX)
        set(MKL_OMP_LIB iomp5)
        set(LIB_EXT ".so")
        if(APPLE)
          set(LIB_EXT ".dylib")
        endif()
      else()
        set(MKL_OMP_LIB libiomp5md)
      endif()
      set(MKL_SDL_THREAD_ENV "INTEL")
      set(OMP_LIBNAME ${LIB_PREFIX}${MKL_OMP_LIB}${LIB_EXT})

      find_library(OMP_LIBRARY ${OMP_LIBNAME}
        HINTS $ENV{LIB} $ENV{LIBRARY_PATH} $ENV{MKLROOT} ${MKL_ROOT} ${CMPLR_ROOT}
        PATH_SUFFIXES "lib" "lib/${MKL_ARCH}"
               "lib/${MKL_ARCH}_lin" "lib/${MKL_ARCH}_win"
               "linux/compiler/lib/${MKL_ARCH}"
               "linux/compiler/lib/${MKL_ARCH}_lin"
               "windows/compiler/lib/${MKL_ARCH}"
               "windows/compiler/lib/${MKL_ARCH}_win"
               "../compiler/lib/${MKL_ARCH}_lin" "../compiler/lib/${MKL_ARCH}_win"
               "../compiler/lib/${MKL_ARCH}" "../compiler/lib"
               "../../compiler/latest/linux/compiler/lib/${MKL_ARCH}"
               "../../compiler/latest/linux/compiler/lib/${MKL_ARCH}_lin"
               "../../compiler/latest/windows/compiler/lib/${MKL_ARCH}"
               "../../compiler/latest/windows/compiler/lib/${MKL_ARCH}_win"
               "../../compiler/latest/mac/compiler/lib")
      if(WIN32)
        set(OMP_DLLNAME ${LIB_PREFIX}${MKL_OMP_LIB}.dll)
        find_path(OMP_DLL_DIR ${OMP_DLLNAME}
          HINTS $ENV{LIB} $ENV{LIBRARY_PATH} $ENV{MKLROOT} ${MKL_ROOT} ${CMPLR_ROOT}
          PATH_SUFFIXES "redist/${MKL_ARCH}"
               "redist/${MKL_ARCH}_win" "redist/${MKL_ARCH}_win/compiler"
               "../redist/${MKL_ARCH}/compiler" "../compiler/lib"
               "../../compiler/latest/windows/redist/${MKL_ARCH}_win"
               "../../compiler/latest/windows/redist/${MKL_ARCH}_win/compiler"
               "../../compiler/latest/windows/compiler/redist/${MKL_ARCH}_win"
               "../../compiler/latest/windows/compiler/redist/${MKL_ARCH}_win/compiler")
        find_package_handle_standard_args(MKL REQUIRED_VARS OMP_DLL_DIR)
        set(MKL_ENV_PATH "${OMP_DLL_DIR}")
      endif()

      if(WIN32 AND SKIP_LIBPATH)
        # Only for Intel OpenMP Offload
        set(OMP_LINK "libiomp5md.lib")
      else()
        set(OMP_LINK "${OMP_LIBRARY}")
      endif()
      find_package_handle_standard_args(MKL REQUIRED_VARS OMP_LIBRARY OMP_LINK)
      set(MKL_THREAD_LIB ${OMP_LINK})
    endif()
  else()
    # Sequential threading
    set(MKL_SDL_THREAD_ENV "SEQUENTIAL")
  endif()
endif() # MKL_THREADING

if (UNIX)
  list(APPEND MKL_SUPP_LINK -lm -ldl -lpthread)
endif()

if(DPCPP_COMPILER OR ENABLE_OMP_OFFLOAD)
  list(APPEND MKL_SUPP_LINK ${LINK_PREFIX}sycl${LINK_SUFFIX} ${LINK_PREFIX}OpenCL${LINK_SUFFIX})
endif()

# Setup link types based on input options
set(LINK_TYPES "")

if(DPCPP_COMPILER)
  add_library(MKL::MKL_DPCPP INTERFACE IMPORTED GLOBAL)
  target_compile_options(MKL::MKL_DPCPP INTERFACE ${MKL_DPCPP_COPT})
  target_link_libraries(MKL::MKL_DPCPP INTERFACE ${MKL_LINK_LINE} ${MKL_THREAD_LIB} ${MKL_SUPP_LINK})
  list(APPEND LINK_TYPES MKL::MKL_DPCPP)
endif()
# Single target for all C, Fortran link-lines
add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
target_compile_options(MKL::MKL INTERFACE
    $<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,C>:${MKL_C_COPT}>
    $<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,Fortran>:${MKL_F_COPT}>
    $<IF:$<BOOL:${ENABLE_OMP_OFFLOAD}>,${MKL_OFFLOAD_COPT},>)
# ifx doesn't support full lib path on Windows
if(WIN32 AND ENABLE_OMP_OFFLOAD AND CMAKE_Fortran_COMPILER MATCHES "ifx.exe")
  target_link_libraries(MKL::MKL INTERFACE ${MKL_OFFLOAD_LOPT} ${EXPORT_DYNAMIC} ${NO_AS_NEEDED} ${MKL_RPATH}
    ${START_GROUP} $<TARGET_FILE_NAME:MKL::${MKL_SYCL}>
    $<TARGET_FILE_NAME:MKL::${MKL_IFACE_LIB}> $<TARGET_FILE_NAME:MKL::${MKL_THREAD}> $<TARGET_FILE_NAME:MKL::${MKL_CORE}>
    ${END_GROUP} ${MKL_THREAD_LIB} ${MKL_SUPP_LINK})
else()
  target_link_libraries(MKL::MKL INTERFACE ${MKL_LINK_LINE} ${MKL_THREAD_LIB} ${MKL_SUPP_LINK})
endif()
list(APPEND LINK_TYPES MKL::MKL)

foreach(link ${LINK_TYPES})
  # Set properties on all INTERFACE targets
  target_include_directories(${link} BEFORE INTERFACE "${MKL_INCLUDE}")
  list(APPEND MKL_IMPORTED_TARGETS ${link})
endforeach(link) # LINK_TYPES

if(MKL_LINK STREQUAL "sdl")
  list(APPEND MKL_ENV "MKL_INTERFACE_LAYER=${MKL_SDL_IFACE_ENV}" "MKL_THREADING_LAYER=${MKL_SDL_THREAD_ENV}")
endif()
if(WIN32 AND NOT MKL_LINK STREQUAL "static")
  list(APPEND MKL_ENV "MKL_BLACS_MPI=${MKL_BLACS_ENV}")
endif()

# Add MKL dynamic libraries if RPATH is not defined on Unix
if(UNIX AND CMAKE_SKIP_BUILD_RPATH)
  if(MKL_LINK STREQUAL "sdl")
    set(MKL_LIB_DIR $<TARGET_FILE_DIR:MKL::${MKL_SDL}>)
  else()
    set(MKL_LIB_DIR $<TARGET_FILE_DIR:MKL::${MKL_CORE}>)
  endif()
  if(APPLE)
    list(APPEND MKL_ENV "DYLD_LIBRARY_PATH=${MKL_LIB_DIR}\;$ENV{DYLD_LIBRARY_PATH}")
  else()
    list(APPEND MKL_ENV "LD_LIBRARY_PATH=${MKL_LIB_DIR}\;$ENV{LD_LIBRARY_PATH}")
  endif()
endif()

# Add MKL dynamic libraries to PATH on Windows
if(WIN32 AND NOT MKL_LINK STREQUAL "static")
  get_filename_component(MKL_DLL_DIR ${MKL_DLL_FILE} DIRECTORY)
  set(MKL_ENV_PATH "${MKL_DLL_DIR}\;${MKL_ENV_PATH}")
endif()

if(MKL_ENV_PATH)
  list(APPEND MKL_ENV "PATH=${MKL_ENV_PATH}\;${OLD_PATH}")
  if(APPLE)
    list(APPEND MKL_ENV "DYLD_LIBRARY_PATH=${MKL_ENV_PATH}\:${OLD_PATH}")
  endif()
endif()

endif() # MKL_LIBRARIES
