if(NOT WITH_ROCM)
  return()
endif()

# Now we don't support RCCL on windows
if(WIN32)
  return()
endif()

if(WITH_RCCL)
  get_filename_component(_ROCM_PATH_REAL "${ROCM_PATH}" REALPATH)
  if(_ROCM_PATH_REAL)
    set(ROCM_PATH "${_ROCM_PATH_REAL}")
  endif()

  set(RCCL_ROOT
      ${ROCM_PATH}/rccl
      CACHE PATH "RCCL ROOT")

  set(_RCCL_HEADER_CANDIDATES
      "${ROCM_PATH}/include/rccl.h"
      "${ROCM_PATH}/include/rccl/rccl.h"
      "${ROCM_PATH}/rccl/include/rccl.h"
      "${ROCM_PATH}/rccl/include/rccl/rccl.h"
      "${RCCL_ROOT}/include/rccl.h"
      "${RCCL_ROOT}/include/rccl/rccl.h"
      "$ENV{RCCL_ROOT}/include/rccl.h"
      "$ENV{RCCL_ROOT}/include/rccl/rccl.h")

  set(RCCL_INCLUDE_DIR "RCCL_INCLUDE_DIR-NOTFOUND")
  foreach(_cand IN LISTS _RCCL_HEADER_CANDIDATES)
    if(EXISTS "${_cand}")
      get_filename_component(RCCL_INCLUDE_DIR "${_cand}" DIRECTORY)
      break()
    endif()
  endforeach()

  if(RCCL_INCLUDE_DIR STREQUAL "RCCL_INCLUDE_DIR-NOTFOUND")
    file(GLOB _RCCL_GLOB "${ROCM_PATH}/include/rccl/**/rccl.h"
         "${ROCM_PATH}/rccl/**/rccl.h")
    foreach(_g IN LISTS _RCCL_GLOB)
      if(EXISTS "${_g}")
        get_filename_component(RCCL_INCLUDE_DIR "${_g}" DIRECTORY)
        break()
      endif()
    endforeach()
  endif()

  if(RCCL_INCLUDE_DIR STREQUAL "RCCL_INCLUDE_DIR-NOTFOUND")
    unset(RCCL_INCLUDE_DIR CACHE)
    find_path(
      RCCL_INCLUDE_DIR
      rccl.h
      PATHS ${ROCM_PATH}/include ${ROCM_PATH}/include/rccl ${ROCM_PATH}/rccl/include
            ${ROCM_PATH}/rccl/include/rccl ${RCCL_ROOT} ${RCCL_ROOT}/include
            ${RCCL_ROOT}/local/include $ENV{RCCL_ROOT} $ENV{RCCL_ROOT}/include
            $ENV{RCCL_ROOT}/local/include
      NO_DEFAULT_PATH)
  endif()

  if(TARGET rccl::rccl)
    get_target_property(_rccl_iface_includes rccl::rccl
                        INTERFACE_INCLUDE_DIRECTORIES)
    if(_rccl_iface_includes)
      foreach(_inc IN LISTS _rccl_iface_includes)
        if(EXISTS "${_inc}/rccl.h")
          set(RCCL_INCLUDE_DIR "${_inc}")
          break()
        elseif(EXISTS "${_inc}/rccl/rccl.h")
          set(RCCL_INCLUDE_DIR "${_inc}/rccl")
          break()
        endif()
      endforeach()
    endif()
  elseif(TARGET rccl)
    get_target_property(_rccl_iface_includes rccl INTERFACE_INCLUDE_DIRECTORIES)
    if(_rccl_iface_includes)
      foreach(_inc IN LISTS _rccl_iface_includes)
        if(EXISTS "${_inc}/rccl.h")
          set(RCCL_INCLUDE_DIR "${_inc}")
          break()
        elseif(EXISTS "${_inc}/rccl/rccl.h")
          set(RCCL_INCLUDE_DIR "${_inc}/rccl")
          break()
        endif()
      endforeach()
    endif()
  endif()

  if(RCCL_INCLUDE_DIR MATCHES "-NOTFOUND$" OR NOT RCCL_INCLUDE_DIR)
    message(
      FATAL_ERROR
        "RCCL header not found. Checked ROCM_PATH=${ROCM_PATH}, RCCL_ROOT=${RCCL_ROOT}, RCCL_ROOT(env)=$ENV{RCCL_ROOT}. "
        "Try: export RCCL_ROOT=/path/to/rccl or install rccl under ROCM_PATH.")
  endif()

  set(_RCCL_HEADER "${RCCL_INCLUDE_DIR}/rccl.h")
  if(NOT EXISTS "${_RCCL_HEADER}")
    set(_RCCL_HEADER "${RCCL_INCLUDE_DIR}/rccl/rccl.h")
  endif()
  if(NOT EXISTS "${_RCCL_HEADER}")
    message(FATAL_ERROR "RCCL header missing under ${RCCL_INCLUDE_DIR}")
  endif()

  file(READ ${_RCCL_HEADER} RCCL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)" RCCL_VERSION
               "${RCCL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1" RCCL_VERSION
                       "${RCCL_VERSION}")

  # 2604 for ROCM3.5 and 2708 for ROCM 3.9
  message(STATUS "Current RCCL header is ${_RCCL_HEADER}. "
                 "Current RCCL version is v${RCCL_VERSION}. ")
endif()
