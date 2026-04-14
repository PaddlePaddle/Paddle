if(NOT WITH_ROCM)
  return()
endif()

# Now we don't support RCCL on windows
if(WIN32)
  return()
endif()

if(WITH_RCCL)
  set(RCCL_ROOT
      ${ROCM_PATH}/rccl
      CACHE PATH "RCCL ROOT")

  # ROCm 7+ often uses include/rccl/rccl.h; older layouts use include/rccl.h
  macro(_paddle_try_rccl_header_parent _base)
    if(NOT _PADDLE_RCCL_INCLUDE_DIR AND _base)
      if(EXISTS "${_base}/rccl.h")
        set(_PADDLE_RCCL_INCLUDE_DIR "${_base}")
      elseif(EXISTS "${_base}/rccl/rccl.h")
        set(_PADDLE_RCCL_INCLUDE_DIR "${_base}/rccl")
      endif()
    endif()
  endmacro()

  set(_PADDLE_RCCL_INCLUDE_DIR "")

  # Include dirs from find_package(rccl) in hip.cmake (imported target).
  foreach(_rccl_tgt IN ITEMS rccl::rccl rccl)
    if(_PADDLE_RCCL_INCLUDE_DIR)
      break()
    endif()
    if(TARGET "${_rccl_tgt}")
      get_target_property(_rccl_iface_includes "${_rccl_tgt}"
                          INTERFACE_INCLUDE_DIRECTORIES)
      if(_rccl_iface_includes AND NOT _rccl_iface_includes STREQUAL
                                        "NOTFOUND")
        foreach(_one IN LISTS _rccl_iface_includes)
          string(FIND "${_one}" "$<" _paddle_rccl_genex)
          if(NOT _paddle_rccl_genex EQUAL -1)
            continue()
          endif()
          _paddle_try_rccl_header_parent("${_one}")
          if(_PADDLE_RCCL_INCLUDE_DIR)
            break()
          endif()
        endforeach()
      endif()
    endif()
  endforeach()

  get_filename_component(_paddle_rocm_real "${ROCM_PATH}" REALPATH)
  if(NOT _paddle_rocm_real)
    set(_paddle_rocm_real "${ROCM_PATH}")
  endif()
  get_filename_component(_paddle_rocm_parent "${ROCM_PATH}" DIRECTORY)

  # Probe unified / alternate layouts (symlinked ROCm root, versioned dirs).
  if(NOT _PADDLE_RCCL_INCLUDE_DIR)
    foreach(
      _rccl_inc IN ITEMS
      "${ROCM_PATH}/include"
      "${_paddle_rocm_real}/include"
      "${ROCM_PATH}/rccl/include"
      "${_paddle_rocm_real}/rccl/include"
      "${RCCL_ROOT}/include"
      "${RCCL_ROOT}"
      "$ENV{RCCL_ROOT}/include"
      "$ENV{RCCL_ROOT}")
      _paddle_try_rccl_header_parent("${_rccl_inc}")
      if(_PADDLE_RCCL_INCLUDE_DIR)
        break()
      endif()
    endforeach()
  endif()

  # e.g. /opt/rocm -> /opt/rocm-7.x; headers may only live under rocm-*.
  if(NOT _PADDLE_RCCL_INCLUDE_DIR AND _paddle_rocm_parent)
    file(GLOB _paddle_rccl_glob
         "${_paddle_rocm_parent}/rocm-*/include/rccl.h"
         "${_paddle_rocm_parent}/rocm-*/include/rccl/rccl.h")
    foreach(_rccl_h IN LISTS _paddle_rccl_glob)
      get_filename_component(_rccl_inc "${_rccl_h}" DIRECTORY)
      if(EXISTS "${_rccl_inc}/rccl.h")
        set(_PADDLE_RCCL_INCLUDE_DIR "${_rccl_inc}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT _PADDLE_RCCL_INCLUDE_DIR)
    unset(RCCL_INCLUDE_DIR CACHE)
    find_path(
      RCCL_INCLUDE_DIR rccl.h
      PATHS ${ROCM_PATH}/include ${_paddle_rocm_real}/include
            ${ROCM_PATH}/rccl/include ${ROCM_PATH}
            ${RCCL_ROOT} ${RCCL_ROOT}/include ${RCCL_ROOT}/local/include
            $ENV{RCCL_ROOT} $ENV{RCCL_ROOT}/include $ENV{RCCL_ROOT}/local/include
      PATH_SUFFIXES "" rccl
      NO_DEFAULT_PATH)
    set(_PADDLE_RCCL_INCLUDE_DIR "${RCCL_INCLUDE_DIR}")
    if(_PADDLE_RCCL_INCLUDE_DIR MATCHES "-NOTFOUND$")
      set(_PADDLE_RCCL_INCLUDE_DIR "")
    endif()
  endif()

  if(NOT _PADDLE_RCCL_INCLUDE_DIR OR NOT EXISTS
      "${_PADDLE_RCCL_INCLUDE_DIR}/rccl.h")
    message(
      FATAL_ERROR
        "rccl.h not found. Set ROCM_PATH (current: '${ROCM_PATH}') or install "
        "rccl development headers (e.g. rccl-dev). "
        "Checked REALPATH '${_paddle_rocm_real}', target rccl includes, and "
        "patterns ${_paddle_rocm_parent}/rocm-*/include/rccl.h and "
        ".../include/rccl/rccl.h.")
  endif()

  set(RCCL_INCLUDE_DIR
      "${_PADDLE_RCCL_INCLUDE_DIR}"
      CACHE PATH "RCCL include directory" FORCE)

  file(READ ${RCCL_INCLUDE_DIR}/rccl.h RCCL_VERSION_FILE_CONTENTS)

  string(REGEX MATCH "define NCCL_VERSION_CODE +([0-9]+)" RCCL_VERSION
               "${RCCL_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define NCCL_VERSION_CODE +([0-9]+)" "\\1" RCCL_VERSION
                       "${RCCL_VERSION}")

  # 2604 for ROCM3.5 and 2708 for ROCM 3.9
  message(STATUS "Current RCCL header is ${RCCL_INCLUDE_DIR}/rccl.h. "
                 "Current RCCL version is v${RCCL_VERSION}. ")
endif()
