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

  # find_package(rccl) in hip.cmake may cache RCCL_INCLUDE_DIR to a versioned
  # path (e.g. /opt/rocm-7.0.0/include) that does not exist in minimal images.
  # Resolve by probing real paths under ROCM_PATH first.
  set(_PADDLE_RCCL_INCLUDE_DIR "")
  foreach(_rccl_inc IN ITEMS "${ROCM_PATH}/include"
      "${ROCM_PATH}/rccl/include" "${RCCL_ROOT}/include" "${RCCL_ROOT}"
      "$ENV{RCCL_ROOT}/include" "$ENV{RCCL_ROOT}")
    if(_rccl_inc AND EXISTS "${_rccl_inc}/rccl.h")
      set(_PADDLE_RCCL_INCLUDE_DIR "${_rccl_inc}")
      break()
    endif()
  endforeach()

  if(NOT _PADDLE_RCCL_INCLUDE_DIR)
    unset(RCCL_INCLUDE_DIR CACHE)
    find_path(
      RCCL_INCLUDE_DIR rccl.h
      PATHS ${ROCM_PATH}/include ${ROCM_PATH}/rccl/include ${ROCM_PATH}
            ${RCCL_ROOT} ${RCCL_ROOT}/include ${RCCL_ROOT}/local/include
            $ENV{RCCL_ROOT} $ENV{RCCL_ROOT}/include $ENV{RCCL_ROOT}/local/include
      NO_DEFAULT_PATH)
    set(_PADDLE_RCCL_INCLUDE_DIR "${RCCL_INCLUDE_DIR}")
  endif()

  if(NOT _PADDLE_RCCL_INCLUDE_DIR OR NOT EXISTS
      "${_PADDLE_RCCL_INCLUDE_DIR}/rccl.h")
    message(
      FATAL_ERROR
        "rccl.h not found. Set ROCM_PATH (current: '${ROCM_PATH}') or install rccl dev. "
        "Tried ${ROCM_PATH}/include and related paths.")
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
