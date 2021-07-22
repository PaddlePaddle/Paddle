file(GLOB UTILS_PY_FILES . ./paddle/legacy/utils/*.py)
file(GLOB_RECURSE FLUID_PY_FILES ./paddle/fluid/*.py)
set(PY_FILES paddle/__init__.py
  ${UTILS_PY_FILES}
  ${FLUID_PY_FILES})

if(WITH_GPU)
  SET(PACKAGE_NAME "paddlepaddle-gpu")
elseif(WITH_ROCM)
  SET(PACKAGE_NAME "paddlepaddle-rocm")
elseif(WITH_ASCEND_CL)
  SET(PACKAGE_NAME "paddlepaddle-npu")
else()
  SET(PACKAGE_NAME "paddlepaddle")
endif()

set(SETUP_LOG_FILE "setup.py.log")

set(FLUID_CORE_NAME "core")
if(WITH_AVX AND AVX_FOUND)
  set(FLUID_CORE_NAME "${FLUID_CORE_NAME}_avx")
  if(NOT DEFINED NOAVX_CORE_FILE OR NOAVX_CORE_FILE STREQUAL "")
    message(STATUS "MESSAGE: This is just a message for publishing release.
      You are building AVX version without NOAVX core.
      So the wheel package may fail on NOAVX machine.
      You can add -DNOAVX_CORE_FILE=/path/to/your/core_noavx.* in cmake command
      to get a full wheel package to resolve this warning.
      While, this version will still work on local machine.")
  endif()

  if(NOAVX_CORE_FILE AND NOT EXISTS "${NOAVX_CORE_FILE}")
    message(FATAL_ERROR "The file ${NOAVX_CORE_FILE} does not exist!")
  endif()

  set(HAS_NOAVX_CORE ON)
else()
  set(FLUID_CORE_NAME "${FLUID_CORE_NAME}_noavx")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

set(FLUID_DST_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/)

IF(WIN32)
    # Python would use the .pyd by default under Windows series platform
    set(FLUID_CORE ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.pyd)
    set(FLUID_CORE_LIB ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.lib)
    
    add_custom_command(OUTPUT ${FLUID_CORE}
      COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${FLUID_CORE}
      COMMAND cmake -E copy $<TARGET_LINKER_FILE:paddle_pybind> ${FLUID_CORE_LIB}
      DEPENDS paddle_pybind)

    set(FLUID_NOAVX_CORE ${FLUID_DST_DIR}/core_noavx.pyd)  
ELSE()
    set(FLUID_CORE ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.so)
    add_custom_command(OUTPUT ${FLUID_CORE}
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${FLUID_CORE}
        DEPENDS paddle_pybind)

    set(FLUID_NOAVX_CORE ${FLUID_DST_DIR}/core_noavx.so)
ENDIF()

set(FLUID_CORE_DEPS ${FLUID_CORE})

if(HAS_NOAVX_CORE AND EXISTS "${NOAVX_CORE_FILE}")
  get_filename_component(NOAVX_CORE_NAME ${NOAVX_CORE_FILE} NAME)
  get_filename_component(NOAVX_CORE_EXT ${NOAVX_CORE_FILE} EXT)
  if(WIN32)
    if(NOT NOAVX_CORE_EXT STREQUAL ".pyd")
      message(FATAL_ERROR "Wrong file ${NOAVX_CORE_NAME}, the ext does not match windows *.pyd!")
    endif()
  else()
    if(NOT NOAVX_CORE_EXT STREQUAL ".so")
      message(FATAL_ERROR "Wrong file ${NOAVX_CORE_NAME}, the ext does not match *.so!")
    endif()
  endif()
  add_custom_command(OUTPUT ${FLUID_NOAVX_CORE}
    COMMAND cmake -E copy ${NOAVX_CORE_FILE} ${FLUID_NOAVX_CORE} DEPENDS paddle_pybind)
  list(APPEND FLUID_CORE_DEPS ${FLUID_NOAVX_CORE})
endif()

add_custom_target(copy_paddle_pybind ALL DEPENDS ${FLUID_CORE_DEPS})

IF(WIN32)
  add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/python/paddle ${PADDLE_BINARY_DIR}/python/paddle/
    COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_paddle_pybind ${FLUID_CORE} framework_py_proto profiler_py_proto ${PY_FILES})
ELSE(WIN32)
  add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND touch stub.cc
    COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle ${PADDLE_BINARY_DIR}/python
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMENT "Packing whl packages------>>>"
    DEPENDS copy_paddle_pybind ${FLUID_CORE} framework_py_proto profiler_py_proto ${PY_FILES})
ENDIF()

add_custom_target(paddle_python ALL DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp)

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if (WITH_TESTING)
  add_subdirectory(paddle/reader/tests)
  add_subdirectory(paddle/dataset/tests)
  add_subdirectory(paddle/tests)
  add_subdirectory(paddle/fluid/tests)
  add_subdirectory(paddle/fluid/contrib/tests)
  add_subdirectory(paddle/fluid/contrib/slim/tests)
endif()
install(DIRECTORY ${PADDLE_PYTHON_PACKAGE_DIR}
    DESTINATION opt/paddle/share/wheels
)

if(APPLE)
  find_program(INSTALL_NAME_TOOL_EXECUTABLE install_name_tool)
  if(NOT INSTALL_NAME_TOOL_EXECUTABLE)
    message(FATAL_ERROR "install_name_tool not found, please check.\n")
  endif()
endif()
if(LINUX AND NOT WITH_SW AND NOT WITH_ARM)
  find_program(PATCHELF_EXECUTABLE patchelf)
  if(NOT PATCHELF_EXECUTABLE)
    message(FATAL_ERROR "patchelf not found, please install it.\n"
            "For Ubuntu, the command is: apt-get install -y patchelf.")
  endif()
endif()
