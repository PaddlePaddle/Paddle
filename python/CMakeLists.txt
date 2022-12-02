file(GLOB UTILS_PY_FILES . ./paddle/legacy/utils/*.py)
file(GLOB_RECURSE FLUID_PY_FILES ./paddle/fluid/*.py)
set(PY_FILES paddle/__init__.py ${UTILS_PY_FILES} ${FLUID_PY_FILES})

if(WITH_GPU)
  set(PACKAGE_NAME "paddlepaddle-gpu")
elseif(WITH_MLU)
  set(PACKAGE_NAME "paddlepaddle-mlu")
elseif(WITH_ROCM)
  set(PACKAGE_NAME "paddlepaddle-rocm")
elseif(WITH_ASCEND_CL)
  set(PACKAGE_NAME "paddlepaddle-npu")
elseif(WITH_XPU)
  set(PACKAGE_NAME "paddlepaddle-xpu")
elseif(WITH_IPU)
  set(PACKAGE_NAME "paddlepaddle-ipu")
else()
  set(PACKAGE_NAME "paddlepaddle")
endif()

set(SETUP_LOG_FILE "setup.py.log")

set(FLUID_CORE_NAME "libpaddle")

if(WITH_SETUP_INSTALL)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/env_dict.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/env_dict.py)
else()
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
endif()

set(FLUID_DST_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/)

if(WIN32)
  # Python would use the .pyd by default under Windows series platform
  set(FLUID_CORE ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.pyd)
  set(FLUID_CORE_LIB ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.lib)
  add_custom_command(
    OUTPUT ${FLUID_CORE}
    COMMAND cmake -E copy $<TARGET_FILE:libpaddle> ${FLUID_CORE}
    COMMAND cmake -E copy $<TARGET_LINKER_FILE:libpaddle> ${FLUID_CORE_LIB}
    DEPENDS libpaddle)
else()
  set(FLUID_CORE ${FLUID_DST_DIR}/${FLUID_CORE_NAME}.so)
  add_custom_command(
    OUTPUT ${FLUID_CORE}
    COMMAND cmake -E copy $<TARGET_FILE:paddle> ${FLUID_CORE}
    DEPENDS paddle)
endif()

set(FLUID_CORE_DEPS ${FLUID_CORE})

add_custom_target(copy_libpaddle ALL DEPENDS ${FLUID_CORE_DEPS})

# NOTE(zhiqiu): WHY?
# In `setup.py.in`, some dynamic libraries (eg, libxpuapi.so) are modified using
# patchelf. In rare cases, if the a linker is linking that dynamic library for
# some executables at the same time, a `file not recognized, file truncated`
# error may occur, result in the compilation error.
# So, add dependency to force the cpp tests built before running `setup.py.in`.
if(WITH_TESTING)
  add_dependencies(copy_libpaddle build_tests)
endif()

if(WIN32)
  if(WITH_SETUP_INSTALL)
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/python/paddle
        ${PADDLE_BINARY_DIR}/python/paddle/
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  else()
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/python/paddle
        ${PADDLE_BINARY_DIR}/python/paddle/
      COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py
              bdist_wheel
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  endif()
else()
  if(WITH_SETUP_INSTALL)
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND touch stub.cc
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle
              ${PADDLE_BINARY_DIR}/python
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  else()
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND touch stub.cc
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle
              ${PADDLE_BINARY_DIR}/python
      COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  endif()
endif()

add_custom_target(paddle_python ALL
                  DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp)

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if(WITH_TESTING)
  add_subdirectory(paddle/reader/tests)
  add_subdirectory(paddle/dataset/tests)
  add_subdirectory(paddle/tests)
  add_subdirectory(paddle/fluid/tests)
  add_subdirectory(paddle/fluid/contrib/tests)
  add_subdirectory(paddle/fluid/contrib/slim/tests)
endif()

if(NOT WITH_SETUP_INSTALL)
  install(DIRECTORY ${PADDLE_PYTHON_PACKAGE_DIR}
          DESTINATION opt/paddle/share/wheels)
endif()

if(APPLE)
  find_program(INSTALL_NAME_TOOL_EXECUTABLE install_name_tool)
  if(NOT INSTALL_NAME_TOOL_EXECUTABLE)
    message(FATAL_ERROR "install_name_tool not found, please check.\n")
  endif()
endif()
if(LINUX
   AND NOT WITH_SW
   AND NOT WITH_ARM)
  find_program(PATCHELF_EXECUTABLE patchelf)
  if(NOT PATCHELF_EXECUTABLE)
    message(
      FATAL_ERROR "patchelf not found, please install it.\n"
                  "For Ubuntu, the command is: apt-get install -y patchelf.")
  endif()
endif()
