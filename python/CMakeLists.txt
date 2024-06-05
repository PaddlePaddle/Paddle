if(WITH_CINN)
  file(GLOB_RECURSE CINN_PY_FILES ${PROJECT_SOURCE_DIR}/python/cinn/*.py)

  if(WITH_GPU)
    set(PACKAGE_NAME "cinn-gpu")
  else()
    set(PACKAGE_NAME "cinn")
  endif()
  set(SETUP_LOG_FILE "setup.py.log")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup_cinn.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/setup_cinn.py)

  if(NOT PYTHON_EXECUTABLE)
    find_package(PythonInterp ${PY_VERSION} REQUIRED)
    find_package(PythonLibs ${PY_VERSION} REQUIRED)
  endif()

  message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
endif()

file(GLOB UTILS_PY_FILES . ./paddle/legacy/utils/*.py)
file(GLOB_RECURSE FLUID_PY_FILES ./paddle/base/*.py)
set(PY_FILES paddle/__init__.py ${UTILS_PY_FILES} ${FLUID_PY_FILES})

if(WITH_GPU)
  set(PACKAGE_NAME "paddlepaddle-gpu")
elseif(WITH_ROCM)
  set(PACKAGE_NAME "paddlepaddle-dcu")
elseif(WITH_XPU)
  set(PACKAGE_NAME "paddlepaddle-xpu")
elseif(WITH_IPU)
  set(PACKAGE_NAME "paddlepaddle-ipu")
else()
  set(PACKAGE_NAME "paddlepaddle")
endif()

set(SETUP_LOG_FILE "setup.py.log")

set(FLUID_CORE_NAME "libpaddle")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/env_dict.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/env_dict.py)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

set(FLUID_DST_DIR ${PADDLE_BINARY_DIR}/python/paddle/base/)

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

# Standard op(phi op) description is defined in ops.yaml and dygraph_ops.yaml.
# When users define composite rules of some nonbasic op, as for definition of args,
# they are supposed to refer to standard op description. However, there exists
# some gap of description between current op and standard ones. So special dictionary
# is needed to record such gap for execution of composite rules.
# Todo: this custom_target will be moved to other place.

set(ops_yaml_path "${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/ops.yaml")

set(ops_legacy_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/inconsistent/dygraph_ops.yaml")

set(ops_compat_yaml_path
    "${PADDLE_SOURCE_DIR}/paddle/phi/ops/yaml/op_compat.yaml")

set(phi_ops_map_path
    "${PADDLE_SOURCE_DIR}/python/paddle/incubate/autograd/phi_ops_map.py")

add_custom_target(
  op_map_codegen ALL
  COMMAND
    "${PYTHON_EXECUTABLE}"
    "${PADDLE_SOURCE_DIR}/python/paddle/incubate/autograd/generate_op_map.py"
    "--ops_yaml_path=${ops_yaml_path}"
    "--ops_legacy_yaml_path=${ops_legacy_yaml_path}"
    "--ops_compat_yaml_path=${ops_compat_yaml_path}"
    "--phi_ops_map_path=${phi_ops_map_path}"
  VERBATIM)

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
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/test
              ${PADDLE_BINARY_DIR}/test
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  else()
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/python/paddle
        ${PADDLE_BINARY_DIR}/python/paddle/
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/test
              ${PADDLE_BINARY_DIR}/test
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})

    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp_wheel
      COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py
              bdist_wheel
      COMMENT "Packing whl packages------>>>"
      DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp)
  endif()
else()
  if(WITH_SETUP_INSTALL)
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND touch stub.cc
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle
              ${PADDLE_BINARY_DIR}/python
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/test ${PADDLE_BINARY_DIR}/
      COMMENT "Packing whl packages------>>>"
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
  else()
    add_custom_command(
      OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
      COMMAND touch stub.cc
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle
              ${PADDLE_BINARY_DIR}/python
      COMMAND cp -r ${PADDLE_SOURCE_DIR}/test ${PADDLE_BINARY_DIR}/
      DEPENDS copy_libpaddle ${FLUID_CORE} framework_py_proto profiler_py_proto
              pass_desc_py_proto ${PY_FILES})
    if(APPLE AND WITH_ARM)
      add_custom_command(
        OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp_wheel
        COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
                --plat-name=macosx_11_0_arm64
        COMMENT "Packing whl packages------>>>")
    else()
      add_custom_command(
        OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp_wheel
        COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        COMMENT "Packing whl packages------>>>")
    endif()
  endif()
endif()

add_custom_target(paddle_python ALL
                  DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp)

if(BUILD_WHL_PACKAGE AND NOT WITH_SETUP_INSTALL)
  add_custom_target(paddle_copy ALL
                    DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp_wheel)
  add_dependencies(paddle_copy paddle_python)
endif()

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

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
