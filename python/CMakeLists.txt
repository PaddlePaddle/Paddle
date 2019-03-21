file(GLOB UTILS_PY_FILES . ./paddle/legacy/utils/*.py)
file(GLOB_RECURSE FLUID_PY_FILES ./paddle/fluid/*.py)
set(PY_FILES paddle/__init__.py
  ${UTILS_PY_FILES}
  ${FLUID_PY_FILES})

if(WITH_GPU)
  SET(PACKAGE_NAME "paddlepaddle-gpu")
else()
  SET(PACKAGE_NAME "paddlepaddle")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
IF(WIN32)
    # Python would use the .pyd by default under Windows series platform
    set(FLUID_DST_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/)
    set(FLUID_CORE ${FLUID_DST_DIR}/core.pyd)
ELSE()
    set(FLUID_CORE ${PADDLE_BINARY_DIR}/python/paddle/fluid/core.so)
ENDIF()
add_custom_command(OUTPUT ${FLUID_CORE}
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${FLUID_CORE}
        DEPENDS paddle_pybind)
add_custom_target(copy_paddle_pybind ALL DEPENDS ${FLUID_CORE})

IF(WIN32)
    add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_SOURCE_DIR}/python/paddle ${PADDLE_BINARY_DIR}/python/paddle/
            COMMAND ${CMAKE_COMMAND} -E env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
            COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PADDLE_PYTHON_BUILD_DIR}/lib-python
            DEPENDS copy_paddle_pybind ${FLUID_CORE} framework_py_proto profiler_py_proto ${PY_FILES})
ELSE(WIN32)
	add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
		COMMAND touch stub.cc
		COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle ${PADDLE_BINARY_DIR}/python
		COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
		COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
		COMMAND ${CMAKE_COMMAND} -E remove_directory ${PADDLE_PYTHON_BUILD_DIR}/lib-python
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_PYTHON_BUILD_DIR}/lib* ${PADDLE_PYTHON_BUILD_DIR}/lib-python
		DEPENDS copy_paddle_pybind ${FLUID_CORE} framework_py_proto profiler_py_proto ${PY_FILES})
ENDIF()

add_custom_target(paddle_python ALL DEPENDS ${PADDLE_PYTHON_BUILD_DIR}/.timestamp)

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if (WITH_TESTING)
  add_subdirectory(paddle/reader/tests)
  add_subdirectory(paddle/dataset/tests)
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
if(LINUX)
  find_program(PATCHELF_EXECUTABLE patchelf)
  if(NOT PATCHELF_EXECUTABLE)
    message(FATAL_ERROR "patchelf not found, please install it.\n"
            "For Ubuntu, the command is: apt-get install -y patchelf.")
  endif()
endif(LINUX)
