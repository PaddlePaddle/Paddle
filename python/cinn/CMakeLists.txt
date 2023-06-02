file(GLOB_RECURSE CINN_PY_FILES ${PROJECT_SOURCE_DIR}/python/cinn/*.py)
set(CINN_PYTHON_DIR ${PROJECT_SOURCE_DIR}/python/cinn)
set(CINN_CORE_API ${PROJECT_SOURCE_DIR}/python/cinn/core_api.so)

if(WITH_GPU)
  set(PACKAGE_NAME "cinn-gpu")
else()
  set(PACKAGE_NAME "cinn")
endif()
set(SETUP_LOG_FILE "setup.py.log")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp ${PY_VERSION} REQUIRED)
  find_package(PythonLibs ${PY_VERSION} REQUIRED)
endif()

message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

# There may be a link file called core_api.so under the dir ${CINN_PYTHON_DIR} due to the `mac_doc`
# function defined in build.sh. So, we need to copy the directory ${CINN_PYTHON_DIR} first and
# then core_api.so.
add_custom_command(
  OUTPUT ${CINN_CORE_API} POST_BUILD
  COMMAND cp -rf --remove-destination ${CINN_PYTHON_DIR}
          ${CMAKE_BINARY_DIR}/python/cinn
  COMMAND cp --remove-destination
          ${CMAKE_BINARY_DIR}/paddle/cinn/pybind/core_api.so ${CINN_CORE_API}
  COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && ${PYTHON_EXECUTABLE} setup.py
          bdist_wheel
  DEPENDS core_api ${CINN_PY_FILES})

add_custom_target(COPY_CINN_CORE_API ALL DEPENDS ${CINN_CORE_API}
                                                 ${CINN_PY_FILES})
