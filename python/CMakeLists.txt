file(GLOB UTILS_PY_FILES . ./paddle/utils/*.py)
file(GLOB_RECURSE FLUID_PY_FILES ./paddle/fluid/*.py)
set(PY_FILES paddle/__init__.py
  ${UTILS_PY_FILES}
  ${FLUID_PY_FILES})

if(NOT WITH_FLUID_ONLY)
  file(GLOB TRAINER_PY_FILES . ./paddle/trainer/*.py)
  file(GLOB HELPERS_PY_FILES . ./paddle/trainer_config_helpers/*.py)
  file(GLOB_RECURSE V2_PY_FILES ./paddle/v2/*.py)
  set(PY_FILES ${PY_FILES}
    ${TRAINER_PY_FILES}
    ${HELPERS_PY_FILES}
    ${V2_PY_FILES})

  add_custom_target(copy_paddle_master)

  SET(COPY_PADDLE_MASTER "")
  if(WITH_GOLANG)
    SET(COPY_PADDLE_MASTER "copy_paddle_master")
    add_custom_command(TARGET ${COPY_PADDLE_MASTER}
      COMMAND cp ${paddle_master_LIB_PATH} ${PADDLE_SOURCE_DIR}/python/paddle/v2/master/
      )
    add_dependencies(copy_paddle_master paddle_master)
  endif(WITH_GOLANG)
endif()

set(MKL_SHARED_LIBS "")
set(MKL_DEPENDS "")
if(WITH_MKLML)
  list(APPEND MKL_SHARED_LIBS ${MKLML_LIB} ${MKLML_IOMP_LIB})
  list(APPEND MKL_DEPENDS mklml)
endif()

if(WITH_MKLDNN)
  list(APPEND MKL_SHARED_LIBS "${MKLDNN_SHARED_LIB}")
  list(APPEND MKL_DEPENDS mkldnn mkldnn_shared_lib)
endif()

if(WITH_GPU)
  SET(PACKAGE_NAME "paddlepaddle-gpu")
else()
  SET(PACKAGE_NAME "paddlepaddle")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

set(FLUID_CORE ${PADDLE_BINARY_DIR}/python/paddle/fluid/core.so)
add_custom_command(OUTPUT ${FLUID_CORE}
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${FLUID_CORE}
        DEPENDS paddle_pybind)
add_custom_target(copy_paddle_pybind ALL DEPENDS ${FLUID_CORE})


add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND touch stub.cc
    COMMAND cp -r ${PADDLE_SOURCE_DIR}/python/paddle ${PADDLE_BINARY_DIR}/python
    COMMAND cp -r ${PADDLE_SOURCE_DIR}/paddle/py_paddle ${PADDLE_BINARY_DIR}/python/
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_PYTHON_BUILD_DIR}/lib* ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    DEPENDS gen_proto_py copy_paddle_pybind ${FLUID_CORE} framework_py_proto profiler_py_proto ${PY_FILES} ${external_project_dependencies} ${COPY_PADDLE_MASTER})

set(paddle_python_deps ${PADDLE_PYTHON_BUILD_DIR}/.timestamp ${MKL_DEPENDS})
if(NOT WITH_FLUID_ONLY)
    set(paddle_python_deps ${paddle_python_deps} paddle_pserver_main paddle_trainer paddle_merge_model)
    if(WITH_SWIG_PY)
        list(APPEND paddle_python_deps python_api_wheel)
    endif()
endif()
add_custom_target(paddle_python ALL DEPENDS ${paddle_python_deps})

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if (WITH_TESTING)
  add_subdirectory(paddle/reader/tests)
  add_subdirectory(paddle/dataset/tests)
  if(NOT WITH_FLUID_ONLY)
    add_subdirectory(paddle/trainer_config_helpers/tests)
    if (WITH_SWIG_PY)
      # enable v2 API unittest only when paddle swig api is compiled
      add_subdirectory(paddle/v2/tests)
      add_subdirectory(paddle/v2/plot/tests)
      add_subdirectory(paddle/v2/reader/tests)
    endif()
  endif()
  add_subdirectory(paddle/fluid/tests)
endif()
install(DIRECTORY ${PADDLE_PYTHON_PACKAGE_DIR}
    DESTINATION opt/paddle/share/wheels
)
