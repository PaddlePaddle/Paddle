
file(GLOB TRAINER_PY_FILES . ./paddle/trainer/*.py)
file(GLOB HELPERS_PY_FILES . ./paddle/trainer_config_helpers/*.py)
file(GLOB UTILS_PY_FILES . ./paddle/utils/*.py)
file(GLOB_RECURSE V2_PY_FILES ./paddle/v2/ *.py)

set(PY_FILES paddle/__init__.py
  ${TRAINER_PY_FILES}
  ${HELPERS_PY_FILES}
  ${UTILS_PY_FILES}
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

set(MKL_SHARED_LIBS "")
set(MKL_DEPENDS "")
if(WITH_MKLML)
  list(APPEND MKL_SHARED_LIBS ${MKLML_LIB} ${MKLML_IOMP_LIB})
  list(APPEND MKL_DEPENDS mklml)
endif()

if(WITH_MKLDNN)
  list(APPEND MKL_SHARED_LIBS "${MKLDNN_LIB}" "${MKLDNN_LIB}.0")
  list(APPEND MKL_DEPENDS mkldnn)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)


add_custom_command(OUTPUT ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so
        DEPENDS paddle_pybind)
add_custom_target(copy_paddle_pybind ALL DEPENDS ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so)


add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND touch stub.cc
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_PYTHON_BUILD_DIR}/lib* ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    DEPENDS gen_proto_py copy_paddle_pybind framework_py_proto ${PY_FILES} ${external_project_dependencies} ${COPY_PADDLE_MASTER})

set(paddle_python_deps ${PADDLE_PYTHON_BUILD_DIR}/.timestamp paddle_pserver_main paddle_trainer paddle_merge_model ${MKL_DEPENDS})
if(WITH_SWIG_PY)
    list(APPEND paddle_python_deps python_api_wheel)
endif()
add_custom_target(paddle_python ALL DEPENDS ${paddle_python_deps})

set(PADDLE_PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/dist/)

if (WITH_TESTING)
  add_subdirectory(paddle/trainer_config_helpers/tests)
  if (WITH_SWIG_PY)
    # enable v2 API unittest only when paddle swig api is compiled
    add_subdirectory(paddle/v2/tests)
    add_subdirectory(paddle/v2/reader/tests)
    add_subdirectory(paddle/v2/plot/tests)
    add_subdirectory(paddle/v2/framework/tests)
  endif()
endif()
install(DIRECTORY ${PADDLE_PYTHON_PACKAGE_DIR}
    DESTINATION opt/paddle/share/wheels
)
