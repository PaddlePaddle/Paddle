
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


# add MKLML and MKLDNN shared lib into wheel if enabled them
set(MKL_SHARED_LIBS "")
set(MKL_SHARED_LIB_NAMES "")
set(MKL_DEPENDS "")
if(WITH_MKLML)
  get_filename_component(mklml_lib_name ${MKLML_LIB} NAME)
  get_filename_component(mklml_iomp_name ${MKLML_IOMP_LIB} NAME)
  list(APPEND MKL_SHARED_LIBS ${MKLML_LIB} ${MKLML_IOMP_LIB})
  list(APPEND MKL_SHARED_LIB_NAMES ${mklml_lib_name} ${mklml_iomp_name})
  list(APPEND MKL_DEPENDS mklml)
endif()

if(WITH_MKLDNN)
  get_filename_component(mkldnn_lib_name ${MKLDNN_LIB} NAME)
  list(APPEND MKL_SHARED_LIBS ${MKLDNN_LIB})
  list(APPEND MKL_SHARED_LIB_NAMES ${mkldnn_lib_name})
  list(APPEND MKL_DEPENDS mkldnn)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

if(MKL_SHARED_LIB_NAMES)
  # Copy the depended shared libraries necessary dir
  set(out_filename ".mkllibs.timestamp")
  set(out_framework_path "${PADDLE_SOURCE_DIR}/python/paddle/v2/framework")
  set(out_framework_file "${out_framework_path}/${out_filename}")
  add_custom_command(OUTPUT ${out_framework_file}
      COMMAND cmake -E copy ${MKL_SHARED_LIBS} ${out_framework_path}
      COMMAND ${CMAKE_COMMAND} -E touch ${out_framework_file}
      DEPENDS ${MKL_DEPENDS})

  set(out_pypaddle_path "${PADDLE_SOURCE_DIR}/paddle/py_paddle")
  set(out_pypaddle_file "${out_pypaddle_path}/${out_filename}")
  if (WITH_SWIG_PY)
    add_custom_command(OUTPUT ${out_pypaddle_file}
        COMMAND cmake -E copy ${MKL_SHARED_LIBS} ${out_pypaddle_path}
        COMMAND ${CMAKE_COMMAND} -E touch ${out_pypaddle_file}
        DEPENDS ${MKL_DEPENDS})
  endif()
  add_custom_target(copy_mkl_shared_libs ALL DEPENDS ${out_framework_file} ${out_pypaddle_file})
else()
  # Dummpy target
  add_custom_target(copy_mkl_shared_libs ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
endif()


add_custom_command(OUTPUT ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so
        COMMAND cmake -E copy $<TARGET_FILE:paddle_pybind> ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so
        DEPENDS paddle_pybind copy_mkl_shared_libs)
add_custom_target(copy_paddle_pybind ALL DEPENDS ${PADDLE_SOURCE_DIR}/python/paddle/v2/framework/core.so)

add_custom_command(OUTPUT ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${PADDLE_PYTHON_BUILD_DIR}/.timestamp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PADDLE_PYTHON_BUILD_DIR}/lib* ${PADDLE_PYTHON_BUILD_DIR}/lib-python
    DEPENDS gen_proto_py copy_paddle_pybind copy_mkl_shared_libs framework_py_proto ${PY_FILES} ${external_project_dependencies} ${COPY_PADDLE_MASTER})

add_custom_target(paddle_python ALL DEPENDS
    ${PADDLE_PYTHON_BUILD_DIR}/.timestamp paddle_pserver_main paddle_trainer paddle_merge_model python_api_wheel)

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
