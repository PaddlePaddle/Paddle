set(OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/build")

file(GLOB TRAINER_PY_FILES . ./paddle/trainer/*.py)
file(GLOB HELPERS_PY_FILES . ./paddle/trainer_config_helpers/*.py)
file(GLOB UTILS_PY_FILES . ./paddle/utils/*.py)
file(GLOB_RECURSE V2_PY_FILES ./paddle/v2/ *.py)
file(GLOB PY_PADDLE . ./paddle/*.py)

set(PY_FILES paddle/__init__.py
             ${TRAINER_PY_FILES}
             ${HELPERS_PY_FILES}
             ${UTILS_PY_FILES}
             ${V2_PY_FILES}
             ${PY_PADDLE})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(OUTPUT ${OUTPUT_DIR}/.timestamp
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_DIR}/.timestamp
    DEPENDS gen_proto_py paddle_api ${PY_FILES} ${external_project_dependencies})

add_custom_target(paddle_python ALL DEPENDS ${OUTPUT_DIR}/.timestamp)

add_subdirectory(paddle/trainer_config_helpers/tests)
add_subdirectory(paddle/v2/reader/tests)
add_subdirectory(paddle/v2/tests)

if(WITH_TESTING)
  IF(NOT PY_PIP_FOUND)
    SET(PIP_SOURCES_DIR ${PYTHON_SOURCES_DIR}/pip)
    ExternalProject_Add(pip
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY      https://github.com/pypa/pip.git
            GIT_TAG             9.0.1
            PREFIX              ${PIP_SOURCES_DIR}
            CONFIGURE_COMMAND   ""
            BUILD_COMMAND       ""
            INSTALL_COMMAND     env ${py_env} ${PYTHON_EXECUTABLE} setup.py install
            BUILD_IN_SOURCE     1
            DEPENDS python setuptools paddle_python
            )
  ENDIF()
  add_subdirectory(test)
endif()

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist/
    DESTINATION opt/paddle/share/wheels
)
