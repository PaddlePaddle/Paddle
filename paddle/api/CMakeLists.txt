FUNCTION(generate_python_api target_name)
    ADD_CUSTOM_COMMAND(OUTPUT ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py
                              ${PROJ_ROOT}/paddle/Paddle_wrap.cxx
                              ${PROJ_ROOT}/paddle/Paddle_wrap.h
        COMMAND ${SWIG_EXECUTABLE} -python -c++ -outcurrentdir -I../ api/Paddle.swig
                && mv ${PROJ_ROOT}/paddle/swig_paddle.py ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py
        DEPENDS ${PROJ_ROOT}/paddle/api/Paddle.swig
                ${PROJ_ROOT}/paddle/api/PaddleAPI.h
                ${external_project_dependencies}
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle
        COMMENT "Generate Python API from swig")
    ADD_CUSTOM_TARGET(${target_name} ALL DEPENDS
                ${PROJ_ROOT}/paddle/Paddle_wrap.cxx
                ${PROJ_ROOT}/paddle/Paddle_wrap.h
                ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py
                ${external_project_dependencies})
ENDFUNCTION(generate_python_api)

set(API_SOURCES
    Arguments.cpp
    ConfigParser.cpp
    Evaluator.cpp
    GradientMachine.cpp
    Matrix.cpp
    Parameter.cpp
    ParameterOptimizer.cpp
    ParameterUpdater.cpp
    SequenceGenerator.cpp
    Trainer.cpp
    Util.cpp
    Vector.cpp)
set(API_HEADER
    PaddleAPI.h
    Internal.h)

add_library(paddle_api STATIC
        ${API_SOURCES})
add_dependencies(paddle_api gen_proto_cpp)

list(LENGTH "${GFLAGS_LIBRARIES}" GFLAGS_LIBRARIES_LENGTH)

if(${GFLAGS_LIBRARIES_LENGTH} EQUAL 0 AND TARGET "${GFLAGS_LIBRARIES}")
# Because gflags compiled by cmake, so it is imported by cmake target,
# not a real library path. Get the real library path here.
message(STATUS "GFLAGS Libraries is ${GFLAGS_LIBRARIES}")
get_target_property(GFLAGS_LOCATION ${GFLAGS_LIBRARIES} LOCATION)
message(STATUS "GFLAGS Target location is ${GFLAGS_LOCATION}")
else()
set(GFLAGS_LOCATION ${GFLAGS_LIBRARIES})
endif()

configure_file(
    paddle_api_config.py.in
    ${PROJ_ROOT}/paddle/api/paddle_api_config.py
)

generate_python_api(python_swig_sources)

file(GLOB PY_PADDLE_PYTHON_FILES ${PROJ_ROOT}/paddle/py_paddle/*.py)

# TODO(yuyang18) : make wheel name calculated by cmake
add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/dist/.timestamp
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch dist/.timestamp
    COMMAND rm -rf py_paddle.egg-info build
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle
    DEPENDS python_swig_sources
            paddle_parameter
            paddle_function
            paddle_math
            paddle_utils
            paddle_gserver
            paddle_pserver
            paddle_trainer
            paddle_api
            paddle_cuda
        ${PY_PADDLE_PYTHON_FILES}
)

install(DIRECTORY ${PROJ_ROOT}/paddle/dist/
    DESTINATION opt/paddle/share/wheels
)

add_custom_target(python_api_wheel ALL DEPENDS
  ${PROJ_ROOT}/paddle/dist/.timestamp)
add_dependencies(python_api_wheel python_swig_sources
  paddle_parameter
  paddle_math
  paddle_utils
  paddle_gserver
  paddle_pserver
  paddle_trainer
  paddle_api
  paddle_cuda)

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
            DEPENDS python setuptools python_api_wheel
        )
    ENDIF()
    add_subdirectory(test)
endif()
