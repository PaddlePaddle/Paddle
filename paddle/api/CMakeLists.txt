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

add_library(paddle_api STATIC ${API_SOURCES})
add_dependencies(paddle_api gen_proto_cpp)

INCLUDE(${SWIG_USE_FILE})
INCLUDE_DIRECTORIES(${PROJ_ROOT}/paddle)

FILE(GLOB PY_PADDLE_PYTHON_FILES ${PROJ_ROOT}/paddle/py_paddle/*.py)

SET_SOURCE_FILES_PROPERTIES(Paddle.i PROPERTIES CPLUSPLUS ON)

SET(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})
SET(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -Wall")
IF(WITH_COVERAGE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
ENDIF(WITH_COVERAGE)

SET(SWIG_MODULE_swig_paddle_EXTRA_DEPS
    paddle_parameter
    paddle_function
    paddle_math
    paddle_utils
    paddle_gserver
    paddle_pserver
    paddle_api
    paddle_cuda
    paddle_trainer_lib
    paddle_network
    paddle_proto
    ${external_project_dependencies}
)

IF(APPLE)
    SET(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load")
ELSE(APPLE)
    SET(START_GROUP "-Xlinker -start-group")
    SET(END_GROUP "-Xlinker -end-group")
    SET(ARCHIVE_START "-Wl,--whole-archive")
    SET(ARCHIVE_END "-Wl,--no-whole-archive")
ENDIF(APPLE)

SWIG_ADD_MODULE(swig_paddle python Paddle.i)
SWIG_LINK_LIBRARIES(swig_paddle
    ${MACOS_LD_FLAGS}
    ${START_GROUP}
    ${ARCHIVE_START}
    paddle_gserver
    paddle_function
    ${METRIC_LIBS}
    ${ARCHIVE_END}
    paddle_pserver
    paddle_trainer_lib
    paddle_network
    paddle_parameter
    paddle_math
    paddle_utils
    paddle_proto
    paddle_cuda
    paddle_api
    ${CMAKE_DL_LIBS}
    ${EXTERNAL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${START_END}
)

add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/py_paddle/_swig_paddle.so
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/swig_paddle.py ${PROJ_ROOT}/paddle/py_paddle
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/_swig_paddle.so ${PROJ_ROOT}/paddle/py_paddle
    COMMAND env ${py_env} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch dist/.timestamp
    COMMAND rm -rf py_paddle.egg-info build
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle
    DEPENDS _swig_paddle
)

# TODO(yuyang18) : make wheel name calculated by cmake
add_custom_target(python_api_wheel ALL DEPENDS ${PROJ_ROOT}/paddle/py_paddle/_swig_paddle.so)

install(DIRECTORY ${PROJ_ROOT}/paddle/dist/ DESTINATION opt/paddle/share/wheels)

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
