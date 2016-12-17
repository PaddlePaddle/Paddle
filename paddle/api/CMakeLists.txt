set(API_SOURCES
    Arguments.cpp
    ConfigParser.cpp
    GradientMachine.cpp
    Matrix.cpp
    Parameter.cpp
    ParameterOptimizer.cpp
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
    COMMAND ${PYTHON_EXECUTABLE} setup.py  bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch dist/.timestamp
    COMMAND rm -rf py_paddle.egg-info build
    WORKING_DIRECTORY ${PROJ_ROOT}/paddle
    DEPENDS python_swig_sources
            paddle_parameter
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

if(WITH_TESTING)
    add_subdirectory(test)
endif()
