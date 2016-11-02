find_program(
    SWIG_BINARY_PATH
    swig)

if(${SWIG_BINARY_PATH} STREQUAL "SWIG_BINARY_PATH-NOTFOUND")
    set(SWIG_FOUND OFF)
else()
    set(SWIG_FOUND ON)
endif()

set(MIN_SWIG_VERSION 2)
if(SWIG_FOUND)
    execute_process(COMMAND sh -c "${SWIG_BINARY_PATH} -version | grep Version | cut -f3 -d' '"
        OUTPUT_VARIABLE _SWIG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(${_SWIG_VERSION} VERSION_LESS ${MIN_SWIG_VERSION})
        message("swig version ${MIN_SWIG_VERSION} or greater is needed for generating python api. "
                 "Only version ${_SWIG_VERSION} is found. Set SWIG_FOUND to FALSE")
        set(SWIG_FOUND FALSE)
    endif(${_SWIG_VERSION} VERSION_LESS ${MIN_SWIG_VERSION})
endif(SWIG_FOUND)

function(generate_python_api target_name)
    add_custom_command(OUTPUT ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py
                              ${PROJ_ROOT}/paddle/Paddle_wrap.cxx
                              ${PROJ_ROOT}/paddle/Paddle_wrap.h
        COMMAND swig -python -c++ -outcurrentdir -I../ api/Paddle.swig
                && mv ${PROJ_ROOT}/paddle/swig_paddle.py ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py
        DEPENDS ${PROJ_ROOT}/paddle/api/Paddle.swig
                ${PROJ_ROOT}/paddle/api/PaddleAPI.h
        WORKING_DIRECTORY ${PROJ_ROOT}/paddle
        COMMENT "Generate Python API from swig")
    add_custom_target(${target_name} ALL DEPENDS
                ${PROJ_ROOT}/paddle/Paddle_wrap.cxx
                ${PROJ_ROOT}/paddle/Paddle_wrap.h
                ${PROJ_ROOT}/paddle/py_paddle/swig_paddle.py)
endfunction(generate_python_api)
