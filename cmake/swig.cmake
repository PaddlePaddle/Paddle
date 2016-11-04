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
