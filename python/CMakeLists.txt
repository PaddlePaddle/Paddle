set(OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/build")


set(PADDLE_INTERNAL_PACKAGE "")
if (PADDLE_WITH_INTERNAL)
    set(PADDLE_INTERNAL_PACKAGE "paddle.internals")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(OUTPUT ${OUTPUT_DIR}/.timestamp
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT_DIR}/.timestamp
    DEPENDS gen_proto_py)

add_custom_target(paddle_python ALL DEPENDS
    ${OUTPUT_DIR}/.timestamp)

find_python_module(pip REQUIRED)
find_python_module(wheel REQUIRED)
find_python_module(google.protobuf REQUIRED)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist/
    DESTINATION opt/paddle/share/wheels
)
