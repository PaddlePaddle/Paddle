######################### socket_test ########################
add_unittest_without_exec(socket_test
    SocketTest.cpp)

add_test(NAME socket_test
    COMMAND ${PROJ_ROOT}/paddle/.set_port.sh -p port
        ${CMAKE_CURRENT_BINARY_DIR}/socket_test --loop_time=10)

####################### test_ProtoServer ####################
add_unittest_without_exec(test_ProtoServer
    test_ProtoServer.cpp)

IF(NOT ON_TRAVIS)
    add_test(NAME test_ProtoServer
        COMMAND ${PROJ_ROOT}/paddle/.set_port.sh -p port
            ${CMAKE_CURRENT_BINARY_DIR}/test_ProtoServer)
ENDIF(NOT ON_TRAVIS)

# TODO(yuyang18): Run test_ProtoServer when with rdma
# add_test(NAME test_ProtoServerRDMA
#   COMMAND ...)

#################### test_ParameterServer2 ####################
add_unittest_without_exec(test_ParameterServer2
    test_ParameterServer2.cpp)
add_test(NAME test_ParameterServer2
    COMMAND ${PROJ_ROOT}/paddle/.set_port.sh -p port -n 4
        ${CMAKE_CURRENT_BINARY_DIR}/test_ParameterServer2)
