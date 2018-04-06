add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/testTrain.py
    COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/*.py ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(copy_api_test ALL DEPENDS testTrain.py)

py_test(testTrain SRCS testTrain.py)
py_test(testMatrix SRCS testMatrix.py)
py_test(testVector SRCS testVector.py)
py_test(testTrainer SRCS testTrainer.py)
py_test(testArguments SRCS testArguments.py)
py_test(testGradientMachine SRCS testGradientMachine.py)
