file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# This test is buggy
# py_test(test_understand_sentiment_dynamic_rnn SRCS
# 	test_understand_sentiment_dynamic_rnn.py SERIAL)
LIST(REMOVE_ITEM TEST_OPS test_understand_sentiment_dynamic_rnn)

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
