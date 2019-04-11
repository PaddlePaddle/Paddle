file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*_new_api.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# This test is buggy
# py_test(test_understand_sentiment_dynamic_rnn SRCS
# 	test_understand_sentiment_dynamic_rnn.py SERIAL)
LIST(REMOVE_ITEM TEST_OPS test_understand_sentiment_dynamic_rnn_new_api)

if(NOT APPLE)
    # default test
    foreach(src ${TEST_OPS})
        py_test(${src} SRCS ${src}.py)
    endforeach()
else()
    foreach(src ${TEST_OPS})
        if(${src} STREQUAL "test_image_classification_vgg_new_api")
            message(WARNING "These tests has been disabled in OSX for random fail: \n" ${src})
        elseif(${src} STREQUAL "test_image_classification_resnet_new_api")
            message(WARNING "These tests has been disabled in OSX for random fail: \n" ${src})
        elseif(${src} STREQUAL "test_recognize_digits_conv_new_api")
            message(WARNING "These tests has been disabled in OSX for random fail: \n" ${src})
        elseif(${src} STREQUAL "test_recognize_digits_mlp_new_api")
            message(WARNING "These tests has been disabled in OSX for random fail: \n" ${src})
        elseif()
            py_test(${src} SRCS ${src}.py)
        endif()
    endforeach()
endif()
