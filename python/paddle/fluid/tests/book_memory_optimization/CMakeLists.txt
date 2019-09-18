file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

list(REMOVE_ITEM TEST_OPS test_memopt_image_classification_train)
py_test(test_memopt_image_classification_train_resnet SRCS test_memopt_image_classification_train.py ARGS resnet)
py_test(test_memopt_image_classification_train_vgg SRCS test_memopt_image_classification_train.py ARGS vgg)

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
