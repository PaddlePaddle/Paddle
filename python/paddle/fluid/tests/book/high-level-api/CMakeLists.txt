file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

# default test
foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()

add_subdirectory(fit_a_line)
add_subdirectory(recognize_digits)
add_subdirectory(image_classification)
add_subdirectory(understand_sentiment)
add_subdirectory(label_semantic_roles)
add_subdirectory(word2vec)
add_subdirectory(recommender_system)
add_subdirectory(machine_translation)
