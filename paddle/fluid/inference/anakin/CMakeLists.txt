cc_library(anakin_engine SRCS engine.cc)
target_link_libraries(anakin_engine anakin anakin_saber_common)
cc_test(test_anakin_engine SRCS test_anakin_engine.cc DEPS anakin_engine)
add_subdirectory(convert)
