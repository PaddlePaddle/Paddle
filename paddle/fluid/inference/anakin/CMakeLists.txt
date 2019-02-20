nv_library(anakin_engine SRCS engine.cc)
target_link_libraries(anakin_engine anakin)
nv_test(test_anakin_engine SRCS test_anakin_engine.cc DEPS anakin_engine)
target_link_libraries(test_anakin_engine anakin)
