cc_test(pten_test_backend SRCS test_backend.cc DEPS gtest)
cc_test(pten_test_data_layout SRCS test_data_layout.cc DEPS gtest)
cc_test(pten_test_data_type SRCS test_data_type.cc DEPS gtest)
cc_test(pten_test_place SRCS test_place.cc DEPS pten_place)
