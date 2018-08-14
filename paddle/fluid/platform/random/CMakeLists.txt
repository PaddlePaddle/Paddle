cc_test(philox_engine_test SRCS philox_engine_test.cc)
nv_test(random_sequence_test SRCS random_sequence_test.cu DEPS device_context)
