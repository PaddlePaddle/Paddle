# internal library.
cc_library(header SRCS header.cc)
cc_test(header_test SRCS header_test.cc DEPS header)
cc_library(chunk SRCS chunk.cc DEPS snappystream snappy header zlib)
cc_test(chunk_test SRCS chunk_test.cc DEPS chunk)
cc_library(recordio DEPS chunk header)
