# internal library.
cc_library(io SRCS io.cc DEPS stringpiece)
cc_test(io_test SRCS io_test.cc DEPS io)
cc_library(header SRCS header.cc DEPS io)
cc_test(header_test SRCS header_test.cc DEPS header)
cc_library(chunk SRCS chunk.cc DEPS snappy)
cc_test(chunk_test SRCS chunk_test.cc DEPS chunk)
cc_library(range_scanner SRCS range_scanner.cc DEPS io chunk)
cc_test(range_scanner_test SRCS range_scanner_test.cc DEPS range_scanner)
cc_library(scanner SRCS scanner.cc DEPS range_scanner)
cc_test(scanner_test SRCS scanner_test.cc DEPS scanner)
# exported library.
cc_library(recordio SRCS recordio.cc DEPS scanner chunk header)
cc_test(recordio_test SRCS recordio_test.cc DEPS scanner)
