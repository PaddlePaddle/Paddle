cc_library(paddle_ciphers SRCS cipher_utils.cc cipher.cc aes_cipher.cc DEPS cryptopp)
cc_test(aes_cipher_test SRCS aes_cipher_test.cc DEPS paddle_ciphers)
cc_test(cipher_utils_test SRCS cipher_utils_test.cc DEPS paddle_ciphers)
