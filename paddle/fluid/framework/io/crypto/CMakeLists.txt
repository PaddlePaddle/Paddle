cc_library(paddle_crypto SRCS cipher_utils.cc cipher.cc aes_cipher.cc DEPS cryptopp enforce)
cc_test(aes_cipher_test SRCS aes_cipher_test.cc DEPS paddle_crypto)
cc_test(cipher_utils_test SRCS cipher_utils_test.cc DEPS paddle_crypto)
