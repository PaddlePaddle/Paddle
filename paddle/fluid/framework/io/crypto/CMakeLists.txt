#cc_library(cipher_factory SRCS cipher_factory.cc DEPS aes_gcm_crypto)
#cc_library(crypto_helper SRCS crypto_helper.cc DEPS cryptopp)
#cc_library(aes_gcm_crypto SRCS aes_gcm_crypto.cc DEPS cryptopp glog)

cc_library(paddle_ciphers SRCS cipher_utils.cc cipher.cc aes_cipher.cc DEPS cryptopp)

#cc_test(crypt_fstream_test SRCS crypt_fstream_test.cc DEPS cryptopp)
cc_test(aes_cipher_test SRCS aes_cipher_test.cc DEPS paddle_ciphers)

cc_test(cipher_utils_test SRCS cipher_utils_test.cc DEPS paddle_ciphers)

set(CMAKE_BUILD_TYPE "Debug")
#add_executable(cryptopp_test cryptopp_test.cc)
#target_link_libraries(cryptopp_test cryptopp crypto_helper)
