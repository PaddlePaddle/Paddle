if(NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
    return()
endif()



# set(CMAKE_TOOLCHAIN_FILE "android.toolchain.cmake")

# arm-v7a

# if [ "${PLATFORM}" = "arm-v7a" ]; then
# ABI="armeabi-v7a with NEON"
# ARM_PLATFORM="V7"
# CXX_FLAGS="-march=armv7-a -mfpu=neon -mfloat-abi=softfp -pie -fPIE -w -Wno-error=format-security"
# elif [ "${PLATFORM}" = "arm-v8a" ]; then
# ABI="arm64-v8a"
# ARM_PLATFORM="V8"
# CXX_FLAGS="-march=armv8-a  -pie -fPIE -w -Wno-error=format-security -llog"
# else
# echo "unknown platform!"
# exit -1
# fi


# # NET default
# if(FPGAV1)
#     set(NET "FPGA_NET_V1" CACHE STRING "select net type")
# elseif(FPGAV2)
#     set(NET "FPGA_NET_V2" CACHE STRING "select net type")
# else()
#     set(NET "default" CACHE STRING "select net type")
# endif()

# set_property(CACHE NET PROPERTY STRINGS "default" "googlenet" "mobilenet" "yolo" "squeezenet" "FPGA_NET_V1" "FPGA_NET_V2" "NLP" "op")
# include("${CMAKE_CURRENT_LIST_DIR}/tools/op.cmake")



# ANDROID_PLATFORM_VERSION="android-22"
# TOOLCHAIN_FILE="./tools/android-cmake/android.toolchain.cmake"
# ANDROID_ARM_MODE="arm"

# cmake .. \
# -DANDROID_ABI="${ABI}" \
# -DCMAKE_BUILD_TYPE="${MODE}" \
# -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
# -DANDROID_PLATFORM="${ANDROID_PLATFORM_VERSION}" \
# -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
# -DANDROID_STL=c++_static \
# -DWITH_PROFILE=ON \
# -DWITH_LOGGING=ON \
# -D"${ARM_PLATFORM}"=true
message(STATUS "compile with lite ARM kernels")
