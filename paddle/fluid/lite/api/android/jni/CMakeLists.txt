if ((NOT LITE_WITH_ARM) OR (NOT LITE_WITH_JAVA))
  return()
endif()

include(UseJava)
find_package(Java REQUIRED)

# We are only interested in finding jni.h: we do not care about extended JVM
# functionality or the AWT library.
set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_JVM_LIBRARY NotNeeded)
set(JAVA_INCLUDE_PATH2 NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)

# Generate PaddlePredictor.jar
include_directories(${JNI_INCLUDE_DIRS})
add_jar(PaddlePredictor 
    src/com/baidu/paddle/lite/PaddlePredictor.java
    src/com/baidu/paddle/lite/Place.java)
get_target_property(_jarFile PaddlePredictor JAR_FILE)
get_target_property(_classDir PaddlePredictor CLASSDIR)
set(_stubDir "${CMAKE_CURRENT_BINARY_DIR}")

# Generate paddle_lite_jni.h
add_custom_target(
    paddle_lite_jni_header ALL
    COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
        -classpath ${_classDir}
        -o paddle_lite_jni.h
        -jni
        com.baidu.paddle.lite.PaddlePredictor
    DEPENDS PaddlePredictor
)

# Generate paddle_lite_jni.so
include_directories(${JNI_INCLUDE_DIRS} ${_classDir} ${_stubDir})
lite_cc_library(paddle_lite_jni MODULE SRCS paddle_lite_jni.cc 
    DEPS light_api_lite cxx_api_lite
        paddle_api_full paddle_api_lite paddle_api_light
    ARM_DEPS ${arm_kernels})
if (APPLE)
    # MacOS only accepts JNI lib ends with .jnilib or .dylib
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".jnilib")
elseif (WIN32)
    # Windows only accepts JNI lib ends with .dll
    set_target_properties(paddle_lite_jni PROPERTIES SUFFIX ".dll")
endif (APPLE)
target_link_libraries(paddle_lite_jni light_api_lite cxx_api_lite
    paddle_api_full paddle_api_lite paddle_api_light ${arm_kernels})
