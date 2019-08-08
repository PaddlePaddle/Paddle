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
    src/com/baidu/paddle/lite/ConfigBase.java
    src/com/baidu/paddle/lite/CxxConfig.java
    src/com/baidu/paddle/lite/MobileConfig.java
    src/com/baidu/paddle/lite/PaddleLiteInitializer.java
    src/com/baidu/paddle/lite/PaddlePredictor.java
    src/com/baidu/paddle/lite/Place.java
    src/com/baidu/paddle/lite/PrecisionType.java
    src/com/baidu/paddle/lite/Tensor.java)
get_target_property(_jarFile PaddlePredictor JAR_FILE)
get_target_property(_classDir PaddlePredictor CLASSDIR)
set(_stubDir "${CMAKE_CURRENT_BINARY_DIR}")

# Generate native headers
add_custom_target(
    paddle_lite_jni_header ALL
    COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
        -classpath ${_classDir}
        -o "${CMAKE_BINARY_DIR}/paddle/fluid/lite/api/android/jni/native/paddle_lite_jni.h"
        -jni
        com.baidu.paddle.lite.PaddlePredictor
    COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
        -classpath ${_classDir}
        -o "${CMAKE_BINARY_DIR}/paddle/fluid/lite/api/android/jni/native/tensor_jni.h"
        -jni
        com.baidu.paddle.lite.Tensor
    COMMAND ${Java_JAVAH_EXECUTABLE} -verbose
        -classpath ${_classDir}
        -o "${CMAKE_BINARY_DIR}/paddle/fluid/lite/api/android/jni/native/paddle_init_jni.h"
        -jni
        com.baidu.paddle.lite.PaddleLiteInitializer
    DEPENDS PaddlePredictor
)

add_subdirectory(native)
