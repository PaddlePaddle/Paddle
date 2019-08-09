if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
    return()
endif()

if(NOT (LITE_WITH_LIGHT_WEIGHT_FRAMEWORK AND LITE_WITH_ARM))
  return()
endif()

set(HAS_ARM_MATH_LIB_DIR OFF)
# will search name as "libmath_arm.${os}.${abi}.${lang}.a"
if(ARM_MATH_LIB_DIR AND EXISTS "${ARM_MATH_LIB_DIR}")
  set(arm_math_name "")
  if(ARM_TARGET_OS STREQUAL "android")
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
      set(arm_math_name "math_arm.android.armv8")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
      set(arm_math_name "math_arm.android.armv7")
    endif()
  endif()

  if(ARM_TARGET_OS STREQUAL "armlinux" )
    if(ARM_TARGET_ARCH_ABI STREQUAL "armv8")
      set(arm_math_name "math_arm.armlinux.armv8")
    elseif(ARM_TARGET_ARCH_ABI STREQUAL "armv7")
      set(arm_math_name "math_arm.armlinux.armv7")
    endif()
  endif()

  if(ARM_TARGET_LANG STREQUAL "clang")
    set(arm_math_name "${arm_math_name}.clang")
  else()
    set(arm_math_name "${arm_math_name}.gcc")
  endif()

  find_library(math_arm_file ${arm_math_name} ${ARM_MATH_LIB_DIR} NO_DEFAULT_PATH)
  if(math_arm_file)
    add_library(math_arm STATIC IMPORTED GLOBAL)
    set_property(TARGET math_arm PROPERTY IMPORTED_LOCATION ${math_arm_file})
    message(STATUS "ARM math library imported: ${math_arm_file}")
    set(HAS_ARM_MATH_LIB_DIR ON)
  else()
    message(WARNING "Can not find arm math library ${arm_math_name} in ${ARM_MATH_LIB_DIR")
  endif()
endif()


if (NOT HAS_ARM_MATH_LIB_DIR)
  # TODO(xxx): seperate them and do not deps proto, eigen3
  cc_library(math_arm SRCS  
      funcs.cc 
      packed_sgemm.cc 
      softmax.cc 
      scale.cc
      pooling.cc
      elementwise.cc
      concat.cc
      sgemv.cc
      type_trans.cc
      conv_impl.cc
      conv_direct_3x3s1.cc
      conv_direct_3x3s2.cc
      conv_direct.cc
      conv_depthwise_3x3_int7.cc
      conv_depthwise_3x3_int8.cc
      conv_depthwise_5x5s1_int8.cc
      conv_depthwise_3x3p0.cc
      conv_depthwise_3x3p1.cc
      conv_depthwise_5x5s1.cc
      conv_depthwise_5x5s2.cc
      conv_depthwise.cc
      conv_gemmlike.cc
      conv_winograd_3x3.cc
      conv_winograd.cc
      split.cc
      activation.cc
      dropout.cc
      gemm_prepacked_int8.cc
      gemv_arm_int8.cc
      conv3x3s1_direct_int8.cc
      conv3x3s2_direct_int8.cc
      DEPS ${lite_kernel_deps} framework_proto_lite)
endif()
