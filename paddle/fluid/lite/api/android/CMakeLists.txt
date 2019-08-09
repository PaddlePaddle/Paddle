if ((NOT LITE_WITH_JAVA) OR (NOT LITE_WITH_ARM))
  return()
endif()

add_subdirectory(jni)
