if(WITH_CUSTOM_DEVICE)
  py_test(test_custom_cpu_plugin SRCS test_custom_cpu_plugin.py)
  set_tests_properties(test_custom_cpu_plugin PROPERTIES TIMEOUT 120)
endif()
