if(WITH_CUSTOM_DEVICE)
  paddle_test(custom_device_test SRCS custom_device_test.cc)

  if(WITH_CINN)
    # fake_cinn_stub_test: exercises the CINN CustomDevice code path
    # (CinnCustomDevicePlugin, DefaultCompilerToolchain, DefaultRuntimeStrategy,
    # CustomBackendAPI, cinn_call_custom_device_kernel) using a pure CPU-based
    # stub device. NO GPU REQUIRED - achieves CINN CustomDevice code coverage
    # on any CI environment with WITH_CUSTOM_DEVICE=ON and WITH_CINN=ON.
    paddle_test(fake_cinn_stub_test SRCS fake_cinn_stub_test.cc DEPS phi
                cinnapi)
  endif()
endif()
