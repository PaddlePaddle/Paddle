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

    # fake_cinn_compiler_test: exercises CINN_WITH_CUSTOM_DEVICE code paths in
    # group_tile_config.cc (BuildScheduleConfig), compiler_custom_device.cc
    # (cdrtc::Compiler), and target.cc (DefaultCustomDeviceTarget). Uses CPU
    # stub device. NO GPU REQUIRED.
    paddle_test(fake_cinn_compiler_test SRCS fake_cinn_compiler_test.cc DEPS
                phi cinnapi)

    # fake_cinn_stub_plugin.so: Shared library plugin for Python-level CINN
    # CustomDevice tests. Loaded via CUSTOM_DEVICE_ROOT at runtime.
    add_library(
      fake_cinn_stub_plugin SHARED
      ${CMAKE_SOURCE_DIR}/test/custom_runtime/fake_cinn_stub_plugin.cc)
    target_include_directories(fake_cinn_stub_plugin
                               PRIVATE ${CMAKE_SOURCE_DIR})
    set_target_properties(
      fake_cinn_stub_plugin
      PROPERTIES PREFIX ""
                 LIBRARY_OUTPUT_DIRECTORY
                 ${CMAKE_BINARY_DIR}/test/custom_runtime/fake_cinn_stub)
  endif()

  if(WITH_CINN)
    # fake_nvgpu_cinn_test: exercises the CINN CustomDevice code path
    # (C_CinnInterface + C_DeviceInterface) using real CUDA/NVRTC/Driver APIs
    # loaded at runtime via dlopen. No WITH_GPU=ON required at compile time.
    # End-to-end: NVRTC compile -> cuModuleLoad -> cuLaunchKernel with result
    # verification. Requires NVIDIA GPU hardware + CUDA libs at runtime.
    paddle_test(fake_nvgpu_cinn_test SRCS fake_nvgpu_cinn_test.cc DEPS phi
                cinnapi)
  endif()
endif()
