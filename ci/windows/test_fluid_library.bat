setlocal enabledelayedexpansion

call :test_inference || goto test_inference_error
goto :eof

:test_inference
echo    ========================================
echo    Step 5. Testing fluid library for inference ...
echo    ========================================

cd %BUILD_DIR%
echo %vcvars64_dir%
call "%vcvars64_dir%"
tree /F %cd%\paddle_inference_install_dir\paddle
%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\paddle_inference_install_dir > lib_size.txt
type lib_size.txt
set /p libsize=< lib_size.txt
for /F %%i in ("%libsize%") do echo "Windows Paddle_Inference Size: !libsize_m!M"
for /F %%i in ("%libsize%") do echo ipipe_log_param_Windows_Paddle_Inference_Size: !libsize_m!M

cd /d %work_dir%\paddle\fluid\inference\api\demo_ci
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %WITH_TENSORRT% %TENSORRT_ROOT% %WITH_ONNXRUNTIME% %MSVC_STATIC_CRT% "%CUDA_TOOLKIT_ROOT_DIR%"

goto:eof

:test_inference_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo    ==========================================
echo    Testing inference library failed!
echo    ==========================================
exit /b 1
