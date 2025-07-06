echo    ========================================
echo    Step 7. Testing fluid library with infer_ut for inference ...
echo    ========================================

call "%PYTHON_VENV_ROOT%\Scripts\activate.bat"
cd /d %work_dir%\test\cpp\inference\infer_ut
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %TENSORRT_ROOT% %WITH_ONNXRUNTIME% %MSVC_STATIC_CRT% "%CUDA_TOOLKIT_ROOT_DIR%"
goto:eof

:test_inference_ut_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Testing fluid library with infer_ut for inference failed!
exit /b 1
