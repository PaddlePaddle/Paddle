@ECHO OFF

set source_path=%SOURCE_PATH%

cd /d %source_path%
if %errorlevel% NEQ 0 GOTO END

set "release_dir=%source_path%\paddle_release"
mkdir %release_dir%

cd ..
set script_path=%~dp0
echo %script_path%

REM source_path PYTHON_DIR WITH_GPU WITH_MKL WITH_AVX BATDIR
call %script_path%build.bat %source_path% %PYTHON_ROOT% ON ON ON %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% ON OFF ON %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% ON ON OFF %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% ON OFF OFF %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% OFF ON ON %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% OFF OFF ON %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% OFF ON OFF %script_path% %release_dir%
call %script_path%build.bat %source_path% %PYTHON_ROOT% OFF OFF OFF %script_path% %release_dir%

:END

