call :build || goto build_error
goto:eof

:build
@ECHO ON
echo    ========================================
echo    Step 2. Build Paddle ...
echo    ========================================

for /F %%# in ('wmic cpu get NumberOfLogicalProcessors^|findstr [0-9]') do set /a PARALLEL_PROJECT_COUNT=%%#*4/5
echo "PARALLEL PROJECT COUNT is %PARALLEL_PROJECT_COUNT%"

set build_times=1
rem MSbuild will build third_party first to improve compiler stability.
if NOT %GENERATOR% == "Ninja" (
    goto :build_tp
) else (
    goto :build_paddle
)

:build_tp
echo Build third_party the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja third_party
) else (
    MSBuild /m /p:PreferredToolArchitecture=x64 /p:Configuration=Release /verbosity:%LOG_LEVEL% third_party.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build third_party failed, will retry!
        goto :build_tp
    )
)
echo Build third_party successfully!

set build_times=1

:build_paddle
:: reset clcache zero stats for collect PR's actual hit rate
rem clcache.exe -z

rem -------clean up environment again-----------
call %ci_scripts%\clean_env.bat

if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)

echo Build Paddle the %build_times% time:
if %GENERATOR% == "Ninja" (
    ninja all
) else (
    MSBuild /m:%PARALLEL_PROJECT_COUNT% /p:PreferredToolArchitecture=x64 /p:TrackFileAccess=false /p:Configuration=Release /verbosity:%LOG_LEVEL% ALL_BUILD.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!
echo 0 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt

:: ci will collect sccache hit rate
if "%WITH_SCCACHE%"=="ON" (
    call :collect_sccache_hits
)

goto:eof

:build_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Build Paddle failed, will exit!
exit /b 7
