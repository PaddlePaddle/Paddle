taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im ninja.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im python.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im op_function_generator.exe /t 2>NUL
taskkill /f /im busybox64.exe /t 2>NUL
taskkill /f /im eager_generator.exe /t 2>NUL
taskkill /f /im eager_legacy_op_function_generator.exe /t 2>NUL
powershell -Command "Stop-Process -Name 'eager_generator' -Force 2>$null"
powershell -Command "Stop-Process -Name 'eager_legacy_op_function_generator' -Force 2>$null"
powershell -Command "Stop-Process -Name 'cvtres' -Force 2>$null"
powershell -Command "Stop-Process -Name 'rc' -Force 2>$null"
powershell -Command "Stop-Process -Name 'cl' -Force 2>$null"
powershell -Command "Stop-Process -Name 'lib' -Force 2>$null"
powershell -Command "Stop-Process -Name 'python' -Force 2>$null" || ver >nul
