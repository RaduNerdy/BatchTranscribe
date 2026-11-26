@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "OUTPUT=%CD%\build\windows"
set "EXE=%OUTPUT%\batchtranscribe.exe"
set "FLET_BUILD_LOG=%CD%\flet-build.log"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul

echo === BatchTranscribe Windows build ===
REM Flutter requires a complete Visual Studio installation with C++ and CMake.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_INSTALL="
if not exist "%VSWHERE%" goto missing_vs
for /f "usebackq delims=" %%V in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Workload.NativeDesktop Microsoft.VisualStudio.Component.VC.CMake.Project -property installationPath`) do set "VS_INSTALL=%%V"
if not defined VS_INSTALL goto missing_vs
REM Prefer the same SDK as the existing Flet project instead of a competing SDK on PATH.
if exist "%USERPROFILE%\flutter\3.41.7\bin\flutter.bat" set "PATH=%USERPROFILE%\flutter\3.41.7\bin;%PATH%"
REM Use the same Python interpreter for dependencies, checks and Flet CLI.
if defined BATCHTRANSCRIBE_PYTHON goto python_ready
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "BATCHTRANSCRIBE_PYTHON=%%P"
)
if not defined BATCHTRANSCRIBE_PYTHON (
    for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "BATCHTRANSCRIBE_PYTHON=%%P"
)
:python_ready
if not defined BATCHTRANSCRIBE_PYTHON (
    echo ERROR: No working Python found. Install Python 3.12 or 3.13 from python.org.
    goto failed
)
"%BATCHTRANSCRIBE_PYTHON%" -c "import sys; assert sys.version_info >= (3, 10); print(sys.executable)"
if errorlevel 1 goto failed

echo [1/4] Installing dependencies...
"%BATCHTRANSCRIBE_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
"%BATCHTRANSCRIBE_PYTHON%" -m pip install --find-links https://k2-fsa.github.io/sherpa/onnx/cuda.html "sherpa-onnx==1.13.4+cuda12.cudnn9"
if errorlevel 1 goto failed

echo [2/4] Generating icons...
"%BATCHTRANSCRIBE_PYTHON%" build_icon.py
if errorlevel 1 goto failed

echo [3/4] Checking backend and build tools...
"%BATCHTRANSCRIBE_PYTHON%" -c "from transcriber import get_import_error; err=get_import_error(); assert err is None, err; import flet_cli.cli"
if errorlevel 1 goto failed
"%BATCHTRANSCRIBE_PYTHON%" prepare_windows_runtime.py
if errorlevel 1 goto failed

echo [4/4] Building Windows application...
REM Flet may report success even when its expected build output is absent.
REM Record the start time so an old executable cannot count as a fresh build.
for /f "delims=" %%T in ('powershell -NoProfile -Command "[DateTime]::UtcNow.Ticks"') do set "BUILD_STARTED=%%T"
echo Flet output will be saved to: %FLET_BUILD_LOG%
powershell -NoProfile -Command "& $env:BATCHTRANSCRIBE_PYTHON -m flet_cli.cli build windows --output $env:OUTPUT --product BatchTranscribe --build-version 2.0.0 --verbose --yes 2>&1 | Tee-Object -FilePath $env:FLET_BUILD_LOG; exit $LASTEXITCODE"
if errorlevel 1 goto failed
if not exist "%EXE%" goto missing_output
if not exist "%OUTPUT%\flutter_windows.dll" goto missing_output
if not exist "%OUTPUT%\data\flutter_assets\" goto missing_output
powershell -NoProfile -Command "if ((Get-Item -LiteralPath $env:EXE).LastWriteTimeUtc.Ticks -lt [long]$env:BUILD_STARTED) { exit 1 }"
if errorlevel 1 (
    echo ERROR: The executable is older than this build. No fresh build was verified.
    goto failed
)

REM Optional: bundle licensed CUDA 12/cuDNN 9 runtime DLLs from an existing
REM NVIDIA installation or a directory containing trusted NVIDIA packages.
if defined BATCHTRANSCRIBE_CUDA_DIR (
    powershell -NoProfile -Command "$src=$env:BATCHTRANSCRIBE_CUDA_DIR; $dst=Join-Path $env:OUTPUT 'cuda\bin'; if (-not (Test-Path -LiteralPath $src -PathType Container)) { throw 'CUDA runtime source does not exist' }; New-Item -ItemType Directory -Path $dst -Force | Out-Null; Get-ChildItem -LiteralPath $src -Recurse -File -Filter '*.dll' | Where-Object { $_.Name -match '^(cublas|cudnn|cufft|nvblas|nvrtc|nvJitLink)' -and $_.Name -notlike '*.alt.dll' } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $dst $_.Name) -Force }; foreach ($dll in 'cublas64_12.dll','cublasLt64_12.dll','cudnn64_9.dll') { if (-not (Test-Path -LiteralPath (Join-Path $dst $dll))) { throw ('Missing required CUDA runtime file: '+$dll) } }"
    if errorlevel 1 goto failed
)

"%BATCHTRANSCRIBE_PYTHON%" prepare_windows_runtime.py "%OUTPUT%"
if errorlevel 1 goto failed

echo.
echo === Build complete ===
echo Executable: %EXE%
echo Keep the entire output folder together, including DLLs and data.
explorer "%OUTPUT%"
exit /b 0

:missing_vs
echo ERROR: Visual Studio 2022 is incomplete or missing the Desktop development with C++ workload and CMake tools.
echo Open Visual Studio Installer, resume or modify the installation, and install that workload with recommended components.
goto failed

:missing_output
echo ERROR: Flet did not produce a complete Windows application in %OUTPUT%.
echo Check the Flutter build errors above.
:failed
echo Build failed. No new executable was verified.
if exist "%FLET_BUILD_LOG%" echo Flet log: %FLET_BUILD_LOG%
pause
exit /b 1



