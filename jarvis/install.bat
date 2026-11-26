@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 949 >nul 2>&1
title 자비스 음성 비서 설치 (가입 없이)
cd /d "%~dp0"

echo.
echo ============================================================
echo    자비스 음성 비서 설치  (가입. 키 필요 없음)
echo ============================================================
echo.
echo  이 창이 스스로 설치를 진행합니다. 5분 안팎 걸릴 수 있어요.
echo  중간에 물어보는 것은 마지막 한 번뿐입니다.
echo.

REM ============================================================
REM  1. 파이썬 찾기
REM ============================================================
echo [1/7] 파이썬을 확인합니다...
set "PY="
REM 3.12 를 가장 먼저 찾습니다. (vosk 등 음성 라이브러리가 3.13 에서는 설치가 안 되는 경우가 있습니다)
for %%V in (3.12 3.11 3.10) do call :try_py %%V
if not defined PY call :try_py_default
if not defined PY goto :no_python

echo   파이썬 확인 완료: %PY%
%PY% --version
goto :make_venv

:try_py
py -%1 -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
if not defined PY set "PY=py -%1"
exit /b 0

:try_py_default
python -c "import sys" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "tokens=2" %%A in ('python -V 2^>^&1') do set "PYVER=%%A"
echo   파이썬 %PYVER% 을 찾았습니다.
set "PY=python"
exit /b 0

:no_python
echo.
echo   [안내] 파이썬이 아직 설치되어 있지 않습니다.
echo          자비스는 파이썬이 필요합니다. 무료입니다.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :no_winget
echo   지금 자동으로 파이썬 3.12를 설치할 수 있습니다.
set /p ANS=  자동 설치할까요? Y 또는 N 입력 후 Enter:
if /i not "%ANS%"=="Y" goto :manual_python
echo   파이썬을 내려받아 설치합니다. 몇 분 걸립니다...
winget install -e --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
echo.
echo   설치가 끝났습니다. 이 창을 닫고 install.bat 을 다시 실행해 주세요.
echo   설치 직후에는 이 창이 새 파이썬을 못 찾을 수 있습니다.
pause
exit /b 1

:no_winget
echo   아래 주소를 브라우저에 붙여 넣어 파이썬을 받아 설치해 주세요.
echo   https://www.python.org/downloads/release/python-3127/
echo   설치 화면에서 Add python.exe to PATH 를 꼭 체크하세요.
echo.
:manual_python
echo   파이썬을 설치한 뒤 이 창을 닫고 install.bat 을 다시 실행해 주세요.
pause
exit /b 1

REM ============================================================
REM  2. 가상환경 만들기
REM ============================================================
:make_venv
echo.
echo [2/7] 자비스 전용 파이썬 공간을 만듭니다...
if exist ".venv\Scripts\python.exe" goto :venv_ready
%PY% -m venv .venv
if exist ".venv\Scripts\python.exe" goto :venv_ready
echo.
echo   [문제] .venv 를 만들지 못했습니다.
echo          폴더 경로에 한글이나 특수문자가 많으면 문제가 될 수 있습니다.
echo          폴더를 C:\davis 처럼 단순한 곳에 두고 다시 실행해 보세요.
pause
exit /b 1

:venv_ready
set "VENVPY=.venv\Scripts\python.exe"

REM ============================================================
REM  3. 필요한 패키지 설치
REM ============================================================
echo.
echo [3/7] 필요한 프로그램을 설치합니다.
echo      2~5분 걸립니다. 창이 멈춘 것처럼 보여도 기다려 주세요.
"%VENVPY%" -m pip install --upgrade pip --disable-pip-version-check
"%VENVPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto :pip_fail
"%VENVPY%" -c "import numpy, sounddevice, vosk, faster_whisper, edge_tts, pygame" >nul 2>&1
if errorlevel 1 goto :pip_check_fail
goto :get_model

:pip_check_fail
echo.
echo   [문제] 설치가 끝났는데 일부 프로그램을 불러오지 못합니다.
echo          파이썬 버전이 3.13 이면 음성 라이브러리가 아직 없을 수 있습니다.
echo          https://www.python.org/downloads/release/python-3127/ 에서
echo          파이썬 3.12 를 설치한 뒤, 이 폴더의 .venv 폴더를 지우고 다시 실행해 주세요.
pause
exit /b 1

:pip_fail
echo.
echo   [문제] 설치에 실패했습니다.
echo          인터넷 연결을 확인한 뒤 install.bat 을 다시 실행해 주세요.
echo          회사 보안 프로그램이 막는 경우도 있습니다.
pause
exit /b 1

REM ============================================================
REM  4. 한국어 음성모델 내려받기 (가입 없이 받는 무료 모델)
REM ============================================================
:get_model
echo.
echo [4/7] 한국어 음성모델을 준비합니다. 약 87MB, 처음 한 번만 받습니다.
echo      "자비스" 라고 부르는 말을 알아듣는 데 쓰는 파일입니다.
"%VENVPY%" davis.py --setup
if errorlevel 1 goto :model_fail
goto :register_startup

:model_fail
echo.
echo   [주의] 음성모델을 자동으로 받지 못했습니다.
echo          인터넷 연결을 확인한 뒤 install.bat 을 다시 실행하면 이어서 받습니다.
echo          지금은 설치를 계속하지만, '자비스' 라고 불러도 반응하지 않습니다.
echo          (자세한 안내는 README 2장)

REM ============================================================
REM  5. 윈도우 시작프로그램에 등록
REM ============================================================
:register_startup
echo.
echo [5/7] 컴퓨터를 켤 때 자동으로 시작되도록 등록합니다...
echo      (예전 '다비스' 판이 남긴 자동 시작 항목이 있으면 함께 정리합니다)
cscript //nologo "%~dp0자비스 시작.vbs" /install
if errorlevel 1 goto :shortcut_fail
echo   [정상] 자동 시작 등록 완료 (이 폴더 기준)
goto :check_claude

:shortcut_fail
echo   [주의] 자동 시작 등록에 실패했습니다.
echo          README 의 안내대로 직접 등록할 수 있습니다.

REM ============================================================
REM  6. 클로드 코드 확인 + 상태 점검
REM ============================================================
:check_claude
echo.
echo [6/7] 클로드 코드와 마이크를 확인합니다...
where claude >nul 2>&1
if errorlevel 1 goto :no_claude
for /f "delims=" %%A in ('claude --version 2^>nul') do echo   클로드 코드: %%A
echo   로그인이 되어 있는지는 첫 질문을 해 보면 알 수 있습니다.
goto :run_check

:no_claude
echo.
echo   [주의] 클로드 코드를 찾지 못했습니다.
echo          자비스는 클로드 코드로 답을 만듭니다. 아래 순서로 설치하세요.
echo          1번 Node.js 설치 - https://nodejs.org 의 LTS 버전
echo          2번 새 cmd 창에서 npm install -g @anthropic-ai/claude-code
echo          3번 cmd 창에서 claude 실행 후 구글 계정으로 로그인
echo.

:run_check
echo.
"%VENVPY%" davis.py --check

echo.
echo ============================================================
echo   설치가 끝났습니다.
echo ============================================================
echo.
echo   부르는 말:  "자비스"   (길게 "헤이 자비스" 라고 해도 됩니다)
echo   가입이나 키는 필요 없습니다.
echo.
echo   컴퓨터를 켜면 이 폴더의 자비스가 저절로 켜집니다.
echo   켜지면 작은 '자비스' 창이 하나 뜹니다. 그 창이 보이면 정상입니다.
echo   (창을 닫아도 자비스는 계속 듣습니다. 끄려면 창의 [종료] 버튼)
echo.
echo   잘 안 되면 이 폴더의 logs 폴더에 날짜별 기록이 남습니다.
echo.
echo   예전 '다비스' 폴더가 아직 남아 있다면, 아래에서 자비스가 잘 되는 것을
echo   확인한 뒤 그 폴더를 지워 주세요. (자동 시작은 이 폴더로 바뀌었습니다)
echo.
set /p ANS=  지금 자비스를 시작할까요? Y 또는 N 입력 후 Enter:
if /i not "%ANS%"=="Y" goto :not_now
start "" wscript.exe "%~dp0자비스 시작.vbs"
echo   자비스가 켜졌습니다. 작은 '자비스' 창이 뜨면 정상입니다.
echo   3초쯤 뒤에 마이크에 대고 "자비스" 라고 말해 보세요.
goto :done

:not_now
echo   나중에 이 폴더의 "자비스 시작.vbs" 를 더블클릭하면 켜집니다.

:done
echo.
echo   이 창은 닫으셔도 됩니다.
echo.
pause
exit /b 0
