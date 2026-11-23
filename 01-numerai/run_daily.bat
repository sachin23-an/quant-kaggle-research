@echo off
REM =============================================================================
REM Numerai Daily Automated Runner
REM Model: sachin23_an
REM =============================================================================

cd /d "D:\Projects\Quant\quant-kaggle-research"

echo ======================================================== >> "01-numerai\numerai.log"
echo [RUN] Numerai Daily Submission Started at %DATE% %TIME% >> "01-numerai\numerai.log"
echo ======================================================== >> "01-numerai\numerai.log"

"D:\Projects\Quant\quant-kaggle-research\.venv\Scripts\python.exe" "01-numerai\submit.py" --model-name sachin23_an >> "01-numerai\numerai.log" 2>&1

echo [STATUS] Execution finished with exit code %ERRORLEVEL% at %DATE% %TIME% >> "01-numerai\numerai.log"
echo. >> "01-numerai\numerai.log"
