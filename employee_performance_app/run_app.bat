@echo off
echo Starting INX Employee Performance Streamlit App...
echo.

cd /d "C:\Users\tonyn\Desktop\IABAC exams (Lennie)\employee_performance_app"

REM Activate virtual environment
call "C:\Users\tonyn\Desktop\IABAC exams (Lennie)\.venv\Scripts\activate.bat"

REM Run the Streamlit app
streamlit run streamlit_app.py --server.port 8501

pause
