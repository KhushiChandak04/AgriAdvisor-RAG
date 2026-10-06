@echo off
echo === AgriAdvisor setup ===
python -m venv venv || goto :err
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
if not exist .env copy .env.example .env
echo.
echo Setup complete. Next: edit .env (add GROQ_API_KEY), put PDFs in data\, run run_ingest.bat
exit /b 0
:err
echo Setup failed. Check the messages above.
exit /b 1
