@echo off
setlocal


cd /d "%~dp0"


pip install --upgrade pip
pip install -r requirements.txt


streamlit run data_story_source_code.py


pause