@echo off

cd /d %~dp0

call C:\Users\ADMIN\anaconda3\Scripts\activate.bat rt_env

streamlit run app.py

pause
