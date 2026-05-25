@echo off
call conda activate rt_env
cd /d D:\RT_project\RT_overlay_viewer
streamlit run app.py
pause