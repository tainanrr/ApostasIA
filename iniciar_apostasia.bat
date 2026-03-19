@echo off
title ApostasIA Engine
echo ========================================
echo   Iniciando ApostasIA Engine...
echo ========================================
cd /d "c:\ProjetosCursor\ApostasIA"
call venv\Scripts\activate.bat
python app.py
pause
