@echo off
chcp 65001 >nul
cd /d "%~dp0"
call "启动应用.cmd" --lan %*
