@echo off
title Capstone Web Server - Online Sharing
echo ========================================================
echo   Launching Capstone Web Server + Worldwide Public Link
echo ========================================================
echo.

:: Start Flask app in background if not already running
start /b python app.py

:: Give Flask 2 seconds to initialize
timeout /t 2 /nobreak >nul

echo.
echo ========================================================
echo   Local Network (Same Wi-Fi):  http://10.123.32.108:5000
echo   Global Public Link (Anywhere in the World):
echo ========================================================
echo.
echo Connecting tunnel... please wait 3 seconds for your HTTPS link:
echo.

:run_tunnel
ssh -R 80:127.0.0.1:5000 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no nokey@localhost.run
echo.
echo Tunnel connection dropped. Reconnecting in 3 seconds...
timeout /t 3 /nobreak >nul
goto run_tunnel

