@echo off
rem Starts the lab dashboard (needs hosts.json next to this file) and opens it.
start "" /min pythonw "%~dp0dashboard.py"
start "" http://127.0.0.1:8765
