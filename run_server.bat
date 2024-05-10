@echo off
start cmd /k "python manage.py runserver"

timeout /t 5

start "" "https://127.0.0.1:8080"