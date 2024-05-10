Start-Process cmd -ArgumentList "/c python mange.py runserver"
Start-Sleep -Seconds 5
Start-Process "https://127.0.0.1:8080"