@echo off
:: Di chuyển dấu nhắc lệnh đến đúng thư mục chứa file .bat này
cd /d "%~dp0"

:: 1. Kiểm tra và kích hoạt venv (giả sử venv nằm trong thư mục viterbox)
if exist "viterbox\venv\Scripts\activate.bat" (
    call "viterbox\venv\Scripts\activate.bat"
) else (
    echo [ERROR] Khong tim thay thu muc venv tai: %~dp0viterbox\venv
    pause
    exit /b
)

:: 2. Mở trình duyệt (dùng start để không chặn tiến trình tiếp theo)
echo Dang mo giao dien tai http://127.0.0.1:7860/ ...
start http://127.0.0.1:7860/

:: 3. Chạy ứng dụng Python
:: Vì file .bat đang ở cùng chỗ với app.py nên gọi trực tiếp
python app.py

pause