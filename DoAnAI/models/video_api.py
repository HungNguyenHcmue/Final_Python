from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS: Cho phép tất cả các nguồn hoặc chỉ những nguồn bạn muốn
origins = [
    "http://localhost",  # Thêm địa chỉ frontend của bạn, ví dụ như http://localhost hoặc http://localhost:3000
    "http://localhost:3000",  # Nếu frontend của bạn đang chạy trên cổng 3000
    "https://your-frontend-domain.com",  # Nếu bạn có tên miền frontend riêng
]

# Thêm middleware CORS vào ứng dụng FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Cho phép các miền này gửi yêu cầu
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],  # Cho phép tất cả các header
)

# Đường dẫn cố định để lưu video, với tên tệp là 'temp.mp4'
UPLOAD_PATH = r"C:\xampp\htdocs\DoAnAI\Video\temp.mp4"


# API endpoint để người dùng tải video lên
@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Kiểm tra định dạng của tệp video (ví dụ chỉ chấp nhận các định dạng mp4)
        if not file.filename.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="Invalid file format. Only MP4 files are allowed.")

        # Mở tệp video và ghi vào đường dẫn cố định với tên tệp là 'temp.mp4'
        with open(UPLOAD_PATH, "wb") as video_file:
            video_file.write(await file.read())

        return JSONResponse(content={"message": "Video uploaded successfully!", "file_name": "temp.mp4"},
                            status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading video: {e}")



@app.get("/run-model")
async def run_model():
    try:
        # Đảm bảo đường dẫn chính xác đến file run_model.py
        result = subprocess.run(
            ['python', r'C:\xampp\htdocs\DoAnAI\models\run_model.py'],  # Lưu ý sử dụng raw string (r"") để tránh lỗi với dấu "\"
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return JSONResponse(content={"status": "success", "output": result.stdout})
        else:
            return JSONResponse(content={"status": "error", "message": result.stderr}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)