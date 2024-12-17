<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}

// Truy vấn lấy video từ các lớp Class A, Class B, Class C
$sql = "SELECT * FROM videos WHERE class_name IN ('Class A', 'Class B', 'Class C') ORDER BY added_at DESC LIMIT 4";
$result = mysqli_query($conn, $sql);

$videos = ['Class A' => [], 'Class B' => [], 'Class C' => []];  // Mảng chứa các video theo lớp học

// Lấy kết quả và phân loại theo lớp học
while ($row = mysqli_fetch_assoc($result)) {
    // Ghép với thư mục uploads để tạo đường dẫn đầy đủ
    $videoUrl = '../Class/' . $row['video_path']; // Ghép với thư mục uploads
    $row['video_url'] = $videoUrl; // Lưu đường dẫn video đầy đủ vào mảng

    // Phân loại video theo lớp học
    $videos[$row['class_name']][] = $row; 
}

// Trả kết quả dưới dạng JSON
echo json_encode($videos);
mysqli_close($conn);
?>