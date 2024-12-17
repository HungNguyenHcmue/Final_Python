<?php
// Lấy video ID từ tham số GET
$videoId = $_GET['id'];

// Kết nối đến cơ sở dữ liệu
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}

// Truy vấn thông tin video
$sql = "SELECT video_path, class_name, added_at FROM videos WHERE id = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("i", $videoId);
$stmt->execute();
$stmt->bind_result($videoPath, $className, $addedAt);
$stmt->fetch();

if ($videoPath) {
    echo json_encode([
        'video_path' => $videoPath,
        'class_name' => $className,
        'added_at' => $addedAt
    ]);
} else {
    echo json_encode(['error' => 'Video not found']);
}

$stmt->close();
$conn->close();
?>
