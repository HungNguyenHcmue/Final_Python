<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}

$sql = "SELECT * FROM videos WHERE deleted_at IS NOT NULL";
$result = $conn->query($sql);
$videos = [];

if ($result->num_rows > 0) {
    // Lấy tất cả video và đẩy vào mảng $videos
    while ($row = $result->fetch_assoc()) {
        $videos[] = $row;
    }
}

// Chuyển mảng $videos thành JSON và trả về
echo json_encode($videos);
$conn->close();
?>
