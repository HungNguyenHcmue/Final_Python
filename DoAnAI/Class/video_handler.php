<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai"; // Tên cơ sở dữ liệu của bạn

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}


if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $class_name = $_POST['class_name'];
    $video_path = "uploads/" . basename($_FILES['video']['name']);

    if (move_uploaded_file($_FILES['video']['tmp_name'], $video_path)) {
        $sql = "INSERT INTO videos (class_name, video_path) VALUES (?, ?)";
        $stmt = $conn->prepare($sql);
        $stmt->bind_param("ss", $class_name, $video_path);
        $stmt->execute();
        echo "Video đã được tải lên thành công.";
    } else {
        echo "Lỗi tải lên video.";
    }
} elseif ($_SERVER['REQUEST_METHOD'] === 'DELETE' && isset($_GET['id'])) {
    $id = intval($_GET['id']);
    $sql = "UPDATE videos SET deleted_at = NOW() WHERE id = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("i", $id);
    $stmt->execute();
    echo "Video đã chuyển vào Thùng Rác.";
}
?>
