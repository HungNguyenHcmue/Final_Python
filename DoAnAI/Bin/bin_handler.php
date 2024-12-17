<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_GET['id']) && $_GET['action'] === 'restore') {
    $id = intval($_GET['id']);
    $sql = "UPDATE videos SET deleted_at = NULL WHERE id = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("i", $id);
    $stmt->execute();
    echo "Video đã được khôi phục.";
} elseif ($_SERVER['REQUEST_METHOD'] === 'DELETE' && isset($_GET['id']) && $_GET['action'] === 'delete_permanently') {
    $id = intval($_GET['id']);
    $sql = "DELETE FROM videos WHERE id = ?";
    $stmt = $conn->prepare($sql);
    $stmt->bind_param("i", $id);
    $stmt->execute();
    echo "Video đã xóa vĩnh viễn.";
    $stmt->close();
    $conn->close();
}

?>
