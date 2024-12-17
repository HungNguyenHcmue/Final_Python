<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}


$class_name = $_GET['class_name'];
$sql = "SELECT * FROM videos WHERE class_name = ? AND deleted_at IS NULL";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $class_name);
$stmt->execute();
$result = $stmt->get_result();
echo json_encode($result->fetch_all(MYSQLI_ASSOC));

$stmt->close();
$conn->close();
?>
