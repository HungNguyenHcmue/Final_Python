<?php
session_start();
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}

if (!isset($_SESSION['user'])) {
    echo json_encode(['error' => 'Người dùng chưa đăng nhập!']);
    exit();
}

// Lấy thông tin từ session
$user = $_SESSION['user'];

// Truy vấn thông tin từ bảng giangvien
$sql = "SELECT * FROM giangvien WHERE Username = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("s", $user);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows > 0) {
    $data = $result->fetch_assoc();
    // Kiểm tra avatar có giá trị hay không
    if (empty($data['Avatar']) || $data['Avatar'] === "NULL") {
        $data['Avatar'] = "../Img/DefaultAvatar.png"; // Avatar mặc định
    }
    echo json_encode($data);
} else {
    echo json_encode(['error' => 'Không tìm thấy thông tin cá nhân!']);
}

$stmt->close();
$conn->close();
?>
