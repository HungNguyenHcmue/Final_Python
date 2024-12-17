<?php
session_start();
header('Content-Type: application/json');

$email = $_POST['email'] ?? '';
$password = $_POST['password'] ?? '';

if (empty($email) || empty($password)) {
    echo json_encode(['success' => false, 'message' => 'Vui lòng nhập đầy đủ thông tin.']);
    exit;
}

$servername = "localhost";
$username = "root";
$dbname = "databaseai";

// Kết nối database
$conn = new mysqli($servername, $username, "", $dbname);

if ($conn->connect_error) {
    echo json_encode(['success' => false, 'message' => 'Kết nối database thất bại.']);
    exit;
}

$sql = "SELECT * FROM TaiKhoan WHERE Username = ? AND Password = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("ss", $email, $password);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows > 0) {
    $_SESSION['user'] = $email; // Tạo session
    $_SESSION['avatar'] = $user['Avatar'] ?? '../Img/DefaultAvatar.png';
    echo json_encode(['success' => true, 'message' => 'Đăng nhập thành công.']);
} else {
    echo json_encode(['success' => false, 'message' => 'Sai email hoặc mật khẩu.']);
}

$stmt->close();
$conn->close();