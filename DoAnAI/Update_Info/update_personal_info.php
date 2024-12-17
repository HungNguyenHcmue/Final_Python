<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

header('Content-Type: application/json; charset=utf-8'); // Đảm bảo JSON được trả về

// Kết nối đến cơ sở dữ liệu
$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die(json_encode(['error' => 'Kết nối thất bại: ' . $conn->connect_error]));
}

// Bắt đầu session
session_start();
if (!isset($_SESSION['user'])) {
    echo json_encode(['error' => 'Bạn chưa đăng nhập hoặc session đã hết hạn']);
    exit();
}

// Lấy Username từ session
$username = $_SESSION['user'];

// Nhận dữ liệu từ POST
$name = $_POST['name'] ?? '';
$gender = $_POST['gender'] ?? '';
$birthplace = $_POST['birthplace'] ?? '';
$dob = $_POST['dob'] ?? '';
$address = $_POST['address'] ?? '';
$phone = $_POST['phone'] ?? '';
$subject = $_POST['subject'] ?? '';
$position = $_POST['position'] ?? '';

// Chuyển đổi ngày sinh về định dạng SQL
$dobFormatted = !empty($dob) ? date('Y-m-d', strtotime($dob)) : null;

// Kiểm tra MaBoMon hợp lệ
$sqlCheckSubject = "SELECT COUNT(*) as count FROM bomon WHERE MaBoMon = ?";
$stmtCheckSubject = $conn->prepare($sqlCheckSubject);
$stmtCheckSubject->bind_param("s", $subject);
$stmtCheckSubject->execute();
$result = $stmtCheckSubject->get_result();
$row = $result->fetch_assoc();

if ($row['count'] == 0) {
    echo json_encode(['error' => 'Mã bộ môn không hợp lệ']);
    exit();
}
$stmtCheckSubject->close();

// Lấy MaGiangVien dựa trên Username
$sqlGetID = "SELECT MaGiangVien FROM giangvien WHERE Username = ?";
$stmtGetID = $conn->prepare($sqlGetID);
$stmtGetID->bind_param("s", $username);
$stmtGetID->execute();
$result = $stmtGetID->get_result();
if ($result->num_rows === 0) {
    echo json_encode(['error' => 'Không tìm thấy giảng viên với Username này']);
    exit();
}

$row = $result->fetch_assoc();
$userId = $row['MaGiangVien']; // Lấy MaGiangVien từ kết quả

// Cập nhật thông tin giảng viên
$sqlUpdate = "UPDATE giangvien 
              SET TenGiangVien = ?, GT = ?, NoiSinh = ?, NgaySinh = ?, DiaChi = ?, SDT = ?, MaBoMon = ?, ChucVu = ? 
              WHERE MaGiangVien = ?";
$stmtUpdate = $conn->prepare($sqlUpdate);

try {
    $stmtUpdate->bind_param(
        "ssssssssi",
        $name,
        $gender,
        $birthplace,
        $dobFormatted,
        $address,
        $phone,
        $subject,
        $position,
        $userId
    );

    $stmtUpdate->execute();
    echo json_encode(['success' => true, 'message' => 'Cập nhật thông tin thành công']);
} catch (mysqli_sql_exception $e) {
    echo json_encode(['error' => 'Lỗi cơ sở dữ liệu: ' . $e->getMessage()]);
}

$stmtUpdate->close();
$stmtGetID->close();
$conn->close();
?>
