<?php
session_start();
header('Content-Type: application/json'); // Đặt header để trả về JSON

if (!isset($_SESSION['user'])) {
    echo json_encode(['success' => false, 'message' => 'Bạn chưa đăng nhập.']);
    exit;
}

$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    echo json_encode(['success' => false, 'message' => 'Kết nối database thất bại.']);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['avatar'])) {
    $user = $_SESSION['user'];
    $avatarDir = "../uploads/avatars/";
    $fileTmpPath = $_FILES['avatar']['tmp_name'];
    $fileName = $_FILES['avatar']['name'];
    $fileExtension = pathinfo($fileName, PATHINFO_EXTENSION);

    // Kiểm tra định dạng file
    $allowedExtensions = ['jpg', 'jpeg', 'png', 'gif'];
    if (!in_array(strtolower($fileExtension), $allowedExtensions)) {
        echo json_encode(['success' => false, 'message' => 'Định dạng file không được hỗ trợ.']);
        exit;
    }

    // Tạo tên file duy nhất
    $newFileName = $user . "_" . time() . "." . $fileExtension;
    $destPath = $avatarDir . $newFileName;

    // Tạo thư mục nếu chưa tồn tại
    if (!is_dir($avatarDir)) {
        mkdir($avatarDir, 0755, true);
    }

    // Xóa avatar cũ
    $oldAvatarQuery = "SELECT Avatar FROM giangvien WHERE Username = ?";
    $stmt = $conn->prepare($oldAvatarQuery);
    $stmt->bind_param("s", $user);
    $stmt->execute();
    $stmt->bind_result($oldAvatar);
    if ($stmt->fetch() && file_exists($oldAvatar)) {
        unlink($oldAvatar);
    }
    $stmt->close();

    // Di chuyển file mới vào thư mục
    if (move_uploaded_file($fileTmpPath, $destPath)) {
        // Lưu đường dẫn avatar mới vào database
        $updateQuery = "UPDATE giangvien SET Avatar = ? WHERE Username = ?";
        $stmt = $conn->prepare($updateQuery);
        $stmt->bind_param("ss", $destPath, $user);
        if ($stmt->execute()) {
            $_SESSION['avatar'] = $destPath; // Cập nhật session
            echo json_encode(['success' => true, 'message' => 'Cập nhật avatar thành công!', 'avatar' => $destPath]);
        } else {
            echo json_encode(['success' => false, 'message' => 'Lỗi khi cập nhật vào database.']);
        }
        $stmt->close();
    } else {
        echo json_encode(['success' => false, 'message' => 'Lỗi khi tải file.']);
    }
} else {
    echo json_encode(['success' => false, 'message' => 'Không có file nào được tải lên.']);
}

$conn->close();
?>
