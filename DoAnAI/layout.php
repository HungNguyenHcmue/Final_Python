<?php
session_start();
if (!isset($_SESSION['user'])) {
    header('Location: ../Login/Login.html'); // Chuyển hướng nếu chưa đăng nhập
    exit;
}

$servername = "localhost";
$username = "root";
$password = ""; // Thay bằng mật khẩu thực tế
$dbname = "databaseai";

$conn = new mysqli($servername, $username, $password, $dbname);
if ($conn->connect_error) {
    die("Kết nối database thất bại: " . $conn->connect_error);
}

$user = $_SESSION['user'];
$sql = "SELECT Avatar FROM giangvien WHERE Username = ?";
$stmt = $conn->prepare($sql);
if (!$stmt) {
    die("Lỗi trong câu lệnh prepare: " . $conn->error);
}
$stmt->bind_param("s", $user);
$stmt->execute();
$result = $stmt->get_result();

if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    $avatar = $row['Avatar'] ?? '../Img/Avatar.png';
} else {
    $avatar = '../Img/Avatar.png';
}

$stmt->close();
$conn->close();
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layout</title>
    <link rel="stylesheet" href="../layout.css">
</head>
<body>
    <div id="header">
        <header>
            <img src="../Img/Logo.png" alt="Logo" class="logo">
            <div class="user-actions">
                <a href="../Personal_Info/Personal_Info.html">
                    <!-- Hiển thị avatar cá nhân -->
                    <img src="<?php echo htmlspecialchars($avatar, ENT_QUOTES, 'UTF-8'); ?>" alt="User" class="icon">
                </a>
                <button class="logout-btn" onclick="logout()">Đăng xuất</button>
            </div>
        </header>
    </div>

    <div id="sidebar">
        <aside>
            <nav>
                <ul>
                    <li><img src="../Img/Home.png" class="menu-icon"><a href="../Home/Home.html" class="menu-name">Trang chủ</a></li>
                    <li>
                        <div onclick="toggleClassList()" class="menu-item">
                            <img src="../Img/List.png" class="menu-icon">
                            <span class="menu-name">Danh sách lớp</span>
                        </div>
                        <ul id="class-list" class="sub-menu">
                            <li><a href="../Class/ClassA.html" class="class-link">Lớp A</a></li>
                            <li><a href="../Class/ClassB.html" class="class-link">Lớp B</a></li>
                            <li><a href="../Class/ClassC.html" class="class-link">Lớp C</a></li>
                        </ul>
                    </li>
                    <li><img src="../Img/Recently.png" class="menu-icon"><a href="../Recently/Recently.html" class="menu-name">Gần đây</a></li>
                    <li><img src="../Img/Bin.png" class="menu-icon"><a href="../Bin/Bin.html" class="menu-name">Thùng rác</a></li>
                    <li><img src="../Img/8.png" class="menu-icon">Bộ nhớ</li>
                </ul>
            </nav>
        </aside>
    </div>
</body>
</html>
