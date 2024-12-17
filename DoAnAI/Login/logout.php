<?php
session_start();
if (!isset($_SESSION['user'])) {
    echo json_encode(['success' => false, 'message' => 'Bạn chưa đăng nhập.']);
    exit;
}

session_destroy();
echo json_encode(['success' => true, 'message' => 'Đăng xuất thành công.']);
exit;
?>
