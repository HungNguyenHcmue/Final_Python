<?php
// Đảm bảo trả về định dạng JSON
header('Content-Type: application/json');

// Lấy video ID từ tham số GET
$videoId = $_GET['id']; 

// Kết nối đến cơ sở dữ liệu
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "databaseai"; // Thay bằng tên cơ sở dữ liệu của bạn

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Kết nối thất bại: " . $conn->connect_error);
}

// Truy vấn thông tin video
$sql = "SELECT video_path FROM videos WHERE id = ?";
$stmt = $conn->prepare($sql);
$stmt->bind_param("i", $videoId);
$stmt->execute();
$stmt->bind_result($videoPath);
$stmt->fetch();

$response = [];

if ($videoPath) {
    // Chạy mô hình Python và truyền đường dẫn video
    $videoFullPath = "../Class/" . $videoPath; // Đảm bảo đường dẫn tuyệt đối hoặc phù hợp
    $command = escapeshellcmd("python Model/run_model.py $videoFullPath");
    $output = shell_exec($command);

    // Trả về kết quả phân tích video từ mô hình
    $response = [
        'status' => 'success',
        'video_path' => $videoPath,
        'analysis_output' => $output // Phần output của mô hình Python
    ];
} else {
    // Nếu không tìm thấy video trong cơ sở dữ liệu
    $response = ['error' => 'Video not found'];
}

$stmt->close();
$conn->close();

// Trả về kết quả dưới dạng JSON
echo json_encode($response);

?>
