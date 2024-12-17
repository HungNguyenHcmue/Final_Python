async function login(event) {
  event.preventDefault();

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  const formData = new FormData();
  formData.append("email", email);
  formData.append("password", password);

  try {
      const response = await fetch("login.php", {
          method: "POST",
          body: formData,
      });
    
      if (!response.ok) {
          throw new Error("Lỗi kết nối đến máy chủ.");
      }

      const data = await response.json();
      if (data.success) {
          // Chuyển đến trang Home
          window.location.href = "../Home/Home.html";
      } else {
          alert(data.message || "Đăng nhập thất bại");
      }
  } catch (error) {
      alert("Có lỗi xảy ra: " + error.message);
  }
}
