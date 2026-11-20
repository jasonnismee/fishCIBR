# Bước 3 — CSDL lưu trữ và tìm kiếm ảnh cá (trích xuất đặc trưng)

## Dữ liệu đã kiểm tra

- `images.zip` có **2000 PNG**, trong đó 1000 file dạng `images/0000.png` đến `images/0999.png` và 1000 bản sao **trùng byte** mang tên `species_*`.
- Chỉ xử lý **1000 file số**; tên gốc `species_*` được giữ ở cột `original_filename` để truy vết.
- Ảnh 640×320 RGB, nền trắng. Mã ảnh **khác** bản báo cáo phần 1 đang mô tả 0001–1000, và một số ví dụ minh họa trong báo cáo không trùng với dữ liệu ZIP đã gửi. Không giả định có thể sửa bằng cách dịch chỉ số.
- Nhãn loài trong tên file **không dùng** để tính điểm giống thị giác.

## Chạy trên Windows (Python đã cài)

```powershell
cd fish_step3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m fishcbir.cli audit --zip "C:\duong-dan\images.zip"
python -m fishcbir.cli extract --zip "C:\duong-dan\images.zip" --db data\fish.db --out outputs
```

`images.zip` không cần giải nén. Chỉ cần thay đường dẫn bằng vị trí thực trên máy bạn.

## Chạy trên Kaggle

- Upload `images.zip` thành Kaggle Dataset, attach vào Notebook.
- Mở `notebooks/step3_kaggle.ipynb` rồi **Run All**. Notebook tự tìm `images.zip` trong `/kaggle/input` và tạo `fish.db` trong `/kaggle/working`.
- **Save Version / download output** vì file trong phiên chạy không phải nơi lưu lâu dài.
- Dùng CPU là đủ cho các thuật toán này. Không cần GPU.

## 8 nhóm vector mặc định

| Nhóm | Số chiều | Cấu hình |
|---|---:|---|
| GLCM | 40 | 32 gray levels, d=1/2, 4 angles, ASM/contrast/homogeneity/correlation/entropy |
| Circular non-rotation-invariant Uniform LBP | 59 | P=8,R=1, `nri_uniform` |
| Geometry | 3 | Aspect Ratio, Solidity, Extent |
| Contour Profile | 64 | arc-length sampled, centroid distance, normalized |
| Fourier Descriptor | 16 | N=256 contour samples, coefficients F2–F17, normalized |
| HSV Color Histogram | 192 | 12×4×4 |
| RGB Color Moment | 9 | mean, std, signed cube-root third moment |
| Spatial HSV | 288 | 2 rows × 3 cols, each region HSV 8×3×2 |

Tổng: **671**. Các tham số là **cấu hình mặc định để thử nghiệm**, chưa phải cấu hình tối ưu. Lưu riêng mỗi nhóm để bước 4 có thể chọn tập con, khoảng cách và trọng số mà không trích xuất lại.

## CSDL

- `images`: metadata, zip member, tên gốc, SHA256, vùng bao, trạng thái mask.
- `feature_configs`: tên nhóm, tham số, số chiều, phiên bản và key kiểm tra.
- `image_features`: (image_id, config_key) → vector little-endian float32 BLOB.
- `extraction_logs`: nhật ký thành công/thất bại/thời gian.

DB không nhúng ảnh gốc. Giữ file `images.zip` để có thể hiển thị ảnh khi demo. Vector lấy qua hàm `load_feature_matrix(...)` trong `fishcbir.database`.

## Tập kết quả

- `data/fish.db`: CSDL SQLite đã chứa giá trị trích xuất (nếu đã chạy).
- `outputs/feature_matrices.npz`: 8 ma trận vector dạng NumPy, tiện cho bước 4.
- `outputs/image_qc.csv`: số pixel mask, bounding box, thời gian, trạng thái.
- `outputs/feature_samples.csv`: ví dụ các vector thực tế.
- `outputs/run_summary.json`: số lượng ảnh xử lý và lỗi.
- `outputs/figures/`: ảnh minh họa RGB/mask/contour/LBP/HSV/profile.

## Chất lượng và giới hạn

Mask tính từ vùng khác nền trắng, rồi giữ contour ngoài lớn nhất và tô đầy; phù hợp nhất với ảnh cá một đối tượng, nền trắng. Cá màu trắng hoặc vây mỏng có thể cần kiểm tra trực quan. Trạng thái `ok` không đảm bảo phân đoạn hoàn hảo; cần mở mẫu ảnh minh họa và kiểm tra trường hợp ngoại lệ.

GLCM chỉ thống kê pixel-pair mà cả hai nằm trong mask. LBP chỉ thống kê tâm có cả lân cận thuộc mask. Spatial color histogram được chia theo **bounding box** (trên-trái → trên-phải, dưới-trái → dưới-phải). Kết quả mỗi nhóm có `float32` và được kiểm tra số chiều và finite.

**Bước 3 chưa khẳng định chất lượng tìm kiếm theo thị giác người**. So sánh tổ hợp, GA/Optuna, đánh giá NDCG@5 và Ground Truth thuộc **bước 4**.
