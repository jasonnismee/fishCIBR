# PHẦN 3. TRIỂN KHAI TRÍCH RÚT ĐẶC TRƯNG VÀ XÂY DỰNG CƠ SỞ DỮ LIỆU

> Nội dung dưới đây được tổng hợp từ **kết quả chạy thật trên bộ images.zip đã cung cấp**, không phải kết quả truy hồi ảnh hay tối ưu chất lượng thị giác. Khi tích hợp vào báo cáo chính thức, cần thống nhất lại ví dụ minh họa ở phần 1–2 với bộ dữ liệu hiện tại.

## 3.1. Mục tiêu và môi trường triển khai

Phần 3 xây dựng pipeline chuyển một ảnh cá RGB (640×320 pixel, nền trắng, đầu hướng trái) thành các bộ vector mô tả **kết cấu, hình dạng và màu sắc**, đồng thời quản lý chúng bằng CSDL SQLite. Các vector được lưu **riêng từng nhóm, theo phiên bản cấu hình**, tạo tiền đề cho bước 4 thử nghiệm các tổ hợp và các trọng số khác nhau.

Sử dụng Python, OpenCV, NumPy, scikit-image, SciPy, SQLite và Matplotlib. Code chạy được trên Windows hoặc Kaggle (CPU), không yêu cầu GPU. Không sử dụng nhãn loài để tính đặc trưng hay làm tiêu chí đúng/sai cho việc tìm kiếm.

## 3.2. Kiểm tra dữ liệu đầu vào

File ZIP chứa **2.000 ảnh PNG** trong một thư mục `images/`, gồm **1.000 ảnh có tên `0000.png`–`0999.png`** và **1.000 ảnh có tên `species_*.png`**. Đối chiếu SHA-256 xác nhận có 1.000 cặp file giống hệt nhau về byte dữ liệu; do đó chỉ xử lý một bản cho mỗi ảnh, giữ tên `species_*` ở cột metadata để truy vết nguồn. Tập có **1.000 nội dung ảnh duy nhất**, đúng yêu cầu số lượng.

Toàn bộ ảnh của bộ đánh số có kích thước **640×320**, 3 kênh màu. Lưu ý **khác biệt với phần 1–2 của bản báo cáo đang có**: dãy tên ảnh hiện bắt đầu từ `0000` thay vì `0001`; một số ảnh minh họa theo mã ảnh trong báo cáo không khớp với ZIP. Ví dụ, ảnh `0187.png` thực tế là cá đỏ; không thể coi phần mô tả cá vàng nâu đốm tối trong báo cáo là kết quả của ảnh này. Cần đối chiếu và điều chỉnh phần minh họa trước khi nộp.

## 3.3. Tiền xử lý và trích xuất đối tượng cá

Quy trình:

1. Giải mã ảnh PNG thành RGB `uint8` và xác nhận dữ liệu hợp lệ.
2. Tách các pixel khác nền trắng gần thuần, ngưỡng kênh RGB `<250`; dùng phép đóng hình thái học (kernel ellipse 3×3) để nối các ranh giới nhỏ.
3. Tìm các thành phần đường bao ngoài và giữ đường bao ngoài có diện tích lớn nhất. Tô đầy đường bao này để tạo **binary mask** của cá, tránh việc các vùng sáng trong thân cá bị nhầm là nền.
4. Tính hộp bao nhỏ nhất song song trục (bounding box), diện tích mask, grayscale và HSV. Kiểm tra mask quá nhỏ, quá lớn, chạm biên hoặc có nhiều thành phần rời.
5. Dùng **cùng một mặt nạ và bounding box** cho tất cả nhóm đặc trưng. Một số phương pháp chỉ sử dụng vị trí có đủ láng giềng nằm trong mask.

Hạn chế: ngưỡng gần trắng có thể bỏ sót vây rất nhạt hoặc vây tách rời, vì vậy phải kết hợp kiểm tra trực quan ảnh gốc/mask/contour. Kết quả kiểm tra tự động `ok` không thay thế việc đối chiếu bằng mắt.

## 3.4. Các nhóm đặc trưng và số chiều

| Nhóm | Cấu hình triển khai thử nghiệm | Số chiều |
|---|---|---:|
| GLCM | 32 mức xám, d={1,2}, 4 hướng, 5 thuộc tính ASM/Contrast/Homogeneity/Correlation/Entropy | 40 |
| Circular Uniform LBP | P=8,R=1, non-rotation-invariant uniform (`nri_uniform`), histogram L1 | 59 |
| Hình học | Aspect Ratio, Solidity (tỷ lệ diện tích theo bao lồi trong miền pixel), Extent | 3 |
| Contour Profile | 64 điểm cách đều theo chiều dài cung, khoảng cách tới centroid / cạnh lớn nhất bounding box | 64 |
| Fourier Descriptor | Chuẩn hóa contour 256 điểm, FFT phức, lấy biên độ F2–F17 chuẩn hóa theo biên độ điều hòa bậc 1 nổi trội | 16 |
| HSV Color Histogram | Lượng tử hóa HSV 12×4×4, chuẩn hóa histogram trên vùng cá | 192 |
| RGB Color Moment | Mean, standard deviation và căn bậc ba có dấu của moment trung tâm bậc 3 trên R/G/B đã quy đổi về [0,1] | 9 |
| Spatial Color Histogram | Chia hộp bao thành 2 hàng × 3 cột; mỗi vùng có histogram HSV 8×3×2 | 288 |
| **Tổng** | Tám vector riêng; có thể ghép để thử nghiệm | **671** |

Lưu ý: `nri_uniform` là quy tắc Uniform LBP không bất biến quay, tạo **59 bin** với P=8; khác với `uniform` (bất biến quay). Các giá trị 671 chiều là *cấu hình khởi đầu để nghiên cứu*, không phải kết luận tối ưu. Bước 4 sẽ kiểm nghiệm số chiều, số bin, trọng số và tổ hợp.

## 3.5. Thiết kế CSDL

CSDL `fish.db` lưu thông tin ảnh và đặc trưng, không nhúng nội dung PNG gốc:

- `images(image_id, zip_member, original_filename, sha256, width, height, bbox_*, mask_area, mask_ratio, mask_status)`: quản lý ảnh, tên trùng tương ứng và metadata tiền xử lý.
- `feature_configs(config_key, family, dimensions, parameters_json, version)`: phân biệt các bộ tham số và phiên bản thuật toán.
- `image_features(image_id, config_key, dtype, dimensions, data)`: mỗi cặp ảnh/cấu hình một vector kiểu `float32`, lưu nhị phân BLOB.
- `extraction_logs(log_id, image_id, status, elapsed_seconds, error_message)`: nhật ký thao tác và trạng thái.

Khóa chính ghép `(image_id,config_key)` cho phép cùng một ảnh có nhiều biến thể đặc trưng. Các file tên `species_*` chỉ là metadata tên nguồn, **không được dùng làm nhãn độ tương đồng thị giác**.

## 3.6. Kết quả triển khai thực tế

| Đại lượng | Kết quả |
|---|---:|
| Số PNG trong ZIP | 2.000 |
| Số ảnh cá duy nhất (trùng byte đã đối chiếu) | 1.000 |
| Số ảnh trích xuất thành công | **1.000 / 1.000** |
| Số ảnh báo lỗi trong lần trích xuất | **0** |
| Số cấu hình đặc trưng mặc định | 8 |
| Số bản ghi vector | **8.000** |
| Tổng số chiều mỗi ảnh (nếu ghép) | 671 |
| Tổng số giá trị đặc trưng | 671.000 |
| Mask ratio thấp nhất | 0,270215 |
| Mask ratio cao nhất | 0,279893 |
| Số ảnh không có cờ cảnh báo mask tự động | 1.000 |
| Trung vị thời gian trích xuất trên môi trường chạy thử | Khoảng 0,0506 giây/ảnh |

Dữ liệu lưu trữ đã được kiểm tra bằng `PRAGMA integrity_check`, kết quả **ok**; 8 ma trận có số hàng 1.000 và đúng chiều đã thiết kế; tất cả giá trị đặc trưng đều hữu hạn (không NaN hoặc vô hạn). Histogram LBP và HSV chuẩn hóa tổng bằng 1; sáu histogram màu không gian cũng có tổng mỗi vùng bằng 1 trong dữ liệu hiện tại.

### Ví dụ giá trị hình học thực tế

| Mã ảnh | Aspect Ratio | Solidity | Extent |
|---|---:|---:|---:|
| 0000 | 2,4877 | 0,7469 | 0,5475 |
| 0032 | 2,0905 | 0,8686 | 0,6119 |
| 0187 | 2,1611 | 0,8375 | 0,5925 |
| 0334 | 1,8340 | 0,8047 | 0,5364 |
| 0917 | 1,9506 | 0,7251 | 0,4962 |

Bảng trên tính trực tiếp từ CSDL. Bộ giá trị đặc trưng đầy đủ của năm ảnh này được xuất trong `outputs/feature_samples.csv`. Các hình minh họa tương ứng (ảnh gốc, mask, contour, LBP, HSV, Contour Profile) nằm trong `outputs/figures`.

## 3.7. Kết luận và phạm vi đánh giá

Đã xây dựng hệ thống tự động trích xuất tám nhóm đặc trưng từ 1.000 ảnh cá và quản lý chúng trong SQLite. Pipeline có thể xử lý ảnh mới bằng cùng quy trình và có thể mở rộng theo các phiên bản cấu hình.

**Chưa thể kết luận** nhóm đặc trưng nào truy hồi giống thị giác người nhất chỉ từ việc vector tính đúng và CSDL toàn vẹn. Bước 4 cần xây dựng đánh giá thị giác người (ground truth), tính khoảng cách đã hiệu chuẩn, so sánh tổ hợp, tối ưu trọng số và đo Human NDCG@5/Precision@5 trên tập kiểm thử độc lập.
