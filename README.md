# ĐẶC TẢ PIPELINE PHÂN LOẠI ĐA NHIỆM (SENTIMENT & TOXICITY)

## 1. TIỀN XỬ LÝ DỮ LIỆU (MINIMALIST PREPROCESSING)

- **Đồng nhất bảng mã (Unicode NFC & Lọc ký tự điều khiển):**
  - **Cơ chế:** Chuyển toàn bộ văn bản về chuẩn dựng sẵn NFC; loại bỏ ký tự BOM đầu file, ký tự trắng không độ rộng (zero-width space) và các mã điều khiển ẩn.
  - **Mục đích:** Đảm bảo các từ tiếng Việt hiển thị giống nhau luôn tạo ra cùng một chuỗi token, tránh phân mảnh từ vựng do lệch bảng mã.
- **Làm sạch nhiễu phi ngôn ngữ:**
  - **Cơ chế:** Xóa thẻ HTML, đường dẫn URL và gộp các khoảng trắng liên tiếp thành một khoảng trắng duy nhất.
  - **Mục đích:** Loại bỏ rác kỹ thuật sinh ra trong quá trình thu thập dữ liệu mà không làm thay đổi cấu trúc câu.
- **Bảo toàn nguyên trạng tín hiệu ngôn ngữ tự nhiên:**
  - **Cơ chế:** Không tách từ thủ công; giữ nguyên chữ viết hoa, dấu câu lặp lại, biểu tượng cảm xúc (emoji) và từ lóng/viết tắt (teencode).
  - **Mục đích:** Giữ lại toàn bộ tín hiệu biểu cảm nguyên bản (viết hoa toàn câu, spam dấu chấm than/chấm hỏi) để đánh giá đúng năng lực tự thân của bộ tách từ subword và mô hình gốc.
- **Khử trùng lặp trước khi chia tập (Pre-split Deduplication):**
  - **Cơ chế:** Đối chiếu và loại bỏ các bình luận trùng khớp hoàn toàn sau bước chuẩn hóa văn bản, thực hiện trước khi chia tập dữ liệu.
  - **Mục đích:** Triệt tiêu rò rỉ dữ liệu (Data Leakage) giữa tập huấn luyện và tập kiểm định/đánh giá.
- **Đệm động theo lô (Dynamic Padding):**
  - **Cơ chế:** Đệm token `<pad>` theo độ dài thực tế của câu dài nhất trong từng lô (batch) thay vì đệm cứng toàn bộ tập dữ liệu về ngưỡng trần.
  - **Mục đích:** Giảm khối lượng tính toán dư thừa trên các vị trí trống và tăng tốc độ huấn luyện mà không làm mất thông tin.

---

## 2. KIẾN TRÚC HỌC ĐA NHIỆM (MULTI-TASK ARCHITECTURE)

- **Tổng hợp đa tầng đặc trưng (Multi-Layer Feature Fusion):**
  - **Cơ chế:** Trích xuất và kết hợp biểu diễn ẩn từ các tầng cuối của bộ mã hóa (Encoder) thay vì chỉ sử dụng tầng cuối cùng.
  - **Mục đích:** Kết hợp đồng thời đặc trưng từ vựng/cú pháp bề mặt ở các tầng dưới (nhạy với từ ngữ độc hại, chửi thề) và đặc trưng ngữ nghĩa toàn cục ở tầng trên cùng (nhạy với sắc thái tình cảm).
- **Gộp thông tin bằng cơ chế chú ý có mặt nạ (Masked Attention Pooling):**
  - **Cơ chế:** Tính trọng số chú ý cho từng token trong câu, gán giá trị âm vô cùng cho các vị trí `<pad>` trước khi chuẩn hóa xác suất (Softmax), sau đó lấy trung bình có trọng số để tạo vector đại diện câu.
  - **Mục đích:** Quét toàn bộ câu để tập trung vào các từ khóa mang tính quyết định ở bất kỳ vị trí nào, đồng thời ngăn chặn hoàn toàn việc cộng lẫn các token đệm vào vector biểu diễn.
- **Nhánh suy luận chuyên biệt (Task-Specific Heads):**
  - **Cơ chế:** Tách vector biểu diễn chung thành hai nhánh độc lập cho Sentiment và Toxicity. Mỗi nhánh đi qua cụm biến đổi riêng: `Linear → GELU → LayerNorm`.
  - **Mục đích:** Tách biệt không gian đặc trưng của hai nhiệm vụ, giải quyết xung đột ngữ nghĩa trên các câu phê bình lịch sự (cảm xúc tiêu cực nhưng không độc hại).
- **Dropout đa mẫu (Multi-Sample Dropout):**
  - **Cơ chế:** Đưa đặc trưng của mỗi nhánh qua nhiều mặt nạ Dropout độc lập song song, thực hiện dự đoán trên từng luồng rồi lấy trung bình cộng kết quả đầu ra.
  - **Mục đích:** Giảm phương sai dự đoán, tăng tốc độ hội tụ và chống học vẹt (overfitting) trên các lớp thiểu số.

---

## 3. XỬ LÝ LỆCH LỚP & RANH GIỚI MỜ (IMBALANCE & AMBIGUITY HANDLING)

- **Nhánh Toxicity — Hiệu chỉnh Logit theo tần suất lớp (Logit-Adjusted Cross-Entropy):**
  - **Cơ chế:** Cộng bù một lượng tỷ lệ với logarit tần suất xuất hiện của từng lớp (điều chỉnh qua hệ số τ) trực tiếp vào giá trị dự đoán thô (logits) trước khi tính Cross-Entropy.
  - **Mục đích:** Mở rộng biên quyết định cho các mức độ độc hại hiếm gặp một cách mượt mà, tránh hiện tượng nổ đạo hàm (gradient explosion) thường gặp khi nhân trọng số nghịch đảo trực tiếp vào hàm mất mát.
- **Nhánh Sentiment — Làm mịn nhãn (Label-Smoothed Cross-Entropy):**
  - **Cơ chế:** Phân bổ một phần nhỏ xác suất mục tiêu (hệ số ε) từ nhãn đúng sang các nhãn còn lại.
  - **Mục đích:** Giảm sự tự tin cực đoan của mô hình tại vùng ranh giới chủ quan, chồng lấn giữa các sắc thái cảm xúc (đặc biệt giữa Trung tính và Tiêu cực).

---

## 4. HÀM MỤC TIÊU TỔNG HỢP (JOINT OBJECTIVE FUNCTION)

- **Tổ hợp mất mát có kiểm soát:**
  - **Cơ chế:** Kết hợp hàm mất mát của hai nhánh thông qua tổ hợp tuyến tính có trọng số cố định khớp với thước đo đánh giá (hoặc cơ chế tự cân bằng có kẹp biên độ trên/dưới).
  - **Mục đích:** Duy trì sự cân bằng đạo hàm giữa hai nhiệm vụ có cùng bản chất Cross-Entropy, ngăn chặn việc mô hình dồn toàn bộ trọng số vào nhánh dễ tối ưu hơn và gây học vẹt trên các lớp hiếm.

---

## 5. CHIẾN LƯỢC HUẤN LUYỆN & KIỂM ĐỊNH (TRAINING & VALIDATION STRATEGY)

- **Chia tập phân tầng theo tổ hợp nhãn (Jointly Stratified Splitting):**
  - **Cơ chế:** Phân chia dữ liệu dựa trên tổ hợp nhãn ghép `(Sentiment, Toxicity)`, đồng thời gom nhóm dự phòng đối với các tổ hợp nhãn có số lượng mẫu quá nhỏ trước khi chia.
  - **Mục đích:** Giữ nguyên phân phối đồng thời của cả hai nhãn trên mọi tập con mà không gây lỗi chia tập trên các tổ hợp cực hiếm.
- **Tốc độ học phân tầng (Differential Learning Rate):**
  - **Cơ chế:** Thiết lập tốc độ học thấp cho bộ khung ngôn ngữ gốc (Backbone) và tốc độ học cao hơn cho các nhánh suy luận mới khởi tạo (Heads).
  - **Mục đích:** Bảo tồn tri thức tiếng Việt đã được tiền huấn luyện của bộ khung gốc (tránh Catastrophic Forgetting) trong khi cho phép hai nhánh phân loại mới hội tụ nhanh.
- **Tối ưu hóa và điều hòa đạo hàm (Optimization & Gradient Control):**
  - **Cơ chế:** Sử dụng bộ tối ưu AdamW kèm suy giảm trọng số (Weight Decay), lịch trình tăng tốc độ học tuyến tính đầu chu kỳ (Warmup) kết hợp giảm dần theo hàm Cosine, và cắt ngưỡng chuẩn đạo hàm (Gradient Clipping).
  - **Mục đích:** Ổn định quá trình cập nhật trọng số ở giai đoạn đầu khi hai nhánh mới chưa khớp với bộ khung gốc, đồng thời chặn đứng các xung đột đạo hàm đột biến giữa hai nhiệm vụ.
- **Dừng sớm đa mục tiêu (Composite Early Stopping):**
  - **Cơ chế:** Lưu trạng thái mô hình tốt nhất và dừng huấn luyện dựa trên điểm số tổng hợp có trọng số từ Macro-F1 của cả hai nhánh trên tập kiểm định.
  - **Mục đích:** Chọn đúng thời điểm cả hai nhiệm vụ cùng đạt hiệu năng đỉnh, tránh hiện tượng một nhánh đã học vẹt trong khi nhánh kia chưa hội tụ xong.
- **Học kết hợp chéo (K-Fold Cross-Validation Ensemble):**
  - **Cơ chế:** Huấn luyện các mô hình độc lập trên K phần dữ liệu giao thoa, sau đó lấy trung bình cộng phân phối xác suất của toàn bộ K mô hình khi suy luận.
  - **Mục đích:** Giảm thiểu sai số ngẫu nhiên giữa các lần chia tập và nâng cao tính ổn định trên dữ liệu thực tế.

---

## 6. ĐÁNH GIÁ & SUY LUẬN (EVALUATION & INFERENCE)

- **Quyết định nhãn trực tiếp từ xác suất đã hiệu chỉnh:**
  - **Cơ chế:** Lấy nhãn có xác suất trung bình cao nhất sau khi đã được cân bằng tự nhiên bởi cơ chế hiệu chỉnh Logit ở Giai đoạn 3; hạn chế dò ngưỡng hậu kỳ phức tạp trên các tập kiểm định kích thước nhỏ.
  - **Mục đích:** Tránh hiện tượng khớp quá mức (overfitting) ngưỡng cắt vào một số ít mẫu hiếm của tập kiểm định, bảo toàn khả năng tổng quát hóa khi gặp phân phối dữ liệu mới.
- **Hệ thống thước đo đánh giá:**
  - **Macro-F1:** Thước đo chính, đánh giá bình đẳng hiệu năng của mọi lớp bất kể số lượng mẫu nhiều hay ít.
  - **Per-class F1:** Theo dõi chi tiết năng lực phân loại trên từng mức độ cảm xúc và từng cấp độ độc hại.
  - **PR-AUC (Precision-Recall AUC):** Đánh giá chất lượng phân tách các lớp độc hại thiểu số độc lập với ngưỡng quyết định.
  - **Confusion Matrix:** Ma trận nhầm lẫn phục vụ kiểm tra trực quan các lỗi giao thoa giữa hai chiều cảm xúc và độc hại.
