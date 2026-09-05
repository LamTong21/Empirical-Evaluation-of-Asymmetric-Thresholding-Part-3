## 📌 Mô tả Thay đổi
<!-- Tóm tắt ngắn gọn tính năng hoặc sửa đổi trong PR -->

## 🔬 Kiểm tra Tính toàn vẹn Kinh tế lượng (Econometric Checklist)
- [ ] **Không có Lookahead Bias:** Các biến trễ, tín hiệu Primary và giá trị target đã được shift phù hợp.
- [ ] **Time Barrier Integrity:** Tập test không bị rò rỉ vào quá trình trích xuất đặc trưng (Stage 2, 3, 4 đều fit trên tập train).
- [ ] **Purged & Embargoed:** Khoảng cách `purge_gap` bằng hoặc lớn hơn horizon gắn nhãn ($h=5$).
- [ ] **Market Friction:** Logic khớp lệnh tuân thủ trễ $T+1$, deadband tối thiểu 15% và thời gian giữ lệnh $T+2.5$.

## 📊 Kết quả Thực thi
<!-- Đính kèm Sharpe Net, Max Drawdown hoặc kết quả chạy kiểm thử CI -->