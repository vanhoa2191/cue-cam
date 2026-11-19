# CueCam

Teleprompter tiếng Việt chạy trên web, dùng cho cá nhân.

- Cuộn tự động hoặc **cuộn theo giọng nói** (Web Speech, tiếng Việt; tốt nhất trên Chrome)
- **Quay video** bằng camera trước/sau, chữ chạy ở nửa trên màn hình sát ống kính, lưu từng take
- **Làm đẹp khi quay** (nút ✨): mịn da (chỉ vùng da mặt, giữ nét mắt môi), sáng da, tông ấm/lạnh, đèn studio, 10 bộ lọc màu, làm mờ hoặc thay phông nền. Hiệu ứng có trong file video đã quay. Chạy hoàn toàn trên máy.
- Cỡ chữ, giãn dòng, vạch đọc, lật gương, đếm ngược, phím tắt, remote bấm slide (PageUp/PageDown)
- Nút Dán, nhập file PDF, Word (.docx), TXT, Markdown (thư viện pdf.js và mammoth nằm trong `lib/`)
- Kịch bản lưu ngay trong trình duyệt; nút "Sao lưu" xuất ra file JSON
- Mục "Liên hệ & ủng hộ" có mã VietQR

Mở: https://vanhoa2191.github.io/cue-cam/

Toàn bộ app nằm trong `index.html` (cộng thư viện trong `lib/`), không cần build.

- Ghi âm MP3 dùng [lamejs](https://github.com/zhuker/lamejs) (LGPL-3.0), file gốc ở `lib/lame.min.js`.
- Lọc ồn chế độ Studio dùng RNNoise qua [@sapphi-red/web-noise-suppressor](https://github.com/sapphi-red/web-noise-suppressor) (MIT), file ở `lib/ns/`.
- Làm đẹp và phông nền: WebGL trong `lib/beauty.js`; nhận diện người và khuôn mặt bằng [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe) 1.2.2 (Apache-2.0) chạy trong Web Worker `lib/beauty-worker.js`, file ở `lib/mp/` (selfie segmenter, BlazeFace short range).
