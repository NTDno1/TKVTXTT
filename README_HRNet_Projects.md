# HRNet Projects - Tài liệu học tập Bộ môn Tìm kiếm và Truy xuất Thông tin

## Thông tin

- **Sinh viên**: Thạc sĩ Học viện Bưu chính Viễn thông
- **Bộ môn**: Tìm kiếm và Truy xuất Thông tin (Information Retrieval)
- **Các dự án**: 3 repo HRNet từ Microsoft Research
- **Ngày cập nhật**: 2026-05-17

---

## Cấu trúc thư mục

```
TKVTXTT/
├── README.md                              ← File này (tổng quan)
├── HRNet_TongQuanHeThong_PhanTichKyThuat.md  ← Phân tích kỹ thuật gốc
│
├── HuongDanCaiDatVaChay_3DuAn_HRNet.md  ← Hướng dẫn cài đặt & chạy
│   ├── Cài đặt môi trường chung
│   ├── Cài đặt chi tiết từng dự án
│   ├── Cách chạy training/inference
│   └── FAQ - Lỗi thường gặp
│
├── LuongCodeChiTiet_3DuAn_HRNet.md      ← Luồn code chi tiết
│   ├── Entry points (train.py, test.py)
│   ├── Luồn data loading
│   ├── Luồn model forward
│   ├── Luồn inference
│   └── So sánh 3 dự án
│
├── HRNet-Object-Detection/              ← Repo 1: Phát hiện đối tượng
│   ├── tools/train.py                   ← Entry point training
│   ├── tools/test.py                    ← Entry point testing
│   ├── configs/hrnet/                   ← Config files
│   └── mmdet/                          ← Core library
│       ├── models/backbones/hrnet.py    ← HRNet backbone
│       ├── models/necks/hrfpn.py        ← HRFPN neck
│       └── models/detectors/           ← Detector models
│
├── HigherHRNet-Human-Pose-Estimation/   ← Repo 2: Ước lượng tư thế người
│   ├── tools/dist_train.py             ← Entry point training
│   ├── tools/valid.py                  ← Entry point validation
│   ├── lib/models/pose_higher_hrnet.py  ← HigherHRNet model
│   ├── lib/core/inference.py           ← Inference logic
│   └── lib/core/group.py               ← Keypoint grouping (AE)
│
└── HRNet-Facial-Landmark-Detection/    ← Repo 3: Phát hiện điểm mốc khuôn mặt
    ├── tools/train.py                  ← Entry point training
    ├── tools/test.py                   ← Entry point testing
    ├── lib/models/hrnet.py             ← HRNetV2-W18 model
    ├── lib/core/function.py            ← Training/testing logic
    └── lib/core/evaluation.py         ← NME metric computation
```

---

## Tổng quan 3 dự án

### 1. HRNet-Object-Detection (Phát hiện đối tượng)

| Thông tin | Chi tiết |
|-----------|-----------|
| Paper | High-Resolution Representations for Object Detection |
| Framework | mmdetection (PyTorch) |
| Input | Ảnh RGB |
| Output | Bounding boxes + Class labels + Masks |
| Dataset | COCO (80 classes) |
| Metric | COCO mAP |
| Best model | Cascade Mask R-CNN + HRNetV2-W48: **47.0 mAP** |

**Các bước chính:**
1. HRNet backbone trích xuất đặc trưng đa mức phân giải
2. HRFPN tổng hợp đặc trưng
3. RPN sinh proposals
4. RoIAlign trích xuất đặc trưng cho từng proposal
5. BBox Head phân loại và regress vị trí

### 2. HigherHRNet-Human-Pose-Estimation (Ước lượng tư thế người)

| Thông tin | Chi tiết |
|-----------|-----------|
| Paper | Bottom-Up Human Pose Estimation via Hierarchical Multi-Resolution Representation |
| Framework | PyTorch tự viết |
| Input | Ảnh RGB |
| Output | 17 keypoints × (x, y, score) cho mỗi người |
| Dataset | COCO Keypoints (17 joints) |
| Metric | COCO Keypoint AP |
| Best model | HigherHRNet-W48: **67.1 AP** |

**Các bước chính:**
1. HigherHRNet backbone với deconv layers tạo multi-resolution heatmaps
2. Trích xuất heatmaps + associative embedding tags
3. NMS trên heatmaps tìm peaks
4. Associative Embedding grouping ghép keypoints thành từng người
5. Transform về tọa độ gốc

### 3. HRNet-Facial-Landmark-Detection (Phát hiện điểm mốc khuôn mặt)

| Thông tin | Chi tiết |
|-----------|-----------|
| Paper | High-Resolution Representations for Facial Landmark Detection |
| Framework | PyTorch tự viết |
| Input | Ảnh khuôn mặt |
| Output | N × (x, y) landmark coordinates |
| Dataset | 300W (68 pts), WFLW (98 pts), AFLW (19 pts), COFW (29 pts) |
| Metric | NME (Normalized Mean Error) |
| Best model | HRNetV2-W18 + WFLW: **4.60 NME** |

**Các bước chính:**
1. HRNetV2-W18 backbone trích xuất đặc trưng
2. Fuse tất cả nhánh về 1 tensor
3. Head conv tạo heatmaps cho từng landmark
4. Argmax + sub-pixel refinement decode tọa độ
5. Inverse transform về tọa độ ảnh gốc

---

## Nhanh nhất - Bắt đầu trong 5 phút

### Cách chạy nhanh nhất (Inference với pretrained model)

#### HRNet-Object-Detection
```bash
# Clone và setup
cd HRNet-Object-Detection
pip install mmcv==0.5.1 pycocotools
./compile.sh  # Biên dịch CUDA ops
python setup.py develop

# Tải pretrained model (xem MODEL_ZOO.md)
# Chạy inference
python tools/test.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py \
    HRNetV2-W18-FasterRCNN.pth --eval bbox
```

#### HigherHRNet-Pose
```bash
# Clone và setup
cd HigherHRNet-Human-Pose-Estimation
pip install -r requirements.txt

# Tải pretrained model
# Chạy inference
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE pose_higher_hrnet_w32_512.pth
```

#### HRNet-Facial-Landmark
```bash
# Clone và setup
cd HRNet-Facial-Landmark-Detection
pip install -r requirements.txt

# Tải pretrained model
# Chạy inference
python tools/test.py \
    --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
    --model-file HR18-WFLW.pth
```

---

## Đọc tài liệu nào trước?

```
Bạn muốn...                    → Đọc file
─────────────────────────────────────────────────────────────
1. Hiểu project làm gì        → README.md này
2. Cài đặt và chạy được      → HuongDanCaiDatVaChay_3DuAn_HRNet.md
3. Hiểu luồn code chi tiết   → LuongCodeChiTiet_3DuAn_HRNet.md
4. Hiểu lý thuyết HRNet       → HRNet_TongQuanHeThong_PhanTichKyThuat.md
5. Tìm hiểu ứng dụng IR      → Chương 7 trong HuongDanCaiDatVaChay_3DuAn_HRNet.md
```

---

## So sánh nhanh 3 dự án

```
                    Object Detection    Pose Estimation    Facial Landmark
                    ────────────────   ────────────────   ────────────────
Mục đích             Phát hiện vật      Ước lượng tư thế    Xác định điểm mốc
                    thể trong ảnh       người               trên khuôn mặt

Output               BBox + Label       17 keypoints        N điểm (68-98)

Kỹ thuật đặc biệt   RPN + RoIAlign    Associative         Heatmap regression
                                        Embedding           + Gaussian target

Loại detection       Two-stage          Bottom-up           Single-stage

Metric               mAP               Keypoint AP         NME (%)

Độ khó setup         Cao (CUDA ops)    Trung bình          Thấp

Thời gian train       ~2 ngày (8 GPU)  ~1 ngày (4 GPU)     ~4 giờ (1 GPU)

Điểm tương đồng      ─────────────────────────────────────────
                     Đều dùng HRNet backbone + đều có input preprocessing
                     + heatmap/head output + post-processing + metric evaluation
```

---

## Tài liệu tham khảo gốc

1. HRNet-Object-Detection: https://github.com/HRNet/HRNet-Object-Detection
2. HigherHRNet-Human-Pose-Estimation: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
3. HRNet-Facial-Landmark-Detection: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
4. HRNet-Image-Classification (pretrained weights): https://github.com/HRNet/HRNet-Image-Classification
5. mmdetection: https://github.com/open-mmlab/mmdetection

---

## License

Tất cả 3 dự án thuộc về **Microsoft Research** và được phân phối theo giấy phép **MIT** (HigherHRNet, HRNet-Facial-Landmark) hoặc **Apache 2.0** (HRNet-Object-Detection).
