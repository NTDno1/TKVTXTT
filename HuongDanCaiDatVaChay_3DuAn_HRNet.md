# Hướng Dẫn Cài Đặt và Chạy 3 Dự Án HRNet
## Bộ môn: Tìm kiếm và Truy xuất Thông tin - Thạc sĩ Học viện Bưu chính Viễn thông

---

## Mục lục

1. [Tổng quan 3 dự án](#1-tổng-quan-3-dự-án)
2. [Cài đặt môi trường chung](#2-cài-đặt-môi-trường-chung)
3. [Dự án 1: HRNet-Object-Detection (Phát hiện đối tượng)](#3-dự-án-1-hrnet-object-detection-phát-hiện-đối-tượng)
4. [Dự án 2: HigherHRNet-Human-Pose-Estimation (Ước lượng tư thế người)](#4-dự-án-2-higherhrnet-human-pose-estimation-ước-lượng-tư-thế-người)
5. [Dự án 3: HRNet-Facial-Landmark-Detection (Phát hiện điểm mốc khuôn mặt)](#5-dự-án-3-hrnet-facial-landmark-detection-phát-hiện-điểm-mốc-khuôn-mặt)
6. [Luồng xử lý dữ liệu chung](#6-luồn-xử-lý-dữ-liệu-chung)
7. [Ứng dụng trong Tìm kiếm và Truy xuất Thông tin](#7-ứng-dụng-trong-tìm-kiếm-và-truy-xuất-thông-tin)
8. [FAQ - Các lỗi thường gặp](#8-faq---các-lỗi-thường-gặp)

---

## 1. Tổng quan 3 dự án

Cả 3 dự án đều được phát triển bởi **Microsoft Research** và sử dụng chung kiến trúc **HRNet (High-Resolution Network)** - mạng học sâu duy trì biểu diễn độ phân giải cao xuyên suốt.

| Dự án | Input | Output | Mục đích |
|-------|-------|--------|----------|
| **HRNet-Object-Detection** | Ảnh RGB | Bounding boxes + Nhãn lớp + Mask (tuỳ model) | Phát hiện và phân loại đối tượng trong ảnh |
| **HigherHRNet-Human-Pose-Estimation** | Ảnh RGB | Tọa độ (x,y) của 17 điểm khớp xương trên cơ thể người | Ước lượng tư thế/con người trong ảnh (multi-person, bottom-up) |
| **HRNet-Facial-Landmark-Detection** | Ảnh khuôn mặt | Tọa độ (x,y) của N điểm mốc trên khuôn mặt | Xác định vị trí các điểm quan trọng trên khuôn mặt |

### Kiến trúc HRNet - Nền tảng chung

```
Đầu vào (Image)
     │
     ▼
┌──────────────────────────────┐
│  STEM: conv1 → bn1 → conv2  │  Giảm độ phân giải 4 lần
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Stage 2: 2 nhánh song song  │  Nhánh 1: 1/4 độ phân giải, 18 kênh
│  (BasicBlock × 4 mỗi nhánh)  │  Nhánh 2: 1/8 độ phân giải, 36 kênh
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Stage 3: 3 nhánh song song  │  Thêm nhánh 1/16 độ phân giải, 72 kênh
│  (BasicBlock × 4 mỗi nhánh)  │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  Stage 4: 4 nhánh song song  │  Thêm nhánh 1/32 độ phân giải, 144 kênh
│  (BasicBlock × 4 mỗi nhánh)  │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│  FUSE: Trộn đặc trưng các   │  Upsample + Concatenate các nhánh
│  nhánh để tạo đặc trưng     │
│  đa mức phân giải            │
└──────────────────────────────┘
     │
     ▼
Đặc trưng đa mức phân giải
```

**Điểm khác biệt so với mạng truyền thống (ResNet, VGG):**
- Mạng truyền thống: Encoder (giảm phân giải → tăng semantic) → Decoder (tăng phân giải lại)
- HRNet: Duy trì **đồng thời** nhiều mức phân giải xuyên suốt → giữ được thông tin không gian chi tiết

---

## 2. Cài đặt môi trường chung

### 2.1. Yêu cầu phần cứng

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|-----------|------------------|-------------|
| GPU | NVIDIA GPU 4GB VRAM | NVIDIA GPU 8GB+ VRAM (RTX 2070+) |
| RAM | 8GB | 16GB+ |
| Ổ cứng | 50GB trống | 100GB+ SSD |
| CUDA | CUDA 9.0+ | CUDA 10.2+ |
| cuDNN | cuDNN 7.0+ | cuDNN 7.6+ |

### 2.2. Cài đặt Anaconda

```bash
# 1. Tải Anaconda
# https://www.anaconda.com/download

# 2. Tạo môi trường conda mới
conda create -n hrnet python=3.6
conda activate hrnet

# 3. Cài đặt PyTorch với CUDA
# Với CUDA 10.2:
pip install torch==1.8.0 torchvision==0.9.0

# Với CUDA 11.3:
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 4. Kiểm tra cài đặt
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Kết quả mong đợi: PyTorch: 1.x.x, CUDA: True
```

### 2.3. Cài đặt Git (nếu chưa có)

```bash
# Windows: Tải từ https://git-scm.com/download/win
# Hoặc dùng PowerShell
winget install Git.Git
```

### 2.4. Clone 3 repo về máy

```bash
cd C:\Users\Admin\Documents\MyGit\TKVTXTT

git clone https://github.com/HRNet/HRNet-Object-Detection.git --depth 1
git clone https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.git --depth 1
git clone https://github.com/HRNet/HRNet-Facial-Landmark-Detection.git --depth 1
```

> **Lưu ý:** Tất cả 3 repo đã được clone sẵn trong thư mục workspace này.

---

## 3. Dự án 1: HRNet-Object-Detection (Phát hiện đối tượng)

### 3.1. Giới thiệu

Dự án triển khai paper **"High-Resolution Representations for Object Detection"** (HRNet-V2). Xây dựng trên **mmdetection framework**.

**Các model được hỗ trợ:**
- Faster R-CNN (2-stage detection)
- Mask R-CNN (detection + instance segmentation)
- Cascade R-CNN (multi-stage refinement)
- Cascade Mask R-CNN / HTC (best: 47.0 mAP)
- RetinaNet (1-stage detection)

**Hiệu suất trên COCO val2017:**

| Model | mAP |
|-------|-----|
| HRNetV2-W18 + Faster R-CNN | 37.6 |
| HRNetV2-W18 + Faster R-CNN (SyncBN + Multi-scale) | 39.4 |
| HRNetV2-W32 + Faster R-CNN | 42.6 |
| HRNetV2-W48 + Cascade Mask R-CNN | 47.0 |

### 3.2. Cài đặt chi tiết

```bash
cd C:\Users\Admin\Documents\MyGit\TKVTXTT\HRNet-Object-Detection

# 1. Cài đặt mmcv
pip install mmcv==0.5.1

# 2. Cài đặt pycocotools
git clone https://github.com/cocodataset/cocoapi.git $env:TEMP\cocoapi
cd $env:TEMP\cocoapi\PythonAPI
python setup.py build_ext install
cd C:\Users\Admin\Documents\MyGit\TKVTXTT\HRNet-Object-Detection

# 3. Cài đặt NVIDIA Apex (cho SyncBatchNorm)
git clone https://github.com/NVIDIA/apex $env:TEMP\apex
cd $env:TEMP\apex
python setup.py install --cuda_ext --cpp_ext

# 4. Biên dịch CUDA extensions
chmod +x compile.sh
# Windows PowerShell: thực thi từng dòng trong compile.sh
nvcc -std=c++11 -O3 -DINFO -DMSDEBUG -arch=sm_60 -I../include -I../mmdet/ops -c roi_align/src/roi_align_cuda.cu -o roi_align/src/roi_align_cuda.o
# ... (hoặc sử dụng bash nếu có Git Bash)

# 5. Cài đặt mmdetection
python setup.py develop
# Hoặc: pip install -e .

# 6. Tải pretrained backbone
mkdir hrnetv2_pretrained
# Tải HRNetV2-W18 ImageNet pretrained từ:
# https://github.com/HRNet/HRNet-Image-Classification
```

### 3.3. Tải dữ liệu COCO

```bash
mkdir -p data/coco
cd data/coco

# Tải từ http://cocodataset.org/#download
# Cần các file:
# - train2017.zip
# - val2017.zip
# - annotations_trainval2017.zip

# Giải nén
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### 3.4. Cấu trúc thư mục dữ liệu

```
HRNet-Object-Detection/
├── data/
│   └── coco/
│       ├── annotations/
│       │   ├── instances_train2017.json
│       │   └── instances_val2017.json
│       ├── train2017/
│       │   ├── 000000000009.jpg
│       │   └── ...
│       └── val2017/
│           ├── 000000000001.jpg
│           └── ...
├── hrnetv2_pretrained/
│   └── hrnetv2_w18_imagenet_pretrained.pth
├── configs/
│   └── hrnet/
│       └── faster_rcnn_hrnetv2p_w18_1x.py
├── tools/
│   ├── train.py
│   └── test.py
└── mmdet/
```

### 3.5. Cách chạy

**Training 1 GPU:**
```bash
python tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py --gpus 1
```

**Training đa GPU:**
```bash
python tools/train.py configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py --gpus 8
```

**Training với SyncBatchNorm + Multi-scale:**
```bash
python tools/train.py \
    configs/hrnet/faster_rcnn_hrnetv2p_w18_syncbn_mstrain_2x.py \
    --gpus 4
```

**Test / Inference:**
```bash
python tools/test.py \
    configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py \
    HRNet-Object-Detection.pth \
    --gpus 1 \
    --eval bbox
```

### 3.6. Luồng xử lý code

```
Ảnh đầu vào
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 1: Data Loading (mmdet/datasets/coco.py)               │
│ - Load ảnh từ COCO dataset                                   │
│ - Random scale jitter, flip, normalize ImageNet             │
│ - Output: tensor [3, H, W], gt_bboxes, gt_labels            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 2: Feature Extraction (mmdet/models/backbones/hrnet.py)│
│ Stage 1: 1 nhánh (1/4 resolution) - 64 channels             │
│ Stage 2: 2 nhánh (1/4, 1/8) - 18, 36 channels               │
│ Stage 3: 3 nhánh (1/4, 1/8, 1/16) - 18, 36, 72 channels    │
│ Stage 4: 4 nhánh (1/4, 1/8, 1/16, 1/32) - 18,36,72,144     │
│ Fuse layers: Trộn đặc trưng giữa các nhánh                  │
│ Output: Feature maps đa mức phân giải                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 3: HRFPN Neck (mmdet/models/necks/hrfpn.py)            │
│ - Upsample tất cả các nhánh về độ phân giải cao nhất       │
│ - Concatenate theo chiều kênh                               │
│ - 1x1 conv giảm về 256 kênh                                │
│ - Tạo 5 mức đặc trưng: P0, P1, P2, P3, P4                   │
│ Output: [P0, P1, P2, P3, P4] mỗi mức 256 kênh              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 4: RPN Head (Region Proposal Network)                 │
│ mmdet/models/anchor_heads/rpn_head.py                       │
│ - Conv 3x3 + cls branch (số anchors) + reg branch (delta)  │
│ - Sinh anchors: ~240K anchors trên 5 mức đặc trưng          │
│ - Anchor Target Assignment (MaxIoUAssigner)                │
│ - NMS loại bỏ trùng lặp                                     │
│ Output: ~2000 proposals                                      │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 5: ROI Extractor (RoIAlign)                           │
│ mmdet/models/roi_extractors/single_level.py                │
│ - Ánh xạ proposals → mức đặc trưng phù hợp theo scale    │
│ - RoIAlign 7x7, sample=2                                   │
│ Output: ROI features kích thước cố định [N, 256, 7, 7]     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 6: BBox Head (Classification + Localization)          │
│ mmdet/models/bbox_heads/convfc_bbox_head.py                 │
│ - 2 FC layers (1024 dim)                                    │
│ - cls: num_classes (81 cho COCO)                            │
│ - reg: 4*num_classes (bbox delta)                           │
│ - BBox Target Assignment + Sampling                        │
│ Output: Class scores + Bounding box coordinates            │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 7: (Tuỳ model) Mask Head                               │
│ mmdet/models/mask_heads/fcn_mask_head.py                    │
│ - Cho Mask R-CNN, Cascade Mask R-CNN, HTC                  │
│ - 4 conv layers + deconv 2x → mask 28x28                   │
│ - Sigmoid per class                                         │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Kết quả: [x1, y1, x2, y2, score, class_id] + (mask)
```

### 3.7. Các file config quan trọng

```python
# configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py

# backbone: HRNetV2-W18
# - Stage 2: 2 branches (18, 36 channels)
# - Stage 3: 3 branches (18, 36, 72 channels)
# - Stage 4: 4 branches (18, 36, 72, 144 channels)

# neck: HRFPN
# - in_channels: [18, 36, 72, 144]
# - out_channels: 256

# rpn_head: RPN
# - anchor_scales: [8]
# - anchor_ratios: [0.5, 1.0, 2.0]
# - anchor_strides: [4, 8, 16, 32, 64]

# bbox_head: SharedFCBBoxHead
# - num_classes: 81 (COCO: 80 classes + background)
# - roi_feat_size: 7

# train_cfg:
# - RPN: rpn_proposal (2000 nms_thr=0.7)
# - RCNN: rcnn (256 samples, 25% positive)
# - Learning rate: 0.02, Step at [8, 11]

# test_cfg:
# - max_per_img: 100
# - score_thr: 0.05
# - nms_thr: 0.5
```

---

## 4. Dự án 2: HigherHRNet-Human-Pose-Estimation (Ước lượng tư thế người)

### 4.1. Giới thiệu

Triển khai paper **"Bottom-Up Human Pose Estimation via Hierarchical Multi-Resolution Representation"** (HigherHRNet), CVPR 2020.

**Điểm khác biệt Bottom-up vs Top-down:**
- **Top-down**: Phát hiện từng người trước → ước lượng keypoints cho từng người (phụ thuộc detector)
- **Bottom-up**: Phát hiện tất cả keypoints cùng lúc → nhóm các keypoints thành từng người (độc lập detector)

**Hiệu suất trên COCO val2017:**

| Model | Input | AP |
|-------|-------|-----|
| HigherHRNet-W32 | 512 | 64.1 |
| HigherHRNet-W48 | 640 | **67.1** |

**COCO có 17 keypoints:**
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

### 4.2. Cài đặt chi tiết

```bash
cd C:\Users\Admin\Documents\MyGit\TKVTXTT\HigherHRNet-Human-Pose-Estimation

# 1. Cài đặt dependencies
pip install -r requirements.txt

# Các package chính:
# torch>=1.1.0, opencv-python, tensorboardX, munkres, scipy,
# scikit-image, pandas, yacs, Cython, json_tricks, tqdm

# 2. Cài đặt COCO API
git clone https://github.com/cocodataset/cocoapi.git $env:TEMP\cocoapi
cd $env:TEMP\cocoapi\PythonAPI
python setup.py build_ext install

# 3. Tạo thư mục output
mkdir output log
```

### 4.3. Chuẩn bị dữ liệu COCO Keypoints

```bash
mkdir -p data/coco
cd data/coco

# Tải từ http://cocodataset.org/#download
# - images/train2017.zip
# - images/val2017.zip
# - annotations/person_keypoints_train2017.json
# - annotations/person_keypoints_val2017.json

# Cấu trúc thư mục:
data/coco/
├── annotations/
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   └── ...
└── val2017/
    ├── 000000000001.jpg
    └── ...
```

### 4.4. Cách chạy

**Training (Distributed):**
```bash
# Training trên 4 GPU
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    GPUS 4
```

**Training với Mixed-Precision (FP16):**
```bash
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    GPUS 4 \
    FP16.ENABLED True \
    FP16.DYNAMIC_LOSS_SCALE True
```

**Validation / Test:**
```bash
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth
```

### 4.5. Luồng xử lý code

```
Ảnh đầu vào
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 1: Preprocessing (tools/valid.py)                      │
│ - Resize về input_size (512 hoặc 640)                      │
│ - Chuẩn hóa ImageNet                                        │
│ - Tính center và scale từ bounding box                     │
│ - Multi-scale inference (1.0, 1.0, 1.0,... đã config sẵn)    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 2: HigherHRNet Forward                                 │
│ lib/models/pose_higher_hrnet.py                            │
│                                                              │
│ Stem: conv1 → bn1 → conv2 → bn2 → layer1                   │
│   Output: [B, 64, H/4, W/4]                                 │
│                                                              │
│ Stage 2: 2 nhánh song song                                  │
│   Nhánh 1: [B, 32, H/4, W/4]                               │
│   Nhánh 2: [B, 64, H/8, W/8]                               │
│   Fuse: upsampling + concatenation                          │
│                                                              │
│ Stage 3: 3 nhánh song song                                  │
│   Nhánh 1: [B, 32, H/4, W/4]                               │
│   Nhánh 2: [B, 64, H/8, W/8]                               │
│   Nhánh 3: [B, 128, H/16, W/16]                            │
│                                                              │
│ Stage 4: 4 nhánh song song                                  │
│   Nhánh 1: [B, 32, H/4, W/4]                               │
│   Nhánh 2: [B, 64, H/8, W/8]                               │
│   Nhánh 3: [B, 128, H/16, W/16]                            │
│   Nhánh 4: [B, 256, H/32, W/32]                             │
│                                                              │
│ Deconv layers: Transposed Convolution                        │
│   Upsample để tạo multi-resolution heatmaps                 │
│   Output: List[heatmaps, tags] at multiple scales           │
│                                                              │
│ Heatmaps: 17 kênh (sigmoid activation)                     │
│ Tags: M kênh (associative embedding)                        │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 3: Multi-scale Aggregation (lib/core/inference.py)     │
│ lib/core/inference.py :: get_multi_stage_outputs()          │
│                                                              │
│ - Interpolation tất cả outputs về cùng kích thước           │
│ - Tách heatmap và tags                                      │
│ - Heatmaps: lấy trung bình các scale                        │
│ - Tags: concatenate theo chiều kênh                         │
│                                                              │
│ Flip Test:                                                   │
│ - Xử lý ảnh flip ngang                                      │
│ - Flip back heatmaps, swap left-right joints                │
│ - Trung bình với kết quả gốc                                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 4: Keypoint Detection (lib/core/group.py)               │
│ HeatmapParser.parse()                                       │
│                                                              │
│ 4a. NMS: Non-Maximum Suppression trên mỗi heatmap           │
│     - Tìm các đỉnh cục bộ (local maximum)                  │
│     - Giữ lại đỉnh có giá trị > ngưỡng                     │
│                                                              │
│ 4b. Top-K: Giữ top K điểm cho mỗi joint                    │
│     - K = 30 mặc định                                       │
│                                                              │
│ 4c. Tag-based Grouping (Associative Embedding):             │
│     - Mỗi keypoint có tag vector (1D hoặc multi-D)          │
│     - Keypoints cùng 1 người có tags gần nhau              │
│     - Hungarian algorithm ghép keypoints thành từng người  │
│                                                              │
│ 4d. Sub-pixel Refinement:                                   │
│     - Tinh chỉnh tọa độ bằng gradient around peak          │
│     - Độ chính xác sub-pixel                                │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 5: Transform về tọa độ gốc                             │
│ lib/utils/transforms.py :: transform_preds()                │
│                                                              │
│ - Inverse affine transform từ heatmap → ảnh gốc            │
│ - Sử dụng center, scale đã tính ở bước 1                   │
│ Output: List of persons, mỗi người có 17 điểm [x, y, score]│
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Kết quả: Danh sách tư thế người (multi-person)
```

### 4.6. Associative Embedding - Chi tiết kỹ thuật

**Bài toán:** Làm sao biết keypoint nào thuộc về người nào?

**Giải pháp:** Gán mỗi keypoint một **tag** (embedding vector). Keypoints cùng người → tags gần nhau.

```
Tag vector cho mỗi keypoint: [t1, t2, ..., tk]

Loss function:
- Pull loss: Keypoints cùng người → kéo tags lại gần nhau
  L_pull = Σ exp(||tag_i - tag_j||²) cho cùng 1 người

- Push loss: Keypoints khác người → đẩy tags ra xa nhau
  L_push = max(0, 1 - ||tag_i - tag_j||) cho khác người
```

### 4.7. Các file config quan trọng

```yaml
# experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml

MODEL:
  NAME: pose_higher_hrnet
  NUM_JOINTS: 17          # COCO: 17 keypoints
  EXTRA:
    STAGE2: {NUM_BRANCHES: 2, ...}
    STAGE3: {NUM_BRANCHES: 3, ...}
    STAGE4: {NUM_BRANCHES: 4, ...}
    DECONV:               # Transposed convolutions
      NUM_DECONVS: 1
      CHANNELS: [256]

LOSS:
  WITH_HEATMAPS_LOSS: [True, True]    # Multi-scale supervision
  WITH_AE_LOSS: [True, False]         # Associative embedding
  HEATMAP_LOSS_WEIGHT: 1.0
  AE_LOSS_WEIGHT: 1.0

DATASET:
  INPUT_SIZE: 512          # Kích thước ảnh đầu vào
  OUTPUT_SIZE: [128, 256]  # Kích thước heatmap đầu ra

TRAIN:
  BATCH_SIZE: 32
  OPTIMIZER: adam
  LR: 0.001
  END_EPOCH: 210

TEST:
  FLIP: True              # Flip test
  SCALE_FACTOR: [1.0]     # Multi-scale factor
```

---

## 5. Dự án 3: HRNet-Facial-Landmark-Detection (Phát hiện điểm mốc khuôn mặt)

### 5.1. Giới thiệu

Triển khai paper **"High-Resolution Representations for Facial Landmark Detection"**, sử dụng HRNetV2 cho bài toán xác định vị trí các điểm mốc (landmarks) trên khuôn mặt.

**Các dataset được hỗ trợ:**

| Dataset | Số landmarks | Metric | NME |
|---------|-------------|--------|-----|
| **300W** | 68 | Interocular norm | 3.34 |
| **AFLW** | 19 | Face bounding box norm | 1.57 |
| **COFW** | 29 | Interocular norm | 3.45 |
| **WFLW** | 98 | Interocular norm | 4.60 |

**Điểm mốc 68 điểm (300W):**
```
0-16: Jawline (đường viền hàm)
17-26: Eyebrows (lông mày)
27-35: Nose (mũi)
36-47: Eyes (mắt)
48-67: Mouth (miệng)
```

### 5.2. Cài đặt chi tiết

```bash
cd C:\Users\Admin\Documents\MyGit\TKVTXTT\HRNet-Facial-Landmark-Detection

# 1. Cài đặt dependencies
pip install -r requirements.txt

# Package chính:
# - torch>=1.0.0
# - yacs (config system)
# - hdf5storage (đọc file .mat của COFW)
# - tensorboardX (logging)
# - pandas, opencv-python, scipy, numpy

# 2. Tải pretrained HRNetV2 backbone
mkdir hrnetv2_pretrained
# Tải HRNetV2-W18 ImageNet pretrained:
# https://github.com/HRNet/HRNet-Image-Classification
# File: hrnetv2_w18_imagenet_pretrained.pth
```

### 5.3. Chuẩn bị dữ liệu

```bash
mkdir -p data
cd data

# Tải annotations từ Google Drive links trong README.md
# hoặc Cloudstor / BaiduYun (xem README.md chi tiết)

# Cấu trúc dữ liệu 300W (ví dụ):
data/
├── 300W/
│   ├── images/
│   │   ├── lfpw/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── afw/
│   │   ├── ibug/
│   │   └── helen/
│   └── labels/
│       ├── 300w_train.csv
│       └── 300w_test.csv
│
├── WFLW/
│   ├── images/
│   └── labels/
│       ├── wflw_train.csv
│       └── wflw_test.csv
```

**Định dạng CSV (ví dụ 300W):**
```csv
path,scale,center_x,center_y,# 68 landmarks (x1,y1,x2,y2,...,x68,y68)
helen/image_001.jpg,1.2,256,256,120,85,122,83,...,200,220
```

### 5.4. Cách chạy

**Training:**
```bash
# 300W dataset
python tools/train.py \
    --cfg experiments/300w/face_alignment_300w_hrnet_w18.yaml

# WFLW dataset
python tools/train.py \
    --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml
```

**Test / Inference:**
```bash
python tools/test.py \
    --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
    --model-file path/to/HR18-WFLW.pth
```

**Thay đổi số epochs:**
```bash
python tools/train.py \
    --cfg experiments/wflw/face_alignment_wflw_hrnet_w18.yaml \
    TRAIN.END_EPOCH 100
```

### 5.5. Luồng xử lý code

```
Ảnh khuôn mặt đầu vào
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 1: Data Loading                                       │
│ lib/datasets/wflw.py (WFLW.__getitem__)                     │
│                                                              │
│ 1a. Load ảnh (PIL.Image)                                    │
│ 1b. Đọc CSV: image_path, scale, center, (x,y) của 98 điểm  │
│ 1c. Data Augmentation:                                     │
│     - Random scale jitter (±25%)                           │
│     - Random rotation (±30°, p=0.6)                        │
│     - Random horizontal flip (p=0.5)                       │
│ 1d. Crop: affine transform về input_size (256x256)          │
│ 1e. Normalize: (img/255 - mean) / std                      │
│ Output: [3, 256, 256] tensor + heatmap targets [98, 64, 64]│
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 2: Tạo Heatmap Targets                                 │
│ lib/utils/transforms.py :: generate_target()                 │
│                                                              │
│ Với mỗi landmark (x, y):                                    │
│ - Tính tọa độ trong heatmap: (x / 4) vì output = input/4  │
│ - Tạo Gaussian heatmap 64x64 centered tại (x, y)           │
│ - sigma = 2 pixels                                          │
│                                                              │
│ Heatmap = Σ Gaussian_i(x, y) cho tất cả landmarks          │
│          (hoặc tách riêng mỗi landmark 1 kênh)            │
│ Output: [NUM_LANDMARKS, 64, 64] heatmaps                   │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 3: HigherHRNet Forward                                 │
│ lib/models/hrnet.py :: HighResolutionNet.forward()           │
│                                                              │
│ Stem: conv1(3→64) → bn1 → conv2(64→64) → bn2 → relu        │
│       → layer1(Bottleneck×4, 64 channels)                   │
│                                                              │
│ Stage 2: 2 nhánh song song                                   │
│   - Nhánh 1: [B, 18, 64, 64] (1/4 resolution)             │
│   - Nhánh 2: [B, 36, 32, 32] (1/8 resolution)             │
│   → HRModule: fuse đặc trưng giữa 2 nhánh                   │
│                                                              │
│ Stage 3: 3 nhánh song song                                   │
│   - Thêm nhánh 3: [B, 72, 16, 16] (1/16 resolution)       │
│   → HRModule: fuse 3 nhánh                                  │
│                                                              │
│ Stage 4: 4 nhánh song song                                   │
│   - Thêm nhánh 4: [B, 144, 8, 8] (1/32 resolution)       │
│   → HRModule: fuse 4 nhánh                                  │
│                                                              │
│ Final:                                                       │
│ - Upsample tất cả về stage4 resolution (8×8)               │
│ - Concatenate: [B, 18+36+72+144=270, 8, 8]                 │
│ - conv1x1(270→256) + bn + relu                             │
│ - conv1x1(256→NUM_LANDMARKS) → heatmaps                    │
│ Output: [B, NUM_LANDMARKS, 64, 64] (Interpolate về 64×64) │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 4: Loss Computation (Training)                         │
│ lib/core/function.py :: train()                            │
│                                                              │
│ Loss = MSELoss(predicted_heatmaps, target_heatmaps)         │
│                                                              │
│ Backpropagation: loss.backward()                            │
│ Optimizer: Adam (lr=0.0001)                                 │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Bước 5: Decode Predictions (Inference)                      │
│ lib/core/evaluation.py :: decode_preds()                    │
│                                                              │
│ 5a. get_preds(output):                                      │
│     - Argmax: tìm vị trí có giá trị lớn nhất              │
│       trên mỗi heatmap channel                              │
│     - Tọa độ trong không gian heatmap [1, 64]             │
│                                                              │
│ 5b. Sub-pixel Refinement:                                   │
│     - Xung quanh peak, dùng gradient để tinh chỉnh        │
│     - dx = 0.25 * (H[y+1,x] - H[y-1,x])                    │
│     - dy = 0.25 * (H[y,x+1] - H[y,x-1])                    │
│     - refined_pos = peak_pos + (dx, dy)                    │
│                                                              │
│ 5c. transform_preds():                                      │
│     - Inverse affine transform từ heatmap → ảnh gốc       │
│     - Sử dụng center, scale, rotation đã crop              │
│ Output: [B, NUM_LANDMARKS, 2] tọa độ (x, y) pixel gốc    │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
Kết quả: Tọa độ (x,y) của N điểm mốc khuôn mặt
```

### 5.6. Các file config quan trọng

```yaml
# experiments/wflw/face_alignment_wflw_hrnet_w18.yaml

MODEL:
  TYPE: hrnet
  EXTRA:
    STAGE2: {NUM_BRANCHES: 2, ...}      # 2 nhánh
    STAGE3: {NUM_BRANCHES: 3, ...}      # 3 nhánh
    STAGE4: {NUM_BRANCHES: 4, ...}      # 4 nhánh
    DECONV: {NUM_DECONVS: 0}            # Không có deconv (khác HigherHRNet)

DATASET:
  DATASET: WFLW                        # Dataset: WFLW, 300W, AFLW, COFW
  NUM_JOINTS: 98                       # 98 landmarks cho WFLW
  INPUT_SIZE: [256, 256]
  OUTPUT_SIZE: [64, 64]               # Heatmap 64x64
  FLIP: True                           # Flip augmentation

TRAIN:
  BATCH_SIZE: 16
  BEGIN_EPOCH: 0
  END_EPOCH: 60
  OPTIMIZER: adam
  LR: 0.0001
  LR_STEP: [30, 50]                   # Decay LR tại epoch 30, 50
  DATASET_AUG: True
  ROT_FACTOR: 30                       # Rotation ±30 độ
  SCALE_FACTOR: 0.25                  # Scale ±25%

TEST:
  BATCH_SIZE: 8
  FLIP: True
  MODEL_FILE: ''                       # Đường dẫn pretrained model
```

---

## 6. Luồng xử lý dữ liệu chung

### 6.1. So sánh 3 dự án

```
                    HRNet-Object-Detection    HigherHRNet-Pose        HRNet-Facial-Landmark
                    ─────────────────────     ─────────────────       ─────────────────────
Input:              Ảnh RGB                   Ảnh RGB                Ảnh khuôn mặt
Backbone:           HRNetV2-W18/W32/W48       HigherHRNet            HRNetV2-W18
Output:             BBox + Class + Mask       Keypoints (17)         Landmarks (N)
Detection type:      Object Detection           Pose Estimation        Facial Landmark
Approach:           Two-stage (RPN)             Bottom-up (AE)         Heatmap Regression
Loss:               CE + SmoothL1              MSE + AE Loss          MSE
Metric:             COCO mAP                   COCO Keypoint AP       NME
Dataset:            COCO Detection              COCO Keypoints         300W, WFLW, AFLW, COFW
GPU Memory:          ~8GB (batch=2)             ~6GB (batch=16)        ~4GB (batch=16)
Training time:      ~2 ngày (8 GPU)            ~1 ngày (4 GPU)        ~4 giờ (1 GPU)
```

### 6.2. Common Pipeline Pattern

Tất cả 3 dự án đều tuân theo pattern chung:

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT PROCESSING                        │
│  Image → Resize → Normalize → Data Augmentation → Tensor  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     FEATURE EXTRACTION                      │
│              HRNet Backbone (Parallel Branches)             │
│  Stage1 → Stage2 → Stage3 → Stage4 → Feature Fusion         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   TASK-SPECIFIC HEAD                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ Detection    │ │ Pose         │ │ Landmark     │        │
│  │ Head         │ │ Head + AE    │ │ Head         │        │
│  │ (RPN+BBox)   │ │ (Heatmap)    │ │ (Heatmap)    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     POST-PROCESSING                         │
│  NMS / Grouping / Coordinate Transform                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                        OUTPUT
```

---

## 7. Ứng dụng trong Tìm kiếm và Truy xuất Thông tin

### 7.1. Image Retrieval (Tìm kiếm ảnh theo nội dung)

**Mô tả:** Cho ảnh truy vấn, tìm các ảnh tương tự trong cơ sở dữ liệu.

**Ứng dụng HRNet:**

1. **Feature Extraction:**
   - Dùng HRNet backbone để trích xuất đặc trưng đa mức phân giải
   - Các đặc trưng spatial giàu thông tin hình học

2. **Human Pose based Retrieval:**
   - Trích xuất pose signatures từ HigherHRNet
   - Tìm kiếm người có tư thế tương tự (hành động, hoạt động)
   - Ứng dụng: tìm video/ảnh cùng hành động thể thao

3. **Face Landmark based Retrieval:**
   - Trích xuất face embeddings từ landmark positions
   - Tìm kiếm khuôn mặt tương tự (biểu cảm, góc nghiêng)
   - Ứng dụng: nhận diện khuôn mặt, verification

### 7.2. Object Detection for Image Indexing

**Mô tả:** Đánh chỉ mục ảnh dựa trên nội dung đối tượng.

**Luồng:**
```
Image Database
    │
    ▼
┌──────────────────┐
│ HRNet Detector   │  ← Đánh chỉ mục tự động
└──────────────────┘
    │
    ├── "person" + "bicycle" + "road" → Image 1
    ├── "car" + "building" + "tree" → Image 2
    └── ...
    │
    ▼
Search: "person riding bicycle"
    │
    ▼
Intersection of indexes → Top results
```

### 7.3. Pose-based Action Recognition

**Mô tả:** Nhận diện hành động từ pose sequences.

**Ứng dụng HigherHRNet:**
1. Trích xuất pose từng frame bằng HigherHRNet
2. Tạo pose sequence → temporal features
3. Phân loại hành động (running, walking, sitting)

### 7.4. Facial Expression Retrieval

**Mô tả:** Tìm kiếm ảnh theo biểu cảm khuôn mặt.

**Ứng dụng HRNet-Facial-Landmark:**
1. Phát hiện facial landmarks từ 98 điểm (WFLW)
2. Tính toán Facial Action Coding System (FACS) features
3. Tìm kiếm theo emotion categories

### 7.5. Multi-modal Retrieval

**Mô tả:** Kết hợp nhiều modalities (image + pose + face) để tìm kiếm.

```
Query: "Running person"
    │
    ├── Image: person + motion blur + outdoor
    ├── Pose: arms raised + legs split → "running" signature
    └── Face: neutral expression
    │
    ▼
Combined Feature Vector → Search in Database → Top Matches
```

### 7.6. Demo đơn giản: Pose-based Image Search

```python
# Ví dụ concept - Tìm kiếm ảnh theo pose
import torch
import numpy as np

# Bước 1: Trích xuất pose từ HigherHRNet
def extract_pose_features(landmarks):
    """
    landmarks: numpy array [17, 2] - tọa độ 17 keypoints
    Trả về: feature vector mô tả pose
    """
    # Normalize theo tỷ lệ
    keypoint_distances = []
    for i in range(17):
        for j in range(i+1, 17):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            keypoint_distances.append(dist)

    # Relative angles
    angles = []
    for i in range(17):
        neighbors = get_neighbors(i)  # Keypoints liền kề
        if len(neighbors) >= 2:
            angle = compute_angle(landmarks, i, neighbors)
            angles.append(angle)

    return np.concatenate([keypoint_distances, angles])

# Bước 2: Tính similarity
def find_similar_poses(query_landmarks, db_landmarks, top_k=10):
    query_feat = extract_pose_features(query_landmarks)
    similarities = []
    for db_pose in db_landmarks:
        db_feat = extract_pose_features(db_pose)
        sim = cosine_similarity(query_feat, db_feat)
        similarities.append(sim)

    # Trả về top-k kết quả
    top_indices = np.argsort(similarities)[-top_k:]
    return top_indices
```

---

## 8. FAQ - Các lỗi thường gặp

### Q1: Lỗi "CUDA out of memory"

**Nguyên nhân:** GPU không đủ VRAM.

**Giải pháp:**
```bash
# Giảm batch size trong config
# Ví dụ: batch_size 16 → 4

# Hoặc thêm vào config:
MODEL.DEVICE = 'cuda'
TRAIN.BATCH_SIZE = 2
```

### Q2: Lỗi "No module named 'mmcv'"

**Giải pháp:**
```bash
pip install mmcv==0.5.1
# Hoặc: pip install mmcv-full
```

### Q3: Lỗi "No module named 'yacs'"

**Giải pháp:**
```bash
pip install yacs
```

### Q4: Lỗi "RuntimeError: Dataset not found"

**Giải pháp:**
- Kiểm tra đường dẫn dataset trong config
- Đảm bảo cấu trúc thư mục đúng như documentation
- Kiểm tra file annotations tồn tại

### Q5: Training chậm trên Windows

**Giải pháp:**
```bash
# Sử dụng DataLoader workers = 0 (tránh multiprocessing issues)
# Thêm vào config:
TRAIN.NUM_WORKERS = 0
TEST.NUM_WORKERS = 0
```

### Q6: Lỗi "Unable to find nvcc"

**Giải pháp:**
- Thêm CUDA vào PATH:
```powershell
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin"
```
- Hoặc sử dụng pre-built wheels thay vì compile từ source

### Q7: Lỗi "Invalid archive" khi giải nén COCO

**Giải pháp:**
- Tải lại file từ trang chủ
- Sử dụng 7-Zip hoặc WinRAR thay vì Windows built-in extractor

### Q8: Lỗi "ValueError: optimizer got an empty parameter group"

**Giải pháp:**
- Kiểm tra pretrained model path trong config
- Đảm bảo pretrained model tồn tại và đúng format

### Q9: Cài đặt Apex thất bại trên Windows

**Giải pháp:**
- Bỏ qua Apex nếu không dùng SyncBatchNorm
- Hoặc sử dụng Windows Subsystem for Linux (WSL)

### Q10: Lỗi "KeyError: 'COCO'" trong HRNet-Object-Detection

**Giải pháp:**
- Kiểm tra file annotations có đúng format COCO
- File instances_train2017.json phải tồn tại trong data/coco/annotations/

---

## Tài liệu tham khảo

1. **HRNet-Object-Detection**: https://github.com/HRNet/HRNet-Object-Detection
2. **HigherHRNet-Human-Pose-Estimation**: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
3. **HRNet-Facial-Landmark-Detection**: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
4. **HRNet Image Classification** (pretrained weights): https://github.com/HRNet/HRNet-Image-Classification
5. **mmdetection**: https://github.com/open-mmlab/mmdetection
6. **COCO Dataset**: https://cocodataset.org/

---

*Hướng dẫn này được viết cho bộ môn Tìm kiếm và Truy xuất Thông tin, Thạc sĩ Học viện Bưu chính Viễn thông. Các dự án thuộc về Microsoft Research và được phân phối theo giấy phép MIT.*
