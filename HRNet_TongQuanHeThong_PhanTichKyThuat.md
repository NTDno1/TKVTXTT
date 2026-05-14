# Tổng Quan Hệ Thống HRNet - Phân Tích Kỹ Thuật & Thuật Toán

> Tài liệu phân tích 3 repos: **HRNet-Object-Detection**, **HigherHRNet-Human-Pose-Estimation**, **HRNet-Facial-Landmark-Detection**
>
> Bản quyền: MIT License | Phát triển bởi Microsoft Research

---

## Mục Lục

1. [Kiến trúc chung - Góc nhìn tổng thể](#1-kiến-trúc-chung---góc-nhìn-tổng-thể)
2. [Nền tảng Thuật toán - High-Resolution Network (HRNet)](#2-nền-tảng-thuật-toán---high-resolution-network-hrnet)
3. [Các Thành phần Kỹ thuật Triển khai](#3-các-thành-phần-kỹ-thuật-triển-khai)
4. [HRNet-Object-Detection - Phát hiện Đối tượng](#4-hrnet-object-detection---phát-hiện-đối-tượng)
5. [HigherHRNet - Ước lượng Tư thế Con người](#5-higherhrnet---ước-lượng-tư-thế-con-người)
6. [HRNet-Facial-Landmark-Detection - Phát hiện Điểm đặc trưng Khuôn mặt](#6-hrnet-facial-landmark-detection---phát-hiện-điểm-đặc-trưng-khuôn-mặt)
7. [Danh sách các Thuật toán và Tác dụng](#7-danh-sách-các -thuật-toán-và-tác-dụng)
8. [Kết luận](#8-kết-luận)

---

## 1. Kiến trúc chung - Góc nhìn tổng thể

### 1.1 Điểm chung của cả 3 hệ thống

Cả 3 hệ thống đều chia sẻ một nguyên tắc thiết kế cơ bản: **duy trì đại diện dữ liệu ở độ phân giải cao (high-resolution representation) xuyên suốt qua các tiến trình xử lý**, thay vì lần lượt giảm phân giải như các mạng CNN truyền thống (như VGG, ResNet).

```
Điểm khác biệt lớn nhất giữa CNN truyền thống và HRNet:

CNN truyền thống ( Sequential Down-Sampling ):
  Đầu vào
    |
  [Conv 7x7 stride=2]  -->  H/2 x W/2
    |
  [MaxPool stride=2]    -->  H/4 x W/4
    |
  [Conv ... stride=2]   -->  H/8 x W/8
    |
  [Conv ... stride=2]   -->  H/16 x W/16
    |
  [Conv ... stride=2]  -->  H/32 x W/32
    |                          (mất thông tin không gian)
    v
  Up-Sampling --> Đầu ra

HRNet ( Parallel Multi-Resolution ):
  Đầu vào
    |
  [Stem: Conv 3x3 x2]   -->  H/4 x W/4, 64 kênh
    |
  +--------+--------+--------+
  |        |        |        |
  v        v        v        v
 Branch0  Branch1  Branch2  Branch3
 (1/4)    (1/8)    (1/16)   (1/32)
  HR       MR       LR       VLR
 256ch    512ch    1024ch   2048ch
  |        |        |        |
  v        v        v        v
 Fuse0 <-> Fuse1 <-> Fuse2 <-> Fuse3
  |        |        |        |
  +--------+--------+--------+
    |
  ĐẦU RA (Giữ nguyên độ phân giải cao)
```

### 1.2 Tầm quan trọng của bài toán "High-Resolution"

Các bài toán như phát hiện điểm đặc trưng khuôn mặt, ước lượng tư thế, phát hiện vật thể đều yêu cầu **vị trí chính xác theo không gian**. Nếu giảm phân giải quá sớm (quá nhiều lần stride-down), thông tin vị trí sẽ bị mất. HRNet giải quyết vấn đề này bằng cách:

- **Giữ nguyên độ phân giải cao** trong nhánh chính (branch 0)
- **Trao đổi thông tin** giữa các nhánh có độ phân giải khác nhau qua các lớp fusion

### 1.3 Hệ thống Pipeline tổng quát

```
Hình ảnh đầu vào
    |
    v
[STEM] - 2 lần Conv 3x3, stride=2 --> Giảm kích thước 4 lần
    |
    v
[Layer 1] - Bottleneck blocks (H/4, W/4, 256 kênh)
    |
    v
[Stage 2] - 2 nhánh (1/4, 1/8)  --> Multi-scale representation bắt đầu
    |
    v
[Stage 3] - 3 nhánh (1/4, 1/8, 1/16)
    |
    v
[Stage 4] - 4 nhánh (1/4, 1/8, 1/16, 1/32)
    |
    v
[HEAD] - Lớp đầu ra cho từng bài toán cụ thể
    |
    v
Kết quả đầu ra
```

---

## 2. Nền tảng Thuật toán - High-Resolution Network (HRNet)

### 2.1 Cấu trúc chi tiết của HRNet

HRNet gồm 4 giai đoạn (stages), mỗi giai đoạn có nhiều nhánh song song và các lớp fusion trao đổi thông tin.

**Tham số cấu hình (với W32 - width=32):**

| Stage | Số nhánh | Độ phân giải | Kênh | Blocks |
|-------|-----------|-------------|-------|--------|
| 1     | 1         | H/4 x W/4   | 64    | 4 Bottleneck |
| 2     | 2         | H/4, H/8    | 32, 64 | 1 module |
| 3     | 3         | H/4, H/8, H/16 | 32, 64, 128 | 4 modules |
| 4     | 4         | H/4, H/8, H/16, H/32 | 32, 64, 128, 256 | 3 modules |

### 2.2 HighResolutionModule - Đơn vị xử lý chính

Đây là thành phần trung tâm nhất của HRNet. Mỗi module gồm:

- **Branches**: các nhánh xử lý song song, mỗi nhánh chứa nhiều BasicBlock hoặc Bottleneck
- **Fuse Layers**: trao đổi thông tin giữa các nhánh

**Logic của Fuse Layer (Exchange Unit):**

```python
# Pseudocode của fuse layer
for i in range(num_branches):  # mỗi nhánh i là đầu ra
    y = x[0] if i == 0 else fuse_layers[i][0](x[0])
    for j in range(1, num_branches):
        if i == j:
            # cùng độ phân giải --> cộng trực tiếp
            y = y + x[j]
        elif j > i:
            # nhánh j có độ phân giải cao hơn --> upsampling
            y = y + upsample(fuse_layers[i][j](x[j]))
        else:
            # nhánh j có độ phân giải thấp hơn --> downsampling (stride conv)
            y = y + downsample(fuse_layers[i][j](x[j]))
    x_fuse.append(relu(y))
```

**Điều này có nghĩa là:** Thông tin từ bất kỳ độ phân giải nào đều có thể được chuyển đổi để làm giàu các độ phân giải khác, đảm bảo tất cả các nhánh đều có quyền truy cập thông tin từ đầy đủ các mức độ phân giải.

### 2.3 Transition Layer - Chuyển đổi giữa các Stage

Khi chuyển từ stage này sang stage khác, số lượng branches tăng dần (từ 1 --> 2 --> 3 --> 4). Transition layer đảm bảo việc chuyển đổi này:

- Nếu kênh khác nhau: sử dụng Conv 3x3 để chuyển đổi
- Nếu cần tạo nhánh mới: sử dụng Conv 3x3 stride=2 để giảm phân giải

---

## 3. Các Thành phần Kỹ thuật Triển khai

### 3.1 Residual Block (BasicBlock & Bottleneck)

**BasicBlock (expansion=1):**
- 2 lớp Conv 3x3
- Skip connection
- Sử dụng khi số kênh nhỏ

**Bottleneck (expansion=4):**
```
Đầu vào (in_channels)
  |
[Conv 1x1, reduce] --> (in_channels/4)
  |
[Conv 3x3] --> (in_channels/4)
  |
[Conv 1x1, expand] --> (in_channels)
  |
+--> [Shortcut] (nếu in != out hoặc stride != 1)
  |
ReLU
```

### 3.2 Batch Normalization

- **BN_MOMENTUM = 0.1** (Object Detection & Human Pose Estimation)
- **BN_MOMENTUM = 0.01** (Facial Landmark Detection - nhanh hơn, ổn định hơn)

### 3.3 Multi-Scale Training

**Kỹ thuật:** Chọn ngẫu nhiên một tỷ lệ scale từ tập `[1600x1000, 1000x600, 1333x800]`, giúp model học được đặc trưng ở nhiều kích thước vật thể khác nhau.

### 3.4 Synchronized BatchNorm (SyncBN)

- Tính toán BatchNorm trên **nhiều GPU cùng lúc**, giải quyết vấn đề batch nhỏ trên mỗi GPU
- Đặc biệt quan trọng khi training trên nhiều node

### 3.5 Mixed-Precision Training (FP16)

- Sử dụng NVIDIA Apex API
- Tính toán Forward/Backward ở FP16, weight update ở FP32
- Giảm bộ nhớ GPU, tăng tốc độ training

---

## 4. HRNet-Object-Detection - Phát hiện Đối tượng

### 4.1 Kiến trúc tổng thể

```
Hình ảnh
  |
  v
[HRNet Backbone] --> Feature maps nhiều mức độ phân giải
  |                     (1/4, 1/8, 1/16, 1/32)
  v
[HRFPN Neck] --> Multi-level feature pyramids (5 mức)
  |
  v
[Detector Head] --> Faster R-CNN / Mask R-CNN / Cascade R-CNN / FCOS
  |
  v
Kết quả: Bounding boxes + Class labels (+ Masks nếu có)
```

### 4.2 HRFPN (HRNet Feature Pyramid Network)

**Vai trò:** Chuyển đổi các feature maps đa mức độ phân giải từ HRNet thành Feature Pyramid phục vụ phát hiện vật thể.

**Cơ chế hoạt động:**

```
HRNet Output:
  Branch0 (1/4 scale)   --> [C0]  --> upsample x2 --> [C0']
  Branch1 (1/8 scale)   --> [C1]  --> upsample x4 --> [C1']
  Branch2 (1/16 scale)  --> [C2]  --> upsample x8 --> [C2']
  Branch3 (1/32 scale)  --> [C3]  --> upsample x16 --> [C3']

  Concatenate: [C0', C1', C2', C3'] --> [sum_channels]
    |
    v
  Reduction Conv 1x1 --> out_channels (256)
    |
    v
  Multi-level outputs:
    P5 = out (không pooling)
    P4 = avg_pool(out, 2x2)  stride=2
    P3 = avg_pool(out, 4x4)  stride=4
    P2 = avg_pool(out, 8x8)  stride=8
    P1 = avg_pool(out, 16x16) stride=16
    |
    v
  [P1, P2, P3, P4, P5] --> 5 Feature Pyramid Levels
```

### 4.3 Các Framework Phát hiện Đối tượng

#### 4.3.1 Faster R-CNN (Two-Stage Detector)

**Giai đoạn 1 - RPN (Region Proposal Network):**
- Trượt trên feature pyramid để tạo anchor boxes
- Phân loại: có vật thể / không có vật thể
- Regression: tính chỉnh bounding box

**Giai đoạn 2 - RCNN (Region CNN):**
- RoI Align: trích xuất features từ proposal regions
- FC layers: phân loại + regression bounding box cuối cùng

#### 4.3.2 Mask R-CNN

Giới thiệu thêm **Mask Head** ngoài bounding box detection:
- Trích xuất features từ mỗi proposal
- Phân loại pixel-by-pixel (segmentation)
- Cho phép phát hiện vật thể + tạo mask phân segment

#### 4.3.3 Cascade R-CNN

**Vấn đề:** Một giai đoạn R-CNN không tối ưu cho mọi IoU threshold

**Giải pháp:** Chuẩn bị nhiều giai đoạn liên tiếp, mỗi giai đoạn nhắm với threshold cao hơn:

```
Stage 1: IoU threshold = 0.5  --> refine --> proposals mới
Stage 2: IoU threshold = 0.6  --> refine --> proposals mới
Stage 3: IoU threshold = 0.7  --> refine --> proposals mới
Kết quả: Chuẩn bị chính xác hơn, loại bỏ false positives
```

**Cascade Mask R-CNN:** Thêm mask head cho từng stage.

#### 4.3.4 FCOS (Anchor-Free)

**Khác biệt:** Không sử dụng anchor boxes. Thay vào đó, mỗi pixel là một center point, và model đoán:
- Khoảng cách từ center đến 4 cạnh bounding box
- Class của vật thể

**Ưu điểm:** Đơn giản hơn, tránh phụ thuộc vào anchor design.

#### 4.3.5 Hybrid Task Cascade (HTC)

Kết hợp giữa Cascade R-CNN và Mask R-CNN nhưng có thêm **inter-channel feature fusion** giữa các stage, cho phép thông tin mask và bbox được trao đổi nhiều hơn.

### 4.4 Kỹ thuật Train/Test

**Multi-scale Training:**
- Default: Padding 1600x1024, chọn scale ngẫu nhiên từ 3 mức
- ResizeCrop: Hiệu quả hơn về bộ nhớ, crop ngẫu nhiên trước khi resize

**Inference:**
- Test time augmentation: Horizontal flip
- Multi-scale testing
- Merge predictions từ nhiều stages

---

## 5. HigherHRNet - Ước lượng Tư thế Con người

### 5.1 Giới thiệu Bài toán

**Bottom-up Human Pose Estimation:** Phát hiện tất cả các keypoints (joints) của tất cả người trong ảnh mà không cần biết trước số lượng người.

**Thái độ:** Top-down (dùng detector trước rồi ước lượng tư thế từng người) | **Bottom-up** (tìm keypoints trước rồi nhóm lại thành từng người)

### 5.2 Kiến trúc của HigherHRNet

```
Hình ảnh đầu vào
    |
    v
[STEM] 2x Conv 3x3, stride=2 --> H/4, W/4
    |
    v
[Layer 1] 4 Bottleneck blocks --> 256 kênh
    |
    v
[Stage 2] 2 nhánh (HR + MR)   [Transition1]
    |
    v
[Stage 3] 3 nhánh (HR + MR + LR) [Transition2]
    |
    v
[Stage 4] 4 nhánh (HR + MR + LR + VLR) [Transition3]
    |           multi_scale_output=False (chỉ đầu ra HR branch)
    v
[Final Layers] --> Heatmaps + Tags
    |
    v
[Deconv Layers] --> Transposed Convolution
    |         Tăng độ phân giải lên gấp 4 (1/4 input size)
    v
[Final Layers 2] --> Higher-resolution Heatmaps + Tags
    |
    v
[Grouping] --> Kết nối keypoints thành từng người
    |
    v
Kết quả: Vị trí 17 keypoints cho nhiều người
```

### 5.3 Multi-Resolution Supervision

**Vấn đề:** Bottom-up phải phát hiện nhiều người với kích thước rất khác nhau (nhỏ, trung bình, lớn).

**Giải pháp:** Huấn luyện ở **nhiều độ phân giải khác nhau**:
- Heatmap ở mức độ phân giải đầu ra
- Thay vì chỉ train ở 1 resolution, train ở nhiều resolution

### 5.4 Deconvolution Layers (Transposed Conv)

Dùng để tăng độ phân giải của heatmap đầu ra:

```
Đầu vào (1/32 scale) --> Transposed Conv 4x4 stride=4 --> 1/8 scale
                  --> Transposed Conv 4x4 stride=2 --> 1/4 scale (cùng hiện tại)

Kết hợp (Concatenation):
  x = concat(feature_1/4, deconv_output_1/4)
```

### 5.5 Loss Function

**Heatmap Loss (MSE):**
```
L_heatmap = MSE(predicted_heatmap, ground_truth_heatmap) * mask
```
- Heatmap là bản đồ xác suất, giá trị cao nhất tại vị trí keypoint
- MSE lỗi tương tác giữa giá trị dự đoán và giá trị thật (Gaussian centered)

**Associative Embedding Loss (AE):**
```
L_push = Khuyến khích tag khác nhau cho người khác nhau
L_pull = Kéo tag cùng người gần nhau
L_AE = L_push + L_pull
```
- Mỗi keypoint được gán một "tag vector"
- Keypoints cùng một người có tag giống nhau
- Keypoints khác người có tag khác nhau
- Như vậy, chỉ cần group keypoints theo tag là biết người nào

### 5.6 Multi-Stage Output

HigherHRNet xuất ra nhiều heatmap outputs ở các độ phân giải khác nhau:
- Heatmap 1: từ HRNet branch chính (1/4 scale)
- Heatmap 2: từ deconv layer (1/4 scale, chi tiết hơn)

Các heatmaps được average lại để tăng độ chính xác.

### 5.7 Grouping Keypoints

Sau khi có heatmap + tag, cần nhóm các keypoints thành từng người:
1. Tìm các peak trong heatmap (vị trí keypoint)
2. Lấy tag vector tại mỗi peak
3. So sánh tag vectors: giá trị gần nhau --> cùng một người

---

## 6. HRNet-Facial-Landmark-Detection - Phát hiện Điểm Đặc trưng Khuôn mặt

### 6.1 Bài toán

Cho một ảnh khuôn mặt, xác định vị trí của N điểm đặc trưng (landmarks), ví dụ:
- **68 điểm** (dataset 300W, WFLW)
- **98 điểm** (dataset WFLW mở rộng)
- **19 điểm** (dataset AFLW)
- **29 điểm** (dataset COFW)

### 6.2 Kiến trúc

```
Hình ảnh
    |
    v
[STEM] 2x Conv 3x3, stride=2
    |
    v
[Layer 1] 4 Bottleneck --> 256 kênh
    |
    v
[Stage 2] 2 nhánh
    |
    v
[Stage 3] 3 nhánh
    |
    v
[Stage 4] 4 nhánh (multi_scale_output=True)
    |         [Khác với Pose: duy trì tất cả 4 nhánh]
    v
[Concatenation Layer]
  Lấy tất cả 4 nhánh, resize về cùng độ phân giải:
    x[0]           --> size gốc
    interpolate(x[1]) --> cùng size
    interpolate(x[2]) --> cùng size
    interpolate(x[3]) --> cùng size
  Concat: [x0, x1', x2', x3'] --> tổng kênh = 480
    |
    v
[Head]
  Conv 1x1, 480 --> 480
  BN + ReLU
  Conv 1x1, 480 --> NUM_JOINTS (số landmarks)
    |
    v
Output: Heatmap Nx64x64 (N = số điểm landmarks)
```

### 6.3 Đặc điểm khác với HigherHRNet

| Đặc điểm | HRNet Object Detection | HigherHRNet | HRNet Facial Landmark |
|-----------|----------------------|-------------|----------------------|
| Multi-scale output | Không | Có (deconv) | Không (chỉ concat) |
| Output scale | 1/32 | 1/4 hoặc 1/2 | 1/4 |
| Fusion method | HRFPN | Multi-stage deconv | Full concat |
| Loss | Classification/Regression | Heatmap + AE Loss | MSE Heatmap |
| BN Momentum | 0.1 | 0.1 | 0.01 |

### 6.4 Loss Function

**JointsMSELoss:**
```python
for each joint:
    loss_joint = MSE(pred_heatmap[joint], gt_heatmap[joint])
L_total = sum(loss_joint) / num_joints
```

**Có thể sử dụng target_weight:** cho phép giảm trọng số của một số landmarks bị che khuất.

### 6.5 Decode Predictions

```python
def get_preds(scores):
    # Tìm vị trí có giá trị cao nhất trong heatmap
    maxval, idx = torch.max(scores.view(-1), dim=-1)
    preds = idx_to_coords(idx, image_size)
    return preds

def decode_preds(output, center, scale, res):
    coords = get_preds(output)
    # Hiệu chỉnh sub-pixel (lấy giá trị trung bình weighted)
    for pixel quan trọng:
        coords += gradient_based_adjustment * 0.25
    # Chuyển đổi tọa độ từ heatmap về tọa độ gốc
    coords = transform_preds(coords, center, scale, res)
    return coords
```

### 6.6 Evaluation Metric - NME

**Normalized Mean Error (NME):**
```
NME = (1/N) * sum(||pred_i - gt_i||_2 / interocular_distance)
```

- `pred_i`: vị trí điểm landmarks thứ i
- `gt_i`: vị trí thật của điểm thứ i
- `interocular_distance`: khoảng cách giữa 2 mắt (dùng làm chuẩn hóa)

---

## 7. Danh sách các Thuật toán và Tác dụng

### 7.1 Thuật toán Mạng (Network Architecture)

| Thuật toán | Tác dụng | Sử dụng trong |
|------------|-----------|---------------|
| **HRNet (High-Resolution Network)** | Duy trì đại diện high-resolution xuyên suốt, trao đổi thông tin đa mức độ phân giải | Tất cả 3 repos |
| **BasicBlock** | Residual block đơn giản, 2 conv 3x3, nhanh hơn Bottleneck | Tất cả 3 repos |
| **Bottleneck Block** | Residual block với 1x1 --> 3x3 --> 1x1 conv, giảm số lượng parameters | Tất cả 3 repos |
| **Multi-branch Parallel Processing** | Xử lý song song ở nhiều độ phân giải | HRNet backbone |
| **Feature Fusion (Exchange Unit)** | Trao đổi thông tin giữa các nhánh, upsampling/downsample để cân chỉnh | HRNet Module |

### 7.2 Thuật toán Phát hiện Đối tượng (Object Detection)

| Thuật toán | Tác dụng | Chi tiết |
|------------|-----------|----------|
| **Faster R-CNN** | Phát hiện vật thể 2 giai đoạn | RPN + RoI Align + Classification |
| **Mask R-CNN** | Phát hiện vật thể + segmentation mask | Faster R-CNN + Mask Head |
| **Cascade R-CNN** | Tăng độ chính xác bounding box | Nhiều giai đoạn R-CNN liên tiếp với IoU tiến dần |
| **Cascade Mask R-CNN** | Phát hiện + mask chính xác hơn | Cascade + Mask nhiều stage |
| **FCOS** | Phát hiện anchor-free | Center-based, tránh thiết kế anchor |
| **Hybrid Task Cascade (HTC)** | Kết hợp bbox và mask refinement | Inter-stage feature fusion |
| **HRFPN** | Build multi-level feature pyramid từ HRNet | Upsample + Concat + Reduction |

### 7.3 Thuật toán Ước lượng Tư thế (Human Pose Estimation)

| Thuật toán | Tác dụng | Chi tiết |
|------------|-----------|----------|
| **Bottom-up Pose Estimation** | Tìm tất cả keypoints trước, group sau | Không cần person detector |
| **Associative Embedding (AE)** | Grouping keypoints | Tag vectors cho mỗi keypoint |
| **Multi-Resolution Supervision** | Huấn luyện ở nhiều độ phân giải | Giải quyết scale variation |
| **Transposed Convolution** | Tăng độ phân giải heatmap | Upsample feature map |
| **Heatmap Regression** | Dự đoán vị trí keypoint | Gaussian heatmap target |
| **Multi-Stage Heatmap** | Average nhiều heatmap | Tăng độ chính xác |

### 7.4 Thuật toán Phát hiện Điểm Đặc trưng Khuôn mặt

| Thuật toán | Tác dụng | Chi tiết |
|------------|-----------|----------|
| **Heatmap Regression** | Dự đoán heatmap cho mỗi landmark | Gaussian target centered tại landmark |
| **Coordinate Regression** | Dự đoán tọa độ trực tiếp | (Chỉ sử dụng trong một số biến thể) |
| **Sub-pixel Refinement** | Tính chỉnh vị trí pixel-level | Gradient-based adjustment |
| **Coordinate Transformation** | Chuyển từ heatmap coords sang image coords | Scale + Center transformation |

### 7.5 Thuật toán Huấn luyện (Training Techniques)

| Thuật toán | Tác dụng | Chi tiết |
|------------|-----------|----------|
| **SyncBatchNorm** | Tính BN đồng bộ trên nhiều GPU | Giải quyết small batch problem |
| **Multi-Scale Training** | Huấn luyện với nhiều input scales | Random scale selection |
| **Mixed Precision (FP16)** | Training nhanh hơn, ít bộ nhớ hơn | NVIDIA Apex |
| **Transfer Learning** | Pretrained trên ImageNet | Khởi tạo weights từ model lớn |
| **Horizontal Flip Augmentation** | Tăng dữ liệu bằng flip ảnh | Lật ngang đối xứng trái/phải |

### 7.6 Thuật toán Inference

| Thuật toán | Tác dụng |
|-------------|----------|
| **RoI Align** | Trích xuất features từ Region of Interest chính xác |
| **NMS (Non-Maximum Suppression)** | Loại bỏ overlapping boxes |
| **Multi-Scale Testing** | Test nhiều scales rồi merge |
| **Test-Time Augmentation** | Flip + Average predictions |

---

## 8. Kết luận

### 8.1 Tầm nhìn Chung

```
              HRNet Backbone (Nền tảng chung)
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
  Object Detection  Pose Estimation  Facial Landmarks
        |               |               |
        v               v               v
  Faster R-CNN    HigherHRNet      HRNetV2 Head
  Mask R-CNN      AE Loss          Heatmap Loss
  Cascade R-CNN   Multi-res        NME Metric
  FCOS            Deconv Upsample  Sub-pixel Refine
```

### 8.2 Điểm mạnh của HRNet

1. **Đại diện High-Resolution**: Giữ nguyên thông tin không gian chính xác
2. **Multi-scale fusion**: Trao đổi thông tin nhiều độ phân giải
3. **Tính năng mạng**: Có thể làm nền tảng cho nhiều bài toán khác nhau
4. **Có thể mở rộng**: Dễ thích nghi với các bài toán mới

### 8.3 So sánh Hiệu năng

| Repo | Task | Metric | Kết quả |
|------|------|--------|---------|
| Object Detection | COCO mAP | Box mAP | 47.0 (HTC) |
| HigherHRNet | COCO Keypoint | AP | 70.5% |
| Facial Landmark | WFLW NME | NME | 4.60 |
| Facial Landmark | 300W NME | NME | 3.34 |

### 8.4 Đường dẫn Repositories

| Repository | URL | Stars |
|------------|-----|-------|
| HRNet-Object-Detection | github.com/HRNet/HRNet-Object-Detection | 666 |
| HigherHRNet-Human-Pose-Estimation | github.com/HRNet/HigherHRNet-Human-Pose-Estimation | 1.5k |
| HRNet-Facial-Landmark-Detection | github.com/HRNet/HRNet-Facial-Landmark-Detection | 1.1k |

### 8.5 Luật trì và Chính sách

Tất cả 3 repositories đều sử dụng **MIT License** - cho phép sử dụng, sửa đổi, phân phối với điều kiện:
- Phi hạn chế hành vi thương mại
- Yêu cầu ghi nhận nguồn gốc và bản quyền

---

*Tài liệu được tạo từ phân tích source code của 3 repositories HRNet. Thông tin chi tiết xem tại source code gốc.*
