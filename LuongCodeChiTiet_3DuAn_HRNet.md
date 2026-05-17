# Luồn Code Chi Tiết - 3 Dự Án HRNet
## Bộ môn: Tìm kiếm và Truy xuất Thông tin - Thạc sĩ Học viện Bưu chính Viễn thông

---

## Mục lục

1. [Luồn Code HRNet-Object-Detection](#1-luồn-code-hrnet-object-detection)
2. [Luồn Code HigherHRNet-Human-Pose-Estimation](#2-luồn-code-higherhrnet-human-pose-estimation)
3. [Luồn Code HRNet-Facial-Landmark-Detection](#3-luồn-code-hrnet-facial-landmark-detection)
4. [So sánh luồn code 3 dự án](#4-so-sánh-luồn-code-3-dự-án)

---

## 1. Luồn Code HRNet-Object-Detection

### 1.1. Entry Point - Training (`tools/train.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ tools/train.py                                              │
│  1. Load config: cfg = Config.fromfile(args.config)        │
│  2. Build detector: model = build_detector(cfg.model)      │
│  3. Build dataset: dataset = get_dataset(cfg.data.train)   │
│  4. Call train_detector(model, dataset, cfg)               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2. Entry Point - Testing (`tools/test.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ tools/test.py                                               │
│  1. Load config + Build model + Load checkpoint            │
│  2. Build dataloader (COCO val2017)                        │
│  3. For each batch: model(return_loss=False, **data)      │
│  4. Save results to .pkl file                             │
│  5. Evaluate: coco_eval(results, types=['bbox'])           │
└─────────────────────────────────────────────────────────────┘
```

### 1.3. Luồn Training Chi Tiết

```
Bước 1: DATA LOADING
─────────────────────────────────────────────────────────────
File: mmdet/datasets/coco.py, mmdet/datasets/custom.py

Class: CocoDataset (hoặc CocoZipDataset)

    __getitem__(idx):
        1. Load image:
           img_bytes = zip_file.read(path) if zip else cv2.imread()
           image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        2. Get annotation:
           ann = self.coco.imgToAnns[img_id]
           gt_bboxes = [ann['bbox'] for ann in ann]
           gt_labels = [ann['category_id'] for ann in ann]
        3. Apply transforms (transforms.py):
           image, bboxes, labels = img_transform(image, bboxes, labels)
           - Random flip (p=0.5)
           - Random scale (0.8 ~ 1.25)
           - Pad to size_divisor=32
        4. Normalize:
           image = (image - mean) / std
           mean = [123.675, 116.28, 103.53]
           std = [58.395, 57.12, 57.375]
        5. Convert to tensor (CHW format)
        Returns: img_tensor [3,H,W], gt_bboxes, gt_labels
```

```
Bước 2: BACKBONE (HRNet)
─────────────────────────────────────────────────────────────
File: mmdet/models/backbones/hrnet.py

Class: HighResolutionNet

    forward(x):
        1. STEM:
           x = self.conv1(x)   # 3 → 64, stride 2
           x = self.bn1(x)
           x = self.conv2(x)   # 64 → 64, stride 2
           x = self.bn2(x)
           x = self.relu(x)
           Output: [B, 64, H/4, W/4]

        2. LAYER 1:
           x = self.layer1(x)   # 4 Bottleneck blocks
           Output: [B, 256, H/4, W/4]

        3. TRANSITION 1:
           x_list = [x]
           x2 = self.transition1[0](x)  # → Nhánh 1 (1/4)
           x_list.append(x2)             # → Nhánh 2 (1/8)
           Output: [Nhánh 1: B,64,H/4,W/4], [Nhánh 2: B,36,H/8,W/8]

        4. STAGE 2:
           y_list = self.stage2(x_list)
           - Mỗi nhánh qua num_blocks BasicBlocks
           - HRModule fuse: upsampling + downsampling + conv
           Output: 2 feature maps đa mức phân giải

        5. TRANSITION 2:
           x_list = [y_list[0], y_list[1]]
           x3 = self.transition2[0](y_list[1])  # → Nhánh 3 (1/16)
           x_list.append(x3)
           Output: 3 nhánh [1/4, 1/8, 1/16]

        6. STAGE 3:
           y_list = self.stage3(x_list)
           - 3 nhánh với HRModule
           - 4 modules (NUM_MODULES=4 trong config)
           Output: 3 feature maps

        7. TRANSITION 3:
           x_list = [y_list[0], y_list[1], y_list[2]]
           x4 = self.transition3[0](y_list[2])  # → Nhánh 4 (1/32)
           x_list.append(x4)
           Output: 4 nhánh [1/4, 1/8, 1/16, 1/32]

        8. STAGE 4:
           y_list = self.stage4(x_list)
           - 4 nhánh với HRModule
           - 3 modules (NUM_MODULES=3 trong config)
           Output: 4 feature maps [B,18], [B,36], [B,72], [B,144]

        9. FUSE:
           y_list = self.fuse_layer(y_list)
           - Upsample smaller resolutions to largest
           - Concatenate along channel dim
           Output: [B, 18+36+72+144 = 270, H/4, W/4]

        Returns: [C1, C2, C3, C4] feature maps (trước fuse)
                 hoặc fused feature map (sau fuse, tùy config)
```

**Chi tiết HRModule (HighResolutionModule):**

```
HighResolutionModule.forward(branches):
    1. For each branch i:
       branches[i] = self.branch[i](branches[i])  # Residual blocks
    2. Fuse:
       new_branches = []
       for i in range(num_branches):
           prev = branches[i]
           # Merge all other branches into branch i
           for j in range(num_branches):
               if j != i:
                   if j < i:  # Downsample
                       prev += downsample(branches[j], target_size)
                   else:      # Upsample
                       prev += upsample(branches[j], target_size)
           new_branches.append(self.conv1x1(prev))
    3. Activation: relu
    Returns: new_branches
```

```
Bước 3: HRFPN NECK
─────────────────────────────────────────────────────────────
File: mmdet/models/necks/hrfpn.py

Class: HRFPN

    forward(features):
        Input: features = [C2, C3, C4, C4] (từ backbone stage 4)
               channels = [18, 36, 72, 144]

        1. For each level:
           if i == 0:  out = conv(concat(upsample(C4) + upsample(C3) + C2))
                      Giải thích: Upsample tất cả về 1/4,
                      concatenate, 1x1 conv giảm kênh
           else:       out = maxpool(previous_out) với stride 2
        2. Output: 5 levels [P0, P1, P2, P3, P4]
                   Mỗi level có out_channels=256
                   P0 = 1/4 resolution, P4 = 1/64 resolution

        Cơ chế: Tạo Feature Pyramid Network đa mức phân giải,
                mỗi level có đặc trưng từ tất cả các nhánh HRNet
```

```
Bước 4: RPN HEAD
─────────────────────────────────────────────────────────────
File: mmdet/models/anchor_heads/rpn_head.py

Class: RPNHead

    forward_single(x):
        Input: x = P0 hoặc P1,...P4 feature map

        x = self.rpn_conv(x)   # 3x3 conv, 256 channels, relu
        rpn_cls_score = self.rpn_cls(x)   # 1x1 conv → num_anchors
        rpn_bbox_pred = self.rpn_reg(x)    # 1x1 conv → num_anchors×4

        Returns: rpn_cls_score, rpn_bbox_pred

    loss():
        1. Tạo anchors:
           anchors = generate_anchors(feature_map, stride, scales, ratios)
           Tổng ~240K anchors cho 5 levels

        2. Anchor Target Assignment:
           assign_result = MaxIoUAssigner.assign()
           - Positive: IoU > 0.7 với gt_bbox
           - Negative: IoU < 0.3
           - Ignore: 0.3 <= IoU <= 0.7

        3. Sampling:
           sampling_result = RandomSampler.sample()
           - 256 samples, 50% positive, 50% negative

        4. Compute losses:
           loss_rpn_cls = CrossEntropyLoss(rpn_cls_score, pos_neg_labels)
           loss_rpn_reg = SmoothL1Loss(rpn_bbox_pred, bbox_targets)

        Returns: loss_rpn_cls + loss_rpn_reg

    get_bboxes_single(rois, scores, deltas):
        1. Decode bboxes:
           bboxes = delta2bbox(anchors, deltas)
           # bboxes = [x_ctr, y_ctr, w, h] → [x1, y1, x2, y2]

        2. Clip to image:
           bboxes[:, ::2] = clamp(bboxes[:, ::2], 0, W)
           bboxes[:, 1::2] = clamp(bboxes[:, 1::2], 0, H)

        3. NMS:
           keep = nms(bboxes, scores, threshold=0.7)

        Returns: bboxes[keep], scores[keep] (top ~2000)
```

```
Bước 5: ROI ALIGN
─────────────────────────────────────────────────────────────
File: mmdet/models/roi_extractors/single_level.py

Class: SingleRoIExtractor

    forward(rois, feat_maps):
        Input:  - rois: [N, 5] (batch_idx, x1, y1, x2, y2)
                - feat_maps: [P0, P1, P2, P3, P4] (mỗi level 256 kênh)

        1. Tính scale level cho mỗi ROI:
           level = log2(sqrt(w * h) / 224) + 1
           → ROI nhỏ → level thấp (P0, P1)
           → ROI lớn → level cao (P3, P4)

        2. Map ROI về feature level:
           feat_idx = clamp(level, 0, num_levels-1)

        3. RoIAlign:
           roi_feat = RoIAlign(output_size=7, sample_num=2)
           - Chia ROI thành 7×7 grid
           - Bilinear interpolation để lấy giá trị
           - Average pooling

        Returns: [N, 256, 7, 7] ROI features
```

```
Bước 6: BBOX HEAD
─────────────────────────────────────────────────────────────
File: mmdet/models/bbox_heads/convfc_bbox_head.py

Class: SharedFCBBoxHead

    forward(self, x):
        x = x.flatten(start_dim=1)  # [N, 256*7*7] → [N, 12544]
        x = self.fc1(x)            # 12544 → 1024
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)            # 1024 → 1024
        x = self.relu2(x)

        cls_score = self.fc_cls(x)    # 1024 → num_classes+1
        bbox_pred = self.fc_reg(x)     # 1024 → num_classes*4 (hoặc 4*4)

        Returns: cls_score, bbox_pred

    loss(cls_score, bbox_pred, labels, bbox_targets):
        1. Classification loss:
           loss_cls = CrossEntropyLoss(cls_score, labels)

        2. BBox regression loss:
           - Chỉ tính cho positive samples
           - L = SmoothL1Loss(pred_delta, target_delta)
           - Normalize theo số positive samples

        Returns: loss_cls + loss_bbox
```

```
Bước 7: MASK HEAD (Mask R-CNN, Cascade Mask R-CNN)
─────────────────────────────────────────────────────────────
File: mmdet/models/mask_heads/fcn_mask_head.py

Class: FCNMaskHead

    forward(self, x):
        1. x = self.conv1(x)  # 256 → 256
        2. x = self.relu1(x)
        3. ... (4 conv layers total)
        4. x = self.deconv(x)  # Upsample 2x → 14×14
        5. mask_pred = self.fc(x)  # → num_classes channels
        6. mask_pred = self.sigmoid(mask_pred)

        Returns: [N, num_classes, 28, 28] mask predictions

    loss(mask_pred, mask_targets):
        1. Chỉ tính loss cho positive ROIs
        2. Sample mask targets theo fg_thr
        3. loss_mask = BinaryCrossEntropy(pos_mask_pred, pos_mask_targets)
```

### 1.4. Luồn Inference Chi Tiết

```
INFERENCE FLOW
─────────────────────────────────────────────────────────────

1. Preprocessing:
   image = cv2.imread(img_path)
   image, scale = resize(image, target_size=(1333, 800))
   mean = [123.675, 116.28, 103.53]
   std = [58.395, 57.12, 57.375]
   image = normalize(image, mean, std)
   image = pad_to_multiple_of_32(image)
   image_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)

2. Feature Extraction:
   features = backbone(image_tensor)  # HRNet → [C1,C2,C3,C4]
   multi_level_features = neck(features)  # HRFPN → [P0,P1,P2,P3,P4]

3. RPN Forward:
   rpn_cls_scores, rpn_bbox_preds = rpn_head(multi_level_features)
   proposals = rpn_head.get_bboxes(rpn_cls_scores, rpn_bbox_preds)
   # ~2000 proposals sau NMS

4. BBox Detection:
   roi_features = roi_extractor(proposals, multi_level_features)
   cls_scores, bbox_deltas = bbox_head(roi_features)
   det_bboxes, det_labels = bbox_head.get_bboxes(cls_scores, bbox_deltas)
   # ~100 detections sau per-class NMS

5. Mask Prediction (Mask R-CNN):
   if has_mask_head:
       mask_rois = det_bboxes[:, :4]  # Lấy tọa độ
       mask_roi_features = roi_extractor(mask_rois, multi_level_features, mask_size=14)
       mask_preds = mask_head(mask_roi_features)
       masks = mask_preds[range(len(det_labels)), det_labels]  # Pick class mask
       masks = (masks > 0.5).float()

6. Output:
   results = [{
       'bbox': [x1,y1,x2,y2,score,label],
       'mask': mask (optional)
   }]
```

### 1.5. Chi tiết các hàm quan trọng

```
Hàm: bbox2roi (mmdet/core/bbox/transforms.py)
─────────────────────────────────────────────
Chuyển từ list of bounding boxes → ROI format

Input: bbox_list = [tensor[N,4], tensor[M,4], ...]
Output: rois = tensor[K,5]
        rois[:, 0] = batch_idx (0, 1, 2, ...)
        rois[:, 1:5] = [x1, y1, x2, y2]

─────────────────────────────────────────────

Hàm: delta2bbox (mmdet/core/bbox/transforms.py)
─────────────────────────────────────────────
Decode anchor-based bbox deltas về tọa độ thực

Input:  anchors[N,4], deltas[N,4], means[4], stds[4]
Output: bboxes[N,4]

Formula:
  dx = delta[:, 0] * std[0] + mean[0]
  dy = delta[:, 1] * std[1] + mean[1]
  dw = delta[:, 2] * std[2] + mean[2]
  dh = delta[:, 3] * std[3] + mean[3]

  x_ctr = anchors[:, 0] + anchors[:, 2] * dx
  y_ctr = anchors[:, 1] + anchors[:, 3] * dy
  w = anchors[:, 2] * exp(dw)
  h = anchors[:, 3] * exp(dh)

  x1 = x_ctr - w/2, y1 = y_ctr - h/2
  x2 = x_ctr + w/2, y2 = y_ctr + h/2

─────────────────────────────────────────────

Hàm: MaxIoUAssigner (mmdet/core/bbox/assigners/max_iou_assigner.py)
─────────────────────────────────────────────
Gán labels và bbox_targets cho anchors

Logic:
  1. Tính IoU giữa mỗi anchor và mỗi gt_bbox
  2. Mỗi gt_bbox → gán cho anchor có IoU cao nhất
  3. Anchors có IoU > pos_iou_thr (0.7) → positive
  4. Anchors có IoU < neg_iou_thr (0.3) → negative
  5. Anchors có 0.3 ≤ IoU ≤ 0.7 → ignore

─────────────────────────────────────────────

Hàm: RandomSampler (mmdet/core/bbox/samplers/random_sampler.py)
─────────────────────────────────────────────
Sampling để cân bằng positive/negative

Input: assign_result với N anchors
Output: sampling_result với num_samples=256
        - 50% positive (128), 50% negative (128)
        - Nếu không đủ positive → pad với negative
```

---

## 2. Luồn Code HigherHRNet-Human-Pose-Estimation

### 2.1. Entry Point

```
tools/valid.py (Testing/Validation)
─────────────────────────────────────────────
1. Load config, build model, load weights
2. Create data_loader (COCO val2017)
3. For each batch:
     a. multi_scale_sizes = get_multi_scale_size(image, input_size, 1.0, scales)
     b. For each scale:
          outputs = model(image)
          heatmaps, tags = split_heatmaps_tags(outputs)
     c. Grouping: grouped_joints = HeatmapParser.parse(heatmaps, tags)
     d. Transform: final_preds = transform_preds(grouped_joints, center, scale)
4. Compute NME, save predictions

tools/dist_train.py (Training)
─────────────────────────────────────────────
1. Load config, build model
2. For each epoch:
     trainer.train(epoch) → forward → loss → backward → step
     trainer.validate(epoch) → compute NME
3. Save checkpoint
```

### 2.2. Luồn Training Chi Tiết

```
Bước 1: DATA LOADING
─────────────────────────────────────────────
File: lib/dataset/COCOKeypoints.py

Class: COCOKeypointsDataset (kế thừa TorchDataLoader)

    __getitem__(idx):
        1. Load image:
           anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
           img_info = self.coco.loadImgs([img_id])[0]
           image_path = os.path.join(self.img_dir, img_info['file_name'])
           image = cv2.imread(image_path)

        2. Get keypoints:
           keypoints = anno[0]['keypoints']  # [51] = 17×3 (x,y,v)
           v = visibility flag: 0=hidden, 1=occluded, 2=visible

        3. Compute bounding box:
           bbox = anno[0]['bbox']  # [x, y, w, h]
           center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
           scale = [bbox[2]/200, bbox[3]/200]  # Normalize

        4. Data Augmentation:
           if random.random() < 0.5:
               image, keypoints = flip(image, keypoints)
               # Swap left-right keypoints (mirror pairs)

           scale = scale * random.uniform(0.8, 1.2)
           if random.random() < 0.6:
               angle = random.uniform(-30, 30)
               image, keypoints = rotate(image, keypoints, angle)

        5. Crop:
           image, keypoints = crop(image, center, scale, input_size)
           # Affine transform: resize + translate về input_size

        6. Generate heatmaps:
           targets = []
           for kp in keypoints:
               # Transform kp coordinates to heatmap space
               x_heatmap = (kp[0] - bbox[0]) / scale / 4
               y_heatmap = (kp[1] - bbox[1]) / scale / 4
               heatmap = gaussian_heatmap(x_heatmap, y_heatmap, output_size)
               targets.append(heatmap)

        7. Generate tags (Associative Embedding):
           # Mỗi keypoint có 1 tag value
           # Tags được predict bởi model và learned via loss
           # (Tag targets không được tạo ở đây, chỉ trong loss)

        8. Normalize image:
           image = image.astype(np.float32) / 255.0
           image = (image - mean) / std

        Returns: image [3,256,256], heatmaps [17,64,64], mask [1]
```

```
Bước 2: HIGHERHRNET BACKBONE
─────────────────────────────────────────────
File: lib/models/pose_higher_hrnet.py

Class: PoseHigherResolutionNet

    forward(x):
        Input: [B, 3, 256, 256]

        1. STEM:
           x = self.conv1(x)   # 3 → 64, stride 2
           x = self.bn1(x)
           x = self.conv2(x)   # 64 → 64, stride 2
           x = self.bn2(x)
           x = self.relu(x)
           x = self.layer1(x)   # Bottleneck × 4, 64 channels
           Output: [B, 64, 64, 64]

        2. TRANSITION 1 (Split):
           x_list[0] = x                                    # Nhánh 1 (1/4)
           x_list[1] = self.transition1[1](x)              # Nhánh 2 (1/8)
           Output: 2 nhánh [64], [64]

        3. STAGE 2:
           y_list = self.stage2(x_list)
           - 2 nhánh, channels [32, 64]
           - HRModule × 1
           Output: [B,32,64,64], [B,64,32,32]

        4. TRANSITION 2 (Add branch):
           x_list = y_list
           x_list.append(self.transition2[2](y_list[1]))    # Nhánh 3 (1/16)
           Output: 3 nhánh

        5. STAGE 3:
           y_list = self.stage3(x_list)
           - 3 nhánh, channels [32, 64, 128]
           - HRModule × 4 (STEPS=4 trong config)
           Output: 3 nhánh [64], [32], [16]

        6. TRANSITION 3 (Add branch):
           x_list = y_list
           x_list.append(self.transition3[3](y_list[2]))    # Nhánh 4 (1/32)
           Output: 4 nhánh

        7. STAGE 4:
           y_list = self.stage4(x_list)
           - 4 nhánh, channels [32, 64, 128, 256]
           - HRModule × 3 (STEPS=3 trong config)
           Output: 4 nhánh [64], [32], [16], [8]

        8. DECONV LAYERS (Transposed Convolutions):
           final_outputs = []

           # Nhánh 1: upsampled từ y_list[0]
           x = y_list[0]
           y = self.final_layers[0](x)
           final_outputs.append(y)  # [B, 17+tag_dim, 64, 64]

           # Deconv → upsampled
           for i in range(self.num_deconvs):
               x = self.deconv_layers[i](x)
               y = self.final_layers[i+1](x)
               final_outputs.append(y)
               # Output: [B, 17+tag_dim, 128, 128]

        9. Output:
           return final_outputs
           # List of multi-scale outputs
           # [0]: [B, 17+tag, 64, 64]
           # [1]: [B, 17+tag, 128, 128]
```

```
Bước 3: MULTI-RESOLUTION SUPERVISION (Training Loss)
─────────────────────────────────────────────
File: lib/core/loss.py

Class: MultiLossFactory

    forward(outputs, targets):
        total_loss = 0

        for i, output in enumerate(outputs):
            heatmaps_pred = output[:, :self.num_joints]
            tags_pred = output[:, self.num_joints:]

            # Heatmap loss (MSE)
            if self.with_heatmaps_loss[i]:
                loss_hm = MSELoss(heatmaps_pred, targets[i], self.keypoint_heatmap_loss_factor[i])

            # AE loss (Associative Embedding)
            if self.with_ae_loss[i]:
                loss_ae = AELoss(tags_pred, target_tags, ...)
                # target_tags: ground truth tags cho grouping

            total_loss += loss_hm + loss_ae

        return total_loss
```

```
Bước 4: INFERENCE - MULTI-SCALE OUTPUTS
─────────────────────────────────────────────
File: lib/core/inference.py

def get_multi_stage_outputs(outputs, with_flip, project2image):
    heatmaps_avg = 0
    tags_list = []

    for i, output in enumerate(outputs):
        # Interpolate to same size (largest)
        if i < len(outputs) - 1:
            output = interpolate(output, size=outputs[-1].shape[2:])

        # Split heatmaps and tags
        heatmaps = output[:, :num_joints]
        tags = output[:, num_joints:]

        heatmaps_avg += heatmaps
        tags_list.append(tags)

    heatmaps_avg /= len(outputs)

    return heatmaps_avg, tags_list

def aggregate_results(heatmaps_avg, tags_list, num_samples, num_joints, project2image):
    # Multi-scale aggregation
    # Concatenate tags from all scales
    tags = torch.cat(tags_list, dim=1)  # [B, num_tags_combined, H, W]
    return final_heatmaps, tags
```

```
Bước 5: KEYPOINT GROUPING (Associative Embedding)
─────────────────────────────────────────────
File: lib/core/group.py

Class: HeatmapParser

    parse(heatmaps, tags):
        1. NMS:
           for joint in range(num_joints):
               hm = heatmaps[joint]
               # Tìm local maxima
               max_pool = max_pool2d(hm, kernel=3, stride=1, padding=1)
               peaks = (hm == max_pool) & (hm > threshold)
               # Trả về: [peak_y, peak_x, peak_score]

        2. Top-K:
           for joint in range(num_joints):
               keep = top_k(scores, k=top_k)  # Giữ top 30 điểm mỗi joint
               ans[joint] = keep

        3. Tag-based Grouping (Associative Embedding):
           tags_joints = tags[ans['y'], ans['x']]  # [num_joints, num_peaks, tag_dim]

           # Khởi tạo persons:
           # Mỗi person = list of joints
           persons = [[joint0_peak0], [joint1_peak0], ...]

           # Hungarian algorithm:
           # Ghép các joints còn lại vào persons dựa trên tag distance
           for joint_idx in range(1, num_joints):
               cost_matrix = tag_distance(tags_joints[joint_idx], tags_joints[0])
               assignment = hungarian(cost_matrix)
               # Gán joint_idx[j] → person[assignment[j]]

        4. Refine:
           # Loại bỏ persons có < 3 joints
           # Interpolate missing joints

        Returns: List of persons, each with [num_joints, 3] (x, y, score)
```

```
Bước 6: COORDINATE TRANSFORM
─────────────────────────────────────────────
File: lib/utils/transforms.py

def transform_preds(coords, center, scale, output_size):
    # Input: coords in heatmap space [K, 2]
    # Output: coords in original image space [K, 2]

    # Scale factor
    scale = scale * 200.0  # Denormalize
    sx = scale[0] * output_size[0] / output_size[0]
    sy = scale[1] * output_size[1] / output_size[1]

    # Transform
    x = coords[:, 0] / output_size[0] * sx + center[0] - sx / 2
    y = coords[:, 1] / output_size[1] * sy + center[1] - sy / 2

    return [x, y]
```

### 2.3. Chi tiết Associative Embedding

```
ASSOCIATIVE EMBEDDING - Chi tiết thuật toán
─────────────────────────────────────────────────────────────

Training Loss:
  L = L_hm + λ * L_ae

  Trong đó:
  L_hm = MSE(heatmap_pred, heatmap_target)
  L_ae = L_pull + L_push

  L_pull (Pull loss):
    - Kéo các keypoints cùng 1 người về gần nhau
    - L_pull = (1/N) * Σ exp(||h_i - h_j||²)
      với h_i, h_j là tags của 2 keypoints cùng 1 người

  L_push (Push loss):
    - Đẩy tags của người khác ra xa
    - L_push = max(0, 1 - ||h_i - h_j|| + margin)
      với h_i, h_j là tags của keypoints khác người

Inference - Grouping:
  1. Với mỗi joint type (vd: nose, left_eye, ...):
     - Tìm top-K peaks trên heatmap
     - Mỗi peak có 1 tag value

  2. Bắt đầu từ center keypoints (thường là nose):
     - Khởi tạo K persons (K = số peaks của center)

  3. Với mỗi joint type còn lại:
     - Tính tag distance từ mỗi peak đến mỗi person
     - Dùng Hungarian algorithm để gán tối ưu
     - Update person với joint position

  4. Output: List of persons
     Person i = [joint_0(x,y), joint_1(x,y), ..., joint_16(x,y)]
```

---

## 3. Luồn Code HRNet-Facial-Landmark-Detection

### 3.1. Entry Point

```
tools/train.py (Training)
─────────────────────────────────────────────
1. update_config() → load YAML config
2. logger = create_logger()
3. model = get_face_alignment_net(cfg)  # HRNetV2-W18
4. model = nn.DataParallel(model).cuda()
5. criterion = MSELoss()
6. optimizer = get_optimizer(model, cfg)
7. train_loader = DataLoader(dataset(cfg, is_train=True), batch_size)
8. val_loader = DataLoader(dataset(cfg, is_train=False), batch_size)
9. for epoch in range(TRAIN.END_EPOCH):
      train(epoch, train_loader, model, criterion, optimizer)
      validate(epoch, val_loader, model, criterion)
      save_checkpoint(epoch, model, optimizer)

tools/test.py (Testing)
─────────────────────────────────────────────
1. Load config, build model
2. Load checkpoint: model.load_state_dict(torch.load(MODEL_FILE))
3. test_loader = DataLoader(dataset(cfg, is_train=False))
4. predictions = inference(test_loader, model)
5. torch.save(predictions, 'predictions.pth')
```

### 3.2. Luồn Training Chi Tiết

```
Bước 1: DATA LOADING - WFLW Dataset
─────────────────────────────────────────────
File: lib/datasets/wflw.py

Class: WFLWDataset

    __getitem__(idx):
        1. Read CSV line:
           row = self.label_list[idx]
           path, scale, center_x, center_y, x1,y1,...,x98,y98 = row

        2. Load image:
           image = cv2.imread(os.path.join(self.img_dir, path))
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        3. Parse landmarks:
           landmarks = np.array([x1,y1, x2,y2, ..., x98,y98])
           landmarks = landmarks.reshape(-1, 2)  # [98, 2]

        4. Data Augmentation (is_train=True):
           a. Random scale:
              scale_factor = random.uniform(1-SCALE_FACTOR, 1+SCALE_FACTOR)
              # SCALE_FACTOR = 0.25
              scale *= scale_factor

           b. Random rotation:
              if random.random() < 0.6:
                 angle = random.uniform(-ROT_FACTOR, ROT_FACTOR)
                 # ROT_FACTOR = 30
                 image, landmarks = rotate(image, landmarks, angle, center)
                 # Cập nhật center và scale

           c. Random flip:
              if random.random() < 0.5:
                 image = cv2.flip(image, 1)  # Horizontal flip
                 landmarks = fliplr_joints(landmarks, image_width)
                 center[0] = image_width - center[0]

        5. Crop face:
           image, landmarks = crop(image, center, scale, INPUT_SIZE)
           # INPUT_SIZE = [256, 256]

           # Affine transform matrix:
           # Scale về 256x256, translate về center
           M = get_affine_matrix(center, scale, INPUT_SIZE)
           image = cv2.warpAffine(image, M, tuple(INPUT_SIZE))

        6. Transform landmarks to heatmap coordinates:
           # Output heatmap = 64x64 (1/4 của 256)
           heatmap_coords = landmarks / 4.0  # [98, 2]

        7. Generate heatmap targets:
           targets = []
           for kp in heatmap_coords:
               heatmap = generate_target(kp, OUTPUT_SIZE, sigma)
               # OUTPUT_SIZE = [64, 64], sigma = 2
               targets.append(heatmap)

           target = np.stack(targets, axis=0)  # [98, 64, 64]

        8. Handle flip:
           if is_flipped:
              target = target[:, :, ::-1].copy()  # Flip heatmaps

        9. Normalize image:
           image = image.astype(np.float32) / 255.0
           image = (image - mean) / std
           # mean = [0.485, 0.456, 0.406] (ImageNet)
           # std = [0.229, 0.224, 0.225]

        10. Convert to tensor:
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW
            target = torch.from_numpy(target)

            Returns: image [3,256,256], target [98,64,64], meta
```

```
Bước 2: MODEL - HRNetV2-W18
─────────────────────────────────────────────
File: lib/models/hrnet.py

Class: HighResolutionNet

    forward(x):
        Input: [B, 3, 256, 256]

        1. STEM:
           x = self.conv1(x)    # 3 → 64, kernel=3, stride=2
           x = self.bn1(x)
           x = self.conv2(x)    # 64 → 64, kernel=3, stride=2
           x = self.bn2(x)
           x = self.relu(x)

           x = self.layer1(x)   # Bottleneck × 4, 64 channels
           Output: [B, 64, 64, 64]

        2. STAGE 2 (2 nhánh):
           x1 = x                                          # Nhánh 1: [B, 18, 64, 64]
           x2 = self.transition2_1(x)                     # Nhánh 2: [B, 36, 32, 32]
           x_list = self.stage2_layer1([x1, x2])
           y_list = self.stage2_layer2(x_list)
           Output: [Nhánh 1, Nhánh 2]

        3. STAGE 3 (3 nhánh):
           x_list = [y_list[0], y_list[1]]
           x_list.append(self.transition3_2(y_list[1]))   # Thêm Nhánh 3
           y_list = self.stage3_layer1(x_list)
           y_list = self.stage3_layer2(y_list)
           Output: [Nhánh 1, Nhánh 2, Nhánh 3]

        4. STAGE 4 (4 nhánh):
           x_list = [y_list[0], y_list[1], y_list[2]]
           x_list.append(self.transition4_3(y_list[2]))   # Thêm Nhánh 4
           y_list = self.stage4_layer1(x_list)
           y_list = self.stage4_layer2(y_list)
           Output: [Nhánh 1, Nhánh 2, Nhánh 3, Nhánh 4]

        5. UPSAMPLE & CONCATENATE:
           # Upsample all branches to stage4 resolution (8×8)
           # Nhánh 1: [B, 18, 64, 64] → upsample ×8 → [B, 18, 8, 8]
           # Nhánh 2: [B, 36, 32, 32] → upsample ×4 → [B, 36, 8, 8]
           # Nhánh 3: [B, 72, 16, 16] → upsample ×2 → [B, 72, 8, 8]
           # Nhánh 4: [B, 144, 8, 8]  → keep         → [B, 144, 8, 8]

           x = concat([upsample(y[0]), upsample(y[1]),
                       upsample(y[2]), y[3]], dim=1)
           # x = [B, 18+36+72+144=270, 8, 8]

        6. HEAD:
           x = self.head_layer1(x)  # conv1x1(270→256), bn, relu
           x = self.head_layer2(x)  # conv1x1(256→NUM_LANDMARKS)

        7. OUTPUT:
           x = interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
           return x  # [B, NUM_LANDMARKS, 64, 64]

Class: HighResolutionModule (cấu trúc tương tự HRNet-Object-Detection)

    forward(branches):
        1. Residual blocks trên mỗi nhánh
        2. Fuse: exchange information giữa các nhánh
           - Upsample nhánh nhỏ, downsample nhánh lớn
           - Concatenate hoặc SUM
        3. Relu activation
        Returns: fused branches
```

```
Bước 3: TRAINING LOOP
─────────────────────────────────────────────
File: lib/core/function.py

def train(cfg, train_loader, model, criterion, optimizer, epoch):
    model.train()
    total_loss = 0

    for i, (images, heatmap_targets, meta) in enumerate(train_loader):
        images = images.cuda()
        heatmap_targets = heatmap_targets.cuda()

        # Forward
        outputs = model(images)  # [B, NUM_LM, 64, 64]

        # Loss
        loss = criterion(outputs, heatmap_targets)  # MSELoss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Decode để compute NME
        preds = decode_preds(outputs, meta['center'], meta['scale'])
        nme_batch = compute_nme(preds, meta)
```

```
Bước 4: DECODE PREDICTIONS
─────────────────────────────────────────────
File: lib/core/evaluation.py

def get_preds(output):
    """Tìm tọa độ (x,y) của mỗi landmark từ heatmap"""
    # output: [B, NUM_LM, 64, 64]

    # Method 1: Argmax (đơn giản)
    coords = output.argmax(dim=2)  # [B, NUM_LM]
    # coords[:, k] = argmax of heatmap for landmark k

    # Method 2: Argmax + sub-pixel refinement
    preds = []
    for b in range(batch_size):
        for k in range(num_landmarks):
            hm = output[b, k]  # [64, 64]

            # Tìm peak
            max_val = hm.max()
            max_idx = hm.argmax()
            y = max_idx // 64
            x = max_idx % 64

            # Sub-pixel refinement
            # dx = 0.25 * (H[y+1,x] - H[y-1,x])
            # dy = 0.25 * (H[y,x+1] - H[y,x-1])
            # refined_x = x + dx
            # refined_y = y + dy

            preds.append([x, y])

    return coords  # [B, NUM_LM, 2]

def decode_preds(outputs, center, scale):
    """Transform từ heatmap coords → original image coords"""

    coords = get_preds(outputs)  # [B, NUM_LM, 2]

    for i in range(coords.size(0)):
        # Denormalize: từ [0,1] → original image size
        coords[i, :, 0] = coords[i, :, 0] * 4 / cfg.MODEL.EXTRA.OUTPUT_SIZE[0] * scale[i, 0] * 200 + center[i, 0]
        coords[i, :, 1] = coords[i, :, 1] * 4 / cfg.MODEL.EXTRA.OUTPUT_SIZE[1] * scale[i, 1] * 200 + center[i, 1]

    return coords  # [B, NUM_LM, 2] in original image pixel coords
```

### 3.3. Chi tiết Heatmap Generation

```
Hàm: generate_target (lib/utils/transforms.py)
─────────────────────────────────────────────
Tạo Gaussian heatmap cho 1 landmark

def generate_target(target_pt, heatmap_size, sigma=2):
    """
    Input:  target_pt = [x, y] trong heatmap coordinates
            heatmap_size = [64, 64]
            sigma = 2 (Gaussian std)
    Output: heatmap [64, 64] - Gaussian centered at target_pt
    """

    heatmap = np.zeros(heatmap_size, dtype=np.float32)

    x, y = int(target_pt[0]), int(target_pt[1])

    # Tạo Gaussian
    for i in range(heatmap_size[0]):
        for j in range(heatmap_size[1]):
            d = (i - x)**2 + (j - y)**2
            heatmap[j, i] = np.exp(-d / (2 * sigma**2))

    return heatmap

─────────────────────────────────────────────

Cơ chế Gaussian:
  Giá trị heatmap tại pixel (i,j):
    H[i,j] = exp(-((i-x)² + (j-y)²) / (2σ²))

  - Tại tâm (x,y): H = 1.0
  - Càng xa tâm: H giảm dần
  - sigma càng lớn: Gaussian càng rộng (nhòe hơn)

  Mục đích:
  - Thay vì dùng hard label (chỉ 1 pixel = 1), dùng soft label
  - Giúp gradient flow tốt hơn trong training
  - Dễ học hơn vì model chỉ cần approximate Gaussian

─────────────────────────────────────────────

Hàm: transform_pixel (lib/utils/transforms.py)
─────────────────────────────────────────────
Transform tọa độ từ original image → heatmap coordinates

def transform_pixel(coords, scale, center, heatmap_size):
    """
    Input:  coords: [x, y] trong original image
            scale: scale factor đã crop
            center: center of face
            heatmap_size: [64, 64]
    Output: [x_hm, y_hm] trong heatmap coordinates
    """

    # Transform từ original → cropped
    x = (coords[0] - center[0]) / (scale * 200.0) + 0.5
    y = (coords[1] - center[1]) / (scale * 200.0) + 0.5

    # Transform từ cropped → heatmap
    x = x * heatmap_size[0]
    y = y * heatmap_size[1]

    return [x, y]
```

### 3.4. Chi tiết NME Metric

```
Hàm: compute_nme (lib/core/evaluation.py)
─────────────────────────────────────────────
Normalized Mean Error - Metric chính cho facial landmark detection

def compute_nme(preds, meta):
    """
    Input:  preds: [B, NUM_LM, 2] - predicted landmark coordinates
            meta: dict chứa ground truth và normalization info
    Output: nme_batch - list of NME values per image
    """

    nme_results = []

    for i in range(len(preds)):
        pred = preds[i]  # [NUM_LM, 2]
        gt = meta['gt'][i]  # [NUM_LM, 2]

        # Normalization factor (tuỳ dataset):
        if dataset == '300W' or 'WFLW':
            # Interocular distance (khoảng cách 2 mắt)
            left_eye = gt[36]  # Left eye center
            right_eye = gt[45]  # Right eye center
            norm_factor = np.linalg.norm(left_eye - right_eye)

        elif dataset == 'AFLW':
            # Face bounding box size
            norm_factor = meta['box_size'][i]

        # Compute NME
        errors = np.linalg.norm(pred - gt, axis=1)  # [NUM_LM]
        nme = errors.mean() / norm_factor
        nme_results.append(nme)

    return nme_results

Công thức:
  NME = (1/K) * Σ ||pred_k - gt_k|| / norm_factor

  - K = số landmarks (68, 98, 19, 29)
  - norm_factor = interocular distance (300W, WFLW, COFW)
                 hoặc bounding box size (AFLW)
  - NME càng nhỏ → accuracy càng cao
  - Thường report NME × 100 (đơn vị: %)

Failure Rate:
  FR @ t = % of images có NME > t (t = 0.1 → 10%)
  → Đo proportion of failed detections
```

---

## 4. So sánh luồn code 3 dự án

### 4.1. Điểm giống nhau

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CHUNG CẢ 3 DỰ ÁN                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT PROCESSING                                                    │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │ Image → Resize → Augmentation → Normalize → Tensor        │        │
│  │  - Random flip (p=0.5)                                   │        │
│  │  - Random scale jitter                                   │        │
│  │  - Random rotation (±30°, p=0.6)                        │        │
│  │  - Center crop/pad to fixed size                         │        │
│  │  - Normalize: (img - mean) / std                        │        │
│  └──────────────────────────────────────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │                    HRNet BACKBONE                         │        │
│  │                                                          │        │
│  │  Stem → Layer1 → Stage2 → Stage3 → Stage4 → Fuse         │        │
│  │                                                          │        │
│  │  Stage 2: 2 branches  [18, 36] channels                  │        │
│  │  Stage 3: 3 branches  [18, 36, 72] channels              │        │
│  │  Stage 4: 4 branches  [18, 36, 72, 144] channels         │        │
│  │                                                          │        │
│  │  Key difference:                                         │        │
│  │  - HRNet-Object-Detection: Feature maps được giữ lại    │        │
│  │  - HigherHRNet: Thêm deconv layers cho multi-scale heatmaps│       │
│  │  - HRNet-Facial: Fuse vào 1 tensor, conv head            │        │
│  │                                                          │        │
│  │  Output: Multi-resolution feature maps                   │        │
│  └──────────────────────────────────────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │                  TASK-SPECIFIC HEAD                      │        │
│  │                                                          │        │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │        │
│  │  │  Detection   │ │    Pose     │ │   Landmark   │      │        │
│  │  │  RPN + BBox  │ │ Heatmap +   │ │  Heatmap     │      │        │
│  │  │  Head        │ │ AE Tag      │ │  Regression │      │        │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │        │
│  └──────────────────────────────────────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │                   POST-PROCESSING                         │        │
│  │                                                          │        │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │        │
│  │  │    NMS +     │ │  AE-based   │ │  Argmax +    │      │        │
│  │  │  BBox Refine │ │  Grouping   │ │  Sub-pixel   │      │        │
│  │  │  Per-class   │ │  Hungarian  │ │  Transform   │      │        │
│  │  │  NMS         │ │  Algorithm  │ │  back to     │      │        │
│  │  │              │ │             │ │  original    │      │        │
│  │  └──────────────┘ └──────────────┘ └──────────────┘      │        │
│  └──────────────────────────────────────────────────────────┘        │
│                              │                                     │
│                              ▼                                     │
│                          RESULTS                                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2. Điểm khác biệt chính

```
┌────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│    Aspect      │  HRNet-Object-Detection │ HigherHRNet-Pose         │ HRNet-Facial-Landmark   │
├────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Output type    │ Bounding boxes + Classes │ (x,y) for 17 joints     │ (x,y) for N landmarks   │
│ Head           │ RPN Head + BBox Head    │ Heatmap Head + Tag Head │ Heatmap Head only       │
│ Loss           │ CrossEntropy + SmoothL1 │ MSE + AE Loss           │ MSE (simple)             │
│ NMS            │ RPN NMS + BBox NMS      │ NMS per joint + Grouping│ Argmax + refine         │
│ Augmentation   │ scale, flip              │ scale, rotate, flip      │ scale, rotate, flip      │
│ Multi-scale    │ FPN (P0-P4)             │ Multi-res outputs       │ Single output           │
│ Metric         │ COCO mAP                │ COCO Keypoint AP        │ NME (Normalized Error)   │
│ Dataset        │ COCO Detection          │ COCO Keypoints          │ 300W, WFLW, AFLW, COFW  │
│ Training time  │ ~2 ngày (8 GPU)         │ ~1 ngày (4 GPU)         │ ~4 giờ (1 GPU)          │
│ GPU memory     │ ~8GB                    │ ~6GB                    │ ~4GB                    │
│ License        │ Apache 2.0              │ MIT                     │ MIT                     │
└────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

### 4.3. So sánh chi tiết Loss Functions

```
HRNet-Object-Detection:
─────────────────────────────────────────────
Total Loss = RPN_Loss + RCNN_Loss (+ Mask_Loss)

  RPN_Loss = RPN_cls_loss + RPN_bbox_loss

    RPN_cls_loss = BinaryCrossEntropy(
                      sigmoid(rpn_cls_score), 
                      pos_neg_labels
                    )
    RPN_bbox_loss = SmoothL1Loss(
                      predicted_deltas, 
                      target_deltas
                    )

  RCNN_Loss = cls_loss + bbox_loss

    cls_loss = CrossEntropyLoss(cls_scores, gt_labels)
    bbox_loss = SmoothL1Loss(predicted_deltas, target_deltas, 
                            weight=positive_mask)

  Mask_Loss = BinaryCrossEntropy(
                sigmoid(mask_preds)[positive_rois],
                mask_targets[positive_rois]
              )

─────────────────────────────────────────────

HigherHRNet-Pose:
─────────────────────────────────────────────
Total Loss = Heatmap_Loss + AE_Loss

  Heatmap_Loss = MSE(
                   sigmoid(predicted_heatmaps),
                   target_heatmaps
                 )
               × keypoint_heatmap_loss_factor (multi-scale)

  AE_Loss = Pull_Loss + Push_Loss

    Pull_Loss = mean(exp(||h_i - h_j||²)) 
                cho cùng 1 người, cùng joint type

    Push_Loss = max(0, 1 - ||h_i - h_j|| + margin)
                cho khác người, cùng joint type

─────────────────────────────────────────────

HRNet-Facial-Landmark:
─────────────────────────────────────────────
Total Loss = Heatmap_Loss

  Heatmap_Loss = MSE(
                   predicted_heatmaps,
                   target_heatmaps
                 )

  Trong đó target_heatmaps là Gaussian centered tại landmark location
```

### 4.4. So sánh kiến trúc đầu ra (Output Heads)

```
HRNet-Object-Detection - Detection Head:
─────────────────────────────────────────────
Feature Pyramid [P0,P1,P2,P3,P4]
    │
    ├── RPN Head (5 levels)
    │   ├── conv3x3(256→256) → relu
    │   ├── cls: conv1x1(256→num_anchors×3)
    │   └── reg: conv1x1(256→num_anchors×4)
    │
    ├── proposals (NMS)
    │
    ├── RoIAlign → [N,256,7,7]
    │
    └── BBox Head
        ├── flatten → fc(12544→1024)
        ├── fc(1024→1024)
        ├── cls: fc(1024→num_classes+1)
        └── reg: fc(1024→num_classes×4)

─────────────────────────────────────────────

HigherHRNet - Pose Head:
─────────────────────────────────────────────
Multi-resolution features [y1,y2,y3,y4]
    │
    ├── Deconv layers
    │   └── transposed_conv → upsample
    │
    └── Final layers (per resolution)
        ├── conv1x1(ch→128) → relu
        ├── conv1x1(128→64) → relu
        ├── conv1x1(64→num_joints+tag_dim)
        │
        ├── Heatmaps: sigmoid → [B, 17, H, W]
        └── Tags: [B, tag_dim, H, W]

─────────────────────────────────────────────

HRNet-Facial-Landmark - Landmark Head:
─────────────────────────────────────────────
Multi-resolution features [y1,y2,y3,y4]
    │
    ├── Upsample all to [B, 8, 8]
    │
    ├── Concatenate: [B, 18+36+72+144=270, 8, 8]
    │
    └── Head layers
        ├── conv1x1(270→256) → bn → relu
        └── conv1x1(256→num_landmarks) → [B, N_LM, 8, 8]
            │
            └── interpolate → [B, N_LM, 64, 64]
```

---

## Tài liệu tham khảo

1. **HRNet-Object-Detection**: https://github.com/HRNet/HRNet-Object-Detection
2. **HigherHRNet-Human-Pose-Estimation**: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
3. **HRNet-Facial-Landmark-Detection**: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
4. **mmdetection**: https://github.com/open-mmlab/mmdetection
5. **HRNet Paper (Original)**: https://arxiv.org/abs/1904.04514
6. **HigherHRNet Paper**: https://arxiv.org/abs/1908.10357
7. **COCO Dataset**: https://cocodataset.org/

---

*Tài liệu này được viết cho bộ môn Tìm kiếm và Truy xuất Thông tin, Thạc sĩ Học viện Bưu chính Viễn thông.*
