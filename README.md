# 地面視角道路阻塞率演算法程式（V40）

本程式使用地面視角影像的人工或模型 segmentation，重建道路幾何、建立道路橫斷面，並依各位置的剩餘連續通行寬度判定道路阻塞程度。

> **資料安全提醒：本 GitHub repository 只應上傳程式碼、README、requirements 與 `.gitignore`。請勿上傳原始照片、segmentation 標註、COCO JSON、試算表、推論結果或診斷圖。**

## 1. 檔案簡述

`road_aligned_2d_evaluation.py`

內含 V40 的完整道路重建與阻塞率判定流程，主要功能包括：

1. 讀取 COCO polygon／RLE segmentation 並建立各類遮罩。
2. 從多個連續角度及多條 Road 軌跡候選中選出主要道路。
3. 建立道路座標、中心、道路寬度與左右邊界。
4. 重建內部缺口、障礙物遮擋區及有限的道路末端。
5. 提取 Roadline 與道路側邊設施的平滑走向曲線。
6. 在符合條件時使用中央 Roadline 校正中心位置與道路方向。
7. 建立橫斷面並計算最大連續自由寬度與阻塞率。
8. 輸出 `small_or_clear`、`half`、`full` 或 `insufficient`。
9. 產生逐圖結果、混淆矩陣、摘要與診斷圖。

目前程式只保留 V40 判定流程；早期固定 Y 軸方法及其比較輸出已移除。

## 2. 演算法架構

V40 可分為六個主要階段。

### 2.1 Segmentation 與遮罩建立

程式讀取下列標籤：

| 類型 | 標籤 | 用途 |
| --- | --- | --- |
| 路面 | `road` | 建立可見道路、主要道路身份及基本左右邊界 |
| 障礙物 | `rock` | 計算道路阻塞與支持遮擋重建 |
| 障礙物 | `fallen vegetation` | 計算道路阻塞與支持遮擋重建 |
| 障礙物 | `debris pile` | 計算道路阻塞與支持遮擋重建 |
| 方向輔助 | `road line`／`road_line` | 提供道路方向；符合條件時可作中央 Roadline |
| 方向輔助 | `curb` | 提供道路側邊方向與曲率 |
| 方向輔助 | `guardrail` | 提供道路側邊方向與曲率 |
| 方向輔助 | `barricade` | 提供道路側邊方向與曲率 |
| 方向輔助 | `roadside wall` | 提供道路側邊方向與曲率 |
| 弱方向輔助 | `cone` | 由排列方向提供較弱的道路方向證據 |

同類標籤會合併為二值遮罩。分析解析度可透過 `--analysis-long-edge` 限制，影像相關遮罩與像素門檻會同步縮放。

### 2.2 多方向、多軌跡主要道路選擇

程式不假設道路只能水平或垂直，而會：

1. 建立多個道路方向候選。
2. 在每個方向中追蹤最多數條不同 Road 軌跡。
3. 比較各候選的 Road 支持、連續距離、中心平順度、寬度變化、缺口與橫向跳動。
4. 選出最能代表目前主要道路的候選。

此階段主要回答：

> 圖中的哪些 Road 片段屬於同一條、目前需要分析的道路？

### 2.3 基礎道路與缺口重建

選出主要道路後，建立道路座標：

```text
u：沿道路前進方向
v：道路橫向方向
```

沿道路前進方向依序估計中心、寬度及左右邊界，並將各位置分為三個核心狀態：

#### 直接觀測

該位置有足夠 Road segmentation 時，直接使用 Road 提供的左右邊界。這是可靠度最高的幾何來源。

#### 內部重建

形式如下：

```text
Road → 空白或障礙物遮擋 → Road
```

若前後 Road 屬於同一條主要軌跡，且中心、方向、寬度與左右順序相容，則分別連接左右邊界，補回中間缺失道路。障礙物可提供支持，但前後雙錨點明確時，不強制要求缺口內一定具有障礙物標籤。

#### 末端外推

形式如下：

```text
Road → 障礙物 → 後方沒有 Road
```

程式使用近期可靠 Road 的方向、左右邊界與寬度有限延伸道路。末端缺少後方 Road 驗證，因此外推長度受限，且信心低於直接觀測與內部雙錨點重建。

若道路某側在障礙物附近突然內凹，也可依附近道路寬度有限補回缺失側，但不得任意增加整體道路寬度。

### 2.4 Roadline 與道路方向輔助物

程式會從 Roadline、curb、guardrail、barricade、roadside wall 與 cone segmentation 提取平滑走向曲線。

Roadline 會進一步判定角色：

```text
C：中央 Roadline，可影響中心位置與方向
E：道路邊線，主要提供方向與曲率
D：一般方向證據
X：因品質不足或衝突而拒絕
```

同一實體 Roadline 若因車輛、障礙物或 segmentation 斷裂而分成數段，可在位置、方向及曲率相容時合併為一條平滑軌跡。

V40 的中心與橫斷面有兩種模式：

1. **Road 幾何模式**：沒有唯一可信中央 Roadline 時，以主要 Road 軌跡建立中心與橫斷面。
2. **Roadline center 模式**：找到唯一可信中央 Roadline 時，使用 Roadline 的位置、弧長站序及局部切線建立中心，再嘗試配對道路左右邊界。

若 Roadline 專用左右邊界配對品質不足，V40 會回退至穩定的 Road 幾何模式。

### 2.5 三種道路範圍

程式會建立三種用途不同的道路範圍：

#### 完整道路 `full_road_corridor`

```text
原始 Road
＋ 內部缺口重建
＋ 內部遮擋推測
＋ 有限側向補齊
＋ 末端外推
```

代表障礙物不存在時，道路原本可能具有的完整範圍。

#### 分析走廊 `analysis_geometry_corridor`

由平滑後的左右道路邊界形成，用於中心線、道路寬度及橫斷面幾何計算。分析走廊是幾何範圍，不表示其中每個像素都是直接觀測到的 Road。

#### 自由通道 `free_lane_corridor`

從完整道路中扣除有效障礙物後，選出沿道路方向可連續通行的主要區域。

### 2.6 橫斷面、阻塞率與分類

沿道路由近到遠建立多條橫斷面。每條有效橫斷面會取得：

- 完整道路寬度；
- 障礙物占用區間；
- 最大連續可通行寬度；
- 自由寬度比例；
- 阻塞率。

阻塞率定義為：

```text
阻塞率 = 1 − 最大連續可通行寬度 ÷ 完整道路寬度
```

分類規則如下：

| 輸出 | 條件 |
| --- | --- |
| `full` | P90 阻塞率 ≥ 0.75，且至少連續 3 條橫斷面阻塞率 ≥ 0.75 |
| `half` | 未達 `full`，P90 阻塞率 ≥ 0.30，且至少連續 3 條橫斷面阻塞率 ≥ 0.30 |
| `small_or_clear` | 未達 `full` 或 `half` 條件 |
| `insufficient` | Road 太少、主要道路無法建立，或有效橫斷面比例低於 60% |

無效橫斷面會中斷連續區段，不會將前後不相鄰的高阻塞位置接在一起。

## 3. 判定信心

道路分類完成後，程式依真正支撐該分類的橫斷面來源估計信心。信心權重不會直接修改阻塞率、P90 或最終類別。

| 道路幾何來源 | 信心權重 |
| --- | ---: |
| 直接 Road 觀測 | 1.00 |
| Roadline 支持的內部重建 | 0.80 |
| 內部缺口插值 | 0.70 |
| 內部遮擋推測 | 0.70 |
| 輔助邊界推測 | 0.65 |
| 輔助物支持的末端外推 | 0.55 |
| 單側邊界推測 | 0.50 |
| 一般末端外推 | 0.35 |

信心分級：

```text
平均權重 ≥ 0.80         → high
0.55 ≤ 平均權重 < 0.80  → medium
平均權重 < 0.55         → low
```

## 4. 必要套件

建議使用 Python 3.10 以上。

```bash
python -m pip install -r requirements.txt
```

必要套件：

- `numpy`
- `Pillow`

程式內建 COCO polygon、未壓縮 RLE 及壓縮 RLE 解碼器，因此沒有安裝 `pycocotools` 也可執行。如需使用官方 COCO 解碼器，可另外安裝：

```bash
python -m pip install pycocotools
```

## 5. 輸入資料格式

### 5.1 COCO segmentation

`--coco` 使用標準 COCO JSON，至少需要：

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "example.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "categories": [
    {"id": 1, "name": "road"},
    {"id": 2, "name": "rock"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[100, 900, 800, 400, 1500, 900]],
      "iscrowd": 0
    }
  ]
}
```

`segmentation` 可使用 polygon 或 COCO RLE。

### 5.2 人工阻塞標準答案

若需要輸出混淆矩陣，影像項目可在 `extra.uavg_ground_truth` 中提供：

```json
{
  "id": 1,
  "file_name": "example.jpg",
  "width": 1920,
  "height": 1080,
  "extra": {
    "uavg_ground_truth": "half"
  }
}
```

可接受：

```text
small_or_clear
half
full
not_evaluable
```

`not_evaluable` 或缺少有效人工答案的影像會從正式評估中排除。

### 5.3 模型 segmentation

若不使用 COCO JSON 內的人工 segmentation，可用 `--predictions` 指定另一份 COCO result JSON。此時評估結果代表 segmentation 加上道路阻塞演算法的整體輸出。

## 6. 執行方式

請先進入程式所在資料夾。輸出資料夾必須位於程式目錄下。

### 6.1 基本評估

```bash
python road_aligned_2d_evaluation.py \
  --coco /path/to/annotations.coco.json \
  --output ./results
```

### 6.2 限制分析解析度

```bash
python road_aligned_2d_evaluation.py \
  --coco /path/to/annotations.coco.json \
  --output ./results_1600 \
  --analysis-long-edge 1600
```

### 6.3 產生診斷圖

```bash
python road_aligned_2d_evaluation.py \
  --coco /path/to/annotations.coco.json \
  --images-dir /path/to/images \
  --output ./results_1600 \
  --analysis-long-edge 1600 \
  --save-diagnostics \
  --diagnostics-all \
  --diagnostic-layout d
```

### 6.4 指定部分影像

```bash
python road_aligned_2d_evaluation.py \
  --coco /path/to/annotations.coco.json \
  --images-dir /path/to/images \
  --output ./selected_results \
  --image-ids 1,5,12 \
  --save-diagnostics \
  --diagnostics-all \
  --diagnostic-layout d
```

可使用下列指令查看完整參數：

```bash
python road_aligned_2d_evaluation.py --help
```

## 7. 輸出內容

### `per_image_results.json`

保存每張影像的完整結果，包括：

- 預測類別與人工答案；
- 中心線、左右邊界與橫斷面；
- P90、最大阻塞率及最小自由寬度比例；
- 道路重建來源；
- Roadline 與方向輔助狀態；
- 道路品質警告及失敗原因；
- 判定信心及支撐來源。

### `per_image_summary.csv`

每張影像一列，包含預測、道路幾何、方向模式、重建數量、有效橫斷面比例、阻塞統計與判定信心，適合進行試算表分析。

### `confusion_table.csv`

以人工答案為列、V40 預測為欄，統計：

```text
small_or_clear
half
full
insufficient
```

### `summary.json` 與 `summary.txt`

記錄：

- Coverage；
- Accuracy 與 Macro-F1；
- 各類別 Precision／Recall／F1；
- `insufficient` 數量；
- 道路重建來源統計；
- 方向輔助使用狀態；
- 中心線失敗原因及品質警告。

### `diagnostics_*/`

保存道路重建、橫斷面、阻塞率及判定資訊的診斷圖。

### `D_aux_debug/`

顯示 Roadline、curb、guardrail、barricade、roadside wall 等輔助走向曲線的候選、角色及採用狀態，用於檢查道路方向判定。

## 8. 診斷圖顏色

- 青色輪廓：原始 Road segmentation。
- 藍色半透明區域：分析走廊。
- 綠色半透明區域：主要自由通道。
- 粉紅色區域：位於道路內、會計入阻塞的障礙物。
- 暗紅色區域：道路外障礙物。
- 橙色輪廓或區域：插值、遮擋推測或外推道路。
- 黃色線：最終道路中心／導航軸。
- 白點及白線：道路分析範圍的左右外邊界。
- 彩色橫線：各位置的道路橫斷面及阻塞程度。

`D_aux_debug` 中，輔助物保留各自的類別顏色，並以線型及 `C／E／D／X` 顯示角色與是否採用。

## 9. 目前限制

- 演算法品質仍依賴 Road 與障礙物 segmentation 是否正確。
- 平行道路、岔路或非常零碎的 Road 可能使主要道路身份選擇錯誤。
- 大型內部缺口若缺少足夠的前後 Road 幾何，可能無法可靠接回。
- 末端外推沒有後方 Road 錨點，因此採保守策略且信心較低。
- 斜拍與強透視場景中，影像上的道路中點不一定等於真實道路中心。
- 中央 Roadline 的角色或左右邊界配對不穩定時，V40 會回退到 Road 幾何模式。
- 使用人工 COCO segmentation 的結果，只代表道路重建與阻塞率演算法表現，不代表端到端 segmentation 模型表現。

## 10. GitHub 上傳安全規則

建議只提交：

```text
road_aligned_2d_evaluation.py
README.md
requirements.txt
.gitignore
```

禁止提交：

```text
原始照片或影片
segmentation 標註與 COCO JSON
Excel／CSV 人工答案
per_image_results.json
診斷圖、overlay 及推論結果資料夾
模型權重、壓縮檔與暫存檔
```

提交前應執行：

```bash
git status
git diff --cached --name-only
```

確認 staged files 中只有程式碼與說明文件後，再執行 commit 及 push。

