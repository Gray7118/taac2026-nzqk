# TAAC 2026 数据集分析与时间特征优化方案

> **数据验证**: 基于 `demo_1000.parquet` (1000样本) 的探索性分析，结论已于2026-05-11更新确认。

---

## 一、数据特性总览

### 1.1 数据集规模与结构

| 维度 | Academic Track | Industrial Track |
|------|---------------|-----------------|
| 样本量 | 100万 | 200万 |
| 总列数 | 120 | 120 |
| 存储格式 | Parquet (Flat Layout) | Parquet (Flat Layout) |
| 数据组织形式 | Row Group 分片 | Row Group 分片 |
| 正负样本比 | ~1:7 (12.4%转化率, demo确认) | 待验证 |

### 1.2 六大特征类别分布

| 类别 | 列数 | 数据类型 | 关键特征 |
|------|------|----------|----------|
| ID & Label | 5 | int64/int32 | user_id, item_id, label_type, label_time, timestamp |
| User Int | 46 (35标量+11数组) | int64/list<int64> | 年龄、性别、偏好等多维离散特征 |
| User Dense | 10 | list<float> | SUM嵌入(256维)、LMF4Ads嵌入(320维)、8对齐特征 |
| Item Int | 14 (13标量+1数组) | int64/list<int64> | 物品类目、类型、多标签 |
| Domain Sequence | 45 | list<int64> | 4域行为序列: a(9), b(14), c(12), d(10) |

---

## 二、特征深度分析（Demo数据验证）

### 2.0 Demo数据实测统计

以下统计基于 `demo_1000.parquet` 的1000条样本：

| 指标 | 值 |
|------|-----|
| 总样本数 | 1000 |
| 唯一用户数 | 1000 (人均1条，demo数据无重复用户) |
| 唯一物品数 | 837 |
| 正样本(label_type==2) | 12.4% |
| 负样本(label_type==1) | 87.6% |
| timestamp 范围 | 1772725000 ~ 1772725781 (2026-03-05, ~13分钟窗口) |
| label_time 范围 | 1772725027 ~ 1772725910 |
| timestamp - label_time | min=-832, max=-2, mean=-228 |

> **注意**: demo数据时间窗口极窄（13分钟），不能完全代表全量数据的时间跨度分布。label_type仅有1和2，说明demo中全为点击后样本(CVR问题)。

### 2.1 User Int Features 分析

**基数分布 (Vocab Size, 括号内为slot数):**

| 规模 | 特征ID | 说明 |
|------|--------|------|
| 极低基数(1-5) | 1, 3, 49, 50, 58, 60(×2), 92, 95-109, 100, 107 | 二元/三元标志位，如性别、开关状态 |
| 低基数(6-50) | 4, 55, 59, 63(×11), 64(×18), 82, 89-91(×10), 93 | 类目型特征，如年龄段、地区编码 |
| 中基数(51-500) | 48, 51, 52, 53, 57, 65(×49), 86 | 细分属性，如职业、兴趣标签 |
| 高基数(501-3000) | 54(2844), 56(1435), 66(×66), 15(×13), 62(×5) | 细粒度用户标识特征 |
| 超基数(>3000) | 80(×5, vocab未知) | 原始ID特征 |

**Demo实测发现：**
- `user_int_feats_1`: non-zero占比100%, 实际唯一值=3, 范围[1,4]
- `user_int_feats_15`: array, avg_len=3.2, median=3, 13.9%为空
- `user_int_feats_60`: array, avg_len=0.6, median=0, **59.2%为空** — 稀疏度极高
- 特征基数差异极大，从1到2844+，需要分层处理

### 2.2 User Dense Features 分析

**两类Dense特征：**

1. **独立Embedding特征 (2个)**
   - `user_dense_feats_61` (256维): SUM embedding，实际维度256，value range=[-0.25, 0.20]
   - `user_dense_feats_87` (320维): LMF4Ads embedding，value range=[-0.68, 0.68]

2. **对齐特征 (8个)**
   - 与user_int_feats_{62-66, 89-91}一一对应，维度完全一致
   - 含义：对离散ID对应的统计量（如停留时长、点击率、得分等连续值）

**使用建议：**
- Dense特征当前仅通过线性投影转为1个NS token，信息压缩严重
- 对齐特征可以采用Cross-Attention方式强化其与离散特征的交互

### 2.3 Item Int Features 分析

**基数分布：**

| 特征ID | Vocab | Slot | 推测含义 |
|--------|-------|------|----------|
| 5 | 326 | 1 | 一级类目 |
| 6 | 978 | 1 | 二级类目 |
| 7 | 2807 | 1 | 三级类目 |
| 8 | 2432 | 1 | 品牌/店铺 |
| 9 | 38 | 1 | 物品状态 |
| 10 | 310 | 1 | 物品属性A |
| 11 | 32506 | 20 | 物品多标签(高基数) |
| 12 | 2778 | 1 | 物品属性B |
| 13 | 9 | 1 | 物品类型 |
| 16 | 35260 | 1 | 原始物品ID/物料ID |
| 81 | 3 | 1 | 标志位 |
| 83 | 32 | 1 | 物品属性C |
| 84 | 227 | 1 | 物品属性D |
| 85 | 1002 | 1 | 物品属性E |

**Demo实测发现：**
- `item_int_feats_11`: array, avg_len=2.1, max_len=20, **43.9%为空** — 大量物品无多标签
- `item_int_feats_16` (35260 vocab): 最高基数单值特征，推测为物料ID

### 2.4 Domain Sequence Features 分析（核心）

#### 2.4.1 序列长度分布（实测）

| 域 | 平均长度 | 中位数 | P75 | P95 | Max | 空序列占比 |
|----|---------|--------|-----|-----|-----|-----------|
| domain_a | 701.1 | 578 | 1118 | 1673 | 1888 | 0.5% |
| domain_b | 570.8 | 405 | 923 | 1563 | 1952 | 1.2% |
| domain_c | 449.4 | 322 | 533 | 1214 | 3894 | 0.2% |
| domain_d | 1099.9 | 1036 | 1687 | 2451 | 3951 | **8.0%** |

#### 2.4.2 序列截断灾难（实测关键发现）

默认 truncation (a:256, b:256, c:512, d:512) 对 demo 数据的实际影响：

| 域 | 截断值 | 被截断样本占比 | 被截断样本平均原长 | 平均信息损失 |
|----|--------|-------------|-----------------|------------|
| domain_a | 256 | **71.9%** | 926 | **262%** |
| domain_b | 256 | **62.4%** | 853 | **233%** |
| domain_c | 512 | 26.6% | 991 | 94% |
| domain_d | 512 | **71.8%** | 1460 | **185%** |

> **警告**: 默认 truncation 对 domain_a/b/d 来说过于激进。71.9%的 domain_a 样本被截断，平均丢失了262%的信息。这可能是 baseline 性能的瓶颈之一。

**建议调整：**
- domain_a: 256 → 1024 或更多
- domain_b: 256 → 768 或更多  
- domain_d: 512 → 1536 或更多

#### 2.4.3 序列特征类型识别（实测确认）

**关键发现: 时间戳特征隐藏在序列特征中！**

4个行为域各有一个特征的实际语义是Unix时间戳（秒级），但当前 schema.json 将其标记为普通离散特征、且 `ts_fid` 设为 `null`，导致时间信息被浪费。

| 域 | 时间戳FID | Vocab(名义) | 实际值范围 | 对应日期范围 | 实际unique数 |
|----|-----------|------------|-----------|-------------|-------------|
| domain_a | **seq_39** | 1,772,725,488 | [1712419199, 1772725487] | 2024-04-06 ~ 2026-03-05 | 321,733 |
| domain_b | **seq_67** | 1,772,725,643 | [1752645918, 1772725642] | 2025-07-16 ~ 2026-03-05 | 421,843 |
| domain_c | **seq_27** | 1,772,725,681 | [1717589186, 1772725680] | 2024-06-05 ~ 2026-03-05 | 441,676 |
| domain_d | **seq_26** | 1,772,725,621 | [1726916040, 1772725620] | 2024-09-21 ~ 2026-03-05 | 50,828 |

**验证**: 所有序列事件时间戳 ≤ 当前交互 timestamp（无时间泄漏）。

#### 2.4.4 其他序列特征的语义推断

| 域 | FID | Vocab | 实际Unique | 推测语义 |
|----|-----|-------|-----------|---------|
| domain_a | seq_40 | 19 | 16 | **Action Type** (行为类型) |
| domain_a | seq_41 | 12 | 9 | Action子类型 |
| domain_a | seq_46 | 18 | 11 | 行为标志 |
| domain_d | seq_17 | 5 | 4 | **Action Type** (4种行为) |
| domain_d | seq_25 | 15 | 10 | 渠道/场景标志 |
| domain_c | seq_32 | 7 | 6 | 行为类型 |
| domain_c | seq_33 | 4 | 3 | 行为标志 |
| domain_c | seq_47 | 278M | 287K | **Item ID** (与目标item_id有91个交集) |

**各域Action Type特征映射：**

| 域 | Action Type FID | 类别数 | 行为粒度推测 |
|----|----------------|--------|------------|
| domain_a | seq_40 | 16 | 细粒度行为(16种) |
| domain_b | seq_68 | 22 | 中等粒度行为(22种) |
| domain_c | seq_32 | 6 | 粗粒度行为(6种) |
| domain_d | seq_17 | 4 | 极粗粒度行为(4种) |

#### 2.4.5 物品ID在序列中的存在形式

| 域 | 可能的Item ID FID | Vocab | Unique | 与目标item_id交集 |
|----|------------------|-------|--------|-----------------|
| domain_c | seq_47 | 278M | 287K | **91个** |
| domain_b | seq_69 | 143M | 192K | 待确认 |
| domain_a | seq_38 | 1.2M | 18K | 待确认 |
| domain_d | seq_23 | 674K | 123K | 1个 |

> domain_c_seq_47 与 目标 item_id 有91个交集，表明它大概率是序列中的物品ID。这为目标物品在用户历史序列中的定位(history matching)提供了可能。

### 2.5 标签与时间分析

**Label定义（demo实测）：**
- `label_type == 2` 为正样本（转化），12.4%
- `label_type == 1` 为负样本（点击但未转化），87.6%
- Demo数据无 label_type==0，说明全量数据可能包含曝光层
- CVR任务：点击后的转化率预估

**时间特征现状：**
- 全局 `timestamp` 和 `label_time` 存在、无缺失
- 序列级时间戳以 **普通序列特征** 形式存在（见2.4.3），未被标记为 ts_fid
- 当前baseline的 time-diff 计算依赖于 ts_fid 配置，**由于 ts_fid 为 null，时间差分桶完全未生效**

---

## 三、当前Baseline时间特征使用分析

### 3.1 Baseline中的时间处理流程

```python
# dataset.py: L110-121
BUCKET_BOUNDARIES = np.array([5, 10, ..., 31536000])  # 64个边界
NUM_TIME_BUCKETS = 65  # 0=padding + 64个桶

# dataset.py: L630-665 时间差分桶计算
# 依赖 ts_fid 配置来定位时间戳列
# 由于 ts_fid = null，此分支不执行！
if ts_ci is not None:   # ← 永远为 False!
    ...

# model.py: L1376-1378
self.time_embedding = nn.Embedding(65, d_model, padding_idx=0)
# 这个embedding被创建但从未接收有效输入（因为seq_time_buckets全为0）
```

### 3.2 致命问题确认

**当前时间特征的3个致命缺陷：**

1. **时间戳覆没**: 序列时间戳特征(seq_39/67/27/26)存在于Parquet中，但 `schema.json` 中 `ts_fid: null`，导致时间特征**完全未被利用**
2. **时间戳被当作类别ID**: 这些时间戳特征vocab高达17.7亿，以超高基数离散特征的形式被送入Embedding，信息扭曲严重
3. **time_embedding不工作**: 65桶的 time_embedding 被创建但从未接收到有效bucket_id（全为0=padding）

### 3.3 现有方案的局限性

1. **无相对时间**: 缺少每个序列事件到当前交互的时间差信息
2. **无绝对时间**: 缺少事件发生时的日期-时间上下文
3. **无时间序列表征**: 时间信息完全未影响序列建模
4. **无周期性特征**: 缺少小时/天/周等周期性时间模式
5. **NS特征无时间维度**: User/Item特征不考虑时间动态性

---

## 四、时间特征优化方案

### 4.0 前置修复：激活基础时间特征（立即执行，0改造代价）

**修复 schema.json，让 time-diff 时序差分桶立即生效：**

```json
// 修改 schema.json 中的 seq 配置
"seq": {
    "domain_a": {
        "prefix": "domain_a_seq",
        "ts_fid": 39,           // 改为 39, 原来是 null
        "features": [
            [38, 1201293],
            [39, 1772725488],   // 时间戳列
            // ... 其余不变
        ]
    },
    "domain_b": {
        "prefix": "domain_b_seq",
        "ts_fid": 67,           // 改为 67
        // ...
    },
    "domain_c": {
        "prefix": "domain_c_seq",
        "ts_fid": 27,           // 改为 27
        // ...
    },
    "domain_d": {
        "prefix": "domain_d_seq",
        "ts_fid": 26,           // 改为 26
        // ...
    }
}
```

**注意事项：**
- 修复后，需要将 `seq_vocab_sizes` 中时间戳列移除（不再送入Embedding），否则会浪费显存
- dataset.py 的 `sideinfo_fids` 逻辑已正确排除了 ts_fid（L323行），但模型侧仍然可能尝试创建它的Embedding
- 修复后 `seq_time_buckets` 将获得有效值，time_embedding 将实际工作

### 4.1 序列时间特征 (Sequence Temporal Features)

#### 4.1.1 多尺度时间桶 (Multi-Scale Time Buckets)

将当前统一的64个bucket扩展为多尺度方案：

```python
# 新增: dataset.py
SCALE_BOUNDARIES = {
    'fine':   np.array([5, 10, 15, 30, 60, 120, 300, 600]),
    'medium': np.array([900, 1800, 3600, 7200, 14400, 43200, 86400]),
    'coarse': np.array([172800, 604800, 2592000, 7776000, 31536000]),
}

def multi_scale_time_bucketing(time_diff, boundaries_dict):
    """
    多尺度时间分桶:
    - fine:   5s~600s   (8+1=9 buckets)  → 秒->分钟级，捕获即时反应模式
    - medium: 15m~24h   (7+1=8 buckets)  → 小时-天级，捕获中期行为节奏  
    - coarse: 2d~365d   (5+1=6 buckets)  → 天-年级，捕获长期兴趣演化
    总计: 9+8+6=23 buckets，相比原来的65大幅压缩
    """
    results = {}
    for scale, boundaries in boundaries_dict.items():
        raw = np.clip(
            np.searchsorted(boundaries, time_diff.ravel()),
            0, len(boundaries) - 1,
        )
        results[scale] = raw.reshape(time_diff.shape) + 1
    return results
```

**模型侧改动：**

```python
# model.py 中扩展 _embed_seq_domain
# 将3个尺度的time bucket embedding分别加入token_emb
if hasattr(self, 'time_embs_fine'):
    token_emb = (token_emb 
                 + self.time_embs_fine(time_bucket_fine)
                 + self.time_embs_medium(time_bucket_medium)
                 + self.time_embs_coarse(time_bucket_coarse))
```

#### 4.1.2 事件间隔特征 (Event Interval Features)

```python
# 新增: dataset.py
def compute_event_intervals(ts_padded, padding_mask, max_len):
    """
    计算序列中相邻事件间的时间间隔
    
    shape: (B, max_len)
    - intervals[i,0] = 0  (第一个事件无前驱)
    - intervals[i,k] = ts[i,k] - ts[i,k-1]  (k>0)
    """
    intervals = np.zeros_like(ts_padded)
    intervals[:, 1:] = np.maximum(ts_padded[:, 1:] - ts_padded[:, :-1], 0)
    intervals[padding_mask] = 0
    return intervals
```

事件间隔可以映射到另一个信息维度——**用户的行为节奏**（快速浏览 vs 深度阅读 vs 长时间间隔后的回归）。

#### 4.1.3 时间感知注意力 (Time-Aware Attention)

让时间差直接影响注意力权重，近期行为获得更高关注：

```python
# model.py: 在 TransformerEncoder 或 LongerEncoder 中注入
def time_decay_self_attention(Q, K, V, time_diffs, padding_mask):
    """
    时间感知自注意力:
    
    attention_score[i,j] = Q_i·K_j / sqrt(d) + log(decay(|t_i-t_j|))
    
    其中 decay(dt) = exp(-dt / tau), tau 为可学习参数
    越近的事件间的注意力权重越高
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 时间衰减矩阵 (B, 1, L_q, L_k)
    td = torch.abs(time_diffs)  # (B, L)
    td_matrix = td.unsqueeze(1) + td.unsqueeze(2)  # 近似: 但实际需要 (B, L, L) pairwise diff
    # 更准确: pairwise time diff
    td_pairwise = td.unsqueeze(2) - td.unsqueeze(1)  # (B, L, L)
    decay = -torch.abs(td_pairwise) / tau  # tau可学习
    decay = decay.unsqueeze(1)  # (B, 1, L, L)
    
    scores = scores + decay
    # ... softmax + weighted sum
```

#### 4.1.4 时间戳Embedding替换 (Timestamp-aware Embedding)

将时间戳特征从离散Embedding改为连续时间编码：

```python
# model.py: 新增时间编码器
class TimestampEncoder(nn.Module):
    """
    将Unix时间戳转为可学习的连续表征，替代当前的Embedding查表
    
    编码维度:
    - Linear position encoding (sin/cos, like positional encoding)
    - 可学习的MLP映射 [minute, hour, dow, ...] → d_emb
    """
    def __init__(self, d_model):
        super().__init__()
        # 周期性时间特征: hour(24), dow(7), dayofmonth(31)
        self.hour_emb = nn.Embedding(24, d_model // 4)
        self.dow_emb = nn.Embedding(7, d_model // 4)
        self.month_emb = nn.Embedding(12, d_model // 4)
        self.fusion = nn.Linear(d_model * 3 // 4, d_model)
    
    def forward(self, timestamps):
        import datetime
        # Convert timestamps to datetime features
        # hour, dow, month etc
        ...
```

#### 4.1.5 时间窗口序列表征 (Temporal Window Pooling)

```python
# 新增: model.py
def temporal_window_pooling(seq_tokens, time_diffs, windows=[300, 900, 3600, 86400]):
    """
    将序列按时间窗口分别池化:
    
    窗口: 5min, 15min, 1h, 24h
    
    对每个窗口做 time-decayed attention pooling:
    - 越近的事件权重越高
    - 输出每个窗口的聚合表征 → concat得到多尺度时间时序画像
    
    Returns: (B, len(windows) * d_model)
    """
    pooled = []
    for w in windows:
        mask = (time_diffs <= w).float().unsqueeze(-1)  # (B, L, 1)
        within_window = seq_tokens * mask
        
        # Time-decayed mean pooling
        decay = torch.exp(-time_diffs / tau)
        decay = decay.unsqueeze(-1) * mask.squeeze(-1)
        weights = decay / (decay.sum(dim=1, keepdim=True) + 1e-8)
        
        pooled.append((within_window * weights.unsqueeze(-1)).sum(dim=1))
    
    return torch.cat(pooled, dim=-1)
```

### 4.2 Item侧时间特征 (Item-Side Temporal Features)

#### 4.2.1 Item历史交互时间统计（基于序列内挖掘）

```python
# 新增: dataset.py 或离线特征工程
def compute_item_temporal_stats(user_seq_fids, timestamp_fid_idx, seq_timestamps, current_time):
    """
    从用户历史序列中提取目标物品的时间统计:
    
    - recency: 最近一次出现距今时间
    - frequency_per_day: 日均交互频率
    - last3_intervals: 最近3次交互的时间间隔
    - position_ratio: 最近出现位置/序列总长
    """
    B, C, L = user_seq_fids.shape
    
    # 找到每个行为域中的物品ID列
    # domain_c seq_47 已确认为物品ID
    
    # 对于每条样本，统计目标物品在序列中的出现模式
    ...
```

#### 4.2.2 Item热度与时间交叉（基于全局统计，离线预计算）

建议离线预计算导出为新特征列：

| 新Item特征 | 计算方式 | 窗口 | 编码方式 |
|-----------|---------|------|---------|
| item_hour_exposure_cnt | 按小时统计曝光量 | 最近24h | 24维向量 |
| item_dow_avg_exposure | 按星期统计平均曝光 | 最近30d | 7维向量 |
| item_ctr_trend | 线性拟合CTR趋势 | 最近7d | 斜率+截距 |
| item_lifecycle_stage | 发布时间分类 | 全局 | 3类(新/成熟/尾) |

### 4.3 Pair特征处理 (Pair特征处理)

#### 4.3.1 时间衰减的行为-目标匹配

```python
# model.py: 在 output_proj 前注入
def temporal_behavior_matching(seq_tokens, seq_time_diffs, target_item_emb, current_time):
    """
    时间衰减的用户行为序列与目标物品的匹配:
    
    1. 计算序列中每个item_emb与目标item_emb的相似度
    2. 用时间衰减加权（近期交互权重高）
    3. 输出加权相似度向量作为pair特征
    """
    # 相似度矩阵 (B, L)
    B, L, D = seq_tokens.shape
    target_expanded = target_item_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)
    similarities = F.cosine_similarity(seq_tokens, target_expanded, dim=-1)  # (B, L)
    
    # 时间衰减加权
    time_weights = torch.exp(-seq_time_diffs / tau)  # (B, L)
    
    # 加权聚合
    weighted_sim = (similarities * time_weights).sum(dim=1) / (time_weights.sum(dim=1) + 1e-8)
    
    return weighted_sim
```

#### 4.3.2 时间交叉特征融合

```python
# model.py: 构建时间感知的用户-物品交叉向量
def temporal_cross_fusion(user_ns_tokens, item_ns_tokens, seq_temporal_context):
    """
    三路时间交叉融合:
    
    cross = TimeGate(t) ⊙ (user_NS ⊙ item_NS + seq_rep ⊙ item_NS)
    
    其中 TimeGate(t) 由当前timestamp编码而来
    """
    # 用户-物品静态交叉
    static_cross = user_ns_tokens * item_ns_tokens  # (B, D)
    
    # 用户历史行为-物品动态交叉
    dynamic_cross = seq_behavior * item_ns_tokens  # (B, D)
    
    # 时间门控
    time_gate = torch.sigmoid(time_encoder(current_timestamp))  # (B, D)
    
    fusion = time_gate * (static_cross + dynamic_cross)
    return fusion
```

### 4.4 综合实施计划

#### 4.4.1 三阶段实施路径

```
Phase 0 (立即修复, <1天):
├── 修复 schema.json: 正确设置 ts_fid
├── 验证 seq_time_buckets 获得有效值
└── 预期: time-diff embedding 首次生效

Phase 1 (初级增强, 1-2天):
├── P1-1: 多尺度时间桶 (4.1.1)
├── P1-2: 事件间隔特征 (4.1.2)  
├── P1-3: 序列长度上限提升 (从256→1024等)
└── 预期: AUC 提升 0.3-1.0%

Phase 2 (中级增强, 2-5天):
├── P2-1: 时间感知注意力 (4.1.3)
├── P2-2: Item时间统计特征 (4.2.1/4.2.2)
├── P2-3: Pair时间交叉特征 (4.3.1/4.3.2)
└── 预期: AUC 提升 0.5-2.0%

Phase 3 (高级增强, 5-10天):
├── P3-1: 时间戳编码器替代Embedding (4.1.4)
├── P3-2: 时间窗口序列表征 (4.1.5)
├── P3-3: 动作类型感知的序列加权
└── 预期: AUC 提升 1.0-3.0%
```

#### 4.4.2 实施检查清单

- [ ] **Phase 0: 修复 ts_fid** — 在 schema.json 中将 `domain_a.ts_fid=39`, `domain_b.ts_fid=67`, `domain_c.ts_fid=27`, `domain_d.ts_fid=26`
- [ ] **Phase 0: 验证** — 运行一次forward确认 `seq_time_buckets` 不再全为零
- [ ] **Phase 1: 多尺度时间桶** — dataset.py 增加 multi_scale_bucketing + model.py 增加3套 time_embedding
- [ ] **Phase 1: 事件间隔** — dataset.py 计算 interval + model.py 增加 interval_embedding
- [ ] **Phase 1: 序列长度** — 提高 seq_max_lens，至少 domain_a:1024, domain_d:1536
- [ ] **Phase 2: 时间感知注意力** — 更新 TransformerEncoder / LongerEncoder 的 attention score 计算
- [ ] **Phase 2: Item时间特征** — 离线预计算 item统计特征并作为新列
- [ ] **Phase 2: Pair交叉** — 在 PCVRHyFormer 的 output_proj 前拼接 time-weighted cross features

---

## 五、数据质量与预处理建议

### 5.1 Demo数据分析总结

| 发现 | 严重程度 | 影响 |
|------|---------|------|
| ts_fid 未配置，时间信息完全浪费 | **致命** | time_embedding零作用 |
| 时间戳被当作17.7亿词汇量的离散特征 | **严重** | Embedding性能与内存浪费 |
| 序列长度截断过激(256截断72%样本) | **严重** | 丢失大量行为信号 |
| domain_d 空序列占比8% | **中等** | 部分用户无此域行为 |
| item多标签特征(iii_11) 43.9%为空 | **低** | 可正常处理 |

### 5.2 全量数据应执行的检查

```python
# 生产环境部署前必查项
1. 正负样本比(全量数据 vs demo的12.4%)
2. 时间跨度分布 (全量数据可能跨越数周/数月)
3. timestamp - label_time 的延迟分布
4. 序列长度在全量数据中的分布 (确认截断值)
5. 用户重复出现情况 (demo为每人1条，全量可能不同)
6. 训练/验证集时间切分 (避免未来信息泄漏)
```

### 5.3 离线预计算统计特征

建议在数据预处理的 PySpark/Hive 阶段新增以下列：

| 新特征名 | 计算逻辑 | 窗口 | 数据类型 |
|---------|---------|------|---------|
| item_7d_exposure_cnt | 全局统计 | 7天滑动窗口 | int64 |
| item_7d_ctr | 全局统计 | 7天 | float32 |
| item_30d_cvr | 全局统计 | 30天 | float32 |
| user_item_last_interact_sec | 用户序列交叉 | 全历史 | int64 |
| item_publish_age_sec | timestamp - 发布时间 | — | int64 |

---

## 六、总结

### 6.1 数据核心特征

- **多域异构长序列**：4个行为域，实际平均长度449-1100，远超当前截断值
- **时间信息隐藏在序列中**：每个域的时间戳作为普通离散特征存在，需修复 schema.json 激活
- **丰富的NS特征**：46+14+10=70个非序列特征，涵盖用户/物品/上下文多维度
- **CVR分类任务**：正负比约1:7，标准的转化率预估问题

### 6.2 最优先动作：修复 ts_fid

这是**零成本、确定性收益**的举措：

1. 修改 `schema.json` 的4个 `ts_fid: null` → `39/67/27/26`
2. 确认 `seq_vocab_sizes` 排除时间戳列（避免为17.7亿vocab创建Embedding）
3. 重新运行训练，time-diff bucket embedding 将首次生效

### 6.3 预期收益评估

| 阶段 | 改造内容 | 改造代价 | 预期AUC提升 |
|------|---------|---------|------------|
| Phase 0 | 修复 ts_fid | 改2行JSON | +0.1~0.5% |
| Phase 1 | 多尺度时间+间隔+长序列 | 改100行Python | +0.5~1.5% |
| Phase 2 | 时间注意力+Item/Pair时间 | 改300行Python | +1.0~3.0% |
| Phase 3 | 高级编码+窗口池 | 改500行Python + 离线计算 | +1.5~4.0% |