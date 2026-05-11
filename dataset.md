Our dataset is a large-scale industrial dataset constructed from real-world advertising logs. It consists of two main components: (1) user behavior sequences and (2) non-sequential multi-field features.

User behavior sequences contain interaction events between users and items (e.g., exposure, click, conversion), each associated with side information such as timestamps and action types. Multi-field features include user attributes, item attributes, contextual signals, and cross features.

To ensure fairness and protect privacy, all sparse features are represented as anonymized integer IDs, and dense features are provided as fixed-length float vectors. No raw content (e.g., text, image, URL) or personally identifiable information is released.

The first-round dataset for Academic Track contains 1 million samples, while the Industrial Track contains 2 million samples. And they both use a flat column layout, where all features are stored as individual top-level columns instead of nested structs/arrays. Here is a detailed explanation of the dataset schema, including the column categories, data types, and descriptions:

## Columns

The 120 columns fall into 6 categories:

| Category | Count | Dataset | Description |
|----------|-------|---------|-------------|
| ID & Label | 5 | int64 / int32 | Core identifiers, label, and timestamp. |
| User Int Features | 46 | int64 / list<int64> | Discrete user features, including both single-value scalar features (such as age, gender, etc.) and multi-value array features (like marital status, etc.), describing user basic attributes and preferences. |
| User Dense Features | 10 | list<float> | Continuous-valued user features, including embeddings and other aligned signals for some corresponding integer features. |
| Item Int Features | 14 | int64 / list<int64> | Discrete item features, including item categories, types, and other basic information, as well as multi-label information for items. |
| Domain Sequence Features | 45 | list<int64> | Behavioral sequence features from 4 domains. |

## Detailed Column Schema
**ID & Label Columns (5 columns)**： All these 5 columns have no null value.

| Column | user_id | item_id | label_type | label_time | timestamp |
|--------|---------|---------|------------|------------|-----------|
| Date Type | int64 | int64 | int32 | int64 | int64 |

**User Int Features (46 columns)**
- user_int_feats_{1, 3, 4, 48-59, 82, 86, 92-109}: Scalar int64, total 35 columns.
- user_int_feats_{15, 60, 62-66, 80, 89-91}: Array list<int64>, total 11 columns.

**User Dense Features (10 columns)**
- user_dense_feats_{61, 87}: Array list<float>, total 2 columns, representing user embedding features (SUM , LMF4Ads).
- user_dense_feats_{62-66, 89-91}: Array list<float>, total 8 columns, corresponding to the integer features user_int_feats_{62-66, 89-91}, with the same length.
    - An Example:
<br>user_int_feats_62: [1, 2, 3], user_dense_feats_62: [10.5, 20, 15.5]
<br>Explanation: Here, the two arrays are aligned element by element. For example, the 1st element in user_int_feats_62 (value 1) denotes a specific entity or category, while the 1st element in user_dense_feats_62 (value 10.5) provides some statistics for that element, such as a dwell time, a score/probability.

**Item Int Features (14 columns)**
- item_int_feats_{5-10, 12-13, 16, 81, 83-85}: Scalar int64, total 13 columns.
- item_int_feats_{11}: Array list<int64>, total 1 column.

**Domain Sequence Features (45 columns)**
list<int64> sequences from 4 behavioral domains:
- domain_a_seq_{38-46}: 9 columns
- domain_b_seq_{67-79, 88}: 14 columns
- domain_c_seq_{27-37, 47}: 12 columns
- domain_d_seq_{17-26}: 10 columns