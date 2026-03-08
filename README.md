# 💊 薬物相互作用予測AIシステム
## Drug-Drug Interaction (DDI) Prediction System

薬剤師の専門知識と機械学習を融合した、薬物相互作用予測システム

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 プロジェクト概要

このプロジェクトは、**機械学習**を用いて2つの薬物間の相互作用リスクを予測するシステムです。薬剤師としての**臨床知識**と**データサイエンス技術**を組み合わせ、実臨床で価値のある予測モデルを構築しています。

### 予測する相互作用の重症度
- **Major（重大）**: 生命を脅かす可能性、入院が必要
- **Moderate（中等度）**: 患者状態の悪化、治療調整が必要
- **Minor（軽度）**: 軽微な影響、経過観察
- **None（なし）**: 臨床的に意義のある相互作用なし

---

## ✨ 主な特徴

### 1. 薬学的特徴量の実装
実際の薬物相互作用メカニズムに基づいた特徴量設計：

**薬物動態学的特徴**
- 脂溶性（LogP値）
- 分子量
- タンパク結合率
- 半減期

**代謝・トランスポーター**
- CYP酵素（CYP3A4, 2D6, 2C9, 2C19）の基質・阻害
- P糖タンパク質の相互作用
- 治療域の狭さ（Narrow Therapeutic Index）

**患者因子**
- 年齢
- 腎機能（eGFR）
- 肝機能
- 併用薬数

### 2. 複数の機械学習モデル
- **Random Forest**: アンサンブル学習、特徴量重要度分析
- **Gradient Boosting**: 高精度な予測
- **Logistic Regression**: 解釈可能性

### 3. 包括的な評価
- 混同行列（Confusion Matrix）
- Precision, Recall, F1-Score
- クラス別パフォーマンス分析
- 特徴量重要度の可視化

---

## 🚀 技術スタック

| カテゴリ | 技術 |
|---------|------|
| **言語** | Python 3.8+ |
| **機械学習** | scikit-learn, Random Forest, Gradient Boosting |
| **データ処理** | pandas, NumPy |
| **可視化** | Matplotlib, Seaborn |
| **評価指標** | Classification metrics, Confusion Matrix |

---

## 📁 プロジェクト構造

```
drug-interaction-prediction/
├── drug_interaction_predictor.py    # メインスクリプト
├── README.md                         # このファイル
├── requirements.txt                  # 依存ライブラリ
├── PORTFOLIO_GUIDE.md               # ポートフォリオとしての活用法
└── outputs/                          # 出力ファイル
    ├── 01_class_distribution.png
    ├── 02_model_comparison.png
    ├── 03_confusion_matrix.png
    ├── 04_feature_importance.png
    └── drug_interaction_data.csv
```

---

## 🔧 セットアップ

### 1. リポジトリのクローン
```bash
git clone https://github.com/yourusername/drug-interaction-prediction.git
cd drug-interaction-prediction
```

### 2. 仮想環境の作成（推奨）
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. 実行
```bash
python drug_interaction_predictor.py
```

---

## 📊 実行結果

### モデル性能（テストセット）

| モデル | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Random Forest | 0.578 | 0.434 | 0.578 | 0.461 |
| Gradient Boosting | 0.573 | 0.466 | 0.573 | 0.476 |
| **Logistic Regression** | **0.588** | **0.453** | **0.588** | **0.484** |

### 最も重要な特徴量（Top 5）
1. **患者年齢** (0.084) - 高齢者は相互作用リスクが高い
2. **腎機能** (0.083) - 腎機能低下で薬物蓄積
3. **分子量** (0.070) - 薬物の吸収・分布に影響
4. **脂溶性** (0.069) - 膜透過性と分布容積
5. **タンパク結合率** (0.069) - 遊離型薬物濃度

---

## 🧠 薬学的知見

### 実装された相互作用メカニズム

#### 1. CYP酵素阻害
```python
# 例: CYP3A4基質薬物 + CYP3A4阻害薬
if drug1_cyp_substrate == 'CYP3A4' and drug2_cyp_inhibitor:
    → 薬物濃度上昇リスク → Major/Moderate
```

#### 2. 既知の高リスク組み合わせ
- **抗凝固薬 + NSAID**: 出血リスク増加
- **SSRI + NSAID**: 消化管出血
- **ACE阻害薬 + 利尿薬**: 低血圧、腎機能障害
- **スタチン + 抗生物質**: 横紋筋融解症

#### 3. 患者因子の考慮
- **高齢者（>75歳）**: 薬物代謝能低下
- **腎機能低下（eGFR<60）**: 薬物排泄遅延
- **多剤併用**: 相互作用リスク増加

---

## 📈 改善の余地と今後の発展

### データの拡張
1. **実データの活用**
   - DrugBank: 薬物相互作用データベース
   - TWOSIDES: 副作用データ
   - FDA Adverse Event Reporting System (FAERS)

2. **追加特徴量**
   - 薬物構造情報（SMILES、分子指紋）
   - 薬物遺伝学（Pharmacogenomics）
   - 電子カルテデータ

### モデルの高度化
1. **深層学習の適用**
   - Graph Neural Networks (GNN)
   - Transformer-based モデル
   - 薬物構造の直接学習

2. **説明可能AI（XAI）**
   - SHAP値による予測理由の説明
   - LIME: 局所的な解釈可能性
   - Attention機構の可視化

3. **マルチタスク学習**
   - 相互作用メカニズムの同時予測
   - 副作用の種類の予測
   - 推奨される代替薬の提案

### 実臨床への応用
1. **リアルタイム警告システム**
   - 電子カルテとの統合
   - 処方時の自動チェック
   - モバイルアプリ

2. **多言語対応**
   - 5ヶ国語での薬物情報提供
   - 国際的な薬物データベース統合

---

## 🎓 このプロジェクトで示せるスキル

### データサイエンススキル
✅ ドメイン知識の特徴量化  
✅ 不均衡データの扱い（クラスの偏り）  
✅ 複数モデルの比較・評価  
✅ 交差検証とハイパーパラメータチューニング  
✅ 特徴量重要度分析

### AIエンジニアリングスキル
✅ scikit-learnでの機械学習パイプライン  
✅ Random Forest、Gradient Boosting の実装  
✅ モデル評価指標の選択と解釈  
✅ 予測結果の可視化

### 薬学専門スキル
✅ 薬物動態学（PK/PD）の理解  
✅ CYP酵素システムの知識  
✅ 臨床的リスク評価  
✅ 実臨床での応用可能性

### ソフトウェアエンジニアリングスキル
✅ オブジェクト指向プログラミング  
✅ クリーンなコード設計  
✅ 詳細なドキュメンテーション  
✅ 再現可能な研究

---

## 💼 面接でのアピールポイント

### 専門性の融合
「薬剤師として培った薬物相互作用の臨床知識を、機械学習モデルの特徴量設計に活かしました。特にCYP酵素阻害や治療域の狭さなど、実際の相互作用メカニズムを特徴量として実装しています。」

### 問題解決能力
「薬物相互作用は複雑で多因子的な問題です。このプロジェクトでは、薬物特性、代謝経路、患者因子を統合的にモデル化し、臨床的に意義のある予測を目指しました。」

### 実用性への意識
「モデルの精度だけでなく、特徴量重要度を分析することで、どの因子が相互作用リスクに寄与しているかを明確にしました。これは実臨床での意思決定支援に重要です。」

### 継続的学習
「現在は教師あり学習を使用していますが、Graph Neural Networksを用いた薬物構造の直接学習や、説明可能AIによる予測根拠の提示など、さらなる改善の余地を認識しています。」

---

## 📚 参考文献・リソース

### 薬物相互作用データベース
- DrugBank: https://go.drugbank.com/
- TWOSIDES: http://tatonettilab.org/offsides/
- Drugs.com Interaction Checker

### 機械学習リソース
- scikit-learn Documentation
- "Hands-On Machine Learning" by Aurélien Géron
- Kaggle: Drug Interaction Prediction Competitions

### 薬学文献
- "Clinical Pharmacokinetics and Pharmacodynamics" by Malcolm Rowland
- FDA Drug Interaction Studies Guidelines

---

## 🤝 貢献

プルリクエストを歓迎します！改善提案や新機能のアイデアがあれば、Issueを開いてください。

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

## 👤 作成者

**古賀勇輝（Yuki Koga）**
- 資格: 薬剤師、英語医療通訳士
- 専門: 薬学 → AIエンジニア/データサイエンティスト
- 言語: 日本語、英語、中国語、韓国語、マレー語
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-profile](https://linkedin.com/in/yourprofile)

---

## ⚠️ 免責事項

このプロジェクトは**教育・研究目的**で作成されています。実際の臨床判断には使用しないでください。薬物相互作用に関する判断は、必ず医療専門家に相談してください。

---

## 🙏 謝辞

このプロジェクトは、薬剤師としての臨床経験とAI技術への情熱を組み合わせて作成しました。医療AIの発展に少しでも貢献できれば幸いです。

---

⭐ **このプロジェクトが参考になったら、スターをお願いします！**

**Keywords**: Drug-Drug Interaction, Machine Learning, Random Forest, Pharmacology, Clinical Decision Support, Healthcare AI, 薬物相互作用, 機械学習, 医療AI
