# Palkia - 屋内位置推定ライブラリ

Palkia は屋内位置推定を実現するためのPythonライブラリです。PDR（Pedestrian Dead Reckoning）をベースとしており、マップマッチング、BLEビーコンを用いた位置補正などが可能です。

## 機能

- 歩行者デッドレコニング（PDR）による位置推定
- 3次元空間での位置推定（階層対応）
- センサーデータの前処理と後処理
- マップマッチングによる位置補正
- BLEビーコンを用いた位置精度の向上
- 気圧データを用いた階層変化の検出

## 依存ライブラリ

- Python ≥ 3.10
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn

## インストール

Poetry を使用してインストールできます:

```bash
pip install palkia
```

あるいはリポジトリをクローンして直接インストールすることもできます:

```bash
git clone https://github.com/kajiLabTeam/palkia.git
cd palkia
make install
```

## 要件

- Python 3.10以上
- Poetry（依存関係管理）

## 使用例

### 基本的な使用方法

```python
from palkia.positioning.pdr.step_estimator import StepEstimator
from palkia.positioning.pdr.orientation_estimator import OrientationEstimator
from palkia.positioning.pdr.trajectory_calculator import TrajectoryCalculator
from palkia.positioning.pdr.pdr_estimator import PDREstimator
from palkia.utils.data_loader import load_sensor_data_from_log
from palkia.utils.visualizer import plot_trajectory

# センサーデータの読み込み
acc_data, gyro_data, _, _, _, _ = load_sensor_data_from_log("path/to/log_file.txt")

# PDRコンポーネントの初期化
step_estimator = StepEstimator()
orientation_estimator = OrientationEstimator()
trajectory_calculator = TrajectoryCalculator(initial_point={"x": 0, "y": 0})

# PDR推定器の初期化
pdr_estimator = PDREstimator(
    step_estimator,
    orientation_estimator,
    trajectory_calculator
)

# 軌跡の推定
trajectory = pdr_estimator.estimate_trajectory(acc_data, gyro_data)

# 推定軌跡の可視化
plot_trajectory(trajectory)
```

### 3次元位置推定

```python
from palkia.positioning.pdr.three_dimensional_estimator import ThreeDimensionalEstimator

# 3D位置推定器の初期化
three_d_estimator = ThreeDimensionalEstimator(pdr_estimator)

# 3D軌跡の推定
trajectory_3d = three_d_estimator.estimate_3d_trajectory(acc_data, gyro_data, baro_data)
```

### 位置補正の適用

```python
from palkia.positioning.correction.map_matching import MapMatchingCorrector
from palkia.utils.floor_map import FloorMap

# フロアマップの読み込み
floor_map = FloorMap(
    floor_name="1F",
    floor_map_path="path/to/floor_map.bmp",
    dx=0.01,
    dy=0.01
)

# マップマッチング補正器の初期化
map_matching = MapMatchingCorrector(floor_map)

# 軌跡の補正
corrected_trajectory = map_matching.correct_trajectory(trajectory)
```

## 開発

### 依存関係のインストール

```bash
make install
```

### テスト実行

```bash
make test
```

### リンターと型チェック

```bash
make ci  # 実行：lint, format-check, pyright
```

### フォーマット修正

```bash
make format-fix
```

## ディレクトリ構造

```
palkia/
├── .github/workflows/    # GitHub Actions CI設定
├── dist/                 # ビルド済みパッケージ
├── docs/                 # ドキュメント
│   ├── api_reference.md  # API リファレンス
│   └── user_guide.md     # ユーザーガイド
├── examples/             # 使用例
│
│
│
├── src/                  # ソースコード
   ├── main.py           # メインエントリーポイント
   └── palkia/           # コアライブラリ
       ├── __init__.py
       ├── const/        # 定数定義
       ├── positioning/  # 位置推定アルゴリズム
       │   ├── correction/ # 位置補正アルゴリズム
       │   └── pdr/      # PDR関連コンポーネント
       └── utils/        # ユーティリティ関数
```


## ライセンス

[MIT License](LICENSE)

## 貢献

バグを発見した場合や機能リクエストがある場合は、GitHubのIssueを作成してください。プルリクエストも歓迎します。


## 開発者

- [Kaji Lab Team](https://github.com/kajiLabTeam)
