#!/bin/bash


mkdir -p src/pdr/filters src/correction src/utils tests/test_pdr/test_filters tests/test_correction tests/test_utils data/{raw,processed,maps} docs examples

# srcディレクトリ内のファイル作成
touch src/__init__.py
touch src/main.py

# pdrディレクトリ内のファイル作成
touch src/pdr/__init__.py
touch src/pdr/step_detector.py
touch src/pdr/orientation_estimator.py
touch src/pdr/trajectory_calculator.py
touch src/pdr/pdr_estimator.py
touch src/pdr/three_dimensional_estimator.py

# pdr/filtersディレクトリ内のファイル作成
touch src/pdr/filters/__init__.py
touch src/pdr/filters/base_filter.py
touch src/pdr/filters/simple_filter.py

# correctionディレクトリ内のファイル作成
touch src/correction/__init__.py
touch src/correction/base_corrector.py
touch src/correction/map_matching_corrector.py
touch src/correction/ble_beacon_corrector.py
touch src/correction/stable_walking_corrector.py
touch src/correction/trajectory_corrector.py

# utilsディレクトリ内のファイル作成
touch src/utils/__init__.py
touch src/utils/data_loader.py
touch src/utils/data_preprocessor.py
touch src/utils/visualizer.py
touch src/utils/barometer_utils.py

# testsディレクトリ内のファイル作成
touch tests/__init__.py

# test_pdrディレクトリ内のファイル作成
touch tests/test_pdr/__init__.py
touch tests/test_pdr/test_step_detector.py
touch tests/test_pdr/test_orientation_estimator.py
touch tests/test_pdr/test_trajectory_calculator.py
touch tests/test_pdr/test_pdr_estimator.py
touch tests/test_pdr/test_three_dimensional_estimator.py

# test_pdr/test_filtersディレクトリ内のファイル作成
touch tests/test_pdr/test_filters/__init__.py
touch tests/test_pdr/test_filters/test_simple_filter.py

# test_correctionディレクトリ内のファイル作成
touch tests/test_correction/__init__.py
touch tests/test_correction/test_map_matching_corrector.py
touch tests/test_correction/test_ble_beacon_corrector.py
touch tests/test_correction/test_stable_walking_corrector.py
touch tests/test_correction/test_trajectory_corrector.py

# test_utilsディレクトリ内のファイル作成
touch tests/test_utils/__init__.py
touch tests/test_utils/test_data_loader.py
touch tests/test_utils/test_data_preprocessor.py
touch tests/test_utils/test_visualizer.py
touch tests/test_utils/test_barometer_utils.py

# docsディレクトリ内のファイル作成
touch docs/api_reference.md
touch docs/user_guide.md

# examplesディレクトリ内のファイル作成
touch examples/basic_usage.py
touch examples/advanced_usage.py

# ルートディレクトリのファイル作成
touch README.md

echo "Project structure created successfully!"