# Palkia: Indoor Positioning Library

Palkia は屋内位置推定を実現するためのPythonライブラリです。
PDR（Pedestrian Dead Reckoning）をベースとしており,マップマッチング、BLEビーコンを用いた位置補正などが行う。

## 特徴

- 歩行者デッドレコニング（PDR）
- 3次元空間での位置推定（階層対応）
- データ前処理と後処理機能 
- マップマッチングによる位置補正(未実装)
- BLEビーコンを用いた位置精度の向上(未実装)


## Makeコマンド

Poetry を使用した依存関係の解決

```bash
make install
```

リント、フォーマットチェック、型チェック
```
make ci
```