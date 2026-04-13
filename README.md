# Delaunay Depth Mesh

Webcam映像をリアルタイムにDelaunay三角形分割してローポリ化するブラウザスケッチ。

2012年にopenFrameworks + Kinectで制作した[.hito project](https://github.com/rettuce/.hito-project)を、Webカメラのみ・ブラウザ完結で再構築したもの。

**Demo**: https://rettuce.github.io/delaunay-depth-mesh/

**元ネタ記事**: https://rettuce.blog/2012/12/11/kinect_of_delaunay/

## やっていること

- グリッド点 + Sobelエッジ検出 + MediaPipe顔ランドマーク(~130点) → Delaunay三角形分割 → 重心色サンプリングでローポリ着色
- selfie_segmenterで人物マスク（元プロジェクトのKinect深度クリッピングに相当）
- Depth Anything V2 Smallで単眼深度推定 → Z押し出しで3D化
- PerspectiveCamera + OrbitControlsでマウス操作可能な3Dビュー

## 技術スタック

- Nuxt 4 + Three.js + Delaunator
- MediaPipe Tasks Vision (FaceLandmarker + ImageSegmenter)
- Depth Anything V2 Small (Transformers.js / WebGPU)

## 注意

- 初回アクセス時にDepth推論モデル(~97MB)のダウンロードが走る
- WebGPU対応ブラウザ推奨（非対応時はWASMフォールバック）
- Webcamの許可が必要

## ローカル実行

```bash
pnpm install
pnpm dev
```

## 元プロジェクトとの差分

| | 2012 (.hito project) | 2026 (このスケッチ) |
|---|---|---|
| 深度取得 | Kinect赤外線センサー | Depth Anything V2 (ML推定) |
| 特徴点 | Cannyエッジ検出 | MediaPipe FaceLandmarker 478点 |
| 人物抽出 | Kinect深度クリッピング | selfie_segmenter |
| 実行環境 | openFrameworks (C++) | ブラウザ (Nuxt + Three.js) |
| 3D操作 | 固定カメラ | OrbitControls |
