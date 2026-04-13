## Phase 1: Basic Delaunay
- [x] Nuxt + Three.js + Delaunator セットアップ
- [x] Webcam → グリッド点 → Delaunay → ローポリ表示
- [ ] ブラウザで動作確認

## Phase 2: Face Landmarks
- [ ] MediaPipe FaceLandmarker (478点) 統合
- [ ] 顔の目・口・眉の密な点群をグリッドに追加
- [ ] 顔部分のポリゴン密度向上を確認

## Phase 3: Depth
- [ ] Depth Anything V2 Small (WebGPU/ONNX) 統合
- [ ] 深度マップから Z 座標を設定
- [ ] PerspectiveCamera に切替 → 3D表示

## Phase 4: Full Body
- [ ] PoseLandmarker (33点 + セグメンテーションマスク)
- [ ] シルエット輪郭 (Canny + findContours) をグリッドに追加
- [ ] HandLandmarker (21点x2)

## Phase 5: Visual Polish
- [ ] flat shader (centroid UV) でGPU完結のローポリ着色
- [ ] ワイヤーフレーム表示切替
- [ ] 音声連動（マイク音量 → ポリゴン密度）

## Phase 6: Deferred Rendering
- [ ] G-Buffer (position/normal/albedo) 出力
- [ ] SSAO
- [ ] Bloom / Rim Lighting

## Phase 7: Export
- [ ] GLTFExporter でメッシュ書き出し
- [ ] 背面追加 + ウォータータイト化
- [ ] 3Dプリント用 STL エクスポート
