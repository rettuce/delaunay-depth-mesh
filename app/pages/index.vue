<template>
  <div ref="containerEl" class="container">
    <video ref="videoEl" autoplay playsinline muted />
    <div v-if="!started" class="overlay" @click="start">
      <span>Click to Start</span>
    </div>
    <div v-if="started" class="controls">
      <label>
        Grid: {{ gridStep }}
        <input v-model.number="gridStep" type="range" min="6" max="60" step="1" @change="rebuildGrid" />
      </label>
      <label>
        Jitter: {{ jitter }}
        <input v-model.number="jitter" type="range" min="0" max="20" step="1" @change="rebuildGrid" />
      </label>
      <label class="toggle">
        <input v-model="useFace" type="checkbox" />
        Face {{ faceStatus }}
      </label>
      <label class="toggle">
        <input v-model="useEdge" type="checkbox" />
        Edge
      </label>
      <label>
        ET: {{ edgeThreshold }}
        <input v-model.number="edgeThreshold" type="range" min="5" max="300" step="5" />
      </label>
      <label>
        ED: {{ edgeSampleStep }}
        <input v-model.number="edgeSampleStep" type="range" min="2" max="16" step="1" />
      </label>
      <label class="toggle">
        <input v-model="useMask" type="checkbox" />
        Mask {{ maskStatus }}
      </label>
      <label class="toggle">
        <input v-model="useDepth" type="checkbox" />
        Depth {{ depthStatus }}
      </label>
      <label v-if="useDepth">
        Z: {{ depthScale.toFixed(1) }}
        <input v-model.number="depthScale" type="range" min="0" max="3" step="0.1" />
      </label>
      <span class="fps">{{ fps }} fps | {{ triCount }} tris</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import Delaunator from 'delaunator'
import { FaceLandmarker, FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'

const containerEl = ref<HTMLDivElement>()
const videoEl = ref<HTMLVideoElement>()
const started = ref(false)
const fps = ref(0)
const triCount = ref(0)
const gridStep = ref(50)
const jitter = ref(10)
const useFace = ref(true)
const faceStatus = ref('(loading...)')
const useEdge = ref(true)
const edgeThreshold = ref(130)
const edgeSampleStep = ref(3)
const useMask = ref(true)
const maskStatus = ref('(loading...)')
const useDepth = ref(true)
const depthStatus = ref('(loading...)')
const depthScale = ref(1.0)

// --- State ---
let renderer: THREE.WebGLRenderer
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let controls: OrbitControls
let mesh: THREE.Mesh
let geometry: THREE.BufferGeometry
let offCtx: CanvasRenderingContext2D
let offCanvas: HTMLCanvasElement
let vw = 0
let vh = 0
let animId: number

// Static grid points
let gridPoints: number[][] = []

// MediaPipe
let faceLandmarker: FaceLandmarker | null = null
let imageSegmenter: ImageSegmenter | null = null
let contourIndices: number[] = []
let cachedBinaryMask: Uint8Array | null = null
let maskFrameCount = 0
const MASK_UPDATE_INTERVAL = 5

// Depth estimation
let depthPipeline: any = null
let cachedDepthMap: Float32Array | null = null
let depthMapW = 0
let depthMapH = 0
let depthRunning = false
let depthCanvas: HTMLCanvasElement
let depthCtx: CanvasRenderingContext2D
const DEPTH_SCALE_FACTOR = 0.25 // 1/4 resolution for inference speed

// Edge detection
let lumBuf: Float32Array
function extractEdgePoints(
  imgData: Uint8ClampedArray,
  w: number, h: number,
  threshold: number, sampleStep: number,
  binaryMask: Uint8Array | null,
): number[][] {
  if (!lumBuf || lumBuf.length < w * h) {
    lumBuf = new Float32Array(w * h)
  }
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4
    lumBuf[i] = imgData[idx] * 0.299 + imgData[idx + 1] * 0.587 + imgData[idx + 2] * 0.114
  }
  const pts: number[][] = []
  const t2 = threshold * threshold
  for (let y = 1; y < h - 1; y += sampleStep) {
    for (let x = 1; x < w - 1; x += sampleStep) {
      if (binaryMask && !binaryMask[y * w + x]) continue
      const gx
        = -lumBuf[(y - 1) * w + (x - 1)] + lumBuf[(y - 1) * w + (x + 1)]
        - 2 * lumBuf[y * w + (x - 1)] + 2 * lumBuf[y * w + (x + 1)]
        - lumBuf[(y + 1) * w + (x - 1)] + lumBuf[(y + 1) * w + (x + 1)]
      const gy
        = -lumBuf[(y - 1) * w + (x - 1)] - 2 * lumBuf[(y - 1) * w + x] - lumBuf[(y - 1) * w + (x + 1)]
        + lumBuf[(y + 1) * w + (x - 1)] + 2 * lumBuf[(y + 1) * w + x] + lumBuf[(y + 1) * w + (x + 1)]
      if (gx * gx + gy * gy > t2) {
        pts.push([x, y])
      }
    }
  }
  return pts
}

// Sample depth at a point (mirrored display coords)
function sampleDepth(x: number, y: number): number {
  if (!cachedDepthMap) return 0
  const dx = Math.min(Math.max(Math.floor(x / vw * depthMapW), 0), depthMapW - 1)
  const dy = Math.min(Math.max(Math.floor(y / vh * depthMapH), 0), depthMapH - 1)
  return cachedDepthMap[dy * depthMapW + dx]
}

// Pre-allocated buffers
const MAX_VERTS = 30000
let posArr = new Float32Array(MAX_VERTS * 3)
let colArr = new Float32Array(MAX_VERTS * 3)
let coordsBuf = new Float64Array(10000)

// --- Init functions ---

async function initSegmenter(vision: any) {
  try {
    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      outputConfidenceMasks: true,
    })
    maskStatus.value = '(ready)'
  }
  catch (e) {
    console.error('ImageSegmenter init failed:', e)
    maskStatus.value = '(error)'
  }
}

async function initFaceLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    )
    initSegmenter(vision)

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
    })
    const indexSet = new Set<number>()
    for (const conn of FaceLandmarker.FACE_LANDMARKS_CONTOURS) {
      indexSet.add(conn.start)
      indexSet.add(conn.end)
    }
    contourIndices = Array.from(indexSet)
    faceStatus.value = `(${contourIndices.length} pts)`
  }
  catch (e) {
    console.error('FaceLandmarker init failed:', e)
    faceStatus.value = '(error)'
  }
}

async function initDepthEstimator() {
  try {
    depthStatus.value = '(downloading...)'
    const { pipeline: createPipeline } = await import('@huggingface/transformers')
    depthPipeline = await createPipeline(
      'depth-estimation',
      'onnx-community/depth-anything-v2-small',
      {
        device: 'webgpu',
        dtype: 'fp32',
      },
    )
    depthStatus.value = '(ready)'
  }
  catch (e) {
    console.error('Depth estimator init failed:', e)
    // Fallback to CPU/wasm
    try {
      depthStatus.value = '(fallback wasm...)'
      const { pipeline: createPipeline } = await import('@huggingface/transformers')
      depthPipeline = await createPipeline(
        'depth-estimation',
        'onnx-community/depth-anything-v2-small',
      )
      depthStatus.value = '(wasm)'
    }
    catch (e2) {
      console.error('Depth estimator fallback failed:', e2)
      depthStatus.value = '(error)'
    }
  }
}

// Non-blocking depth update (runs on downscaled canvas for speed)
async function updateDepthMap() {
  if (depthRunning || !depthPipeline || !depthCanvas) return
  depthRunning = true
  try {
    // Draw current mirrored video to small canvas
    depthCtx.drawImage(offCanvas, 0, 0, depthCanvas.width, depthCanvas.height)
    const result = await depthPipeline(depthCanvas)
    if (result?.depth) {
      const d = result.depth
      cachedDepthMap = d.data as Float32Array
      depthMapW = d.width
      depthMapH = d.height
    }
  }
  catch (e) {
    console.error('Depth estimation error:', e)
  }
  finally {
    depthRunning = false
  }
}

function generateGridPoints(w: number, h: number, step: number, jit: number): number[][] {
  const pts: number[][] = []
  for (let x = 0; x <= w; x += step) {
    pts.push([x, 0], [x, h])
  }
  for (let y = step; y < h; y += step) {
    pts.push([0, y], [w, y])
  }
  for (let y = step; y < h; y += step) {
    for (let x = step; x < w; x += step) {
      pts.push([
        x + (Math.random() - 0.5) * jit * 2,
        y + (Math.random() - 0.5) * jit * 2,
      ])
    }
  }
  return pts
}

function rebuildGrid() {
  gridPoints = generateGridPoints(vw, vh, gridStep.value, jitter.value)
}

async function start() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
  })
  const vid = videoEl.value!
  vid.srcObject = stream
  await vid.play()
  started.value = true

  vw = vid.videoWidth
  vh = vid.videoHeight

  // Offscreen canvas for pixel sampling (mirrored)
  offCanvas = document.createElement('canvas')
  offCanvas.width = vw
  offCanvas.height = vh
  offCtx = offCanvas.getContext('2d', { willReadFrequently: true })!

  // Small canvas for depth inference (1/4 res = ~16x faster)
  depthCanvas = document.createElement('canvas')
  depthCanvas.width = Math.round(vw * DEPTH_SCALE_FACTOR)
  depthCanvas.height = Math.round(vh * DEPTH_SCALE_FACTOR)
  depthCtx = depthCanvas.getContext('2d')!

  // Three.js — PerspectiveCamera for 3D depth view
  renderer = new THREE.WebGLRenderer({ antialias: false })
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
  containerEl.value!.appendChild(renderer.domElement)

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  const fov = 50
  const aspect = window.innerWidth / window.innerHeight
  camera = new THREE.PerspectiveCamera(fov, aspect, 1, 5000)
  // Position camera centered on the mesh, pulled back
  const camDist = (vh / 2) / Math.tan((fov / 2) * Math.PI / 180)
  camera.position.set(vw / 2, vh / 2, camDist)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(vw / 2, vh / 2, 0)
  controls.enableDamping = true
  controls.dampingFactor = 0.1
  controls.update()

  // Pre-allocate geometry
  geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3))
  geometry.setDrawRange(0, 0)

  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide })
  mesh = new THREE.Mesh(geometry, material)
  scene.add(mesh)

  gridPoints = generateGridPoints(vw, vh, gridStep.value, jitter.value)

  // Start loading models (async, non-blocking)
  initFaceLandmarker()
  initDepthEstimator()

  // Handle resize
  const onResize = () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  }
  window.addEventListener('resize', onResize)

  let lastTime = performance.now()
  let frames = 0
  let depthFrameCount = 0
  const DEPTH_UPDATE_INTERVAL = 1

  function animate() {
    animId = requestAnimationFrame(animate)

    // Mirror video to offscreen canvas
    offCtx.save()
    offCtx.scale(-1, 1)
    offCtx.drawImage(vid, -vw, 0, vw, vh)
    offCtx.restore()
    const imgData = offCtx.getImageData(0, 0, vw, vh).data

    // --- Segmentation mask (every N frames) ---
    let binaryMask: Uint8Array | null = null
    if (useMask.value && imageSegmenter) {
      maskFrameCount++
      if (!cachedBinaryMask || maskFrameCount >= MASK_UPDATE_INTERVAL) {
        maskFrameCount = 0
        const segResult = imageSegmenter.segmentForVideo(vid, performance.now())
        if (segResult.confidenceMasks && segResult.confidenceMasks.length > 0) {
          const cm = segResult.confidenceMasks[0]
          const raw = cm.getAsFloat32Array()
          const mw = cm.width
          const mh = cm.height
          if (!cachedBinaryMask || cachedBinaryMask.length !== vw * vh) {
            cachedBinaryMask = new Uint8Array(vw * vh)
          }
          for (let y = 0; y < vh; y++) {
            const sy = Math.min(Math.floor(y / vh * mh), mh - 1)
            for (let x = 0; x < vw; x++) {
              const origX = vw - 1 - x
              const sx = Math.min(Math.floor(origX / vw * mw), mw - 1)
              cachedBinaryMask[y * vw + x] = raw[sy * mw + sx] > 0.5 ? 1 : 0
            }
          }
        }
        segResult.close()
      }
      binaryMask = cachedBinaryMask
    }
    else {
      cachedBinaryMask = null
    }

    // --- Depth estimation (non-blocking, every N frames) ---
    if (useDepth.value && depthPipeline) {
      depthFrameCount++
      if (depthFrameCount >= DEPTH_UPDATE_INTERVAL) {
        depthFrameCount = 0
        updateDepthMap()
      }
    }

    // --- Collect points ---
    let allPoints: number[][] = []

    if (binaryMask) {
      for (let i = 0; i < gridPoints.length; i++) {
        const px = Math.floor(gridPoints[i][0])
        const py = Math.floor(gridPoints[i][1])
        if (px >= 0 && px < vw && py >= 0 && py < vh && binaryMask[py * vw + px]) {
          allPoints.push(gridPoints[i])
        }
      }
    }
    else {
      allPoints = [...gridPoints]
    }

    if (useFace.value && faceLandmarker) {
      const result = faceLandmarker.detectForVideo(vid, performance.now())
      if (result.faceLandmarks.length > 0) {
        const lm = result.faceLandmarks[0]
        for (let i = 0; i < contourIndices.length; i++) {
          const li = contourIndices[i]
          allPoints.push([
            vw - lm[li].x * vw,
            lm[li].y * vh,
          ])
        }
      }
    }

    if (useEdge.value) {
      const edgePts = extractEdgePoints(imgData, vw, vh, edgeThreshold.value, edgeSampleStep.value, binaryMask)
      for (let i = 0; i < edgePts.length; i++) {
        allPoints.push(edgePts[i])
      }
    }

    if (allPoints.length < 3) {
      geometry.setDrawRange(0, 0)
      controls.update()
      renderer.render(scene, camera)
      frames++
      const now = performance.now()
      if (now - lastTime >= 1000) { fps.value = frames; frames = 0; lastTime = now }
      return
    }

    // --- Delaunay ---
    const totalPts = allPoints.length
    if (coordsBuf.length < totalPts * 2) {
      coordsBuf = new Float64Array(totalPts * 4)
    }
    for (let i = 0; i < totalPts; i++) {
      coordsBuf[i * 2] = allPoints[i][0]
      coordsBuf[i * 2 + 1] = allPoints[i][1]
    }
    const del = new Delaunator(coordsBuf.subarray(0, totalPts * 2))
    const numTri = del.triangles.length / 3
    triCount.value = numTri

    const neededVerts = numTri * 3
    if (posArr.length < neededVerts * 3) {
      posArr = new Float32Array(neededVerts * 4)
      colArr = new Float32Array(neededVerts * 4)
      geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3))
    }

    // --- Build mesh with depth Z ---
    const dScale = useDepth.value ? depthScale.value : 0
    for (let t = 0; t < numTri; t++) {
      const i0 = del.triangles[t * 3]
      const i1 = del.triangles[t * 3 + 1]
      const i2 = del.triangles[t * 3 + 2]
      const x0 = allPoints[i0][0], y0 = allPoints[i0][1]
      const x1 = allPoints[i1][0], y1 = allPoints[i1][1]
      const x2 = allPoints[i2][0], y2 = allPoints[i2][1]

      const b = t * 9
      posArr[b] = x0
      posArr[b + 1] = vh - y0
      posArr[b + 2] = sampleDepth(x0, y0) * dScale
      posArr[b + 3] = x1
      posArr[b + 4] = vh - y1
      posArr[b + 5] = sampleDepth(x1, y1) * dScale
      posArr[b + 6] = x2
      posArr[b + 7] = vh - y2
      posArr[b + 8] = sampleDepth(x2, y2) * dScale

      const cx = (x0 + x1 + x2) / 3
      const cy = (y0 + y1 + y2) / 3
      const px = Math.min(Math.max(Math.floor(cx), 0), vw - 1)
      const py = Math.min(Math.max(Math.floor(cy), 0), vh - 1)
      const idx = (py * vw + px) * 4
      const r = imgData[idx] / 255
      const g = imgData[idx + 1] / 255
      const bl = imgData[idx + 2] / 255

      colArr[b] = r; colArr[b + 1] = g; colArr[b + 2] = bl
      colArr[b + 3] = r; colArr[b + 4] = g; colArr[b + 5] = bl
      colArr[b + 6] = r; colArr[b + 7] = g; colArr[b + 8] = bl
    }

    geometry.setDrawRange(0, numTri * 3)
    geometry.getAttribute('position').needsUpdate = true
    geometry.getAttribute('color').needsUpdate = true

    controls.update()
    renderer.render(scene, camera)

    frames++
    const now = performance.now()
    if (now - lastTime >= 1000) {
      fps.value = frames
      frames = 0
      lastTime = now
    }
  }

  animate()
}

onUnmounted(() => {
  if (animId) cancelAnimationFrame(animId)
  if (controls) controls.dispose()
  if (renderer) renderer.dispose()
  if (geometry) geometry.dispose()
  if (faceLandmarker) faceLandmarker.close()
  if (imageSegmenter) imageSegmenter.close()
  const vid = videoEl.value
  if (vid?.srcObject) {
    (vid.srcObject as MediaStream).getTracks().forEach(t => t.stop())
  }
})
</script>

<style>
html, body, #__nuxt {
  margin: 0;
  padding: 0;
  overflow: hidden;
  background: #000;
  width: 100%;
  height: 100%;
}

.container {
  width: 100vw;
  height: 100vh;
  position: relative;
}

.container video {
  display: none;
}

.container canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 10;
}

.overlay span {
  color: #fff;
  font-family: monospace;
  font-size: 24px;
  padding: 16px 32px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 8px;
  transition: border-color 0.2s;
}

.overlay:hover span {
  border-color: rgba(255, 255, 255, 0.8);
}

.controls {
  position: absolute;
  bottom: 16px;
  left: 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 12px 16px;
  align-items: center;
  z-index: 10;
  font-family: monospace;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.6);
  max-width: calc(100vw - 32px);
}

.controls label {
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.controls input[type="range"] {
  width: 80px;
  accent-color: #fff;
}

.toggle input[type="checkbox"] {
  accent-color: #fff;
}

.fps {
  color: rgba(255, 255, 255, 0.4);
}
</style>
