<template>
  <div ref="containerEl" class="container">
    <video ref="videoEl" autoplay playsinline muted />
    <div v-if="!started" class="overlay" @click="start">
      <span>Click to Start</span>
    </div>
    <div v-if="started" class="controls">
      <label>
        Grid: {{ gridStep }}
        <input v-model.number="gridStep" type="range" min="6" max="40" step="1" @change="rebuildGrid" />
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
        Edge Thresh: {{ edgeThreshold }}
        <input v-model.number="edgeThreshold" type="range" min="5" max="300" step="5" />
      </label>
      <label>
        Edge Density: {{ edgeSampleStep }}
        <input v-model.number="edgeSampleStep" type="range" min="2" max="16" step="1" />
      </label>
      <span class="fps">{{ fps }} fps | {{ triCount }} tris</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import * as THREE from 'three'
import Delaunator from 'delaunator'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

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

// --- State ---
let renderer: THREE.WebGLRenderer
let scene: THREE.Scene
let camera: THREE.OrthographicCamera
let mesh: THREE.Mesh
let geometry: THREE.BufferGeometry
let offCtx: CanvasRenderingContext2D
let vw = 0
let vh = 0
let animId: number

// Static grid points (regenerated on param change)
let gridPoints: number[][] = []

// MediaPipe
let faceLandmarker: FaceLandmarker | null = null
let contourIndices: number[] = []

// Edge detection: extract points along strong luminance gradients
// Uses Sobel-like gradient on the existing imgData (no OpenCV needed)
let lumBuf: Float32Array // luminance buffer, allocated once
function extractEdgePoints(
  imgData: Uint8ClampedArray,
  w: number,
  h: number,
  threshold: number,
  sampleStep: number,
): number[][] {
  if (!lumBuf || lumBuf.length < w * h) {
    lumBuf = new Float32Array(w * h)
  }

  // Compute luminance
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4
    lumBuf[i] = imgData[idx] * 0.299 + imgData[idx + 1] * 0.587 + imgData[idx + 2] * 0.114
  }

  // Sobel gradient magnitude, sample at intervals
  const pts: number[][] = []
  const t2 = threshold * threshold
  for (let y = 1; y < h - 1; y += sampleStep) {
    for (let x = 1; x < w - 1; x += sampleStep) {
      // Sobel X
      const gx
        = -lumBuf[(y - 1) * w + (x - 1)] + lumBuf[(y - 1) * w + (x + 1)]
        - 2 * lumBuf[y * w + (x - 1)] + 2 * lumBuf[y * w + (x + 1)]
        - lumBuf[(y + 1) * w + (x - 1)] + lumBuf[(y + 1) * w + (x + 1)]
      // Sobel Y
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

// Pre-allocated buffers (max ~10000 triangles)
const MAX_VERTS = 30000
let posArr = new Float32Array(MAX_VERTS * 3)
let colArr = new Float32Array(MAX_VERTS * 3)
// Reusable coords buffer for Delaunator
let coordsBuf = new Float64Array(10000)

async function initFaceLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    )
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
    })

    // Extract unique point indices from contour connections
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

function generateGridPoints(w: number, h: number, step: number, jit: number): number[][] {
  const pts: number[][] = []

  // Border
  for (let x = 0; x <= w; x += step) {
    pts.push([x, 0], [x, h])
  }
  for (let y = step; y < h; y += step) {
    pts.push([0, y], [w, y])
  }

  // Interior with jitter
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

  // Offscreen canvas for pixel sampling
  const offCanvas = document.createElement('canvas')
  offCanvas.width = vw
  offCanvas.height = vh
  offCtx = offCanvas.getContext('2d', { willReadFrequently: true })!

  // Three.js
  renderer = new THREE.WebGLRenderer({ antialias: false })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(vw, vh)
  containerEl.value!.appendChild(renderer.domElement)

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  camera = new THREE.OrthographicCamera(0, vw, 0, vh, -1, 1)
  camera.position.z = 1

  // Pre-allocate geometry with max capacity
  geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3))
  geometry.setDrawRange(0, 0)

  const material = new THREE.MeshBasicMaterial({ vertexColors: true })
  mesh = new THREE.Mesh(geometry, material)
  scene.add(mesh)

  // Generate initial grid
  gridPoints = generateGridPoints(vw, vh, gridStep.value, jitter.value)

  // Start loading MediaPipe (async, non-blocking)
  initFaceLandmarker()

  let lastTime = performance.now()
  let frames = 0

  function animate() {
    animId = requestAnimationFrame(animate)

    // Mirror video to offscreen canvas
    offCtx.save()
    offCtx.scale(-1, 1)
    offCtx.drawImage(vid, -vw, 0, vw, vh)
    offCtx.restore()
    const imgData = offCtx.getImageData(0, 0, vw, vh).data

    // Collect points: grid + face landmarks + edge points
    let allPoints: number[][] = gridPoints

    // Face landmarks
    let facePts: number[][] = []
    if (useFace.value && faceLandmarker) {
      const result = faceLandmarker.detectForVideo(vid, performance.now())
      if (result.faceLandmarks.length > 0) {
        const lm = result.faceLandmarks[0]
        facePts = contourIndices.map((li) => {
          return [
            vw - lm[li].x * vw, // mirror X to match display
            lm[li].y * vh,
          ]
        })
      }
    }

    // Edge detection on mirrored video pixels
    let edgePts: number[][] = []
    if (useEdge.value) {
      edgePts = extractEdgePoints(imgData, vw, vh, edgeThreshold.value, edgeSampleStep.value)
    }

    // Merge all point sources
    if (facePts.length > 0 || edgePts.length > 0) {
      allPoints = [...gridPoints, ...facePts, ...edgePts]
    }

    // Ensure coords buffer is large enough
    const totalPts = allPoints.length
    if (coordsBuf.length < totalPts * 2) {
      coordsBuf = new Float64Array(totalPts * 4)
    }
    for (let i = 0; i < totalPts; i++) {
      coordsBuf[i * 2] = allPoints[i][0]
      coordsBuf[i * 2 + 1] = allPoints[i][1]
    }

    // Delaunay triangulation
    const del = new Delaunator(coordsBuf.subarray(0, totalPts * 2))
    const numTri = del.triangles.length / 3
    triCount.value = numTri

    // Ensure vertex buffers are large enough
    const neededVerts = numTri * 3
    if (posArr.length < neededVerts * 3) {
      posArr = new Float32Array(neededVerts * 4)
      colArr = new Float32Array(neededVerts * 4)
      geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
      geometry.setAttribute('color', new THREE.BufferAttribute(colArr, 3))
    }

    // Build positions + sample colors
    for (let t = 0; t < numTri; t++) {
      const i0 = del.triangles[t * 3]
      const i1 = del.triangles[t * 3 + 1]
      const i2 = del.triangles[t * 3 + 2]
      const x0 = allPoints[i0][0], y0 = allPoints[i0][1]
      const x1 = allPoints[i1][0], y1 = allPoints[i1][1]
      const x2 = allPoints[i2][0], y2 = allPoints[i2][1]

      const b = t * 9
      posArr[b] = x0; posArr[b + 1] = y0; posArr[b + 2] = 0
      posArr[b + 3] = x1; posArr[b + 4] = y1; posArr[b + 5] = 0
      posArr[b + 6] = x2; posArr[b + 7] = y2; posArr[b + 8] = 0

      // Centroid color from mirrored video
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
    renderer.render(scene, camera)

    // FPS counter
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
  if (renderer) renderer.dispose()
  if (geometry) geometry.dispose()
  if (faceLandmarker) faceLandmarker.close()
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
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.container video {
  display: none;
}

.container canvas {
  max-width: 100vw;
  max-height: 100vh;
  object-fit: contain;
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
  gap: 16px;
  align-items: center;
  z-index: 10;
  font-family: monospace;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.6);
}

.controls label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.controls input[type="range"] {
  width: 100px;
  accent-color: #fff;
}

.toggle input[type="checkbox"] {
  accent-color: #fff;
}

.fps {
  color: rgba(255, 255, 255, 0.4);
}
</style>
