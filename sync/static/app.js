import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const DEFAULT_TANK_DIMENSIONS = { x: 25, y: 25, z: 25, unit: 'in' };
const screenshotMode = new URLSearchParams(window.location.search).has('screenshot');
const TANK_DISPLAY_SCALE = 0.62;
const TANK_DISPLAY_OFFSET = 2.15;

let tankSize = { x: 4.0, y: 2.2, z: 2.6 };
const calmColor = 0xd7dde3;
const selectedColor = 0xffffff;
const featuredColor = 0xf2f2f2;

const state = {
  layout: { tanks: [], nodes: [], cameras: [], scene_items: [], detections: [], health: {} },
  meshes: new Map(),
  frustums: new Map(),
  selectedId: null,
  pendingId: null,
  step: 0.06,
  dirtyTimer: null,
  setupStep: 0,
  activeTankId: 'main-tank',
  setupDraft: {},
  feedSignature: '',
  liveIndex: 0,
  liveTimer: null,
  focusObservationId: null,
  stageFeedSource: '',
  manualViewId: null,
  manualViewUntil: 0,
  manualTankUntil: 0,
  controlsActiveUntil: 0,
  orbitAngle: 0,
  stageCameraId: null,
  controlAxis: 'slide',
  padDrag: null,
  initialSetupPrompted: false,
  identifyPromptedFor: new Set(),
  floaterRevisions: new Map(),
  floaterCards: new Map(),
  lighthouse: {
    pan: 90,
    tilt: 90,
    minPan: 20,
    maxPan: 160,
    minTilt: 45,
    maxTilt: 125,
    step: 3,
    driver: '',
    deviceId: '',
    ready: false,
    status: 'checking',
    holdTimer: null,
    panelClosed: true,
    contextId: '',
  },
  reeflex: {
    base: 90,
    shoulder: 90,
    elbow: 90,
    minBase: 20,
    maxBase: 160,
    minShoulder: 45,
    maxShoulder: 135,
    minElbow: 35,
    maxElbow: 145,
    step: 3,
    driver: '',
    deviceId: '',
    status: 'checking',
    holdTimer: null,
    panelClosed: true,
    contextId: '',
    ready: false,
  },
};

const stage = document.getElementById('tank-stage');
const feedMarquee = document.getElementById('feed-marquee');
const systemGuide = document.getElementById('system-guide');
const tankSummary = document.getElementById('tank-summary');
const placementSection = document.getElementById('placement-section');
const liveSection = document.getElementById('live-section');
const liveFeed = document.getElementById('live-feed');
const liveTitle = document.getElementById('live-title');
const liveDetail = document.getElementById('live-detail');
const liveControls = document.getElementById('live-controls');
const lighthouseControlSection = document.getElementById('lighthouse-control-section');
const lighthouseCurrent = document.getElementById('lighthouse-current');
const lighthouseStatus = document.getElementById('lighthouse-status');
const lighthousePanSlider = document.getElementById('lighthouse-pan-slider');
const lighthouseTiltSlider = document.getElementById('lighthouse-tilt-slider');
const lighthouseClose = document.getElementById('lighthouse-close');
const reeflexControlSection = document.getElementById('reeflex-control-section');
const reeflexCurrent = document.getElementById('reeflex-current');
const reeflexStatus = document.getElementById('reeflex-status');
const reeflexBaseSlider = document.getElementById('reeflex-base-slider');
const reeflexShoulderSlider = document.getElementById('reeflex-shoulder-slider');
const reeflexElbowSlider = document.getElementById('reeflex-elbow-slider');
const reeflexClose = document.getElementById('reeflex-close');
const unplacedList = document.getElementById('unplaced-list');
const selectedSection = document.getElementById('selected-section');
const selectedTitle = document.getElementById('selected-title');
const preview = document.getElementById('feed-preview');
const pip = document.getElementById('stage-pip');
const pipPreview = document.getElementById('pip-preview');
const pipLabel = document.getElementById('pip-label');
const tankTitle = document.getElementById('tank-title');
const tankSubtitle = document.getElementById('tank-subtitle');
const tankTabs = document.getElementById('tank-tabs');
const setupOverlay = document.getElementById('setup-overlay');
const setupQuestion = document.getElementById('setup-question');
const setupCurrent = document.getElementById('setup-current');
const setupOptions = document.getElementById('setup-options');
const setupBack = document.getElementById('setup-back');
const setupSkip = document.getElementById('setup-skip');
const setupButton = document.getElementById('setup-button');
const observationList = document.getElementById('observation-list');
const worldControls = document.getElementById('world-controls');
const sidePanel = document.querySelector('.side-panel');
const dockToggle = document.getElementById('dock-toggle');
const placementPad = document.getElementById('placement-pad');
const placementStick = document.getElementById('placement-stick');
const identifySelected = document.getElementById('identify-selected');
const removeSelected = document.getElementById('remove-selected');
const stageFeedBackdrop = document.getElementById('stage-feed-backdrop');
const stageFeedImage = document.getElementById('stage-feed-image');
const stageFeedLabel = document.getElementById('stage-feed-label');
const floaterLeft = document.getElementById('floater-left');
const floaterLeftImage = document.getElementById('floater-left-image');
const floaterLeftLabel = document.getElementById('floater-left-label');
const floaterRight = document.getElementById('floater-right');
const floaterRightImage = document.getElementById('floater-right-image');
const floaterRightLabel = document.getElementById('floater-right-label');
const feedPrevious = document.getElementById('feed-previous');
const feedNext = document.getElementById('feed-next');
const feedPin = document.getElementById('feed-pin');
const sightingShutter = document.getElementById('sighting-shutter');
const feedEmpty = document.getElementById('feed-empty');
const feedThumbnails = document.getElementById('feed-thumbnails');
const cctvState = document.getElementById('cctv-state');
const sightingsDrawer = document.getElementById('sightings-drawer');
const sightingsGrid = document.getElementById('sightings-grid');
const deepDialog = document.getElementById('deep-dialog');
const deepImage = document.getElementById('deep-image');
const homeButton = document.getElementById('home-button');
const decorateToggle = document.getElementById('decorate-toggle');
const structureToolbar = document.getElementById('structure-toolbar');
let pendingDeepSightingId = null;

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(4.8, 3.4, 5.0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
stage.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.25, 0);
controls.minDistance = 3.2;
controls.maxDistance = 9.0;
controls.addEventListener('start', () => {
  state.controlsActiveUntil = Date.now() + 12000;
});
controls.addEventListener('end', () => {
  state.controlsActiveUntil = Date.now() + 6000;
});

scene.add(new THREE.HemisphereLight(0xd9f5ff, 0x071321, 1.6));
const keyLight = new THREE.DirectionalLight(0xe8f8ff, 0.82);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);

const tankGroup = new THREE.Group();
scene.add(tankGroup);

const tankMesh = new THREE.Mesh(
  new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z),
  new THREE.MeshBasicMaterial({
    color: 0xdce2e8,
    transparent: true,
    opacity: 0.095,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
);
tankMesh.name = 'tank-volume';
tankGroup.add(tankMesh);

const edgeMesh = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z)),
  new THREE.LineBasicMaterial({ color: 0xf2f4f6, transparent: true, opacity: 0.68 })
);
edgeMesh.name = 'tank-edges';
tankGroup.add(edgeMesh);

const floorGrid = new THREE.GridHelper(5.2, 18, 0xb8bec5, 0x30343a);
floorGrid.name = 'tank-floor-grid';
floorGrid.position.y = -tankSize.y / 2 - 0.02;
floorGrid.material.transparent = true;
floorGrid.material.opacity = 0.18;
tankGroup.add(floorGrid);

const waterPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(5.2, 4.6),
  new THREE.MeshBasicMaterial({ color: 0xe8ecef, transparent: true, opacity: 0.075, side: THREE.DoubleSide })
);
waterPlane.rotation.x = Math.PI / 2;
waterPlane.name = 'tank-water-plane';
waterPlane.position.y = tankSize.y * 0.22;
tankGroup.add(waterPlane);

const tableGroup = new THREE.Group();
tableGroup.name = 'tank-table';
const tableTop = new THREE.Mesh(
  new THREE.BoxGeometry(1, 0.12, 1),
  new THREE.MeshStandardMaterial({ color: 0x181a1d, roughness: 0.72, metalness: 0.08 })
);
tableTop.position.y = -tankSize.y / 2 - 0.18;
tableTop.name = 'tank-table-top';
tableGroup.add(tableTop);
for (const x of [-1, 1]) {
  for (const z of [-1, 1]) {
    const leg = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.74, 0.08),
      new THREE.MeshStandardMaterial({ color: 0x0d0f12, roughness: 0.66, metalness: 0.16 })
    );
    leg.name = 'tank-table-leg';
    leg.userData.cornerX = x;
    leg.userData.cornerZ = z;
    tableGroup.add(leg);
  }
}
tankGroup.add(tableGroup);

tankGroup.scale.setScalar(TANK_DISPLAY_SCALE);
tankGroup.position.x = -TANK_DISPLAY_OFFSET;
const companionTankGroup = tankGroup.clone(true);
companionTankGroup.position.x = TANK_DISPLAY_OFFSET;
scene.add(companionTankGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const faceTargets = [tankMesh];

function text(value, fallback = '') {
  return value === undefined || value === null || value === '' ? fallback : String(value);
}

function activeTank() {
  const tanks = state.layout.tanks || [];
  return tanks.find(tank => tank.tank_id === state.activeTankId) || tanks[0] || {
    tank_id: 'main-tank',
    label: 'Main Tank',
    dimensions: { ...DEFAULT_TANK_DIMENSIONS },
    setup_complete: false,
  };
}

function connectedTankIds() {
  const ids = new Set();
  (state.layout.nodes || []).forEach(node => {
    if (node.status === 'online') {
      (node.tank_ids || []).forEach(tankId => ids.add(tankId));
    }
  });
  (state.layout.cameras || []).forEach(camera => {
    if (camera.status === 'online' && camera.tank_id) ids.add(camera.tank_id);
  });
  return ids;
}

function visibleTanks() {
  const tanks = state.layout.tanks || [];
  const connected = connectedTankIds();
  if (!connected.size) return tanks;
  return tanks.filter(tank => connected.has(tank.tank_id));
}

function setActiveTank(tankId) {
  state.activeTankId = tankId;
  state.manualTankUntil = Date.now() + 60000;
  state.selectedId = null;
  state.pendingId = null;
  state.lighthouse.ready = false;
  state.lighthouse.driver = '';
  state.lighthouse.deviceId = '';
  state.reeflex.ready = false;
  state.reeflex.driver = '';
  state.reeflex.deviceId = '';
  applyTankDimensions();
  renderHud();
  renderFeedMarquee();
  renderUnplaced();
  renderSelection();
  updateAllMeshes();
  updateStageFeed();
  refreshLighthouseStatus().catch(() => {});
  refreshReeflexStatus().catch(() => {});
  refreshVisionStatus().catch(() => {});
}

function cameraId(item) {
  return item.camera_id || item.id;
}

function isFloater(item) {
  return item.camera_type === 'floater_cam' || item.item_type === 'floater_cam' || item.source_type === 'esp32_upload';
}

function isLighthouse(item) {
  if (isCameraItem(item)) return item.role_locked && item.camera_type === 'lighthouse_cam';
  return item.item_type === 'lighthouse';
}

function isReeflex(item) {
  if (isCameraItem(item)) return item.role_locked && item.camera_type === 'reeflex_cam';
  return item.item_type === 'reeflex';
}

function isEndoscope(item) {
  return !isLighthouse(item) && !isReeflex(item) && (item.camera_type === 'endoscope_cam' || item.item_type === 'endoscope_cam' || item.source_type === 'usb_camera');
}

function isRobotArm(item) {
  return item.item_type === 'robot_arm';
}

function isCameraItem(item) {
  return Boolean(cameraId(item));
}

function isConnectedItem(item) {
  if (!item) return false;
  if (item.hidden_from_layout) return false;
  if (!isCameraItem(item)) return item.status !== 'offline';
  return item.status !== 'offline' && item.node_active !== false;
}

function itemKind(item) {
  if (isFloater(item)) return 'Floater';
  if (isCameraItem(item) && item.source_type === 'usb_camera' && !item.role_locked) return 'USB feed - identify';
  if (isEndoscope(item)) return 'Reel';
  if (isRobotArm(item)) return 'Robot arm rail';
  if (isReeflex(item)) return 'Reeflex 3-servo arm';
  if (isLighthouse(item)) return 'Raydar pan-tilt';
  if (item.item_type === 'pan_tilt_cam') return 'Reeflex';
  if (item.item_type === 'structure_shape') return `Structure · ${text(item.structure_type, 'shape')}`;
  return text(item.item_type || item.camera_type, 'Scene item');
}

function displayName(item) {
  if (!item) return '';
  const id = cameraId(item) || item.item_id;
  if (isCameraItem(item) && item.source_type === 'usb_camera' && !item.role_locked) {
    return `USB feed ${id}`;
  }
  if (isLighthouse(item)) return text(item.label || item.name, 'Raydar').replace(/lighthouse/ig, 'Raydar');
  if (isEndoscope(item)) return text(item.label || item.name, `Reel ${id}`).replace(/scope|endoscope/ig, 'Reel');
  return text(item.label || item.name, id);
}

function nodeLabel(node) {
  return text(node.label || node.hostname || node.node_id, 'Edge node');
}

function countWord(count) {
  return ['zero', 'one', 'two', 'three', 'four', 'five'][count] || String(count);
}

function activeNode() {
  const nodes = state.layout.nodes || [];
  const tankId = activeTank().tank_id;
  return nodes.find(node => node.status === 'online' && (node.tank_ids || []).includes(tankId))
    || nodes.find(node => (node.tank_ids || []).includes(tankId))
    || nodes.find(node => node.status === 'online')
    || nodes[0]
    || null;
}

function usbVideoCameras() {
  return allCamerasForActiveTank().filter(item => item.source_type === 'usb_camera' || item.camera_type === 'endoscope_cam');
}

function snapshotCameras() {
  return allCamerasForActiveTank().filter(item => item.source_type === 'esp32_upload' || item.camera_type === 'floater_cam');
}

function manifestCounts() {
  const node = activeNode();
  return ((node && node.device_inventory && node.device_inventory.counts) || {});
}

function manifestTotal() {
  const node = activeNode();
  if (node && node.device_inventory && Number.isFinite(Number(node.device_inventory.owned_device_count))) {
    return Number(node.device_inventory.owned_device_count);
  }
  const counts = manifestCounts();
  return Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
}

function expectedUsbVideoCount() {
  const counts = manifestCounts();
  return Number(counts.scope || 0) + Number(counts.reeflex || 0) + Number(counts.lighthouse || 0);
}

function manifestSummary() {
  const counts = manifestCounts();
  const order = [
    ['scope', 'Reel'],
    ['reeflex', 'reeflex'],
    ['lighthouse', 'Raydar'],
    ['floater', 'floater'],
  ];
  const parts = order
    .filter(([key]) => counts[key] !== undefined)
    .map(([key, label]) => `${Number(counts[key] || 0)} ${label}${Number(counts[key] || 0) === 1 ? '' : 's'}`);
  return parts.length ? parts.join(', ') : 'No hand-validated manifest yet';
}

function placementOf(item) {
  if (!item.placement) {
    item.placement = { placed: false, position: null, target: null, fov_degrees: isEndoscope(item) ? 70 : 60 };
  }
  return item.placement;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normToWorld(position) {
  return new THREE.Vector3(
    (position.x - 0.5) * tankSize.x,
    (position.y - 0.5) * tankSize.y,
    (position.z - 0.5) * tankSize.z
  );
}

function tankSceneOffset(tankId = activeTank().tank_id) {
  const tanks = state.layout.tanks || [];
  return Math.max(0, tanks.findIndex(tank => tank.tank_id === tankId)) % 2 === 1 ? TANK_DISPLAY_OFFSET : -TANK_DISPLAY_OFFSET;
}

function worldToNorm(vector) {
  return {
    x: clamp(vector.x / tankSize.x + 0.5, 0, 1),
    y: clamp(vector.y / tankSize.y + 0.5, 0, 1),
    z: clamp(vector.z / tankSize.z + 0.5, 0, 1),
  };
}

function dimensionsToWorld(dimensions = {}) {
  const x = Number(dimensions.x || 48);
  const y = Number(dimensions.y || 24);
  const z = Number(dimensions.z || 24);
  const longest = Math.max(x, y, z, 1);
  return {
    x: clamp((x / longest) * 4.2, 1.8, 4.6),
    y: clamp((y / longest) * 4.2, 1.3, 3.4),
    z: clamp((z / longest) * 4.2, 1.6, 4.2),
  };
}

function updateTableForGroup(group) {
  const top = group.getObjectByName('tank-table-top');
  if (top) {
    top.scale.set(tankSize.x + 0.55, 1, tankSize.z + 0.55);
    top.position.y = -tankSize.y / 2 - 0.2;
  }
  group.getObjectsByProperty('name', 'tank-table-leg').forEach(leg => {
    const x = Number(leg.userData.cornerX || 0);
    const z = Number(leg.userData.cornerZ || 0);
    leg.position.set(x * (tankSize.x / 2 + 0.18), -tankSize.y / 2 - 0.6, z * (tankSize.z / 2 + 0.18));
    leg.scale.y = 1;
  });
}

function updateTableForTank() {
  updateTableForGroup(tankGroup);
  updateTableForGroup(companionTankGroup);
}

function updateCompanionTankGeometry() {
  const volume = companionTankGroup.getObjectByName('tank-volume');
  const edges = companionTankGroup.getObjectByName('tank-edges');
  const grid = companionTankGroup.getObjectByName('tank-floor-grid');
  const water = companionTankGroup.getObjectByName('tank-water-plane');
  if (volume) {
    volume.geometry.dispose();
    volume.geometry = new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z);
  }
  if (edges) {
    edges.geometry.dispose();
    edges.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z));
  }
  if (grid) grid.position.y = -tankSize.y / 2 - 0.02;
  if (water) {
    water.position.y = tankSize.y * 0.22;
    water.scale.set(tankSize.x / 4.2, tankSize.z / 4.2, 1);
  }
}

function applyTankDimensions() {
  const nextSize = dimensionsToWorld(activeTank().dimensions);
  if (
    Math.abs(nextSize.x - tankSize.x) < 0.001 &&
    Math.abs(nextSize.y - tankSize.y) < 0.001 &&
    Math.abs(nextSize.z - tankSize.z) < 0.001
  ) {
    updateTableForTank();
    return;
  }
  tankSize = nextSize;
  tankMesh.geometry.dispose();
  tankMesh.geometry = new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z);
  edgeMesh.geometry.dispose();
  edgeMesh.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(tankSize.x, tankSize.y, tankSize.z));
  floorGrid.position.y = -tankSize.y / 2 - 0.02;
  waterPlane.position.y = tankSize.y * 0.22;
  waterPlane.scale.set(tankSize.x / 4.2, tankSize.z / 4.2, 1);
  updateCompanionTankGeometry();
  controls.minDistance = Math.max(2.8, Math.max(tankSize.x, tankSize.y, tankSize.z) * 0.9);
  controls.maxDistance = Math.max(8, Math.max(tankSize.x, tankSize.y, tankSize.z) * 2.2);
  updateTableForTank();
}

function faceFromNormal(normal) {
  const abs = { x: Math.abs(normal.x), y: Math.abs(normal.y), z: Math.abs(normal.z) };
  if (abs.y >= abs.x && abs.y >= abs.z) return normal.y >= 0 ? 'y+' : 'y-';
  if (abs.x >= abs.z) return normal.x >= 0 ? 'x+' : 'x-';
  return normal.z >= 0 ? 'z+' : 'z-';
}

function normalForFace(face) {
  const normals = {
    'x+': new THREE.Vector3(1, 0, 0),
    'x-': new THREE.Vector3(-1, 0, 0),
    'y+': new THREE.Vector3(0, 1, 0),
    'y-': new THREE.Vector3(0, -1, 0),
    'z+': new THREE.Vector3(0, 0, 1),
    'z-': new THREE.Vector3(0, 0, -1),
  };
  return normals[face] || normals['z+'];
}

function inwardTargetFor(position, face) {
  const world = normToWorld(position);
  const inward = normalForFace(face).multiplyScalar(-1);
  const target = world.clone().add(inward.multiplyScalar(1.2));
  return worldToNorm(target);
}

function snapFloaterToFace(point, face) {
  const pos = worldToNorm(point);
  if (face === 'x+') pos.x = 1;
  if (face === 'x-') pos.x = 0;
  if (face === 'y+') pos.y = 1;
  if (face === 'y-') pos.y = 1;
  if (face === 'z+') pos.z = 1;
  if (face === 'z-') pos.z = 0;
  return pos;
}

function snapRailToPerimeter(point, face) {
  const pos = worldToNorm(point);
  const side = face === 'x+' || face === 'x-' || face === 'z+' || face === 'z-' ? face : 'z+';
  pos.y = clamp(pos.y, 0.05, 0.95);
  if (side === 'x+') pos.x = 1;
  if (side === 'x-') pos.x = 0;
  if (side === 'z+') pos.z = 1;
  if (side === 'z-') pos.z = 0;
  return { position: pos, mount_face: side, target: inwardTargetFor(pos, side) };
}

function snapLighthouseToRim(point, face) {
  const pos = worldToNorm(point);
  const side = face === 'x+' || face === 'x-' || face === 'z+' || face === 'z-' ? face : 'z+';
  pos.y = 1;
  if (side === 'x+') pos.x = 1;
  if (side === 'x-') pos.x = 0;
  if (side === 'z+') pos.z = 1;
  if (side === 'z-') pos.z = 0;
  return { position: pos, mount_face: side, target: { x: 0.5, y: 0.48, z: 0.5 } };
}

function defaultPlacement(item) {
  if (isFloater(item)) {
    const position = { x: 0.5, y: 1, z: 0.5 };
    return { placed: true, mount_face: 'y+', position, target: inwardTargetFor(position, 'y+'), fov_degrees: 60 };
  }
  if (isLighthouse(item)) {
    const position = { x: 0.5, y: 1, z: 1 };
    return { placed: true, mount_face: 'z+', position, target: { x: 0.5, y: 0.48, z: 0.5 }, fov_degrees: 78 };
  }
  const position = { x: 0.5, y: 0.5, z: 0.5 };
  return { placed: true, position, target: { x: 0.5, y: 0.5, z: 0.75 }, fov_degrees: isEndoscope(item) ? 70 : 60 };
}

function createMesh(item) {
  let mesh;
  if (item.item_type === 'structure_shape') {
    const material = new THREE.MeshStandardMaterial({ color: item.color || '#698f88', roughness: 0.82, metalness: 0.02 });
    const kind = item.structure_type || 'block';
    if (kind === 'rounded-rock') mesh = new THREE.Mesh(new THREE.DodecahedronGeometry(0.28, 1), material);
    else if (kind === 'pillar') mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.19, 0.7, 18), material);
    else if (kind === 'mound') mesh = new THREE.Mesh(new THREE.SphereGeometry(0.36, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2), material);
    else if (kind === 'slab') mesh = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.12, 0.42), material);
    else if (kind === 'arch') {
      mesh = new THREE.Group();
      const left = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.48, 0.18), material);
      const right = left.clone(); left.position.x = -0.27; right.position.x = 0.27;
      const top = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.14, 0.18), material); top.position.y = 0.24;
      mesh.add(left, right, top);
    } else mesh = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.35, 0.42), material);
  } else if (isFloater(item)) {
    mesh = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.18, 0.18, 0.045, 32),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.5, metalness: 0.08 })
    );
    const lens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.055, 0.055, 0.055, 20),
      new THREE.MeshStandardMaterial({ color: 0x101817, roughness: 0.32, metalness: 0.18 })
    );
    lens.position.y = 0.052;
    mesh.add(body, lens);
  } else if (isEndoscope(item)) {
    mesh = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.055, 0.055, 0.22, 20),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.42, metalness: 0.12 })
    );
    body.position.y = 0.055;
    const nose = new THREE.Mesh(
      new THREE.CylinderGeometry(0.044, 0.055, 0.05, 20),
      new THREE.MeshStandardMaterial({ color: 0x102233, roughness: 0.34, metalness: 0.16 })
    );
    nose.position.y = 0.19;
    const lens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.024, 0.024, 0.014, 14),
      new THREE.MeshStandardMaterial({ color: 0xf5f5f2, emissive: 0x242424, roughness: 0.18, metalness: 0.12 })
    );
    lens.position.y = 0.222;
    mesh.add(body, nose, lens);
  } else if (item.item_type === 'robot_arm') {
    mesh = new THREE.Group();
    const rail = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.12, 1.0),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.46, metalness: 0.1 })
    );
    const carriage = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.18, 0.18),
      new THREE.MeshStandardMaterial({ color: 0xf0f0eb, roughness: 0.38, metalness: 0.12 })
    );
    carriage.position.z = 0.22;
    mesh.add(rail, carriage);
  } else if (isReeflex(item)) {
    mesh = new THREE.Group();
    const base = new THREE.Mesh(
      new THREE.CylinderGeometry(0.15, 0.18, 0.12, 28),
      new THREE.MeshStandardMaterial({ color: 0x242528, roughness: 0.42, metalness: 0.12 })
    );
    const shoulder = new THREE.Mesh(
      new THREE.SphereGeometry(0.105, 22, 14),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.34, metalness: 0.12 })
    );
    shoulder.position.y = 0.16;
    const upperArm = new THREE.Mesh(
      new THREE.BoxGeometry(0.095, 0.34, 0.095),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.38, metalness: 0.1 })
    );
    upperArm.position.set(0.11, 0.32, 0);
    upperArm.rotation.z = -0.42;
    const elbow = new THREE.Mesh(
      new THREE.SphereGeometry(0.078, 18, 12),
      new THREE.MeshStandardMaterial({ color: 0x17181a, roughness: 0.34, metalness: 0.16 })
    );
    elbow.position.set(0.19, 0.49, 0);
    const forearm = new THREE.Mesh(
      new THREE.BoxGeometry(0.08, 0.3, 0.08),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.38, metalness: 0.1 })
    );
    forearm.position.set(0.25, 0.63, 0);
    forearm.rotation.z = -0.18;
    const wrist = new THREE.Mesh(
      new THREE.BoxGeometry(0.16, 0.08, 0.08),
      new THREE.MeshStandardMaterial({ color: 0x111214, roughness: 0.28, metalness: 0.18 })
    );
    wrist.position.set(0.29, 0.79, 0);
    const lens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.028, 0.028, 0.026, 14),
      new THREE.MeshStandardMaterial({ color: 0xf5f5f2, emissive: 0x242424, roughness: 0.18, metalness: 0.1 })
    );
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.38, 0.79, 0);
    mesh.add(base, shoulder, upperArm, elbow, forearm, wrist, lens);
  } else if (isLighthouse(item)) {
    mesh = new THREE.Group();
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.14, 0.18),
      new THREE.MeshStandardMaterial({ color: 0x242528, roughness: 0.48, metalness: 0.12 })
    );
    const head = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.15, 0.14),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.5, metalness: 0.08 })
    );
    head.position.y = 0.17;
    const lens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.035, 0.035, 0.024, 14),
      new THREE.MeshStandardMaterial({ color: 0x071018, roughness: 0.16, metalness: 0.2 })
    );
    lens.rotation.z = Math.PI / 2;
    lens.position.set(0.102, 0.17, 0);
    mesh.add(base, head, lens);
  } else {
    mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.22, 0.22),
      new THREE.MeshStandardMaterial({ color: calmColor, roughness: 0.55 })
    );
  }
  mesh.userData.itemId = cameraId(item) || item.item_id;
  mesh.userData.kind = 'item';
  mesh.traverse(child => {
    child.userData.itemId = mesh.userData.itemId;
    child.userData.kind = 'item';
  });
  scene.add(mesh);
  return mesh;
}

function setObjectColor(object, color) {
  object.traverse(child => {
    if (child.material && child.material.color) child.material.color.set(color);
    if (child.material && child.material.emissive) {
      const isFeatured = color === featuredColor;
      child.material.emissive.set(isFeatured ? 0x7fdcff : 0x000000);
      child.material.emissiveIntensity = isFeatured ? 0.35 : 0;
    }
  });
}

function displayColorForItem(id) {
  if (id === state.selectedId) return selectedColor;
  if (id && id === state.stageCameraId) return featuredColor;
  return calmColor;
}

function createFrustum(id) {
  const geometry = new THREE.BufferGeometry();
  const material = new THREE.LineBasicMaterial({
    color: selectedColor,
    transparent: true,
    opacity: 0.44,
  });
  const line = new THREE.LineSegments(geometry, material);
  line.userData.itemId = id;
  scene.add(line);
  state.frustums.set(id, line);
  return line;
}

function basisFromDirection(direction) {
  const forward = direction.clone().normalize();
  const worldUp = Math.abs(forward.y) > 0.86 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(forward, worldUp).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();
  return { forward, right, up };
}

function updateFrustum(item) {
  const id = cameraId(item) || item.item_id;
  if (!id) return;
  const placement = placementOf(item);
  let frustum = state.frustums.get(id);
  if (!isFloater(item) && !isEndoscope(item) && item.item_type !== 'pan_tilt_cam' && !isLighthouse(item) && !isReeflex(item)) {
    if (frustum) frustum.visible = false;
    return;
  }
  if (!placement.placed || !placement.position) {
    if (frustum) frustum.visible = false;
    return;
  }
  if (!frustum) frustum = createFrustum(id);
  const origin = normToWorld(placement.position).multiplyScalar(TANK_DISPLAY_SCALE);
  const target = normToWorld(placement.target || { x: 0.5, y: 0.5, z: 0.5 }).multiplyScalar(TANK_DISPLAY_SCALE);
  origin.x += tankSceneOffset(item.tank_id);
  target.x += tankSceneOffset(item.tank_id);
  const direction = target.sub(origin);
  if (direction.lengthSq() < 0.01) {
    frustum.visible = false;
    return;
  }
  const { forward, right, up } = basisFromDirection(direction);
  const nearDistance = 0.16;
  const farDistance = isFloater(item) ? 1.45 : 1.1;
  const fov = ((placement.fov_degrees || 65) * Math.PI) / 180;
  const nearHalf = Math.tan(fov / 2) * nearDistance * 0.38;
  const farHalf = Math.tan(fov / 2) * farDistance * 0.68;
  const centerNear = origin.clone().add(forward.clone().multiplyScalar(nearDistance));
  const centerFar = origin.clone().add(forward.clone().multiplyScalar(farDistance));
  const corners = [
    centerNear.clone().add(right.clone().multiplyScalar(-nearHalf)).add(up.clone().multiplyScalar(-nearHalf)),
    centerNear.clone().add(right.clone().multiplyScalar(nearHalf)).add(up.clone().multiplyScalar(-nearHalf)),
    centerNear.clone().add(right.clone().multiplyScalar(nearHalf)).add(up.clone().multiplyScalar(nearHalf)),
    centerNear.clone().add(right.clone().multiplyScalar(-nearHalf)).add(up.clone().multiplyScalar(nearHalf)),
    centerFar.clone().add(right.clone().multiplyScalar(-farHalf)).add(up.clone().multiplyScalar(-farHalf)),
    centerFar.clone().add(right.clone().multiplyScalar(farHalf)).add(up.clone().multiplyScalar(-farHalf)),
    centerFar.clone().add(right.clone().multiplyScalar(farHalf)).add(up.clone().multiplyScalar(farHalf)),
    centerFar.clone().add(right.clone().multiplyScalar(-farHalf)).add(up.clone().multiplyScalar(farHalf)),
  ];
  const edgePairs = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const points = [];
  edgePairs.forEach(([a, b]) => {
    points.push(corners[a].x, corners[a].y, corners[a].z, corners[b].x, corners[b].y, corners[b].z);
  });
  frustum.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  frustum.geometry.computeBoundingSphere();
  const isSelected = id === state.selectedId;
  const isFeatured = id === state.stageCameraId;
  frustum.material.color.set(isSelected ? selectedColor : isFeatured ? featuredColor : calmColor);
  frustum.material.opacity = isSelected ? 0.72 : isFeatured ? 0.58 : 0.28;
  frustum.visible = true;
}

function orientFloater(mesh, face) {
  const normal = normalForFace(face);
  const up = new THREE.Vector3(0, 1, 0);
  mesh.quaternion.setFromUnitVectors(up, normal);
}

function orientEndoscope(mesh, placement) {
  const origin = normToWorld(placement.position);
  const target = normToWorld(placement.target || { x: 0.5, y: 0.5, z: 0.5 });
  const direction = target.sub(origin).normalize();
  if (direction.lengthSq() < 0.01) return;
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
}

function orientRail(mesh, face) {
  if (face === 'x+' || face === 'x-') {
    mesh.rotation.set(0, Math.PI / 2, 0);
  } else {
    mesh.rotation.set(0, 0, 0);
  }
}

function updateMesh(item) {
  const id = cameraId(item) || item.item_id;
  if (!id) return;
  const placement = placementOf(item);
  let mesh = state.meshes.get(id);
  if (!placement.placed || !placement.position) {
    if (mesh) mesh.visible = false;
    return;
  }
  if (!mesh) {
    mesh = createMesh(item);
    state.meshes.set(id, mesh);
  }
  if (isLighthouse(item) || isReeflex(item)) {
    const face = placement.mount_face || 'z+';
    placement.mount_face = face;
    if (placement.position) {
      if (face === 'x+') placement.position.x = 1;
      if (face === 'x-') placement.position.x = 0;
      if (face === 'z+') placement.position.z = 1;
      if (face === 'z-') placement.position.z = 0;
      placement.position.y = 1;
    }
    placement.target = placement.target || {
      x: isReeflex(item) ? 0.5 : placement.position?.x || 0.5,
      y: isReeflex(item) ? 0.62 : 0.48,
      z: isReeflex(item) ? 0.5 : placement.position?.z || 0.5,
    };
  }
  mesh.visible = true;
  mesh.position.copy(normToWorld(placement.position));
  mesh.position.multiplyScalar(TANK_DISPLAY_SCALE);
  mesh.position.x += tankSceneOffset(item.tank_id);
  if (item.item_type !== 'structure_shape') setObjectColor(mesh, displayColorForItem(id));
  const baseScale = Number(item.scale || 1) * TANK_DISPLAY_SCALE;
  const scale = id === state.stageCameraId && id !== state.selectedId ? baseScale * 1.12 : baseScale;
  mesh.scale.setScalar(scale);
  if (item.item_type === 'structure_shape') mesh.rotation.y = Number(item.rotation || 0) * Math.PI / 180;
  if (isFloater(item)) orientFloater(mesh, placement.mount_face || 'y+');
  else if (isEndoscope(item)) orientEndoscope(mesh, placement);
  else if (isRobotArm(item) || isLighthouse(item) || isReeflex(item)) orientRail(mesh, placement.mount_face || 'z+');
  updateFrustum(item);
}

function allItems() {
  const tankId = activeTank().tank_id || 'main-tank';
  return [
    ...(state.layout.cameras || []),
    ...(state.layout.scene_items || []),
  ].filter(item => !item.hidden_from_layout && (!item.tank_id || item.tank_id === tankId));
}

function allCamerasForActiveTank() {
  const tankId = activeTank().tank_id || 'main-tank';
  return (state.layout.cameras || []).filter(item => !item.hidden_from_layout && (!item.tank_id || item.tank_id === tankId));
}

function allCameraRecordsForActiveTank() {
  const tankId = activeTank().tank_id || 'main-tank';
  return (state.layout.cameras || []).filter(item => !item.tank_id || item.tank_id === tankId);
}

function selectedItem() {
  return allItems().find(item => (cameraId(item) || item.item_id) === state.selectedId) || null;
}

function unidentifiedConnectedCameras() {
  return allCamerasForActiveTank().filter(item =>
    isConnectedItem(item) &&
    isCameraItem(item) &&
    item.source_type === 'usb_camera' &&
    !item.role_locked &&
    !placementOf(item).placed
  );
}

function renderUnplaced() {
  const disconnected = allItems().filter(item => !placementOf(item).placed && !isConnectedItem(item));
  const unplaced = allItems().filter(item => !placementOf(item).placed && isConnectedItem(item));
  if (!unplaced.length) {
    unplacedList.innerHTML = disconnected.length
      ? `<div class="empty">No connected devices need placement. ${disconnected.length} disconnected item${disconnected.length === 1 ? '' : 's'} hidden until the node returns.</div>`
      : '<div class="empty">All connected devices placed</div>';
    return;
  }
  unplacedList.innerHTML = '';
  unplaced.forEach(item => {
    const id = cameraId(item) || item.item_id;
    const button = document.createElement('button');
    button.className = 'feed-button';
    button.innerHTML = `<span class="name"></span><span class="kind"></span>`;
    button.querySelector('.name').textContent = displayName(item);
    button.querySelector('.kind').textContent = `${itemKind(item)}${item.assigned_role ? ' / assigned' : ''}`;
    button.addEventListener('click', () => {
      state.pendingId = id;
      state.selectedId = id;
      renderSelection();
      updateAllMeshes();
    });
    unplacedList.appendChild(button);
  });
}

function assignmentOptionsFor(item) {
  if (!item) return [];
  return [
    {
      label: 'Reel',
      detail: 'Inside-tank Reel camera feed',
      camera_type: 'endoscope_cam',
      item_type: 'endoscope_cam',
      name: 'Reel Camera',
    },
    {
      label: 'Raydar',
      detail: 'Pan-tilt camera on the rim',
      camera_type: 'lighthouse_cam',
      item_type: 'lighthouse',
      name: 'Raydar Camera',
      device_id: 'lighthouse-001',
    },
    {
      label: 'Reeflex',
      detail: 'Robot-arm camera or arm-mounted view',
      camera_type: 'reeflex_cam',
      item_type: 'reeflex',
      name: 'Reeflex Camera',
      device_id: 'reeflex-001',
    },
  ].map(option => ({
    ...option,
    apply: () => assignCameraRole(item, option),
  }));
}

function assignCameraRole(item, option) {
  if (!item || !isCameraItem(item)) return activeTank();
  const label = `${option.name} #1`;
  Object.assign(item, {
    camera_type: option.camera_type,
    item_type: option.item_type,
    label,
    name: label,
    device_id: option.device_id || item.device_id || null,
    assigned_role: option.label.toLowerCase(),
    role_locked: true,
    hidden_from_layout: false,
  });
  if (!item.placement) item.placement = { placed: false, position: null, target: null, fov_degrees: option.camera_type === 'endoscope_cam' ? 70 : 60 };
  state.selectedId = cameraId(item);
  state.pendingId = placementOf(item).placed ? null : cameraId(item);
  return activeTank();
}

function setCameraHidden(item, hidden) {
  if (!item || !isCameraItem(item)) return activeTank();
  item.hidden_from_layout = hidden;
  if (hidden) {
    item.placement = { placed: false, position: null, target: null, fov_degrees: placementOf(item).fov_degrees || 70 };
    if (state.selectedId === cameraId(item)) state.selectedId = null;
    if (state.pendingId === cameraId(item)) state.pendingId = null;
  }
  return activeTank();
}

async function removeSelectedItem() {
  const item = selectedItem();
  if (!item) return;
  const id = cameraId(item) || item.item_id;
  state.selectedId = null;
  state.pendingId = null;
  if (isCameraItem(item)) {
    item.hidden_from_layout = true;
    item.placement = { placed: false, position: null, target: null, fov_degrees: placementOf(item).fov_degrees || 70 };
    state.layout.cameras = (state.layout.cameras || []).filter(camera => cameraId(camera) !== id);
    await fetch('/api/layout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delete_camera_ids: [id], cameras: state.layout.cameras || [] }),
    });
  } else {
    state.layout.scene_items = (state.layout.scene_items || []).filter(sceneItem => sceneItem.item_id !== id);
    await fetch('/api/layout', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delete_scene_item_ids: [id], scene_items: state.layout.scene_items || [] }),
    });
  }
  renderSelection();
  renderUnplaced();
  renderHud();
  renderFeedMarquee();
  renderLiveView();
  updateStageFeed();
  updateAllMeshes();
  refreshLayout().catch(() => {});
}

function sourceForPreview(item) {
  if (!item) return '';
  const id = cameraId(item);
  if (!id) return '';
  if (item.stream_url) return `/api/cameras/${encodeURIComponent(id)}/stream`;
  if (item.snapshot_url || item.latest_image_url) return `/api/cameras/${encodeURIComponent(id)}/snapshot?t=${Date.now()}`;
  return '';
}

function snapshotSource(item, reason = 'snapshot') {
  const id = cameraId(item);
  if (!id) return '';
  if (item.snapshot_url || item.latest_image_url || item.stream_url) {
    return `/api/cameras/${encodeURIComponent(id)}/snapshot?reason=${encodeURIComponent(reason)}&t=${Date.now()}`;
  }
  return '';
}

function observationsForActiveTank() {
  return (state.layout.observations || state.layout.detections || [])
    .filter(item => !item.tank_id || item.tank_id === activeTank().tank_id)
    .sort((a, b) => observationActivityScore(b) - observationActivityScore(a));
}

function observationsForAllTanks() {
  return (state.layout.observations || state.layout.detections || [])
    .slice()
    .sort((a, b) => observationActivityScore(b) - observationActivityScore(a));
}

function observationActivityScore(observation) {
  if (!observation) return 0;
  const nowSeconds = Date.now() / 1000;
  const ageSeconds = Math.max(0, nowSeconds - Number(observation.created_at || nowSeconds));
  const recency = clamp(1 - ageSeconds / 120, 0, 1);
  const motion = clamp(Number(observation.motion_score || 0) * 8, 0, 1);
  const confidence = clamp(Number(observation.classifier_confidence || observation.fish_candidate_score || 0), 0, 1);
  const classifierBoost = observation.classifier_label && observation.classifier_label !== 'motion' ? 0.18 : 0;
  return Number(observation.activity_score || 0) || (recency * 0.42 + motion * 0.4 + confidence * 0.18 + classifierBoost);
}

function bestObservation() {
  const nowSeconds = Date.now() / 1000;
  return observationsForActiveTank()
    .find(observation => nowSeconds - Number(observation.created_at || nowSeconds) <= 120) || null;
}

function cameraForObservation(observation) {
  if (!observation || !observation.camera_id) return null;
  return allCamerasForActiveTank().find(item => cameraId(item) === observation.camera_id) || null;
}

function pinView(cameraIdValue) {
  state.manualViewId = cameraIdValue;
  state.manualViewUntil = Date.now() + 60000;
}

function clearExpiredPin() {
  if (state.manualViewUntil && Date.now() > state.manualViewUntil) {
    state.manualViewId = null;
    state.manualViewUntil = 0;
  }
}

function activeView() {
  clearExpiredPin();
  const selected = selectedItem();
  const selectedSrc = sourceForPreview(selected);
  if (selected && selectedSrc) {
    return {
      type: 'selected',
      camera: selected,
      src: selectedSrc,
      label: `${displayName(selected)} / selected`,
      tank_id: selected.tank_id || activeTank().tank_id,
    };
  }
  if (state.manualViewId) {
    const pinnedCamera = liveCameras().find(item => cameraId(item) === state.manualViewId);
    const pinnedSrc = sourceForPreview(pinnedCamera);
    if (pinnedCamera && pinnedSrc) {
      return {
        type: 'pinned',
        camera: pinnedCamera,
        src: pinnedSrc,
        label: `${displayName(pinnedCamera)} / pinned`,
        tank_id: pinnedCamera.tank_id || activeTank().tank_id,
      };
    }
  }
  const observation = bestObservation();
  const observationCamera = cameraForObservation(observation);
  const observationLiveSrc = sourceForPreview(observationCamera);
  const frame = observation && (observation.frame_urls || [])[0];
  if (observation && (observationLiveSrc || frame)) {
    return {
      type: 'observation',
      observation,
      camera: observationCamera,
      src: observationLiveSrc || frame,
      label: `${text(observation.camera_id, 'camera')} / ${observationLiveSrc ? 'live ' : ''}${text(observation.classifier_label || observation.event_type, 'motion')} ${text(observation.motion_score, '')}`,
      tank_id: observation.tank_id || activeTank().tank_id,
    };
  }
  const cameras = rotatingLiveCameras();
  const cameraItem = cameras[state.liveIndex] || cameras[0];
  const src = sourceForPreview(cameraItem);
  if (cameraItem && src) {
    return {
      type: 'live',
      camera: cameraItem,
      src,
      label: `${displayName(cameraItem)} / live`,
      tank_id: cameraItem.tank_id || activeTank().tank_id,
    };
  }
  return null;
}

function updateStageFeed() {
  if (!stageFeedBackdrop || !stageFeedImage || !stageFeedLabel) return;
  const previousStageCameraId = state.stageCameraId;
  const view = activeView();
  if (!view || !view.src) {
    stageFeedBackdrop.hidden = true;
    state.stageCameraId = null;
    stageFeedBackdrop.dataset.cameraId = '';
    updateLiveControlHighlights();
    if (feedEmpty) feedEmpty.hidden = false;
    renderFeedThumbnails();
    if (previousStageCameraId) updateAllMeshes();
    return;
  }
  if (feedEmpty) feedEmpty.hidden = true;
  if (view.tank_id && view.tank_id !== activeTank().tank_id && Date.now() > state.manualTankUntil) {
    state.activeTankId = view.tank_id;
    applyTankDimensions();
    renderTankTabs();
  }
  state.stageCameraId = cameraId(view.camera) || view.observation?.camera_id || null;
  stageFeedBackdrop.hidden = false;
  stageFeedBackdrop.dataset.viewType = view.type || 'live';
  stageFeedBackdrop.dataset.cameraId = state.stageCameraId || '';
  if (view.src !== state.stageFeedSource) {
    state.stageFeedSource = view.src;
    stageFeedImage.src = view.src;
  }
  const tank = (state.layout.tanks || []).find(item => item.tank_id === view.tank_id);
  stageFeedLabel.textContent = `${text(tank?.label, view.tank_id || 'Tank')} · ${view.label} · ${view.type === 'pinned' ? 'PINNED' : 'LIVE'}`;
  renderFeedThumbnails();
  updateLiveControlHighlights();
  renderLighthouseControls();
  if (previousStageCameraId !== state.stageCameraId) updateAllMeshes();
}

function returnHome() {
  stopLighthouseHold();
  stopReeflexHold();
  state.selectedId = null;
  state.pendingId = null;
  state.setupDraft = {};
  state.initialSetupPrompted = true;
  state.lighthouse.panelClosed = true;
  state.reeflex.panelClosed = true;
  if (setupOverlay) setupOverlay.hidden = true;
  if (sightingsDrawer) sightingsDrawer.hidden = true;
  if (deepDialog?.open) deepDialog.close();
  if (sidePanel) sidePanel.classList.remove('open');
  if (dockToggle) {
    dockToggle.classList.remove('open');
    dockToggle.setAttribute('aria-expanded', 'false');
    dockToggle.setAttribute('aria-label', 'Open placement controls');
  }
  if (structureToolbar) structureToolbar.hidden = true;
  if (decorateToggle) decorateToggle.hidden = false;
  renderSelection();
  renderLighthouseControls();
  renderReeflexControls();
  updateAllMeshes();
}

function updateLiveControlHighlights() {
  if (!liveControls) return;
  liveControls.querySelectorAll('button[data-camera-id]').forEach(button => {
    const isCurrent = button.dataset.cameraId === state.stageCameraId;
    button.classList.toggle('current-view', Boolean(isCurrent));
    const stateLabel = button.querySelector('.view-state');
    if (stateLabel && isCurrent && !stateLabel.textContent.startsWith('motion')) {
      stateLabel.textContent = 'showing now';
    }
  });
}

function renderFloaterSideFeeds() {
  const floaters = snapshotCameras()
    .filter(item => item.status !== 'offline')
    .slice(0, 2);
  const slots = [
    [floaterLeft, floaterLeftImage, floaterLeftLabel],
    [floaterRight, floaterRightImage, floaterRightLabel],
  ];
  slots.forEach(([panel, image, label], index) => {
    const item = floaters[index];
    if (!panel || !image || !label || !item) {
      if (panel) panel.hidden = true;
      return;
    }
    const id = cameraId(item);
    const src = id ? `/api/cameras/${encodeURIComponent(id)}/snapshot?reason=floater-side&t=${Math.floor(Date.now() / 10000)}` : '';
    if (!src) {
      panel.hidden = true;
      return;
    }
    const isCurrent = id && id === state.stageCameraId;
    panel.hidden = false;
    panel.classList.toggle('current-view', Boolean(isCurrent));
    if (image.dataset.src !== src) {
      image.dataset.src = src;
      panel.classList.add('peek');
      panel.classList.remove('flash');
      void panel.offsetWidth;
      panel.classList.add('flash');
      window.clearTimeout(panel._peekTimer);
      panel._peekTimer = window.setTimeout(() => panel.classList.remove('peek'), 2400);
      window.clearTimeout(panel._flashTimer);
      panel._flashTimer = window.setTimeout(() => panel.classList.remove('flash'), 900);
    }
    if (isCurrent) panel.classList.add('peek');
    panel.onclick = () => {
      pinView(id);
      updateStageFeed();
      renderLiveView();
    };
    image.src = src;
    label.textContent = displayName(item);
  });
}

async function pollFloaterFrames() {
  const floaters = (state.layout.cameras || []).filter(item => isFloater(item) && item.status !== 'offline');
  for (const item of floaters) {
    const id = cameraId(item);
    try {
      const response = await fetch(`/api/cameras/${encodeURIComponent(id)}/snapshot?revision=${Date.now()}`, { cache: 'no-store' });
      if (!response.ok) continue;
      const bytes = await response.arrayBuffer();
      let revision = response.headers.get('ETag') || response.headers.get('Last-Modified');
      if (!revision) {
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        revision = Array.from(new Uint8Array(digest)).map(value => value.toString(16).padStart(2, '0')).join('');
      }
      const previous = state.floaterRevisions.get(id);
      state.floaterRevisions.set(id, revision);
      if (previous && previous !== revision) showFloaterMarkerImage(item);
    } catch {
      // Missing floater stills are expected when hardware is disconnected.
    }
  }
}

function showFloaterMarkerImage(item, reopened = false) {
  const id = cameraId(item);
  if (!id) return;
  let card = state.floaterCards.get(id);
  if (!card) {
    card = document.createElement('button');
    card.className = 'floater-marker-card';
    card.innerHTML = '<img alt=""><span></span>';
    stage.appendChild(card);
    state.floaterCards.set(id, card);
  }
  card.querySelector('img').src = `/api/cameras/${encodeURIComponent(id)}/snapshot?popup=${Date.now()}`;
  card.querySelector('span').textContent = displayName(item);
  card.hidden = false;
  const fishCandidate = observationsForAllTanks().some(obs => obs.camera_id === id && String(obs.label || obs.classifier_label).toLowerCase() === 'fish');
  window.clearTimeout(card._hideTimer);
  if (!fishCandidate || reopened) card._hideTimer = window.setTimeout(() => { card.hidden = true; }, 5000);
  card.onclick = () => showFloaterMarkerImage(item, true);
}

function positionFloaterCards() {
  state.floaterCards.forEach((card, id) => {
    if (card.hidden) return;
    const mesh = state.meshes.get(id);
    if (!mesh) { card.hidden = true; return; }
    const point = mesh.position.clone().project(camera);
    card.style.left = `${(point.x * .5 + .5) * stage.clientWidth}px`;
    card.style.top = `${(-point.y * .5 + .5) * stage.clientHeight}px`;
  });
}

function cameraIndexById(id) {
  return liveCameras().findIndex(item => cameraId(item) === id);
}

function liveCameras() {
  const cameras = (state.layout.cameras || []).filter(item => !item.hidden_from_layout && item.status === 'online');
  const usb = cameras.filter(item => item.stream_url);
  const rest = cameras.filter(item => !item.stream_url);
  return [...usb, ...rest];
}

function rotatingLiveCameras() {
  return (state.layout.cameras || []).filter(item => !item.hidden_from_layout && item.status === 'online' && item.stream_url);
}

function placementComplete() {
  return allItems().length > 0 && allItems().every(item => placementOf(item).placed);
}

function setLiveCamera(index) {
  const cameras = rotatingLiveCameras();
  if (!cameras.length || !liveFeed) return;
  state.liveIndex = ((index % cameras.length) + cameras.length) % cameras.length;
  const cameraItem = cameras[state.liveIndex];
  state.stageCameraId = cameraId(cameraItem);
  // The side dock uses a still so it never opens a second reader on a USB camera.
  const src = snapshotSource(cameraItem, 'side-dock');
  if (src && liveFeed.src !== src) liveFeed.src = src;
  liveTitle.textContent = displayName(cameraItem);
  liveDetail.textContent = cameraItem.stream_url
    ? `${itemKind(cameraItem)} / live MJPEG`
    : `${itemKind(cameraItem)} / snapshot refresh`;
}

function renderLiveView() {
  if (!liveSection || !placementSection) return;
  const complete = placementComplete();
  placementSection.hidden = complete;
  liveSection.hidden = !complete;
  if (!complete) return;
  const cameras = rotatingLiveCameras();
  if (!cameras.length) {
    liveTitle.textContent = 'No active streams';
    liveDetail.textContent = 'Waiting for camera video';
    liveFeed.removeAttribute('src');
    liveControls.innerHTML = '';
    return;
  }
  const observation = bestObservation();
  if (observation && observation.observation_id !== state.focusObservationId) {
    const focusIndex = cameraIndexById(observation.camera_id);
    if (focusIndex >= 0) {
      state.focusObservationId = observation.observation_id;
      state.liveIndex = focusIndex;
    }
  }
  setLiveCamera(state.liveIndex);
  liveControls.innerHTML = '';
  cameras.forEach((cameraItem, index) => {
    const button = document.createElement('button');
    const id = cameraId(cameraItem);
    const isCurrent = id && id === state.stageCameraId;
    button.className = [
      index === state.liveIndex ? 'active' : '',
      isCurrent ? 'current-view' : '',
    ].filter(Boolean).join(' ');
    button.dataset.cameraId = id || '';
    const observationOnCamera = observation && observation.camera_id === id;
    button.innerHTML = '<span class="view-name"></span><span class="view-state"></span>';
    button.querySelector('.view-name').textContent = displayName(cameraItem);
    button.querySelector('.view-state').textContent = observationOnCamera
      ? `motion ${text(observation.motion_score, '')}`
      : (isCurrent ? 'showing now' : (cameraItem.stream_url ? 'live' : 'snapshot'));
    button.addEventListener('click', () => {
      state.focusObservationId = null;
      pinView(cameraId(cameraItem));
      setLiveCamera(index);
      renderLiveView();
      updateStageFeed();
    });
    liveControls.appendChild(button);
  });
}

function lighthouseAvailableInTank() {
  const node = activeNode();
  const controlUrls = (node && node.control_urls) || {};
  const ownsRaydar = Number(manifestCounts().lighthouse || 0) > 0 || allItems().some(item => isLighthouse(item));
  return ownsRaydar && Boolean(controlUrls.lighthouse_pose);
}

function reeflexAvailableInTank() {
  const node = activeNode();
  const controlUrls = (node && node.control_urls) || {};
  const ownsReeflex = Number(manifestCounts().reeflex || 0) > 0 || allItems().some(item => isReeflex(item));
  return ownsReeflex && Boolean(controlUrls.reeflex_pose || controlUrls.reeflex_servo || controlUrls.reeflex_stop);
}

function currentLighthouseContextId() {
  const selected = selectedItem();
  if (selected && isLighthouse(selected)) return cameraId(selected) || selected.item_id;
  const stageItem = allItems().find(item => (cameraId(item) || item.item_id) === state.stageCameraId);
  if (stageItem && isLighthouse(stageItem)) return cameraId(stageItem) || stageItem.item_id;
  if (Number(manifestCounts().lighthouse || 0) > 0) return `${activeTank().tank_id}-lighthouse-controls`;
  return '';
}

function currentReeflexContextId() {
  const selected = selectedItem();
  if (selected && isReeflex(selected)) return cameraId(selected) || selected.item_id;
  const stageItem = allItems().find(item => (cameraId(item) || item.item_id) === state.stageCameraId);
  if (stageItem && isReeflex(stageItem)) return cameraId(stageItem) || stageItem.item_id;
  if (Number(manifestCounts().reeflex || 0) > 0) return `${activeTank().tank_id}-reeflex-controls`;
  return '';
}

function readServoState(payload) {
  const arm = payload && (payload.arm || payload);
  const servos = (arm && arm.servos) || {};
  const devices = (arm && arm.devices) || {};
  const pan = servos.lighthouse_pan || {};
  const tilt = servos.lighthouse_tilt || {};
  const hasPan = Number.isFinite(Number(pan.angle));
  const hasTilt = Number.isFinite(Number(tilt.angle));
  const device = Object.entries(devices).find(([, value]) => value && value.type === 'lighthouse');
  if (hasPan) state.lighthouse.pan = Number(pan.angle);
  if (hasTilt) state.lighthouse.tilt = Number(tilt.angle);
  if (pan.min_angle !== undefined) state.lighthouse.minPan = Number(pan.min_angle);
  if (pan.max_angle !== undefined) state.lighthouse.maxPan = Number(pan.max_angle);
  if (tilt.min_angle !== undefined) state.lighthouse.minTilt = Number(tilt.min_angle);
  if (tilt.max_angle !== undefined) state.lighthouse.maxTilt = Number(tilt.max_angle);
  state.lighthouse.driver = text((arm && arm.driver) || payload?.driver, '');
  state.lighthouse.deviceId = device ? device[0] : '';
  const hasPcaDriver = state.lighthouse.driver.startsWith('pca9685');
  state.lighthouse.ready = hasPcaDriver && hasPan && hasTilt && Boolean(state.lighthouse.deviceId);
  state.lighthouse.status = state.lighthouse.ready
    ? 'ready'
    : state.lighthouse.driver.startsWith('mock_')
      ? 'mock'
      : hasPcaDriver
        ? 'missing Raydar axes or device'
        : text(state.lighthouse.driver, 'unavailable');
}

function readReeflexState(payload) {
  const arm = payload && (payload.arm || payload);
  const servos = (arm && arm.servos) || {};
  const devices = (arm && arm.devices) || {};
  const base = servos.reeflex_base || servos.base || {};
  const shoulder = servos.reeflex_shoulder || servos.shoulder || {};
  const elbow = servos.reeflex_elbow || servos.elbow || {};
  const hasBase = Number.isFinite(Number(base.angle));
  const hasShoulder = Number.isFinite(Number(shoulder.angle));
  const hasElbow = Number.isFinite(Number(elbow.angle));
  const device = Object.entries(devices).find(([, value]) => value && value.type === 'reeflex');
  if (hasBase) state.reeflex.base = Number(base.angle);
  if (hasShoulder) state.reeflex.shoulder = Number(shoulder.angle);
  if (hasElbow) state.reeflex.elbow = Number(elbow.angle);
  if (base.min_angle !== undefined) state.reeflex.minBase = Number(base.min_angle);
  if (base.max_angle !== undefined) state.reeflex.maxBase = Number(base.max_angle);
  if (shoulder.min_angle !== undefined) state.reeflex.minShoulder = Number(shoulder.min_angle);
  if (shoulder.max_angle !== undefined) state.reeflex.maxShoulder = Number(shoulder.max_angle);
  if (elbow.min_angle !== undefined) state.reeflex.minElbow = Number(elbow.min_angle);
  if (elbow.max_angle !== undefined) state.reeflex.maxElbow = Number(elbow.max_angle);
  state.reeflex.driver = text((arm && arm.driver) || payload?.driver, '');
  state.reeflex.deviceId = device ? device[0] : '';
  const hasPcaDriver = state.reeflex.driver.startsWith('pca9685');
  state.reeflex.ready = hasPcaDriver && hasBase && hasShoulder && hasElbow && Boolean(state.reeflex.deviceId);
  state.reeflex.status = state.reeflex.ready
    ? 'ready'
    : state.reeflex.driver.startsWith('mock_')
      ? 'mock'
      : hasPcaDriver
        ? 'missing reeflex axes'
        : text(state.reeflex.driver, 'unavailable');
}

function lighthouseEnabled() {
  return Boolean(state.lighthouse.ready);
}

function reeflexEnabled() {
  return Boolean(state.reeflex.ready);
}

function renderLighthouseControls() {
  if (!lighthouseControlSection) return;
  const available = lighthouseAvailableInTank();
  const contextId = currentLighthouseContextId();
  if (contextId && contextId !== state.lighthouse.contextId) {
    const selected = selectedItem();
    const stageItem = allItems().find(item => (cameraId(item) || item.item_id) === state.stageCameraId);
    state.lighthouse.panelClosed = !((selected && isLighthouse(selected)) || (stageItem && isLighthouse(stageItem)));
    state.lighthouse.contextId = contextId;
  }
  const visible = Boolean(available && contextId && !state.lighthouse.panelClosed);
  lighthouseControlSection.hidden = !visible;
  if (!visible) return;
  const enabled = lighthouseEnabled();
  lighthouseControlSection.classList.toggle('disabled', !enabled);
  lighthouseCurrent.textContent = `Pan ${Math.round(state.lighthouse.pan)} / Tilt ${Math.round(state.lighthouse.tilt)}`;
  lighthouseStatus.textContent = enabled
    ? `Driver: ${state.lighthouse.driver}`
    : state.lighthouse.status === 'mock'
      ? 'Servo controller not connected'
      : 'Servo status unavailable';
  lighthousePanSlider.min = state.lighthouse.minPan;
  lighthousePanSlider.max = state.lighthouse.maxPan;
  lighthousePanSlider.value = clamp(state.lighthouse.pan, state.lighthouse.minPan, state.lighthouse.maxPan);
  lighthouseTiltSlider.min = state.lighthouse.minTilt;
  lighthouseTiltSlider.max = state.lighthouse.maxTilt;
  lighthouseTiltSlider.value = clamp(state.lighthouse.tilt, state.lighthouse.minTilt, state.lighthouse.maxTilt);
  document.querySelectorAll('[data-lighthouse-move]').forEach(button => {
    button.disabled = !enabled;
  });
  document.querySelectorAll('.lighthouse-sliders input').forEach(input => {
    input.disabled = !enabled;
  });
}

function renderReeflexControls() {
  if (!reeflexControlSection) return;
  const available = reeflexAvailableInTank();
  const contextId = currentReeflexContextId();
  if (contextId && contextId !== state.reeflex.contextId) {
    const selected = selectedItem();
    const stageItem = allItems().find(item => (cameraId(item) || item.item_id) === state.stageCameraId);
    state.reeflex.panelClosed = !((selected && isReeflex(selected)) || (stageItem && isReeflex(stageItem)));
    state.reeflex.contextId = contextId;
  }
  const visible = Boolean(available && contextId && !state.reeflex.panelClosed);
  reeflexControlSection.hidden = !visible;
  if (!visible) return;
  const enabled = reeflexEnabled();
  reeflexControlSection.classList.toggle('disabled', !enabled);
  reeflexCurrent.textContent = `Base ${Math.round(state.reeflex.base)} / Shoulder ${Math.round(state.reeflex.shoulder)} / Elbow ${Math.round(state.reeflex.elbow)}`;
  reeflexStatus.textContent = enabled
    ? `Driver: ${state.reeflex.driver}`
    : state.reeflex.status === 'mock'
      ? 'Servo controller not connected'
      : state.reeflex.status === 'missing reeflex axes'
        ? 'Reeflex axes not reported by tank node'
        : 'Servo status unavailable';
  reeflexBaseSlider.min = state.reeflex.minBase;
  reeflexBaseSlider.max = state.reeflex.maxBase;
  reeflexBaseSlider.value = clamp(state.reeflex.base, state.reeflex.minBase, state.reeflex.maxBase);
  reeflexShoulderSlider.min = state.reeflex.minShoulder;
  reeflexShoulderSlider.max = state.reeflex.maxShoulder;
  reeflexShoulderSlider.value = clamp(state.reeflex.shoulder, state.reeflex.minShoulder, state.reeflex.maxShoulder);
  reeflexElbowSlider.min = state.reeflex.minElbow;
  reeflexElbowSlider.max = state.reeflex.maxElbow;
  reeflexElbowSlider.value = clamp(state.reeflex.elbow, state.reeflex.minElbow, state.reeflex.maxElbow);
  document.querySelectorAll('[data-reeflex-move], [data-reeflex-center], [data-reeflex-stop]').forEach(button => {
    button.disabled = !enabled && !button.dataset.reeflexStop;
  });
  document.querySelectorAll('.reeflex-sliders input').forEach(input => {
    input.disabled = !enabled;
  });
}

async function refreshLighthouseStatus() {
  if (!lighthouseAvailableInTank()) {
    state.lighthouse.ready = false;
    state.lighthouse.driver = '';
    state.lighthouse.deviceId = '';
    renderLighthouseControls();
    return;
  }
  const tankId = activeTank().tank_id;
  try {
    const response = await fetch(`/api/controls/arm?tank_id=${encodeURIComponent(tankId)}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`status ${response.status}`);
    const payload = await response.json();
    if (activeTank().tank_id !== tankId) return;
    readServoState(payload);
  } catch {
    if (activeTank().tank_id !== tankId) return;
    state.lighthouse.status = 'unavailable';
    state.lighthouse.driver = '';
    state.lighthouse.deviceId = '';
    state.lighthouse.ready = false;
  }
  renderLighthouseControls();
}

async function refreshReeflexStatus() {
  if (!reeflexAvailableInTank()) {
    state.reeflex.ready = false;
    state.reeflex.driver = '';
    state.reeflex.deviceId = '';
    renderReeflexControls();
    return;
  }
  const tankId = activeTank().tank_id;
  try {
    const response = await fetch(`/api/controls/arm?tank_id=${encodeURIComponent(tankId)}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`status ${response.status}`);
    const payload = await response.json();
    if (activeTank().tank_id !== tankId) return;
    readReeflexState(payload);
  } catch {
    if (activeTank().tank_id !== tankId) return;
    state.reeflex.status = 'unavailable';
    state.reeflex.driver = '';
    state.reeflex.deviceId = '';
    state.reeflex.ready = false;
  }
  renderReeflexControls();
}

function lighthousePoseForMove(move) {
  const step = state.lighthouse.step;
  let pan = state.lighthouse.pan;
  let tilt = state.lighthouse.tilt;
  if (move === 'center') {
    pan = 90;
    tilt = 90;
  }
  if (move === 'left') pan -= step;
  if (move === 'right') pan += step;
  if (move === 'up') tilt += step;
  if (move === 'down') tilt -= step;
  return {
    pan: clamp(pan, state.lighthouse.minPan, state.lighthouse.maxPan),
    tilt: clamp(tilt, state.lighthouse.minTilt, state.lighthouse.maxTilt),
  };
}

function stopLighthouseHold() {
  window.clearInterval(state.lighthouse.holdTimer);
  state.lighthouse.holdTimer = null;
}

function stopReeflexHold() {
  window.clearInterval(state.reeflex.holdTimer);
  state.reeflex.holdTimer = null;
}

async function sendLighthousePose(pan, tilt) {
  if (!lighthouseEnabled()) return;
  const tankId = activeTank().tank_id;
  const payload = {
    device_id: state.lighthouse.deviceId,
    tank_id: tankId,
    pan: Math.round(clamp(pan, state.lighthouse.minPan, state.lighthouse.maxPan)),
    tilt: Math.round(clamp(tilt, state.lighthouse.minTilt, state.lighthouse.maxTilt)),
  };
  try {
    const response = await fetch('/api/controls/lighthouse/pose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      stopLighthouseHold();
      lighthouseStatus.textContent = text(body.error || body.message, `Command rejected ${response.status}`);
      return;
    }
    if (activeTank().tank_id !== tankId) return;
    readServoState(body);
    renderLighthouseControls();
  } catch {
    stopLighthouseHold();
    lighthouseStatus.textContent = 'Command timed out';
  }
}

function reeflexPoseForMove(move) {
  const step = state.reeflex.step;
  let base = state.reeflex.base;
  let shoulder = state.reeflex.shoulder;
  let elbow = state.reeflex.elbow;
  if (move === 'base-') base -= step;
  if (move === 'base+') base += step;
  if (move === 'shoulder-') shoulder -= step;
  if (move === 'shoulder+') shoulder += step;
  if (move === 'elbow-') elbow -= step;
  if (move === 'elbow+') elbow += step;
  return {
    base: clamp(base, state.reeflex.minBase, state.reeflex.maxBase),
    shoulder: clamp(shoulder, state.reeflex.minShoulder, state.reeflex.maxShoulder),
    elbow: clamp(elbow, state.reeflex.minElbow, state.reeflex.maxElbow),
  };
}

async function sendReeflexPose(base, shoulder, elbow) {
  if (!reeflexEnabled()) return;
  const tankId = activeTank().tank_id;
  const payload = {
    device_id: state.reeflex.deviceId,
    tank_id: tankId,
    base: Math.round(clamp(base, state.reeflex.minBase, state.reeflex.maxBase)),
    shoulder: Math.round(clamp(shoulder, state.reeflex.minShoulder, state.reeflex.maxShoulder)),
    elbow: Math.round(clamp(elbow, state.reeflex.minElbow, state.reeflex.maxElbow)),
  };
  try {
    const response = await fetch('/api/controls/reeflex/pose', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      stopReeflexHold();
      reeflexStatus.textContent = text(body.error || body.message, `Command rejected ${response.status}`);
      return;
    }
    if (activeTank().tank_id !== tankId) return;
    readReeflexState(body);
    renderReeflexControls();
  } catch {
    stopReeflexHold();
    reeflexStatus.textContent = 'Command timed out';
  }
}

function moveReeflex(move) {
  const pose = reeflexPoseForMove(move);
  sendReeflexPose(pose.base, pose.shoulder, pose.elbow);
}

function startReeflexHold(move) {
  stopReeflexHold();
  moveReeflex(move);
  state.reeflex.holdTimer = window.setInterval(() => moveReeflex(move), 220);
}

async function stopReeflexMotion() {
  stopReeflexHold();
  try {
    await fetch('/api/controls/reeflex/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device_id: state.reeflex.deviceId, tank_id: activeTank().tank_id }),
    });
  } catch {
    if (reeflexStatus) reeflexStatus.textContent = 'Stop command unavailable';
  }
}

function moveLighthouse(move) {
  const pose = lighthousePoseForMove(move);
  sendLighthousePose(pose.pan, pose.tilt);
}

function startLighthouseHold(move) {
  stopLighthouseHold();
  moveLighthouse(move);
  if (move === 'center') return;
  state.lighthouse.holdTimer = window.setInterval(() => moveLighthouse(move), 220);
}

function ensureLiveRotation() {
  if (state.liveTimer) return;
  state.liveTimer = window.setInterval(() => {
    if (setupOverlay.hidden === false) return;
    clearExpiredPin();
    if (state.manualViewId) return;
    const cameras = rotatingLiveCameras();
    if (cameras.length > 1) {
      setLiveCamera(state.liveIndex + 1);
      renderLiveView();
      updateStageFeed();
    } else if (cameras.length === 1 && !cameras[0].stream_url) {
      setLiveCamera(0);
      updateStageFeed();
    }
  }, 8000);
}

function renderFeedThumbnails() {
  if (!feedThumbnails) return;
  feedThumbnails.innerHTML = '';
  const cameras = rotatingLiveCameras();
  cameras.forEach((item, index) => {
    const button = document.createElement('button');
    const id = cameraId(item);
    const tank = (state.layout.tanks || []).find(entry => entry.tank_id === item.tank_id);
    button.className = `feed-thumb${id === state.stageCameraId ? ' active' : ''}`;
    button.innerHTML = `<img alt=""><span></span>`;
    button.querySelector('img').src = `/api/cameras/${encodeURIComponent(id)}/snapshot?thumb=1`;
    button.querySelector('span').textContent = `${text(tank?.label, item.tank_id)} · ${displayName(item)}`;
    button.addEventListener('click', () => {
      state.liveIndex = index;
      pinView(id);
      updateStageFeed();
    });
    feedThumbnails.appendChild(button);
  });
}

async function captureActiveSighting() {
  if (!state.stageCameraId) return;
  sightingShutter.disabled = true;
  const cameraItem = (state.layout.cameras || []).find(item => cameraId(item) === state.stageCameraId);
  try {
    const response = await fetch('/api/sightings/capture', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: state.stageCameraId, tank_id: cameraItem?.tank_id, trigger: 'manual', label: 'Unknown' }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Capture failed');
    cctvState.textContent = 'Sighting captured';
    await loadSightings();
  } catch (error) {
    cctvState.textContent = error.message;
  } finally {
    sightingShutter.disabled = false;
  }
}

function fieldNoteMarkup(note) {
  if (!note) return '';
  return `<div class="field-note"><strong>${text(note.label)}</strong><p>${text(note.visual_evidence)}</p><p><b>Possible subject:</b> ${text(note.possible_subject, 'Unknown')} · <b>Uncertainty:</b> ${text(note.uncertainty, 'High')}</p><p>${text(note.interesting_fact)}</p><em>${text(note.narration)}</em></div>`;
}

async function loadSightings() {
  if (!sightingsGrid) return;
  const response = await fetch('/api/sightings', { cache: 'no-store' });
  const data = await response.json();
  sightingsGrid.innerHTML = '';
  if (!(data.sightings || []).length) sightingsGrid.innerHTML = '<div class="empty">No sightings yet. Use Capture on a live feed.</div>';
  (data.sightings || []).forEach(sighting => {
    const card = document.createElement('article');
    card.className = 'sighting-card';
    card.innerHTML = `<img src="${sighting.image_url}" alt="Captured aquarium sighting"><div class="sighting-meta"><strong>${text(sighting.label, 'Unknown')}</strong><span>${text(sighting.tank_id)} · ${text(sighting.camera_id)}</span><time>${new Date(sighting.timestamp * 1000).toLocaleString()}</time><div class="sighting-labels"><select aria-label="Sighting label">${(data.labels || []).map(label => `<option${label === sighting.label ? ' selected' : ''}>${label}</option>`).join('')}</select><button class="favorite">${sighting.favorite ? '★ Favorite' : '☆ Favorite'}</button></div><button class="ask-deep">✦ Ask the Deep</button><small>Sends this captured image to OpenAI for analysis</small>${fieldNoteMarkup(sighting.ai_field_note)}</div>`;
    const update = payload => fetch(`/api/sightings/${encodeURIComponent(sighting.sighting_id)}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).then(() => loadSightings());
    card.querySelector('select').addEventListener('change', event => update({ label: event.target.value }));
    card.querySelector('.favorite').addEventListener('click', () => update({ favorite: !sighting.favorite }));
    card.querySelector('.ask-deep').disabled = !data.ask_the_deep.enabled;
    card.querySelector('.ask-deep').addEventListener('click', () => {
      pendingDeepSightingId = sighting.sighting_id;
      deepImage.src = sighting.image_url;
      deepDialog.showModal();
    });
    sightingsGrid.appendChild(card);
  });
}

async function analyzePendingSighting(event) {
  event.preventDefault();
  if (!pendingDeepSightingId) return;
  const response = await fetch(`/api/sightings/${encodeURIComponent(pendingDeepSightingId)}/analyze`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ confirmed: true }),
  });
  const result = await response.json();
  deepDialog.close();
  cctvState.textContent = response.ok ? 'AI field note saved' : text(result.error, 'Ask the Deep unavailable');
  await loadSightings();
}

function structureRecord(kind, index = 0) {
  const snap = value => Math.round(clamp(value, 0.05, 0.95) / 0.05) * 0.05;
  return {
    item_id: `structure-${Date.now()}-${index}`,
    item_type: 'structure_shape', structure_type: kind, tank_id: activeTank().tank_id,
    label: `${kind.replace('-', ' ')} ${index + 1}`, color: ['#698f88', '#927c62', '#647f94', '#8c6d78'][index % 4],
    rotation: (index * 45) % 360, scale: 1,
    placement: { placed: true, position: { x: snap(0.2 + ((index * 0.23) % 0.65)), y: 0.08, z: snap(0.2 + ((index * 0.31) % 0.65)) }, target: null },
  };
}

async function saveStructures(items) {
  state.layout.scene_items = [...(state.layout.scene_items || []), ...items];
  await fetch('/api/layout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scene_items: items }) });
  updateAllMeshes();
}

function addStructure() {
  const kind = document.getElementById('structure-type')?.value || 'block';
  saveStructures([structureRecord(kind)]).catch(() => {});
}

function scatterStructures() {
  const kinds = ['block', 'slab', 'rounded-rock', 'pillar', 'arch', 'mound'];
  const count = 3 + Math.floor(Math.random() * 4);
  saveStructures(Array.from({ length: count }, (_, index) => structureRecord(kinds[index % kinds.length], index))).catch(() => {});
}

function associatedRigCamera(kind) {
  const tankId = activeTank().tank_id;
  return (state.layout.cameras || []).find(item => item.tank_id === tankId && (kind === 'raydar' ? isLighthouse(item) : isReeflex(item)));
}

async function setAutonomy(rig, action) {
  const cameraItem = associatedRigCamera(rig);
  const response = await fetch(`/api/vision/${rig}/${action}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tank_id: activeTank().tank_id,
      node_id: cameraItem?.node_id,
      camera_id: cameraId(cameraItem),
      reason: action === 'stop' ? 'Dashboard STOP' : undefined,
    }),
  });
  const result = await response.json();
  if (!response.ok) cctvState.textContent = result.error || `${rig} unavailable`;
  await refreshVisionStatus();
}

async function refreshVisionStatus() {
  const response = await fetch('/api/vision/status', { cache: 'no-store' });
  if (!response.ok) return;
  const status = await response.json();
  const raydarMode = document.getElementById('raydar-mode');
  const reeflexMode = document.getElementById('reeflex-mode');
  const raydarAvailable = lighthouseAvailableInTank();
  const reeflexAvailable = reeflexAvailableInTank();
  [raydarMode, document.getElementById('raydar-survey'), document.getElementById('raydar-stop')]
    .forEach(element => { if (element) element.hidden = !raydarAvailable; });
  [reeflexMode, document.getElementById('reeflex-survey'), document.getElementById('reeflex-auto-stop')]
    .forEach(element => { if (element) element.hidden = !reeflexAvailable; });
  const toolbar = document.querySelector('.autonomy-toolbar');
  if (toolbar) toolbar.hidden = !raydarAvailable && !reeflexAvailable;
  if (raydarMode) { raydarMode.textContent = `Raydar · ${status.raydar.state}`; raydarMode.title = status.raydar.detail; }
  if (reeflexMode) { reeflexMode.textContent = `Reeflex · ${status.reeflex.state}`; reeflexMode.title = status.reeflex.detail; }
  if (stageFeedBackdrop) {
    stageFeedBackdrop.classList.toggle('tracking-crop', status.raydar.state === 'Track' && state.stageCameraId === status.raydar.camera_id);
    if (status.raydar.target) stageFeedImage.style.transformOrigin = `${status.raydar.target.x * 100}% ${status.raydar.target.y * 100}%`;
  }
}

function renderFeedMarquee() {
  const cameras = allCamerasForActiveTank();
  const online = cameras.filter(item => item.status === 'online').length;
  const usb = usbVideoCameras();
  const usbOnline = usb.filter(item => item.status === 'online').length;
  const unplaced = allItems().filter(item => !placementOf(item).placed).length;
  const observers = (state.layout.observations || state.layout.detections || [])
    .filter(item => !item.tank_id || item.tank_id === activeTank().tank_id).length;
  const signature = cameras.map(item => [
    cameraId(item),
    item.status || '',
    placementOf(item).placed ? 'placed' : 'unplaced',
  ].join('|')).join('::') + `:${online}:${usbOnline}:${usb.length}:${unplaced}:${observers}:${allItems().length}`;
  if (signature === state.feedSignature && feedMarquee.querySelector('.status-pill')) {
    return;
  }
  state.feedSignature = signature;
  feedMarquee.innerHTML = '';
  const stats = [
    ['Habitat', text(activeTank().label, activeTank().tank_id || 'Main')],
    ['Link', online ? 'online' : 'waiting'],
    ['Live USB', `${usbOnline}/${usb.length}`],
    ['Assigned', `${allItems().filter(item => item.role_locked || !isCameraItem(item)).length} item${allItems().filter(item => item.role_locked || !isCameraItem(item)).length === 1 ? '' : 's'}`],
    ['Unplaced', String(unplaced)],
    ['Vision', observers ? `${observers} event${observers === 1 ? '' : 's'}` : 'idle'],
  ];
  stats.forEach(([label, value]) => {
    const pill = document.createElement('div');
    pill.className = 'status-pill';
    pill.innerHTML = '<span></span><strong></strong>';
    pill.querySelector('span').textContent = label;
    pill.querySelector('strong').textContent = value;
    feedMarquee.appendChild(pill);
  });
}

function renderSelection() {
  const item = selectedItem();
  selectedSection.hidden = !item;
  pip.hidden = !item;
  worldControls.hidden = !item;
  if (!item) {
    updateStageFeed();
    return;
  }
  selectedTitle.textContent = displayName(item);
  const src = sourceForPreview(item);
  preview.hidden = !src;
  if (src) preview.src = src;
  if (src) pipPreview.src = src;
  pipLabel.textContent = displayName(item) + ' / ' + itemKind(item);
  const faceRow = selectedSection.querySelector('.face-row');
  if (faceRow) faceRow.hidden = !(isFloater(item) || isLighthouse(item) || isReeflex(item) || isRobotArm(item));
  document.querySelectorAll('.endoscope-only').forEach(button => {
    button.hidden = !isEndoscope(item);
  });
  if (identifySelected) identifySelected.hidden = !isCameraItem(item);
  if (removeSelected) {
    removeSelected.textContent = isCameraItem(item) ? 'Hide feed from tank' : 'Remove from tank';
  }
  const id = cameraId(item) || item.item_id;
  if (!placementOf(item).placed && state.pendingId === id) {
    tankSubtitle.textContent = isFloater(item)
      ? 'Click a side wall or the top to mount this camera.'
      : isReeflex(item) || isLighthouse(item)
        ? 'Click a tank rim or side to place this hardware.'
        : 'Click the tank to place this item.';
  } else {
    tankSubtitle.textContent = 'Use the controller buttons to nudge the selected item.';
  }
  renderLighthouseControls();
  renderReeflexControls();
  updateStageFeed();
}

function updateWorldControls() {
  const item = selectedItem();
  if (!item || worldControls.hidden) return;
  const placement = placementOf(item);
  if (!placement.placed || !placement.position) {
    worldControls.hidden = true;
    return;
  }
  const vector = normToWorld(placement.position).project(camera);
  const x = (vector.x * 0.5 + 0.5) * stage.clientWidth;
  const y = (-vector.y * 0.5 + 0.5) * stage.clientHeight;
  const visible = vector.z > -1 && vector.z < 1;
  worldControls.hidden = !visible;
  if (visible) {
    worldControls.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px) translate(-50%, -115%)`;
  }
}

function renderHud() {
  const tank = activeTank();
  const nodes = state.layout.nodes || [];
  const activeNodeCount = nodes.filter(node => node.status === 'online').length;
  const usbCount = usbVideoCameras().length;
  const unplacedCount = allItems().filter(item => !placementOf(item).placed && isConnectedItem(item)).length;
  const unidentifiedCount = unidentifiedConnectedCameras().length;
  tankTitle.textContent = text(tank.label, 'Sync Tank');
  if (!nodes.length && !allCamerasForActiveTank().length) {
    tankTitle.textContent = 'No tanks detected';
    tankSubtitle.textContent = 'Turn on an edge node, connect it over LAN, or set up a tank offline.';
    if (systemGuide) systemGuide.textContent = 'Waiting for tank node.';
  } else {
    tankTitle.textContent = unidentifiedCount
      ? `${unidentifiedCount} USB feed${unidentifiedCount === 1 ? '' : 's'} need setup`
      : unplacedCount
      ? `${unplacedCount} assigned item${unplacedCount === 1 ? '' : 's'} need placement`
      : 'Habitat view ready';
    tankSubtitle.textContent = `${usbCount} USB feed${usbCount === 1 ? '' : 's'} visible / ${allItems().filter(item => item.role_locked || !isCameraItem(item)).length} assigned tank item${allItems().filter(item => item.role_locked || !isCameraItem(item)).length === 1 ? '' : 's'}`;
    if (systemGuide) systemGuide.textContent = unplacedCount
      ? 'Identify feeds or place assigned hardware from Setup.'
      : 'Live habitat view is ready.';
  }
  renderTankTabs();
}

function renderTankSummary() {
  if (!tankSummary) return;
  const usb = usbVideoCameras();
  const snapshots = snapshotCameras();
  const node = activeNode();
  const setupState = node && node.setup_state ? node.setup_state.status || 'default' : 'unknown';
  const rows = [
    ['Node', node ? text(node.label || node.node_id, 'active') : 'none'],
    ['Observer', `${usb.filter(item => item.status === 'online').length} USB feed${usb.filter(item => item.status === 'online').length === 1 ? '' : 's'} live`],
    ['USB video', `${usb.filter(item => item.status === 'online').length}/${usb.length} online`],
    ['Snapshots', `${snapshots.length} endpoint${snapshots.length === 1 ? '' : 's'}`],
    ['Setup', setupState],
  ];
  tankSummary.innerHTML = '';
  rows.forEach(([label, value]) => {
    const row = document.createElement('div');
    row.className = 'summary-row';
    row.innerHTML = '<span></span><strong></strong>';
    row.querySelector('span').textContent = label;
    row.querySelector('strong').textContent = value;
    tankSummary.appendChild(row);
  });
}

async function labelObservation(observationId, label) {
  await fetch('/api/observations/label', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observation_id: observationId, label }),
  });
  await refreshLayout();
}

function renderObservations() {
  const observations = observationsForActiveTank().slice(0, 4);
  if (!observations.length) {
    observationList.innerHTML = '<div class="empty">Watching for motion. The strongest feed will take over this dock when something moves.</div>';
    renderObservationFocus(null);
    return;
  }
  observationList.innerHTML = '';
  observations.forEach(observation => {
    const card = document.createElement('article');
    card.className = 'observation-card';
    const frame = (observation.frame_urls || [])[0] || '';
    card.innerHTML = `
      ${frame ? '<div class="observation-frame"><img alt=""><span class="focus-reticle"></span></div>' : ''}
      <div class="observation-body">
        <div class="observation-title"></div>
        <div class="label-grid"></div>
      </div>
    `;
    const img = card.querySelector('img');
    if (img) img.src = frame;
    const focus = observation.focus_region || observation.focus_point || null;
    const reticle = card.querySelector('.focus-reticle');
    if (reticle && focus) {
      const x = Number(focus.x !== undefined ? focus.x : (focus.cx || 0.5));
      const y = Number(focus.y !== undefined ? focus.y : (focus.cy || 0.5));
      reticle.style.left = `${Math.round(clamp(x, 0, 1) * 100)}%`;
      reticle.style.top = `${Math.round(clamp(y, 0, 1) * 100)}%`;
    } else if (reticle) {
      reticle.hidden = true;
    }
    card.querySelector('.observation-title').textContent =
      observation.label ||
      `${text(observation.camera_id, 'camera')} ${text(observation.classifier_label || observation.event_type, 'motion')} ${text(observation.motion_score, '')}`;
    card.addEventListener('click', () => {
      const index = cameraIndexById(observation.camera_id);
      if (index >= 0) {
        state.focusObservationId = observation.observation_id;
        pinView(observation.camera_id);
        setLiveCamera(index);
        renderLiveView();
        updateStageFeed();
      }
    });
    const labels = ['Fish', 'Shrimp', 'Snail', 'Coral', 'Unknown'];
    const grid = card.querySelector('.label-grid');
    labels.forEach(label => {
      const button = document.createElement('button');
      button.textContent = label;
      button.addEventListener('click', event => {
        event.stopPropagation();
        labelObservation(observation.observation_id, label).catch(() => {});
      });
      grid.appendChild(button);
    });
    observationList.appendChild(card);
  });
  renderObservationFocus(observations[0]);
}

function renderObservationFocus(observation) {
  if (selectedItem()) return;
  const frame = observation && (observation.frame_urls || [])[0];
  if (!frame) {
    pip.hidden = true;
    return;
  }
  const cameras = liveCameras();
  const cameraIndex = cameras.findIndex(item => cameraId(item) === observation.camera_id);
  if (cameraIndex >= 0 && !liveSection.hidden) {
    setLiveCamera(cameraIndex);
  }
  pip.hidden = false;
  pipPreview.src = frame;
  pipLabel.textContent = `${text(observation.camera_id, 'camera')} / motion focus`;
}

function renderTankTabs() {
  const tanks = visibleTanks();
  tankTabs.innerHTML = '';
  tankTabs.hidden = tanks.length <= 1;
  tanks.forEach(tank => {
    const button = document.createElement('button');
    button.className = tank.tank_id === activeTank().tank_id ? 'tank-tab active' : 'tank-tab';
    button.textContent = text(tank.label, tank.tank_id);
    button.addEventListener('click', () => setActiveTank(tank.tank_id));
    tankTabs.appendChild(button);
  });
}

function ensureSceneItem(itemId, label, itemType) {
  return tank => {
    const existing = (state.layout.scene_items || []).find(item => item.item_id === itemId);
    if (!existing) {
      state.layout.scene_items = state.layout.scene_items || [];
      state.layout.scene_items.push({
        item_id: itemId,
        label,
        tank_id: tank.tank_id || 'main-tank',
        node_id: tank.primary_node_id || null,
        item_type: itemType,
        placement: { placed: false, position: null, target: null },
      });
    }
    return tank;
  };
}

function ensureSceneItemForActive(itemId, label, itemType) {
  const tank = activeTank();
  const scopedId = `${tank.tank_id}-${itemId}`;
  let item = (state.layout.scene_items || []).find(sceneItem => sceneItem.item_id === scopedId);
  if (!item) {
    state.layout.scene_items = state.layout.scene_items || [];
    item = {
      item_id: scopedId,
      label,
      tank_id: tank.tank_id || 'main-tank',
      node_id: tank.primary_node_id || null,
      item_type: itemType,
      placement: { placed: false, position: null, target: null },
      profile_only: true,
      status: 'planned',
    };
    state.layout.scene_items.push(item);
  }
  return item;
}

const DEVICE_CATALOG = [
  {
    itemId: 'floater-manual-001',
    label: 'Floater',
    itemType: 'floater_cam',
    detail: 'ESP32 puck camera mounted on a wall or the top',
  },
  {
    itemId: 'reeflex-001',
    label: 'Reeflex',
    itemType: 'reeflex',
    detail: '3-servo rim robot arm looking over the water',
  },
  {
    itemId: 'lighthouse-001',
    label: 'Raydar',
    itemType: 'lighthouse',
    detail: 'Pan-tilt camera head mounted just above the rim',
  },
  {
    itemId: 'scope-001',
    label: 'Reel',
    itemType: 'endoscope_cam',
    detail: 'Endoscope-style probe camera inside the tank',
  },
];

function addCatalogDevice(device) {
  const item = ensureSceneItemForActive(device.itemId, device.label, device.itemType);
  const id = item.item_id;
  state.pendingId = id;
  state.selectedId = id;
  return activeTank();
}

function hasSavedSetup() {
  return (state.layout.tanks || []).some(tank => tank.hardware_validated);
}

function configuredTanks() {
  return (state.layout.tanks || []).filter(tank =>
    tank.setup_complete ||
    tank.hardware_validated ||
    tank.primary_node_id ||
    tank.metrics_profile
  );
}

function createDraftTanks(count) {
  state.setupDraft.count = count;
  state.layout.tanks = Array.from({ length: count }, (_, index) => ({
    tank_id: index === 0 ? 'main-tank' : `tank-${index + 1}`,
    label: index === 0 ? 'Main Tank' : `Tank ${index + 1}`,
    dimensions: { ...DEFAULT_TANK_DIMENSIONS },
    tank_type: 'freshwater_shrimp',
    metrics_profile: 'default',
    setup_complete: false,
  }));
  state.layout.scene_items = [];
  state.activeTankId = 'main-tank';
  (state.layout.cameras || []).forEach(camera => {
    camera.tank_id = camera.tank_id || 'main-tank';
    camera.placement = { placed: false, position: null, target: null, fov_degrees: isEndoscope(camera) ? 70 : 60 };
  });
}

function setTankDimension(tank, axis, value) {
  const dimensions = { ...DEFAULT_TANK_DIMENSIONS, ...(tank.dimensions || {}) };
  dimensions[axis] = Math.round(clamp(Number(value) || DEFAULT_TANK_DIMENSIONS[axis], 1, 240) * 10) / 10;
  Object.assign(tank, {
    dimensions,
    setup_complete: false,
    hardware_validated: false,
  });
  applyTankDimensions();
  return tank;
}

function adjustTankDimension(tank, axis, delta) {
  const dimensions = { ...DEFAULT_TANK_DIMENSIONS, ...(tank.dimensions || {}) };
  return setTankDimension(tank, axis, Number(dimensions[axis] || DEFAULT_TANK_DIMENSIONS[axis]) + delta);
}

function sizeOptionsForTank(tank, includeName = false, index = 0) {
  const named = (name, dimensions) => includeName ? { name, dimensions } : { dimensions };
  return [
    {
      label: 'Small square',
      detail: '25 x 25 x 25 in default',
      ...named(index === 0 ? 'Main Tank' : `Tank ${index + 1}`, { ...DEFAULT_TANK_DIMENSIONS }),
    },
    {
      label: 'Medium square',
      detail: '36 x 36 x 24 in',
      ...named(`Square Tank ${index + 1}`, { x: 36, y: 24, z: 36, unit: 'in' }),
    },
    {
      label: 'Long shallow',
      detail: '48 x 18 x 18 in',
      ...named(`Long Tank ${index + 1}`, { x: 48, y: 18, z: 18, unit: 'in' }),
    },
  ].map(option => ({
    ...option,
    apply: () => {
      if (option.adjustAxis) return adjustTankDimension(tank, option.adjustAxis, option.adjustDelta);
      const update = { dimensions: option.dimensions, setup_complete: false, hardware_validated: false };
      if (option.name) update.label = option.name;
      Object.assign(tank, update);
      applyTankDimensions();
      return tank;
    },
  }));
}

function setupSteps() {
  const nodes = state.layout.nodes || [];
  const nodeOptions = (nodes.length ? nodes : [{ node_id: 'tank-pi-001', label: 'Edge Node 1' }]);
  const savedTanks = configuredTanks();
  if (!state.setupDraft.mode) {
    const nodesConnected = nodeOptions.length && (state.layout.nodes || []).length;
    const activeNodeCount = nodes.filter(node => node.status === 'online').length;
    const usbCount = usbVideoCameras().length;
    const unidentifiedCount = unidentifiedConnectedCameras().length;
    const unplacedCount = allItems().filter(item => !placementOf(item).placed && isConnectedItem(item)).length;
    return [
      {
        question: usbCount
          ? `${usbCount} USB feed${usbCount === 1 ? '' : 's'} detected${unidentifiedCount ? ` / ${unidentifiedCount} need setup` : ''}`
          : activeNodeCount
            ? 'Tank node online; no USB feeds detected'
            : 'Waiting for tank node',
        options: [
          { label: 'Measure tank', detail: 'Set tank type, size, and node before placing hardware', mode: 'measure', apply: () => activeTank() },
          { label: 'Manage feeds', detail: usbCount ? 'Identify, preview, enable, or hide USB feeds' : 'No USB feeds are online yet', mode: 'feeds', apply: () => activeTank() },
          { label: 'Add hardware', detail: unplacedCount ? `Place ${unplacedCount} assigned item${unplacedCount === 1 ? '' : 's'} or add optional hardware` : 'Add floater, Reeflex, Raydar, or Reel hardware by hand', mode: 'edit', apply: () => activeTank() },
          { label: 'Tanks', detail: 'Create or reset one/two tank profiles', mode: 'fresh', apply: () => activeTank() },
        ].concat(nodesConnected
          ? [
              { label: 'Review node', detail: manifestSummary(), mode: 'manifest', apply: () => activeTank() },
            ]
          : []),
      },
    ];
  }
  if (state.setupDraft.mode === 'default') {
    state.setupDraft.mode = 'fresh';
    state.setupDraft.count = 1;
  }
  if (state.setupDraft.mode === 'measure') {
    const tank = activeTank();
    return [
      {
        question: `Choose size for ${text(tank.label, 'this tank')}`,
        dimensionEditor: true,
        options: sizeOptionsForTank(tank),
      },
      {
        question: `Choose type for ${text(tank.label, 'this tank')}`,
        options: [
          { label: 'Freshwater shrimp', detail: 'Shrimp, snails, planted tank, or small freshwater animals', tank_type: 'freshwater_shrimp', profile: 'shrimp_basic' },
          { label: 'Reef or saltwater', detail: 'Marine tank with reef-style camera/sensor assumptions', tank_type: 'reef', profile: 'reef_basic' },
          { label: 'General aquarium', detail: 'Camera-first setup without species-specific assumptions', tank_type: 'general_aquarium', profile: 'camera_only' },
        ].map(option => ({
          ...option,
          apply: () => {
            Object.assign(tank, { tank_type: option.tank_type, metrics_profile: option.profile });
            return tank;
          },
        })),
      },
    ];
  }
  if (state.setupDraft.mode === 'manifest') {
    const cameras = allCamerasForActiveTank();
    const online = cameras.filter(camera => camera.status === 'online').length;
    const usb = usbVideoCameras();
    const usbOnline = usb.filter(camera => camera.status === 'online').length;
    const snapshots = snapshotCameras();
    const expectedUsb = expectedUsbVideoCount();
    return [
      {
        question: `Active node manifest: ${manifestSummary()}`,
        options: [
          { label: 'Manifest looks right', detail: `${usbOnline}/${usb.length} registered USB streams online; manifest expects ${expectedUsb || 'unknown'} USB-video devices; ${snapshots.length} snapshot endpoint${snapshots.length === 1 ? '' : 's'}`, mode: 'edit', apply: () => activeTank() },
          { label: 'Add missing hardware', detail: 'Choose floaters, Reeflex, Raydar, or Reels by hand', mode: 'edit', apply: () => activeTank() },
          { label: 'Use default tank', detail: 'Keep live feeds and default dimensions without marking hardware validated', close: true, apply: () => activeTank() },
        ],
      },
    ];
  }
  if (state.setupDraft.mode === 'feeds') {
    const cameras = allCameraRecordsForActiveTank();
    return [
      {
        question: cameras.length
          ? `Set up ${cameras.length} camera feed${cameras.length === 1 ? '' : 's'}`
          : 'No camera feeds are reporting yet',
        options: cameras.length
          ? cameras.map(camera => {
              const hidden = Boolean(camera.hidden_from_layout);
              const preview = sourceForPreview(camera);
              return {
                label: `${hidden ? 'Enable ' : ''}${displayName(camera)}`,
                detail: `${hidden ? 'Hidden' : itemKind(camera)}${preview ? ' / preview available' : ''}`,
                preview,
                keepOpen: true,
                apply: () => {
                  setCameraHidden(camera, false);
                  state.selectedId = cameraId(camera);
                  state.setupDraft = { mode: 'identify', cameraId: cameraId(camera) };
                  state.setupStep = 0;
                  return activeTank();
                },
              };
            }).concat([
              { label: 'Done', detail: 'Return to tank view', close: true, apply: () => activeTank() },
            ])
          : [
              { label: 'Done', detail: 'Return to tank view', close: true, apply: () => activeTank() },
            ],
      },
    ];
  }
  if (state.setupDraft.mode === 'identify') {
    const item = selectedItem() || allCamerasForActiveTank().find(camera => cameraId(camera) === state.setupDraft.cameraId);
    return [
      {
        question: item
          ? `What is ${displayName(item)}?`
          : 'Choose a camera feed to identify',
        previewItem: item,
        options: item
          ? [
              ...assignmentOptionsFor(item),
              { label: 'Hide this feed', detail: 'Remove it from the tank view without unplugging it', close: true, apply: () => setCameraHidden(item, true) },
              { label: 'Cancel', detail: 'Leave this feed unchanged', close: true, apply: () => activeTank() },
            ]
          : allCamerasForActiveTank().filter(isConnectedItem).map(camera => ({
              label: displayName(camera),
              detail: itemKind(camera),
              keepOpen: true,
              apply: () => {
                state.selectedId = cameraId(camera);
                state.setupDraft.cameraId = cameraId(camera);
                return activeTank();
              },
            })),
      },
    ];
  }
  if (state.setupDraft.mode === 'edit') {
    return [
      {
        question: 'Add hardware to this tank',
        options: [
          ...DEVICE_CATALOG.map(device => ({
            label: device.label,
            detail: device.detail,
            placeNow: true,
            apply: () => addCatalogDevice(device),
          })),
          { label: 'Done', detail: 'Return to tank view', close: true, apply: () => activeTank() },
        ],
      },
    ];
  }
  const steps = [];
  if (!state.setupDraft.count) {
    steps.push({
      question: 'Start from scratch with how many tanks?',
      options: [
        { label: 'One tank', detail: 'Single connected tank display', count: 1 },
        { label: 'Two tanks', detail: 'Main tank plus second connected tank', count: 2 },
      ].map(option => ({
        ...option,
        apply: () => {
          createDraftTanks(option.count);
          return activeTank();
        },
      })),
    });
  }

  const tankCount = Math.max(1, state.layout.tanks.length || state.setupDraft.count || 1);
  for (let index = 0; index < tankCount; index += 1) {
    steps.push({
      question: `Choose type for Tank ${index + 1}`,
      tankIndex: index,
      options: [
        { label: 'Freshwater shrimp', detail: 'Shrimp, snails, planted tank, or small freshwater animals', tank_type: 'freshwater_shrimp', profile: 'shrimp_basic' },
        { label: 'Reef or saltwater', detail: 'Marine tank with reef-style camera/sensor assumptions', tank_type: 'reef', profile: 'reef_basic' },
        { label: 'General aquarium', detail: 'Camera-first setup without species-specific assumptions', tank_type: 'general_aquarium', profile: 'camera_only' },
      ].map(option => ({
        ...option,
        apply: () => {
          const tank = state.layout.tanks[index];
          Object.assign(tank, { tank_type: option.tank_type, metrics_profile: option.profile });
          return tank;
        },
      })),
    });
    steps.push({
      question: `Choose size for Tank ${index + 1}`,
      tankIndex: index,
      dimensionEditor: true,
      options: sizeOptionsForTank(state.layout.tanks[index], true, index),
    });
    steps.push({
      question: `Assign node for Tank ${index + 1}`,
      tankIndex: index,
      options: nodeOptions.map(node => ({
        label: nodeLabel(node),
        detail: text(node.lan_url || node.camera_service_url, 'saved local node'),
        apply: () => {
          const tank = state.layout.tanks[index];
          tank.primary_node_id = node.node_id;
          return tank;
        },
      })),
    });
    steps.push({
      question: `Metrics for Tank ${index + 1}`,
      tankIndex: index,
      options: [
        { label: 'Use defaults', detail: 'Camera status, node health, feed freshness', profile: 'default' },
        { label: 'Reef basics', detail: 'Temperature, pH, flow, and camera status', profile: 'reef_basic' },
        { label: 'Camera only', detail: 'Feeds and placement without sensor metrics', profile: 'camera_only' },
      ].map(option => ({
        ...option,
        apply: () => {
          const tank = state.layout.tanks[index];
          tank.metrics_profile = option.profile;
          return tank;
        },
      })),
    });
  }

  steps.push({
    question: 'Confirm tank hardware',
    options: [
      ...DEVICE_CATALOG.map(device => ({
        label: device.label,
        detail: device.detail,
        placeNow: true,
        apply: () => addCatalogDevice(device),
      })),
      { label: 'Done', detail: 'Save setup and start placing', finish: true, apply: () => activeTank() },
    ],
  });
  return steps;
}

function setupInfoSummary() {
  const tank = activeTank();
  const dimensions = tank.dimensions || {};
  const size = dimensions.x && dimensions.y && dimensions.z
    ? `${dimensions.x} x ${dimensions.y} x ${dimensions.z} ${dimensions.unit || 'in'}`
    : 'size unset';
  const type = text(tank.tank_type || tank.metrics_profile, 'type unset').replaceAll('_', ' ');
  const node = activeNode();
  const feeds = usbVideoCameras().length;
  const assigned = allItems().filter(item => item.role_locked || !isCameraItem(item)).length;
  const unplaced = allItems().filter(item => !placementOf(item).placed && isConnectedItem(item)).length;
  return `Current: ${text(tank.label, tank.tank_id || 'tank')} / ${type} / ${size} / ${node ? nodeLabel(node) : 'no active node'} / ${feeds} USB feed${feeds === 1 ? '' : 's'}, ${assigned} assigned item${assigned === 1 ? '' : 's'}, ${unplaced} to place`;
}

function renderDimensionEditor(tank) {
  const dimensions = { ...DEFAULT_TANK_DIMENSIONS, ...(tank.dimensions || {}) };
  const editor = document.createElement('div');
  editor.className = 'dimension-editor';
  const fields = [
    ['x', 'Width'],
    ['z', 'Depth'],
    ['y', 'Height'],
  ];
  fields.forEach(([axis, label]) => {
    const row = document.createElement('div');
    row.className = 'dimension-row';
    row.innerHTML = `
      <span>${label}</span>
      <button type="button" data-dimension-axis="${axis}" data-dimension-delta="-0.1">-</button>
      <input type="number" min="1" max="240" step="0.1" inputmode="decimal" data-dimension-input="${axis}">
      <button type="button" data-dimension-axis="${axis}" data-dimension-delta="0.1">+</button>
      <em>in</em>
    `;
    const input = row.querySelector('input');
    input.value = Number(dimensions[axis] || DEFAULT_TANK_DIMENSIONS[axis]).toFixed(1);
    input.addEventListener('change', () => {
      setTankDimension(tank, axis, input.value);
      renderSetup();
      queueSave();
    });
    row.querySelectorAll('button').forEach(button => {
      button.addEventListener('click', () => {
        adjustTankDimension(tank, axis, Number(button.dataset.dimensionDelta));
        renderSetup();
        queueSave();
      });
    });
    editor.appendChild(row);
  });
  return editor;
}

function renderSetupFeedPreview(item) {
  const card = document.createElement('div');
  card.className = 'setup-feed-preview';
  const src = sourceForPreview(item);
  const id = cameraId(item) || 'unknown camera';
  const kind = item.stream_url ? 'live MJPEG stream' : (item.snapshot_url || item.latest_image_url ? 'latest snapshot' : 'no feed URL');
  card.innerHTML = `
    <div class="setup-feed-copy">
      <strong>Identify this feed by sight</strong>
      <span>${id} / ${text(item.node_id, 'unknown node')} / ${kind}</span>
    </div>
    <div class="setup-feed-frame">
      ${src ? '<img alt="Camera feed preview">' : '<div class="setup-feed-empty">No preview URL reported by this node</div>'}
    </div>
  `;
  const img = card.querySelector('img');
  if (img) {
    img.src = src;
    img.addEventListener('error', () => {
      card.classList.add('feed-error');
      const copy = card.querySelector('.setup-feed-copy span');
      if (copy) copy.textContent = `${id} / preview did not load`;
    });
  }
  return card;
}

function renderSetup() {
  const steps = setupSteps();
  if (state.setupStep >= steps.length) state.setupStep = Math.max(0, steps.length - 1);
  const step = steps[state.setupStep];
  if (step.tankIndex !== undefined && state.layout.tanks[step.tankIndex]) {
    state.activeTankId = state.layout.tanks[step.tankIndex].tank_id;
    applyTankDimensions();
    renderHud();
  }
  setupQuestion.textContent = step.question;
  if (setupCurrent) setupCurrent.textContent = setupInfoSummary();
  setupOptions.innerHTML = '';
  if (step.dimensionEditor) {
    setupOptions.appendChild(renderDimensionEditor(activeTank()));
  }
  if (step.previewItem) {
    setupOptions.appendChild(renderSetupFeedPreview(step.previewItem));
  }
  setupBack.disabled = false;
  setupBack.textContent = state.setupStep === 0 ? 'Close' : 'Back';
  step.options.forEach(option => {
    const button = document.createElement('button');
    button.className = 'setup-option';
    button.innerHTML = `${option.preview ? '<img class="setup-option-preview" alt="">' : ''}<strong></strong><span></span>`;
    const previewImg = button.querySelector('.setup-option-preview');
    if (previewImg) previewImg.src = option.preview;
    button.querySelector('strong').textContent = option.label;
    button.querySelector('span').textContent = option.detail;
    button.addEventListener('click', () => {
      if (option.mode) state.setupDraft.mode = option.mode;
      if (option.count) createDraftTanks(option.count);
      if (option.reset) {
        state.layout.scene_items = [];
        (state.layout.cameras || []).forEach(camera => {
          camera.placement = { placed: false, position: null, target: null, fov_degrees: isEndoscope(camera) ? 70 : 60 };
        });
      }
      const tank = activeTank();
      const nextTank = option.apply(tank);
      const tankId = nextTank.tank_id || 'main-tank';
      if (step.tankIndex !== undefined) {
        state.layout.tanks[step.tankIndex] = { ...nextTank, tank_id: tankId };
      } else if (!state.layout.tanks.length || option.count) {
        state.layout.tanks = state.layout.tanks.length ? state.layout.tanks : [{ ...nextTank, tank_id: tankId }];
      }
      if ((option.mode || option.count) && !option.close && !option.keepOpen && !option.finish) {
        state.setupStep = 0;
        renderSetup();
      } else if (option.close) {
        closeSetup(false);
      } else if (option.placeNow) {
        setupOverlay.hidden = true;
        sidePanel?.classList.add('open');
        dockToggle?.classList.add('open');
        dockToggle?.setAttribute('aria-expanded', 'true');
        dockToggle?.setAttribute('aria-label', 'Close placement controls');
        renderHud();
        renderUnplaced();
        renderLiveView();
        renderSelection();
        updateAllMeshes();
        saveLayout().catch(() => {});
      } else if (option.keepOpen) {
        renderSetup();
        renderUnplaced();
        saveLayout().catch(() => {});
      } else if (state.setupStep < steps.length - 1) {
        state.setupStep += 1;
        renderSetup();
      } else {
        closeSetup(true);
      }
    });
    setupOptions.appendChild(button);
  });
}

function openSetup() {
  state.setupStep = 0;
  state.setupDraft = {};
  if (placementSection) placementSection.hidden = false;
  if (liveSection) liveSection.hidden = true;
  setupOverlay.hidden = false;
  renderSetup();
}

function shouldPromptInitialSetup() {
  if (state.initialSetupPrompted || !setupOverlay.hidden) return false;
  const tank = activeTank();
  return !tank.setup_complete && !tank.hardware_validated;
}

function openInitialSizeSetup() {
  state.initialSetupPrompted = true;
  state.setupStep = 0;
  state.setupDraft = { mode: 'measure' };
  if (placementSection) placementSection.hidden = false;
  if (liveSection) liveSection.hidden = true;
  setupOverlay.hidden = false;
  renderSetup();
}

function closeSetup(markComplete = true) {
  if (markComplete) {
    (state.layout.tanks || []).forEach(tank => {
      tank.setup_complete = true;
      tank.hardware_validated = true;
      tank.saved_by_user = true;
    });
  }
  setupOverlay.hidden = true;
  applyTankDimensions();
  renderHud();
  renderUnplaced();
  saveLayout().catch(() => {});
}

function updateAllMeshes() {
  const activeIds = new Set(allItems().map(item => cameraId(item) || item.item_id));
  state.meshes.forEach((mesh, id) => {
    if (!activeIds.has(id)) mesh.visible = false;
  });
  state.frustums.forEach((frustum, id) => {
    if (!activeIds.has(id)) frustum.visible = false;
  });
  allItems().forEach(updateMesh);
}

async function saveLayout() {
  const payload = {
    tanks: state.layout.tanks || [],
    cameras: state.layout.cameras || [],
    scene_items: state.layout.scene_items || [],
  };
  await fetch('/api/layout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

function queueSave() {
  window.clearTimeout(state.dirtyTimer);
  state.dirtyTimer = window.setTimeout(() => {
    saveLayout().catch(() => {});
  }, 120);
}

function selectById(id, keepPending = false) {
  state.selectedId = id;
  if (!keepPending) state.pendingId = null;
  renderSelection();
  updateAllMeshes();
}

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function placeItemOnTank(item, point, face) {
  const placement = placementOf(item);
  if (isFloater(item)) {
    const safeFace = face === 'y-' ? 'y+' : face;
    placement.placed = true;
    placement.mount_face = safeFace;
    placement.position = snapFloaterToFace(point, safeFace);
    placement.target = inwardTargetFor(placement.position, safeFace);
  } else if (isRobotArm(item)) {
    const rail = snapRailToPerimeter(point, face);
    placement.placed = true;
    placement.mount_face = rail.mount_face;
    placement.position = rail.position;
    placement.target = rail.target;
  } else if (isLighthouse(item) || isReeflex(item)) {
    const rim = snapLighthouseToRim(point, face);
    placement.placed = true;
    placement.mount_face = rim.mount_face;
    placement.position = rim.position;
    placement.target = isReeflex(item) ? { x: 0.5, y: 0.62, z: 0.5 } : rim.target;
  } else {
    placement.placed = true;
    placement.position = worldToNorm(point);
    placement.target = placement.target || { x: 0.5, y: 0.5, z: 0.5 };
  }
  state.selectedId = null;
  state.pendingId = null;
  renderUnplaced();
  renderFeedMarquee();
  renderSelection();
  if (sidePanel) sidePanel.classList.remove('open');
  if (dockToggle) {
    dockToggle.classList.remove('open');
    dockToggle.setAttribute('aria-expanded', 'false');
  }
  queueSave();
}

function onPointerDown(event) {
  setPointer(event);
  raycaster.setFromCamera(pointer, camera);

  const meshes = [...state.meshes.values()].filter(mesh => mesh.visible);
  const objectHit = raycaster.intersectObjects(meshes, true)[0];
  if (objectHit) {
    const id = objectHit.object.userData.itemId || objectHit.object.parent?.userData.itemId;
    selectById(id);
    const item = allItems().find(entry => (cameraId(entry) || entry.item_id) === id);
    if (item && isFloater(item)) showFloaterMarkerImage(item, true);
    return;
  }

  const tankHit = raycaster.intersectObjects(faceTargets, false)[0];
  if (!tankHit) return;

  const pending = allItems().find(item => (cameraId(item) || item.item_id) === state.pendingId);
  if (pending) {
    const worldNormal = tankHit.face.normal.clone().transformDirection(tankHit.object.matrixWorld);
    placeItemOnTank(pending, tankHit.point, faceFromNormal(worldNormal));
  }
}

function onDoubleClick(event) {
  setPointer(event);
  raycaster.setFromCamera(pointer, camera);
  if (!raycaster.intersectObjects([...state.meshes.values()], true).length) {
    state.selectedId = null;
    state.pendingId = null;
    renderSelection();
    updateAllMeshes();
  }
}

function moveSelected(action) {
  const item = selectedItem();
  if (!item) return;
  const placement = placementOf(item);
  if (!placement.position) {
    Object.assign(placement, defaultPlacement(item));
  }
  const pos = { ...placement.position };
  const target = placement.target ? { ...placement.target } : { x: 0.5, y: 0.5, z: 0.5 };
  const step = state.step;

  if (isRobotArm(item)) {
    const face = placement.mount_face || 'z+';
    const inward = normalForFace(face).multiplyScalar(-1);
    const currentTarget = normToWorld(target);
    const directionStep = action === 'forward' ? step : action === 'back' ? -step : 0;
    const nextTarget = directionStep ? currentTarget.add(inward.multiplyScalar(directionStep * tankSize.z)) : currentTarget;
    placement.position = pos;
    placement.target = worldToNorm(nextTarget);
    if (face === 'x+') placement.position.x = 1;
    if (face === 'x-') placement.position.x = 0;
    if (face === 'z+') placement.position.z = 1;
    if (face === 'z-') placement.position.z = 0;
    if (!directionStep) placement.target = inwardTargetFor(placement.position, face);
    placement.placed = true;
    updateMesh(item);
    queueSave();
    return;
  }

  if (action === 'left') pos.x -= step;
  if (action === 'right') pos.x += step;
  if (action === 'up') pos.y += step;
  if (action === 'down') pos.y -= step;
  if (action === 'forward') pos.z -= step;
  if (action === 'back') pos.z += step;
  if (action === 'aim-up') target.y += step;
  if (action === 'aim-down') target.y -= step;
  if (action === 'rotate-left') target.x -= step;
  if (action === 'rotate-right') target.x += step;

  pos.x = clamp(pos.x, 0, 1);
  pos.y = clamp(pos.y, 0, 1);
  pos.z = clamp(pos.z, 0, 1);
  target.x = clamp(target.x, 0, 1);
  target.y = clamp(target.y, 0, 1);
  target.z = clamp(target.z, 0, 1);

  if (isFloater(item)) {
    const face = placement.mount_face || 'y+';
    if (face === 'x+') pos.x = 1;
    if (face === 'x-') pos.x = 0;
    if (face === 'y+') pos.y = 1;
    if (face === 'z+') pos.z = 1;
    if (face === 'z-') pos.z = 0;
    placement.target = inwardTargetFor(pos, face);
  } else if (isLighthouse(item) || isReeflex(item)) {
    const face = placement.mount_face || 'z+';
    if (face === 'x+') pos.x = 1;
    if (face === 'x-') pos.x = 0;
    if (face === 'z+') pos.z = 1;
    if (face === 'z-') pos.z = 0;
    pos.y = 1;
    placement.target = target;
  } else {
    placement.target = target;
  }
  placement.position = pos;
  placement.placed = true;
  updateMesh(item);
  renderUnplaced();
  renderLiveView();
  queueSave();
}

function applyPlacementFromPad(offsetX, offsetY) {
  const item = selectedItem();
  if (!item || !state.padDrag) return;
  const placement = placementOf(item);
  const startPosition = state.padDrag.startPosition || placement.position || defaultPlacement(item).position;
  const startTarget = state.padDrag.startTarget || placement.target || { x: 0.5, y: 0.5, z: 0.5 };
  const pos = { ...startPosition };
  const target = { ...startTarget };
  const amountX = clamp(offsetX, -1, 1) * state.step * 5;
  const amountY = clamp(offsetY, -1, 1) * state.step * 5;

  if (state.controlAxis === 'depth') {
    pos.z = startPosition.z - amountY;
    if (isRobotArm(item)) target.z = startTarget.z - amountY;
  } else if (state.controlAxis === 'aim') {
    target.x = startTarget.x + amountX;
    target.y = startTarget.y - amountY;
  } else {
    pos.x = startPosition.x + amountX;
    pos.y = startPosition.y - amountY;
  }

  pos.x = clamp(pos.x, 0, 1);
  pos.y = clamp(pos.y, 0, 1);
  pos.z = clamp(pos.z, 0, 1);
  target.x = clamp(target.x, 0, 1);
  target.y = clamp(target.y, 0, 1);
  target.z = clamp(target.z, 0, 1);

  if (isFloater(item)) {
    const face = placement.mount_face || 'y+';
    if (face === 'x+') pos.x = 1;
    if (face === 'x-') pos.x = 0;
    if (face === 'y+') pos.y = 1;
    if (face === 'z+') pos.z = 1;
    if (face === 'z-') pos.z = 0;
    placement.target = inwardTargetFor(pos, face);
  } else if (isLighthouse(item) || isReeflex(item)) {
    const face = placement.mount_face || 'z+';
    if (face === 'x+') pos.x = 1;
    if (face === 'x-') pos.x = 0;
    if (face === 'z+') pos.z = 1;
    if (face === 'z-') pos.z = 0;
    pos.y = 1;
    placement.target = target;
  } else {
    placement.target = target;
  }

  placement.position = pos;
  placement.placed = true;
  updateMesh(item);
  renderUnplaced();
  renderLiveView();
  queueSave();
}

function updatePadStick(offsetX = 0, offsetY = 0) {
  if (!placementStick) return;
  placementStick.style.transform = `translate(${Math.round(offsetX * 72)}px, ${Math.round(offsetY * 72)}px)`;
}

function padOffsets(event) {
  const rect = placementPad.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
  const y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
  return { x: clamp(x, -1, 1), y: clamp(y, -1, 1) };
}

function startPadDrag(event) {
  const item = selectedItem();
  if (!item || !placementPad) return;
  event.preventDefault();
  const placement = placementOf(item);
  if (!placement.position) Object.assign(placement, defaultPlacement(item));
  state.padDrag = {
    pointerId: event.pointerId,
    startPosition: { ...(placement.position || { x: 0.5, y: 0.5, z: 0.5 }) },
    startTarget: { ...(placement.target || { x: 0.5, y: 0.5, z: 0.5 }) },
  };
  placementPad.classList.add('dragging');
  placementPad.setPointerCapture(event.pointerId);
  const offset = padOffsets(event);
  updatePadStick(offset.x, offset.y);
  applyPlacementFromPad(offset.x, offset.y);
}

function movePadDrag(event) {
  if (!state.padDrag || state.padDrag.pointerId !== event.pointerId) return;
  event.preventDefault();
  const offset = padOffsets(event);
  updatePadStick(offset.x, offset.y);
  applyPlacementFromPad(offset.x, offset.y);
}

function endPadDrag(event) {
  if (!state.padDrag || state.padDrag.pointerId !== event.pointerId) return;
  state.padDrag = null;
  placementPad.classList.remove('dragging');
  updatePadStick(0, 0);
  try {
    placementPad.releasePointerCapture(event.pointerId);
  } catch {
    // The browser may already have released capture when the pointer leaves.
  }
}

function mountSelectedToFace(face) {
  const item = selectedItem();
  if (!item) return;
  const placement = placementOf(item);
  if (isEndoscope(item)) return;
  if (isLighthouse(item)) {
    const currentWorld = normToWorld(placement.position || { x: 0.5, y: 1, z: 0.5 });
    const rim = snapLighthouseToRim(currentWorld, face);
    placement.placed = true;
    placement.mount_face = rim.mount_face;
    placement.position = rim.position;
    placement.target = rim.target;
    updateMesh(item);
    renderUnplaced();
    renderFeedMarquee();
    renderLiveView();
    queueSave();
    return;
  }
  const current = placement.position || { x: 0.5, y: 0.5, z: 0.5 };
  const position = { ...current };
  if (face === 'x+') position.x = 1;
  if (face === 'x-') position.x = 0;
  if (face === 'y+') position.y = 1;
  if (face === 'z+') position.z = 1;
  if (face === 'z-') position.z = 0;
  placement.placed = true;
  placement.mount_face = face;
  placement.position = position;
  placement.target = inwardTargetFor(position, face);
  updateMesh(item);
  renderUnplaced();
  renderFeedMarquee();
  renderLiveView();
  queueSave();
}

function bindControls() {
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('dblclick', onDoubleClick);
  document.querySelectorAll('[data-move]').forEach(button => {
    button.addEventListener('click', () => moveSelected(button.dataset.move));
  });
  document.querySelectorAll('[data-step]').forEach(button => {
    button.addEventListener('click', () => {
      state.step = Number(button.dataset.step);
      document.querySelectorAll('[data-step]').forEach(item => item.classList.toggle('active', item === button));
    });
  });
  document.querySelectorAll('[data-axis]').forEach(button => {
    button.addEventListener('click', () => {
      state.controlAxis = button.dataset.axis || 'slide';
      document.querySelectorAll('[data-axis]').forEach(item => item.classList.toggle('active', item === button));
    });
  });
  if (placementPad) {
    placementPad.addEventListener('pointerdown', startPadDrag);
    placementPad.addEventListener('pointermove', movePadDrag);
    placementPad.addEventListener('pointerup', endPadDrag);
    placementPad.addEventListener('pointercancel', endPadDrag);
  }
  document.querySelectorAll('[data-lighthouse-step]').forEach(button => {
    button.addEventListener('click', () => {
      state.lighthouse.step = Number(button.dataset.lighthouseStep || 3);
      document.querySelectorAll('[data-lighthouse-step]').forEach(item => item.classList.toggle('active', item === button));
    });
  });
  document.querySelectorAll('[data-lighthouse-move]').forEach(button => {
    const move = button.dataset.lighthouseMove;
    button.addEventListener('pointerdown', event => {
      event.preventDefault();
      startLighthouseHold(move);
    });
    button.addEventListener('pointerup', stopLighthouseHold);
    button.addEventListener('pointercancel', stopLighthouseHold);
    button.addEventListener('pointerleave', stopLighthouseHold);
  });
  if (lighthousePanSlider) {
    lighthousePanSlider.addEventListener('change', () => sendLighthousePose(Number(lighthousePanSlider.value), state.lighthouse.tilt));
  }
  if (lighthouseTiltSlider) {
    lighthouseTiltSlider.addEventListener('change', () => sendLighthousePose(state.lighthouse.pan, Number(lighthouseTiltSlider.value)));
  }
  if (lighthouseClose) {
    lighthouseClose.addEventListener('click', () => {
      state.lighthouse.panelClosed = true;
      stopLighthouseHold();
      renderLighthouseControls();
    });
  }
  document.querySelectorAll('[data-reeflex-step]').forEach(button => {
    button.addEventListener('click', () => {
      state.reeflex.step = Number(button.dataset.reeflexStep || 3);
      document.querySelectorAll('[data-reeflex-step]').forEach(item => item.classList.toggle('active', item === button));
    });
  });
  document.querySelectorAll('[data-reeflex-move]').forEach(button => {
    const move = button.dataset.reeflexMove;
    button.addEventListener('pointerdown', event => {
      event.preventDefault();
      startReeflexHold(move);
    });
    button.addEventListener('pointerup', stopReeflexHold);
    button.addEventListener('pointercancel', stopReeflexHold);
    button.addEventListener('pointerleave', stopReeflexHold);
  });
  document.querySelectorAll('[data-reeflex-center]').forEach(button => {
    button.addEventListener('click', () => sendReeflexPose(90, 90, 90));
  });
  document.querySelectorAll('[data-reeflex-stop]').forEach(button => {
    button.addEventListener('click', stopReeflexMotion);
  });
  if (reeflexBaseSlider) {
    reeflexBaseSlider.addEventListener('change', () => sendReeflexPose(Number(reeflexBaseSlider.value), state.reeflex.shoulder, state.reeflex.elbow));
  }
  if (reeflexShoulderSlider) {
    reeflexShoulderSlider.addEventListener('change', () => sendReeflexPose(state.reeflex.base, Number(reeflexShoulderSlider.value), state.reeflex.elbow));
  }
  if (reeflexElbowSlider) {
    reeflexElbowSlider.addEventListener('change', () => sendReeflexPose(state.reeflex.base, state.reeflex.shoulder, Number(reeflexElbowSlider.value)));
  }
  if (reeflexClose) {
    reeflexClose.addEventListener('click', () => {
      state.reeflex.panelClosed = true;
      stopReeflexHold();
      renderReeflexControls();
    });
  }
  document.querySelectorAll('[data-face]').forEach(button => {
    button.addEventListener('click', () => mountSelectedToFace(button.dataset.face));
  });
  document.querySelectorAll('[data-world-move]').forEach(button => {
    button.addEventListener('click', () => moveSelected(button.dataset.worldMove));
  });
}

function mergeLayout(next) {
  const oldById = new Map((state.layout.cameras || []).map(item => [cameraId(item), item]));
  (next.cameras || []).forEach(item => {
    const old = oldById.get(cameraId(item));
    if (old && old.placement && !item.placement) item.placement = old.placement;
  });
  state.layout = next;
  const tanks = state.layout.tanks || [];
  const visible = visibleTanks();
  const activeVisible = visible.some(tank => tank.tank_id === state.activeTankId);
  const activeExists = tanks.some(tank => tank.tank_id === state.activeTankId);
  if (!activeExists || !activeVisible) {
    if (visible[0]) state.activeTankId = visible[0].tank_id;
    else if (tanks[0]) state.activeTankId = tanks[0].tank_id;
  }
  clearExpiredPin();
  if (!state.manualViewId && Date.now() > state.manualTankUntil) {
    const activeObservation = observationsForAllTanks()[0];
    const observationTankId = activeObservation && activeObservation.tank_id;
    if (observationTankId && visible.some(tank => tank.tank_id === observationTankId)) {
      state.activeTankId = observationTankId;
    }
  }
  applyTankDimensions();
}

async function refreshLayout() {
  const response = await fetch('/api/layout', { cache: 'no-store' });
  if (!response.ok) throw new Error('layout unavailable');
  mergeLayout(await response.json());
  renderHud();
  renderFeedMarquee();
  renderTankSummary();
  renderUnplaced();
  renderLiveView();
  renderLighthouseControls();
  renderReeflexControls();
  renderSelection();
  renderObservations();
  updateStageFeed();
  updateAllMeshes();
  if (!setupOverlay.hidden) renderSetup();
  else if (!screenshotMode) {
    const unidentified = unidentifiedConnectedCameras().find(item => !state.identifyPromptedFor.has(cameraId(item)));
    if (unidentified) {
      state.identifyPromptedFor.add(cameraId(unidentified));
      state.selectedId = cameraId(unidentified);
      state.setupStep = 0;
      state.setupDraft = { mode: 'identify', cameraId: cameraId(unidentified) };
      setupOverlay.hidden = false;
      renderSetup();
    } else if (shouldPromptInitialSetup()) {
      openInitialSizeSetup();
    }
  }
}

function resize() {
  const width = Math.max(1, stage.clientWidth);
  const height = Math.max(1, stage.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function updateSimulatorCamera() {
  if (selectedItem() || state.pendingId || Date.now() < state.controlsActiveUntil) return;
  const radius = Math.max(tankSize.x, tankSize.z) * 2.15;
  state.orbitAngle += 0.002;
  const target = new THREE.Vector3(Math.sin(state.orbitAngle) * 0.55, 0.15, 0);
  const desired = new THREE.Vector3(
    0,
    Math.max(1.8, tankSize.y * 1.15),
    radius
  );
  camera.position.lerp(desired, 0.018);
  controls.target.lerp(target, 0.025);
}

function renderFrame() {
  resize();
  updateSimulatorCamera();
  controls.update();
  updateWorldControls();
  positionFloaterCards();
  renderer.render(scene, camera);
}

function animate() {
  renderFrame();
  requestAnimationFrame(animate);
}

bindControls();
if (screenshotMode) {
  refreshLayout().then(() => {
    renderFrame();
    document.documentElement.dataset.screenshotReady = 'true';
  }).catch(error => {
    document.documentElement.dataset.screenshotError = error.message || 'layout unavailable';
  });
} else {
  ensureLiveRotation();
  refreshLayout().catch(() => {});
  refreshLighthouseStatus().catch(() => {});
  refreshReeflexStatus().catch(() => {});
  window.setInterval(() => refreshLayout().catch(() => {}), 2000);
  window.setInterval(() => refreshLighthouseStatus().catch(() => {}), 3000);
  window.setInterval(() => refreshReeflexStatus().catch(() => {}), 3000);
  window.setInterval(() => refreshVisionStatus().catch(() => {}), 2000);
  window.setInterval(() => pollFloaterFrames().catch(() => {}), 5000);
  window.setInterval(() => {
    const item = selectedItem();
    const src = sourceForPreview(item);
    if (src && !item.stream_url) {
      preview.src = src;
      pipPreview.src = src;
    }
  }, 10000);
}
window.addEventListener('resize', resize);
feedPrevious?.addEventListener('click', () => {
  state.manualViewId = null; state.liveIndex -= 1; setLiveCamera(state.liveIndex); updateStageFeed();
});
feedNext?.addEventListener('click', () => {
  state.manualViewId = null; state.liveIndex += 1; setLiveCamera(state.liveIndex); updateStageFeed();
});
feedPin?.addEventListener('click', () => {
  if (state.manualViewId) { state.manualViewId = null; state.manualViewUntil = 0; feedPin.textContent = 'Pin'; }
  else if (state.stageCameraId) { pinView(state.stageCameraId); feedPin.textContent = 'Unpin'; }
  updateStageFeed();
});
sightingShutter?.addEventListener('click', () => captureActiveSighting());
document.getElementById('open-sightings')?.addEventListener('click', () => {
  sightingsDrawer.hidden = false; loadSightings().catch(() => {});
});
document.getElementById('close-sightings')?.addEventListener('click', () => { sightingsDrawer.hidden = true; });
document.getElementById('deep-confirm')?.addEventListener('click', analyzePendingSighting);
document.getElementById('add-structure')?.addEventListener('click', addStructure);
document.getElementById('scatter-structures')?.addEventListener('click', scatterStructures);
document.getElementById('raydar-survey')?.addEventListener('click', () => setAutonomy('raydar', 'start'));
document.getElementById('raydar-stop')?.addEventListener('click', () => setAutonomy('raydar', 'stop'));
document.getElementById('reeflex-survey')?.addEventListener('click', () => setAutonomy('reeflex', 'start'));
document.getElementById('reeflex-auto-stop')?.addEventListener('click', () => setAutonomy('reeflex', 'stop'));
homeButton?.addEventListener('click', returnHome);
decorateToggle?.addEventListener('click', () => {
  if (!structureToolbar) return;
  structureToolbar.hidden = false;
  decorateToggle.hidden = true;
});
setupButton.addEventListener('click', openSetup);
dockToggle.addEventListener('click', () => {
  if (!sidePanel) return;
  const open = sidePanel.classList.toggle('open');
  dockToggle.classList.toggle('open', open);
  dockToggle.setAttribute('aria-expanded', String(open));
  dockToggle.setAttribute('aria-label', open ? 'Close placement controls' : 'Open placement controls');
});
setupBack.addEventListener('click', () => {
  if (state.setupStep === 0) {
    if (state.setupDraft.mode) {
      state.setupDraft.mode = null;
      renderSetup();
      return;
    }
    closeSetup(false);
    return;
  }
  state.setupStep = Math.max(0, state.setupStep - 1);
  renderSetup();
});
setupSkip.addEventListener('click', () => closeSetup(true));
identifySelected?.addEventListener('click', () => {
  const item = selectedItem();
  if (!item || !isCameraItem(item)) return;
  state.setupStep = 0;
  state.setupDraft = { mode: 'identify', cameraId: cameraId(item) };
  setupOverlay.hidden = false;
  renderSetup();
});
removeSelected?.addEventListener('click', () => {
  removeSelectedItem().catch(() => {});
});
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') returnHome();
});
document.addEventListener('error', event => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement)) return;
  const current = image.dataset.streamSource || image.getAttribute('src') || '';
  if (!current.includes('/stream')) return;
  image.dataset.streamSource = current.split(/[?&]reconnect=/)[0];
  window.clearTimeout(image._reconnectTimer);
  image._reconnectTimer = window.setTimeout(() => {
    const separator = image.dataset.streamSource.includes('?') ? '&' : '?';
    image.src = `${image.dataset.streamSource}${separator}reconnect=${Date.now()}`;
  }, 1500);
}, true);
animate();
