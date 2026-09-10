const $ = id => document.getElementById(id);
const status = message => { $('save-status').textContent = message; };
const KEY = 'sync-tank-builder-v1';
const kinds = ['hide', 'rock', 'endo', 'floater'];
const faces = ['x-', 'x+', 'z-', 'z+', 'y+'];
const views = ['front', 'right', 'top'];
const copy = value => JSON.parse(JSON.stringify(value));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
let draft = { format: 'sync-tank-builder', version: 1, tank: { label: 'My tank', width: 60, height: 40, depth: 30 }, items: [], references: {} };
let selected = null;
let undo = [], redo = [];
const photos = new Map();
const maxItems = 100;
let storedProblem = false;

function validate(value) {
  const number = (n, a, b) => typeof n === 'number' && Number.isFinite(n) && n >= a && n <= b;
  const text = (s, limit) => typeof s === 'string' && s.length <= limit;
  if (value?.format !== 'sync-tank-builder' || value.version !== 1 || !value.tank || !text(value.tank.label, 80) || !['width', 'height', 'depth'].every(k => number(value.tank[k], 10, 300))) throw new Error('Not a valid version 1 tank-builder file.');
  if (!Array.isArray(value.items) || value.items.length > maxItems) throw new Error('The draft may contain up to 100 objects.');
  const ids = new Set();
  const clean = { format: value.format, version: 1, tank: { label: value.tank.label, width: value.tank.width, height: value.tank.height, depth: value.tank.depth }, items: [], references: {} };
  for (const i of value.items) {
    if (!i || !text(i.id, 80) || !i.id || ids.has(i.id) || !kinds.includes(i.kind) || !text(i.label, 80) || typeof i.placed !== 'boolean' || !['x', 'y', 'z'].every(k => number(i.position?.[k], 0, 1)) || !number(i.size, 1, 30) || !number(i.yaw, -180, 180) || !number(i.pitch, -80, 80) || !number(i.fov, 20, 110) || !faces.includes(i.mount) || !text(i.notes, 1000)) throw new Error('An object has missing or invalid placement values.');
    ids.add(i.id);
    clean.items.push({ id: i.id, kind: i.kind, label: i.label, placed: i.placed, position: { x: i.position.x, y: i.position.y, z: i.position.z }, size: i.size, yaw: i.yaw, pitch: i.pitch, fov: i.fov, mount: i.mount, notes: i.notes });
  }
  for (const view of views) {
    const ref = value.references?.[view];
    if (!ref) continue;
    if (!text(ref.name, 200) || !Array.isArray(ref.crop) || ref.crop.length !== 4 || !ref.crop.every(n => number(n, 0, 1)) || ref.crop[2] - ref.crop[0] < 0.03 || ref.crop[3] - ref.crop[1] < 0.03) throw new Error('Invalid photo crop metadata.');
    clean.references[view] = { name: ref.name, crop: [...ref.crop] };
  }
  return clean;
}

try { const saved = localStorage.getItem(KEY); if (saved) draft = validate(JSON.parse(saved)); } catch { storedProblem = true; }

async function start() {
  const THREE = await import('../sync/static/vendor/three.module.js');
  const { OrbitControls } = await import('../sync/static/vendor/OrbitControls.js');
  const stage = $('stage');
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setClearColor('#09211e');
  stage.replaceChildren(renderer.domElement);
  renderer.domElement.setAttribute('aria-label', '3D tank canvas');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;
  controls.minDistance = 1;
  controls.maxDistance = 18;
  const ambient = new THREE.HemisphereLight('#efffed', '#445f48', 2.4);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight('#fff0d0', 2);
  sun.position.set(-3, 7, 4); scene.add(sun);
  const tankGroup = new THREE.Group(), itemsGroup = new THREE.Group(), guideGroup = new THREE.Group();
  scene.add(tankGroup, itemsGroup, guideGroup);
  let frame = 0;
  function render() { if (!frame) frame = requestAnimationFrame(() => { frame = 0; renderer.render(scene, camera); }); }
  controls.addEventListener('change', render);
  const unit = () => 4 / Math.max(draft.tank.width, draft.tank.height, draft.tank.depth);
  const dimensions = () => ({ x: draft.tank.width * unit(), y: draft.tank.height * unit(), z: draft.tank.depth * unit() });
  const current = () => draft.items.find(i => i.id === selected);
  const position = i => { const d = dimensions(); return new THREE.Vector3((i.position.x - 0.5) * d.x, i.position.y * d.y, (i.position.z - 0.5) * d.z); };
  function dispose(group) { group.traverse(o => { o.geometry?.dispose(); if (o.material) for (const m of [].concat(o.material)) { m.map?.dispose(); m.dispose(); } }); group.clear(); }
  function constrain(item) {
    const d = draft.tank;
    item.size = clamp(item.size, 1, Math.min(30, d.width * 0.8, d.height * 0.8, d.depth * 0.8));
    const halves = { x: item.size / 2 / d.width, y: item.size / 2 / d.height, z: item.size / 2 / d.depth };
    for (const axis of ['x', 'y', 'z']) item.position[axis] = clamp(item.position[axis], halves[axis], 1 - halves[axis]);
    if (item.kind === 'floater') item.position[item.mount[0]] = item.mount[1] === '+' ? 1 : 0;
  }
  function direction(item) {
    if (item.kind === 'floater') { const v = new THREE.Vector3(); v[item.mount[0]] = item.mount[1] === '+' ? -1 : 1; return v; }
    const yaw = THREE.MathUtils.degToRad(item.yaw), pitch = THREE.MathUtils.degToRad(item.pitch);
    return new THREE.Vector3(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
  }
  function label(text, x, y, z) {
    const c = document.createElement('canvas'); c.width = 256; c.height = 64;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#e8dc99'; ctx.font = 'bold 26px monospace'; ctx.textAlign = 'center'; ctx.fillText(text, 128, 40);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
    sprite.position.set(x, y, z); sprite.scale.set(1, 0.25, 1); tankGroup.add(sprite);
  }
  function buildTank() {
    dispose(tankGroup);
    const d = dimensions();
    const box = new THREE.BoxGeometry(d.x, d.y, d.z);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: '#95d8c7' }));
    edges.position.y = d.y / 2; tankGroup.add(edges); box.dispose();
    const floor = new THREE.Mesh(new THREE.BoxGeometry(d.x, 0.04, d.z), new THREE.MeshStandardMaterial({ color: '#567666', roughness: 1 }));
    floor.position.y = -0.025; tankGroup.add(floor);
    const grid = new THREE.GridHelper(1, 12, '#abc49a', '#769287'); grid.scale.set(d.x, 1, d.z); grid.position.y = 0.005; tankGroup.add(grid);
    label('FRONT', 0, -0.16, d.z / 2 + 0.16); label('BACK', 0, -0.16, -d.z / 2 - 0.16);
    label('LEFT', -d.x / 2 - 0.28, 0, 0); label('RIGHT', d.x / 2 + 0.28, 0, 0);
    render();
  }
  function mesh(geometry, color) { return new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness: 0.8 })); }
  function buildItems() {
    dispose(itemsGroup); dispose(guideGroup);
    for (const item of draft.items) {
      constrain(item);
      if (!item.placed) continue;
      const group = new THREE.Group(); group.userData.id = item.id; group.position.copy(position(item));
      const size = item.size * unit();
      const color = item.id === selected ? '#ffdc61' : ({ hide: '#c7919e', rock: '#879f88', endo: '#b2d6d6', floater: '#a3c3e2' })[item.kind];
      if (item.kind === 'hide') group.add(mesh(new THREE.BoxGeometry(size, size, size), color));
      if (item.kind === 'rock') group.add(mesh(new THREE.DodecahedronGeometry(size / 2, 0), color));
      if (['hide', 'rock'].includes(item.kind)) group.rotation.y = THREE.MathUtils.degToRad(item.yaw);
      if (['endo', 'floater'].includes(item.kind)) {
        const radius = size / (item.kind === 'floater' ? 2 : 4), length = item.kind === 'floater' ? size * 0.12 : size * 0.65;
        group.add(mesh(new THREE.CylinderGeometry(radius, radius, length, 20), color));
        const lens = mesh(new THREE.CylinderGeometry(radius * 0.48, radius * 0.48, size * 0.03, 16), '#162b39'); lens.position.y = length / 2 + size * 0.02; group.add(lens);
        group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction(item));
        if (item.id === selected) {
          const view = new THREE.PerspectiveCamera(item.fov, 4 / 3, 0.02, Math.max(dimensions().x, dimensions().z) * 0.6);
          view.position.copy(group.position); view.up.set(0, 1, 0); if (Math.abs(direction(item).y) > 0.99) view.up.set(0, 0, -1);
          view.lookAt(group.position.clone().add(direction(item))); view.updateMatrixWorld();
          guideGroup.add(new THREE.CameraHelper(view));
        }
      }
      itemsGroup.add(group);
    }
    render();
  }
  function setView() {
    const d = dimensions(), mode = $('view').value;
    const distance = Math.max(d.x, d.y, d.z) * 1.9;
    controls.target.set(0, d.y / 2, 0); camera.up.set(0, 1, 0);
    if (mode === 'front') camera.position.set(0, d.y / 2, distance);
    else if (mode === 'right') camera.position.set(distance, d.y / 2, 0);
    else if (mode === 'top') { camera.position.set(0, distance + d.y, 0); camera.up.set(0, 0, -1); }
    else camera.position.set(distance * 0.68, d.y + distance * 0.35, distance * 0.8);
    controls.update(); render();
  }
  function save() {
    if (storedProblem) return status('Stored draft unavailable; it has not been overwritten. Export your work or import a valid draft.');
    try { localStorage.setItem(KEY, JSON.stringify(draft)); status('Draft saved on this browser. Photos are session-only.'); }
    catch { status('Browser storage unavailable. Export JSON to keep this draft.'); }
  }
  function history(before = copy(draft)) { undo.push(before); if (undo.length > 40) undo.shift(); redo = []; updateHistory(); }
  function updateHistory() { $('undo').disabled = !undo.length; $('redo').disabled = !redo.length; }
  function sync() {
    for (const k of ['width', 'height', 'depth']) $(k).value = draft.tank[k]; $('tank-name').value = draft.tank.label;
    $('items').replaceChildren();
    for (const item of draft.items) {
      const li = document.createElement('li'), b = document.createElement('button'); b.textContent = `${item.label} / ${item.placed ? 'Placed' : 'Unplaced'}`;
      b.setAttribute('aria-pressed', String(item.id === selected)); b.onclick = () => { selected = item.id; $('interaction').value = 'move'; sync(); buildItems(); }; li.append(b); $('items').append(li);
    }
    const item = current(); $('inspector').hidden = !item;
    $('selection-status').textContent = item ? `${item.label} / ${item.placed ? 'Editing' : 'Unplaced'}` : 'No object selected';
    if (!item) return;
    $('selection-title').textContent = item.label; $('label').value = item.label; $('notes').value = item.notes; $('mount').value = item.mount;
    $('mount-field').hidden = item.kind !== 'floater'; $('aim-fields').hidden = item.kind === 'floater'; $('pitch-field').hidden = item.kind !== 'endo'; $('fov-field').hidden = !['endo', 'floater'].includes(item.kind);
    for (const [axis, dimension] of [['x', 'width'], ['y', 'height'], ['z', 'depth']]) { $(axis).value = item.position[axis]; $(axis).disabled = item.kind === 'floater' && axis === item.mount[0]; $(axis + '-value').value = `${(item.position[axis] * draft.tank[dimension]).toFixed(1)} cm`; }
    $('size').max = Math.min(30, draft.tank.width * 0.8, draft.tank.height * 0.8, draft.tank.depth * 0.8);
    for (const key of ['size', 'yaw', 'pitch', 'fov']) { $(key).value = item[key]; $(key + '-value').value = `${item[key]}${key === 'size' ? ' cm' : ' deg'}`; }
  }
  function changed() { buildItems(); sync(); save(); }
  function deselect() { selected = null; $('interaction').value = 'orbit'; sync(); buildItems(); }
  $('view').onchange = () => { const plane = { front: 'xy', right: 'zy', top: 'xz' }[$('view').value]; if (plane) $('plane').value = plane; setView(); };
  $('done').onclick = deselect;
  $('add').onclick = () => {
    if (draft.items.length >= maxItems) return status('The draft may contain up to 100 objects.');
    history(); const kind = $('kind').value; const id = crypto.randomUUID(), size = kind === 'hide' ? 8 : 5;
    draft.items.push({ id, kind, label: `${kind === 'endo' ? 'Endoscope' : kind === 'floater' ? 'Floater' : kind === 'hide' ? 'Hide' : 'Rock'} ${draft.items.length + 1}`, placed: false, position: { x: 0.5, y: ['hide', 'rock'].includes(kind) ? size / (2 * draft.tank.height) : 0.5, z: 0.5 }, size, yaw: 0, pitch: 0, fov: 60, mount: 'z+', notes: '' });
    selected = id; $('interaction').value = 'move'; changed();
  };
  $('apply-tank').onclick = () => {
    if (!['width', 'height', 'depth'].every(k => $(k).reportValidity() && Number($(k).value) >= 10 && Number($(k).value) <= 300)) return status('Tank dimensions must be 10 to 300 cm.');
    history(); draft.tank = { label: $('tank-name').value || 'My tank', width: Number($('width').value), height: Number($('height').value), depth: Number($('depth').value) }; buildTank(); changed(); setView();
  };
  for (const id of ['delete', 'unplace']) $(id).onclick = () => { const item = current(); if (!item) return; history(); if (id === 'delete') { draft.items = draft.items.filter(i => i.id !== selected); selected = null; } else item.placed = false; changed(); };
  for (const key of ['label', 'notes', 'mount']) $(key).onchange = () => { const item = current(); if (!item) return; history(); item[key] = $(key).value; changed(); };
  for (const key of ['x', 'y', 'z', 'size', 'yaw', 'pitch', 'fov']) {
    let before = null;
    $(key).addEventListener('focus', () => { before = copy(draft); });
    $(key).addEventListener('pointerdown', () => { before = copy(draft); });
    $(key).oninput = () => { const item = current(); if (!item) return; if (['x', 'y', 'z'].includes(key)) { item.position[key] = Number($(key).value); item.placed = true; } else item[key] = Number($(key).value); changed(); };
    $(key).onchange = () => { if (before) { history(before); before = copy(draft); } };
  }
  function restore(value) { draft = value; if (!current()) selected = null; buildTank(); changed(); showPhoto(); updateHistory(); setView(); }
  $('undo').onclick = () => { if (undo.length) { redo.push(copy(draft)); restore(undo.pop()); } };
  $('redo').onclick = () => { if (redo.length) { undo.push(copy(draft)); restore(redo.pop()); } };
  $('export').onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(draft, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = 'sync-tank-draft.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); status('Draft exported. Reference photos are not included.'); };
  $('import').onchange = async event => {
    const file = event.target.files[0]; if (!file) return;
    try { if (file.size > 1024 * 1024) throw new Error('JSON must be smaller than 1 MB.'); const next = validate(JSON.parse(await file.text())); history(); for (const p of photos.values()) URL.revokeObjectURL(p.url); photos.clear(); storedProblem = false; restore(next); setView(); }
    catch (error) { status(`Import rejected: ${error.message}`); }
    event.target.value = '';
  };
  const ray = new THREE.Raycaster(), pointer = new THREE.Vector2(); let drag = null, down = null, lastDrag = null, firstClick = null;
  function cast(event) { const rect = renderer.domElement.getBoundingClientRect(); pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1); ray.setFromCamera(pointer, camera); }
  function pick() { for (const hit of ray.intersectObjects(itemsGroup.children, true)) { let o = hit.object; while (o && !o.userData.id) o = o.parent; if (o) return o.userData.id; } return null; }
  function movePoint(event, item) {
    cast(event); const pos = position(item), d = dimensions(); let axis = $('plane').value === 'xz' ? 'y' : $('plane').value === 'xy' ? 'z' : 'x';
    if (item.kind === 'floater') axis = item.mount[0];
    const normal = new THREE.Vector3(); normal[axis] = 1;
    const hit = ray.ray.intersectPlane(new THREE.Plane(normal, -pos[axis]), new THREE.Vector3()); if (!hit) return;
    if (drag?.preserve && !drag.offset) drag.offset = pos.clone().sub(hit);
    if (drag?.offset) hit.add(drag.offset);
    item.position = { x: hit.x / d.x + 0.5, y: hit.y / d.y, z: hit.z / d.z + 0.5 }; item.placed = true; constrain(item); buildItems(); sync();
  }
  renderer.domElement.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    lastDrag = null;
    down = { x: event.clientX, y: event.clientY }; cast(event); const hit = pick();
    if ($('interaction').value !== 'move') return;
    if (hit) selected = hit;
    const item = current(); if (!item) return;
    if (event.detail > 1) return;
    event.stopImmediatePropagation(); controls.enabled = false;
    drag = { id: event.pointerId, before: copy(draft), undoBefore: [...undo], redoBefore: [...redo], preserve: Boolean(hit && item.placed) }; renderer.domElement.setPointerCapture(event.pointerId); movePoint(event, item);
  }, true);
  renderer.domElement.addEventListener('pointermove', event => { if (drag && drag.id === event.pointerId && current()) movePoint(event, current()); });
  function finish(event) {
    if (drag) { lastDrag = drag; if (JSON.stringify(drag.before) !== JSON.stringify(draft)) history(drag.before); drag = null; controls.enabled = true; changed(); }
    else if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) < 4 && event.detail < 2) { cast(event); const hit = pick(); if (hit) { selected = hit; $('interaction').value = 'move'; sync(); buildItems(); } }
    down = null;
  }
  renderer.domElement.addEventListener('pointerup', finish);
  renderer.domElement.addEventListener('pointercancel', () => { if (drag) { const before = drag.before; drag = null; controls.enabled = true; restore(before); } down = null; });
  renderer.domElement.addEventListener('click', event => { if (event.detail === 1) firstClick = lastDrag; });
  renderer.domElement.addEventListener('dblclick', event => { event.preventDefault(); if (firstClick) { draft = firstClick.before; undo = firstClick.undoBefore; redo = firstClick.redoBefore; updateHistory(); save(); } firstClick = null; deselect(); });

  const refCanvas = $('reference'), ctx = refCanvas.getContext('2d'); let cropStart = null, pendingCrop = null;
  const refView = () => $('photo-view').value;
  const activePhoto = () => { const p = photos.get(refView()); return p?.name === draft.references[refView()]?.name ? p : null; };
  function photoRect() { const image = activePhoto()?.image; if (!image) return null; const scale = Math.min(600 / image.width, 450 / image.height); return { x: (600 - image.width * scale) / 2, y: (450 - image.height * scale) / 2, w: image.width * scale, h: image.height * scale }; }
  function photoPoint(event) { const r = refCanvas.getBoundingClientRect(), p = photoRect(); if (!p) return null; const x = ((event.clientX - r.left) / r.width * 600 - p.x) / p.w, y = ((event.clientY - r.top) / r.height * 450 - p.y) / p.h; return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? [x, y] : null; }
  function drawPhoto() {
    ctx.fillStyle = '#10181a'; ctx.fillRect(0, 0, 600, 450); const p = photoRect(); if (!p) return;
    ctx.drawImage(photos.get(refView()).image, p.x, p.y, p.w, p.h);
    const crop = pendingCrop || draft.references[refView()]?.crop || [0, 0, 1, 1];
    ctx.strokeStyle = '#ffe173'; ctx.lineWidth = 3; ctx.strokeRect(p.x + crop[0] * p.w, p.y + crop[1] * p.h, (crop[2] - crop[0]) * p.w, (crop[3] - crop[1]) * p.h);
  }
  function showPhoto() { const photo = activePhoto(); $('photo-tools').hidden = !photo; $('photo-name').textContent = photo ? photo.name : draft.references[refView()] ? `${draft.references[refView()].name} / reload photo` : 'No photo loaded'; pendingCrop = null; drawPhoto(); }
  $('photo-view').onchange = showPhoto;
  $('photo-file').onchange = async event => {
    const file = event.target.files[0], view = refView(); if (!file) return;
    let url;
    try {
      if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 15 * 1024 * 1024) throw new Error('Choose a JPEG, PNG or WebP below 15 MB.');
      url = URL.createObjectURL(file); const image = new Image(); image.src = url; await image.decode(); if (image.width * image.height > 40000000) throw new Error('Photo exceeds 40 megapixels. Resize it before loading.');
      history(); const old = photos.get(view); if (old) URL.revokeObjectURL(old.url);
      photos.set(view, { image, url, name: file.name }); const existing = draft.references[view]; draft.references[view] = { name: file.name.slice(0, 200), crop: existing?.name === file.name ? existing.crop : [0, 0, 1, 1] }; showPhoto(); save();
    } catch (error) { if (url) URL.revokeObjectURL(url); status(error.message); }
    event.target.value = '';
  };
  $('clear-photo').onclick = () => { history(); const photo = photos.get(refView()); if (photo) URL.revokeObjectURL(photo.url); photos.delete(refView()); delete draft.references[refView()]; showPhoto(); save(); };
  refCanvas.addEventListener('pointerdown', event => {
    const point = photoPoint(event); if (!point) return;
    if ($('photo-mode').value === 'crop') { cropStart = point; refCanvas.setPointerCapture(event.pointerId); }
    else if ($('photo-mode').value === 'place') {
      const item = current(); if (!item) return status('Select an object before placing from a photo.');
      const crop = draft.references[refView()]?.crop || [0, 0, 1, 1];
      const u = (point[0] - crop[0]) / (crop[2] - crop[0]), v = (point[1] - crop[1]) / (crop[3] - crop[1]); if (u < 0 || u > 1 || v < 0 || v > 1) return;
      history(); if (refView() === 'front') { item.position.x = u; item.position.y = 1 - v; }
      else if (refView() === 'right') { item.position.z = 1 - u; item.position.y = 1 - v; }
      else { item.position.x = u; item.position.z = v; }
      item.placed = true; changed();
    }
  });
  refCanvas.addEventListener('pointermove', event => { const p = photoPoint(event); if (cropStart && p) { pendingCrop = [Math.min(cropStart[0], p[0]), Math.min(cropStart[1], p[1]), Math.max(cropStart[0], p[0]), Math.max(cropStart[1], p[1])]; drawPhoto(); } });
  refCanvas.addEventListener('pointerup', () => { if (pendingCrop && pendingCrop[2] - pendingCrop[0] >= 0.03 && pendingCrop[3] - pendingCrop[1] >= 0.03) { history(); draft.references[refView()].crop = pendingCrop; save(); } cropStart = null; pendingCrop = null; drawPhoto(); });
  refCanvas.addEventListener('pointercancel', () => { cropStart = null; pendingCrop = null; drawPhoto(); });
  new ResizeObserver(() => { const r = stage.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); camera.aspect = r.width / r.height; camera.updateProjectionMatrix(); render(); }).observe(stage);
  renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); status('3D context lost. Export your draft, then reload.'); });
  for (const id of ['add', 'apply-tank', 'export', 'import', 'photo-file']) $(id).disabled = false;
  buildTank(); buildItems(); sync(); setView(); showPhoto(); save();
}

start().catch(error => { $('loading')?.remove(); const message = document.createElement('p'); message.id = 'loading'; message.textContent = 'The 3D editor could not start. Use a browser with WebGL and open this page through the website or a local HTTP server.'; $('stage').append(message); status(`Editor unavailable: ${error.message}`); });
