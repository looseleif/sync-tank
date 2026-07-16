const statusLine = document.querySelector("#statusLine");
const servoControls = document.querySelector("#servoControls");
const cameraGrid = document.querySelector("#cameraGrid");
const cameraCount = document.querySelector("#cameraCount");
const hubState = document.querySelector("#hubState");
const hubMode = document.querySelector("#hubMode");
const hubUrl = document.querySelector("#hubUrl");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function renderArm(arm) {
  servoControls.innerHTML = "";
  Object.values(arm.servos || {}).forEach((servo) => {
    const row = document.createElement("div");
    row.className = "servo-control";
    row.innerHTML = `
      <label><strong>${servo.name}</strong><span>${servo.angle}°</span></label>
      <input type="range" min="${servo.min_angle}" max="${servo.max_angle}" value="${servo.angle}" step="1">
      <div class="muted">${servo.channel === null || servo.channel === undefined ? `GPIO ${servo.gpio_pin}` : `PCA9685 ch ${servo.channel}`} · ${servo.min_angle}°-${servo.max_angle}°</div>
    `;
    const slider = row.querySelector("input");
    const value = row.querySelector("span");
    slider.addEventListener("change", async () => {
      const result = await api(`/api/arm/servo/${servo.id}`, {
        method: "POST",
        body: JSON.stringify({ angle: Number(slider.value) }),
      });
      value.textContent = `${result.angle}°`;
    });
    servoControls.appendChild(row);
  });
}

function renderCameras(cameras) {
  cameraCount.textContent = `${cameras.length} camera${cameras.length === 1 ? "" : "s"}`;
  cameraGrid.innerHTML = "";
  if (!cameras.length) {
    cameraGrid.innerHTML = `<p class="muted">No cameras detected yet. Plug in a USB camera or run ESP32 discovery.</p>`;
    return;
  }
  cameras.forEach((camera) => {
    const card = document.createElement("article");
    card.className = "camera-card";
    const viewUrl = camera.stream_url || `/api/cameras/${camera.id}/snapshot`;
    card.innerHTML = `
      <header>
        <div>
          <strong>${camera.name || camera.id}</strong>
          <div class="muted">${camera.source_type || "camera"} · ${camera.id}</div>
        </div>
        <span class="pill">${camera.status || "unknown"}</span>
      </header>
      <img src="${viewUrl}" alt="${camera.name || camera.id} feed">
      <div class="camera-actions">
        <button data-action="snapshot">Send Snapshot</button>
        <button data-action="start">Start Hub</button>
        <button data-action="stop">Stop Hub</button>
      </div>
    `;
    card.querySelector('[data-action="snapshot"]').addEventListener("click", () => api(`/api/uplink/cameras/${camera.id}/snapshot`, { method: "POST", body: "{}" }).then(refresh));
    card.querySelector('[data-action="start"]').addEventListener("click", () => api(`/api/uplink/cameras/${camera.id}/start`, { method: "POST", body: "{}" }).then(refresh));
    card.querySelector('[data-action="stop"]').addEventListener("click", () => api(`/api/uplink/cameras/${camera.id}/stop`, { method: "POST", body: "{}" }).then(refresh));
    cameraGrid.appendChild(card);
  });
}

async function refresh() {
  try {
    const status = await api("/api/status");
    statusLine.textContent = `${status.tank_id} · arm ${status.arm.driver}`;
    hubState.textContent = status.hub.state;
    hubMode.textContent = status.hub.mode;
    hubUrl.textContent = status.hub.base_url || "not configured";
    renderArm(status.arm);
    renderCameras(status.cameras);
  } catch (error) {
    statusLine.textContent = `Host error: ${error.message}`;
  }
}

document.querySelector("#discoverBtn").addEventListener("click", async () => {
  statusLine.textContent = "Discovering ESP32 cameras...";
  await api("/api/cameras/discover", { method: "POST", body: "{}" });
  await refresh();
});

document.querySelector("#hubTestBtn").addEventListener("click", async () => {
  const result = await api("/api/uplink/test", { method: "POST", body: "{}" });
  hubState.textContent = result.state;
});

document.querySelector("#stopArmBtn").addEventListener("click", () => api("/api/arm/stop", { method: "POST", body: "{}" }).then(refresh));

refresh();
setInterval(refresh, 10000);
