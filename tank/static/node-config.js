const dataEl = document.querySelector("#ingest-data");

if (dataEl) {
  const summary = JSON.parse(dataEl.textContent);
  const config = summary.node_config || {};
  renderBootSetup(summary);
  hydrateInventory(config);
  renderCameraConfig(summary, config);
  renderSystemsManager(summary);
  renderFeederViewports(summary, config);
  document.querySelector("#refreshNode")?.addEventListener("click", () => window.location.reload());
  document.querySelector("#saveNodeConfig")?.addEventListener("click", () => saveNodeConfig(config));
  document.querySelector("#selfTestSystems")?.addEventListener("click", () => runSelfTest(false));
  document.querySelector("#repairSystems")?.addEventListener("click", () => runSelfTest(true));
}

function renderBootSetup(summary) {
  const container = document.querySelector("#bootSetup");
  if (!container) return;
  const boot = summary.boot || {};
  const host = summary.host || {};
  const systems = summary.systems || {};
  const commands = boot.commands || {};
  const floaterCount = (summary.expected_nodes || []).length;
  const usbCount = (summary.usb_cameras || []).length;
  const issueCount = (systems.issues || []).length;
  const dashboardUrl = boot.dashboard_url || window.location.origin;
  const cameraServiceUrl = boot.camera_service_url || window.location.origin.replace(":8080", ":5050");
  const assignedFloaters = (summary.expected_nodes || []).map((nodeId) => `<span>${escapeHtml(nodeId)}</span>`).join("");

  container.innerHTML = `
    <article class="boot-card boot-identity">
      <div class="boot-card-title">
        <strong>${escapeHtml(host.label || "Edge Node")}</strong>
        <span class="status-pill ${issueCount ? "warn" : "ok"}">${issueCount ? `${issueCount} issue${issueCount === 1 ? "" : "s"}` : "ready"}</span>
      </div>
      <dl class="status-grid node-status">
        <dt>Node ID</dt><dd>${escapeHtml(host.id || "unknown")}</dd>
        <dt>Hub ID</dt><dd>${escapeHtml(host.hub_id || "unknown")}</dd>
        <dt>Project</dt><dd>${escapeHtml(boot.project_root || "/home/one/Projects/sync-tank/host")}</dd>
        <dt>Uploads</dt><dd>${escapeHtml(boot.upload_dir || "test_uploads")}</dd>
      </dl>
    </article>

    <article class="boot-card">
      <div class="boot-card-title">
        <strong>Service URLs</strong>
        <span class="count-pill">LAN</span>
      </div>
      <div class="link-stack">
        <a href="${escapeHtml(dashboardUrl)}">${escapeHtml(dashboardUrl)}</a>
        <a href="${escapeHtml(cameraServiceUrl)}">${escapeHtml(cameraServiceUrl)}</a>
      </div>
      <p class="viewport-note">Dashboard receives floater uploads on port 8080. Camera snapshots and MJPEG streams are served on port 5050.</p>
    </article>

    <article class="boot-card">
      <div class="boot-card-title">
        <strong>Attached Systems</strong>
        <span class="count-pill">${usbCount + floaterCount} feeds</span>
      </div>
      <div class="boot-metrics">
        <div><span>${usbCount}</span><small>ReefScope USB</small></div>
        <div><span>${floaterCount}</span><small>Floater nodes</small></div>
        <div><span>${Number(summary.inventory?.robotic_arms || 0)}</span><small>REEFLEX</small></div>
      </div>
      <div class="chip-list">${assignedFloaters || "<span>No floater nodes assigned</span>"}</div>
    </article>

    <article class="boot-card boot-card-wide">
      <div class="boot-card-title">
        <strong>Startup Commands</strong>
        <span class="count-pill">copy + run</span>
      </div>
      <div class="command-list">
        ${commandRow("Dashboard receiver", commands.start_dashboard)}
        ${commandRow("Camera service", commands.start_camera_service)}
        ${commandRow("USB self-test", commands.self_test)}
        ${commandRow("Register PC hub", commands.register_pc_hub)}
      </div>
    </article>
  `;

  container.querySelectorAll("[data-copy-command]").forEach((button) => {
    button.addEventListener("click", async () => {
      const command = button.getAttribute("data-copy-command") || "";
      try {
        await navigator.clipboard.writeText(command);
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = "Copy"; }, 1200);
      } catch {
        button.textContent = "Select";
      }
    });
  });
}

function commandRow(label, command) {
  if (!command) return "";
  return `
    <div class="command-row">
      <div>
        <span>${escapeHtml(label)}</span>
        <code>${escapeHtml(command)}</code>
      </div>
      <button type="button" class="secondary-action" data-copy-command="${escapeHtml(command)}">Copy</button>
    </div>
  `;
}

function hydrateInventory(config) {
  const inventory = config.inventory || {};
  const feeders = config.feeders || {};
  setValue("roboticArms", inventory.robotic_arms ?? 1);
  setValue("endoscopeCameras", inventory.endoscope_cameras ?? 0);
  setValue("floaterCameras", inventory.floater_cameras ?? 0);
  setValue("lighthouses", inventory.lighthouses ?? 0);
  setValue("solidFeeders", feeders.solid ?? 0);
  setValue("liquidFeeders", feeders.liquid ?? 0);
  setValue("miscFeeders", feeders.misc ?? 0);
  const validatedByHand = document.querySelector("#validatedByHand");
  if (validatedByHand) {
    validatedByHand.checked = Boolean(config.validation?.validated_by_hand || config.validation?.status === "validated");
  }
  setValue("nodeNotes", config.notes || "");
}

const CAMERA_TYPE_LABELS = {
  floater_cam: "Floater Camera",
  endoscope_cam: "ReefScope Camera",
  robot_arm_cam: "REEFLEX Camera",
  feeder_cam: "Feeder Camera",
  lighthouse_cam: "Lighthouse Camera",
  overview_cam: "Overview Camera",
};

function renderCameraConfig(summary, config) {
  const container = document.querySelector("#cameraConfigList");
  if (!container) return;
  const cameras = summary.consolidated_cameras || [];
  container.innerHTML = cameras.map((camera) => {
    const configured = (config.cameras || {})[camera.camera_id] || {};
    const label = configured.label || camera.label || camera.camera_id;
    const type = configured.camera_type || camera.camera_type;
    return `
      <div class="source-row" data-camera-id="${escapeHtml(camera.camera_id)}">
        <div>
          <strong>${escapeHtml(label)}</strong>
          <div class="muted">${escapeHtml(camera.camera_id)} · ${escapeHtml(CAMERA_TYPE_LABELS[type] || type)} · ${escapeHtml(camera.status || "unknown")}</div>
        </div>
        <label>Label <input data-field="label" value="${escapeHtml(label)}"></label>
        <label>Type
          <select data-field="camera_type">
            ${option("floater_cam", type)}
            ${option("endoscope_cam", type)}
            ${option("robot_arm_cam", type)}
            ${option("feeder_cam", type)}
            ${option("lighthouse_cam", type)}
            ${option("overview_cam", type)}
          </select>
        </label>
        <label>Owner Node <input data-field="node_id" value="${escapeHtml(configured.node_id || camera.node_id || summary.host?.id || "")}"></label>
        <label>Relay Node <input data-field="relay_node_id" value="${escapeHtml(configured.relay_node_id || camera.relay_node_id || "")}" placeholder="only if relayed"></label>
        <label>Tank ID <input data-field="tank_id" value="${escapeHtml(configured.tank_id || camera.tank_id || "tank-main")}"></label>
        <label class="check-row"><input data-field="enabled" type="checkbox" ${configured.enabled !== false ? "checked" : ""}> Enabled</label>
      </div>
    `;
  }).join("");
}

function renderSystemsManager(summary) {
  const container = document.querySelector("#systemsManager");
  if (!container) return;
  const systems = summary.systems || {};
  const cards = [
    systemCard("ReefScope Cameras", systems.reefscope),
    systemCard("Floater Cameras", systems.floaters),
    systemCard("REEFLEX", systems.reeflex),
    systemCard("Lighthouses", systems.lighthouse),
    systemCard("Feeders", systems.feeders),
  ].join("");
  const issues = (systems.issues || []).length
    ? `<div class="system-issues">${(systems.issues || []).map((issue) => `<div>${escapeHtml(issue)}</div>`).join("")}</div>`
    : '<div class="system-issues ok">All configured edge systems are accounted for.</div>';
  container.innerHTML = cards + issues;
}

function systemCard(label, system = {}) {
  const status = system.status || "idle";
  return `
    <div class="system-card ${escapeHtml(status)}">
      <div class="system-topline">
        <strong>${escapeHtml(label)}</strong>
        <span class="status-pill ${escapeHtml(status)}">${escapeHtml(status)}</span>
      </div>
      <div class="system-count">${Number(system.active || 0)} / ${Number(system.expected || 0)}</div>
    </div>
  `;
}

function renderFeederViewports(summary, config) {
  const container = document.querySelector("#feederViewportList");
  if (!container) return;
  const cameras = summary.consolidated_cameras || [];
  const viewports = config.feeder_viewports || {};
  const feeders = config.feeders || {};
  const sections = [
    ["solid", "Solid Feeder", feeders.solid || 0],
    ["liquid", "Liquid Feeder", feeders.liquid || 0],
    ["misc", "Misc Feeder", feeders.misc || 0],
  ];
  container.innerHTML = sections.map(([id, label, count]) => {
    const selected = viewports[id]?.camera_id || "";
    const camera = cameras.find((item) => item.camera_id === selected);
    return `
      <div class="feeder-card" data-feeder-id="${escapeHtml(id)}">
        <div class="system-topline">
          <strong>${escapeHtml(label)}</strong>
          <span class="count-pill">${Number(count)} configured</span>
        </div>
        <label>Viewport Camera
          <select data-feeder-field="camera_id">
            <option value="">No camera assigned</option>
            ${cameras.map((item) => `<option value="${escapeHtml(item.camera_id)}" ${item.camera_id === selected ? "selected" : ""}>${escapeHtml(item.label || item.camera_id)}</option>`).join("")}
          </select>
        </label>
        ${feederPreview(camera)}
      </div>
    `;
  }).join("");
}

async function runSelfTest(repair) {
  const results = document.querySelector("#selfTestResults");
  const selfTest = document.querySelector("#selfTestSystems");
  const repairButton = document.querySelector("#repairSystems");
  if (!results) return;
  selfTest.disabled = true;
  repairButton.disabled = true;
  results.hidden = false;
  results.innerHTML = `<div class="viewport-note">${repair ? "Repairing and retesting cameras..." : "Testing camera JPEG capture..."}</div>`;
  try {
    const response = await fetch("/api/systems/self-test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ repair }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Self test failed");
    renderSelfTestResults(results, payload);
  } catch (error) {
    results.innerHTML = `<div class="system-issues">${escapeHtml(error.message || "Self test failed")}</div>`;
  } finally {
    selfTest.disabled = false;
    repairButton.disabled = false;
  }
}

function renderSelfTestResults(container, payload) {
  const rows = (payload.cameras || []).map((camera) => {
    const repair = camera.repair;
    const cleared = repair?.cleared?.length ? ` · cleared ${repair.cleared.length}` : "";
    const skipped = repair?.skipped?.length ? ` · skipped ${repair.skipped.length}` : "";
    const retry = camera.retry ? " · retested" : "";
    const detail = camera.ok
      ? `${camera.jpeg_bytes || 0} JPEG bytes${cleared}${retry}`
      : `${camera.error || "capture failed"}${cleared}${skipped}`;
    return `
      <div class="self-test-row ${camera.ok ? "ok" : "warn"}">
        <div>
          <strong>${escapeHtml(camera.camera_id)}</strong>
          <span>${escapeHtml(camera.device)} · ${escapeHtml(camera.stable_match?.id_path || "unknown port")}</span>
        </div>
        <div>${escapeHtml(detail)}</div>
      </div>
    `;
  }).join("");
  container.innerHTML = `
    <div class="self-test-summary ${payload.ok ? "ok" : "warn"}">
      ${payload.ok ? "All USB cameras produced valid JPEG frames." : "One or more USB cameras failed JPEG capture."}
    </div>
    ${rows}
  `;
}

function feederPreview(camera) {
  if (!camera) return '<div class="viewport-note">No viewport assigned</div>';
  if (camera.latest_image_url) {
    return `<img class="viewport-preview" src="${escapeHtml(camera.latest_image_url)}" alt="${escapeHtml(camera.label || camera.camera_id)} viewport">`;
  }
  if (camera.snapshot_url) {
    return `<img class="viewport-preview" src="${escapeHtml(camera.snapshot_url)}?t=${Date.now()}" alt="${escapeHtml(camera.label || camera.camera_id)} viewport">`;
  }
  return `<div class="viewport-note">${escapeHtml(camera.label || camera.camera_id)} has no snapshot URL</div>`;
}

async function saveNodeConfig(config) {
  config.inventory ||= {};
  config.feeders ||= {};
  config.cameras ||= {};
  config.feeder_viewports ||= {};
  config.inventory.robotic_arms = numberValue("roboticArms");
  config.inventory.endoscope_cameras = numberValue("endoscopeCameras");
  config.inventory.floater_cameras = numberValue("floaterCameras");
  config.inventory.lighthouses = numberValue("lighthouses");
  config.feeders.solid = numberValue("solidFeeders");
  config.feeders.liquid = numberValue("liquidFeeders");
  config.feeders.misc = numberValue("miscFeeders");
  const validatedByHand = document.querySelector("#validatedByHand")?.checked || false;
  config.validation = {
    ...(config.validation || {}),
    status: validatedByHand ? "validated" : "default",
    validated_by_hand: validatedByHand,
  };
  config.notes = document.querySelector("#nodeNotes")?.value || "";

  document.querySelectorAll(".source-row").forEach((row) => {
    const cameraId = row.dataset.cameraId;
    config.cameras[cameraId] ||= {};
    row.querySelectorAll("[data-field]").forEach((input) => {
      const field = input.dataset.field;
      config.cameras[cameraId][field] = input.type === "checkbox" ? input.checked : input.value;
    });
  });
  document.querySelectorAll(".feeder-card").forEach((card) => {
    const feederId = card.dataset.feederId;
    config.feeder_viewports[feederId] ||= {};
    const cameraId = card.querySelector("[data-feeder-field='camera_id']")?.value || "";
    if (cameraId) {
      config.feeder_viewports[feederId].camera_id = cameraId;
    } else {
      delete config.feeder_viewports[feederId];
    }
  });

  const response = await fetch("/api/node-config", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(config),
  });
  const payload = await response.json();
  if (!response.ok) {
    alert(payload.error || "Could not save node config");
    return;
  }
  window.location.reload();
}

function setValue(id, value) {
  const el = document.querySelector(`#${id}`);
  if (el) el.value = value;
}

function numberValue(id) {
  return Number.parseInt(document.querySelector(`#${id}`)?.value || "0", 10);
}

function option(value, selected) {
  return `<option value="${value}" ${value === selected ? "selected" : ""}>${CAMERA_TYPE_LABELS[value] || value}</option>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}
