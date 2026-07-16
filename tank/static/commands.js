const streamPollers = new Map();
const burstPlayers = new Map();

document.querySelectorAll(".stream-command").forEach((button) => {
  button.addEventListener("click", async () => {
    const nodeId = button.dataset.nodeId;
    const status = document.querySelector(`[data-command-status="${cssEscape(nodeId)}"]`);
    button.disabled = true;
    setStatus(status, "queueing stream...");
    try {
      const response = await fetch(`/api/node/${encodeURIComponent(nodeId)}/stream`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ stream_seconds: 30 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not queue stream");
      setStatus(status, "queued · waiting for next wake");
      startStreamPolling(nodeId);
    } catch (error) {
      setStatus(status, error.message || "stream request failed");
    } finally {
      button.disabled = false;
    }
  });
});

document.querySelectorAll("[data-stream-preview]").forEach((preview) => {
  startStreamPolling(preview.dataset.streamPreview);
});

function startStreamPolling(nodeId) {
  if (!nodeId || streamPollers.has(nodeId)) return;
  const poller = window.setInterval(() => pollStream(nodeId), 1000);
  streamPollers.set(nodeId, poller);
  pollStream(nodeId);
}

async function pollStream(nodeId) {
  const status = document.querySelector(`[data-command-status="${cssEscape(nodeId)}"]`);
  const preview = document.querySelector(`[data-stream-preview="${cssEscape(nodeId)}"]`);
  try {
    const response = await fetch(`/api/node/${encodeURIComponent(nodeId)}/stream/status`, { cache: "no-store" });
    if (!response.ok) return;
    const payload = await response.json();
    if (payload.status === "pending") {
      setStatus(status, "queued · waiting for next wake");
      setHistoricalPreview(preview, payload);
      renderTimeline(nodeId, payload);
      return;
    }
    const session = payload.session;
    if (!session) {
      setHistoricalPreview(preview, payload);
      renderTimeline(nodeId, payload);
      stopStreamPolling(nodeId);
      return;
    }
    if (session.status === "waiting_for_frames") {
      setStatus(status, "command delivered · waiting for first frame");
      setHistoricalPreview(preview, payload);
    } else if (session.status === "active") {
      const latest = formatTime(session.last_frame_at);
      setStatus(status, `stream active · ${session.frame_count || 0} frames${latest ? ` · latest ${latest}` : ""}`);
      setPreview(preview, session);
    } else if (session.status === "ended") {
      setStatus(status, `stream ended · ${session.frame_count || 0} frames`);
      setPreview(preview, session);
      stopStreamPolling(nodeId);
    }
    renderTimeline(nodeId, payload);
  } catch {
    setStatus(status, "stream status unavailable");
  }
}

function setPreview(preview, session) {
  const frame = session?.latest_frame;
  if (!preview || !frame?.url) return;
  preview.hidden = false;
  preview.title = session.status === "ended" ? "Latest saved stream frame" : "Live stream burst frame";
  preview.src = `${frame.url}?t=${encodeURIComponent(frame.received_at || Date.now())}`;
}

function setHistoricalPreview(preview, payload) {
  const session = (payload.history || []).find((item) => item?.latest_frame?.url);
  if (session) {
    setPreview(preview, session);
  } else if (preview?.dataset.streamPreview) {
    clearPreview(preview.dataset.streamPreview);
  }
}

function clearPreview(nodeId) {
  const preview = document.querySelector(`[data-stream-preview="${cssEscape(nodeId)}"]`);
  if (!preview) return;
  preview.hidden = true;
  preview.removeAttribute("src");
}

function stopStreamPolling(nodeId) {
  const poller = streamPollers.get(nodeId);
  if (poller) window.clearInterval(poller);
  streamPollers.delete(nodeId);
}

function setStatus(element, message) {
  if (element) element.textContent = message;
}

function renderTimeline(nodeId, payload) {
  const timeline = document.querySelector(`[data-stream-timeline="${cssEscape(nodeId)}"]`);
  if (!timeline) return;
  const items = [];
  if (payload.status === "pending") {
    items.push({
      state: "active",
      label: "Queued",
      meta: `waiting for wake${formatTime(payload.pending_queued_at) ? ` · ${formatTime(payload.pending_queued_at)}` : ""}`,
      nodeId,
    });
  }
  if (payload.session) items.push(sessionTimelineItem(payload.session, true));
  (payload.history || []).forEach((session) => {
    if (payload.session && session.session_id === payload.session.session_id) return;
    items.push(sessionTimelineItem(session, false));
  });
  if (!items.length) {
    timeline.innerHTML = '<div class="stream-stage idle"><span class="stream-dot"></span><span>No stream requests yet</span></div>';
    return;
  }
  timeline.innerHTML = items.slice(0, 5).map((item) => renderTimelineItem(nodeId, item)).join("");
  timeline.querySelectorAll("[data-burst-session]").forEach((button) => {
    button.addEventListener("click", () => {
      const session = findTimelineSession(button.dataset.burstSession, payload);
      if (session) playBurst(nodeId, session);
    });
  });
}

function sessionTimelineItem(session, isCurrent) {
  const status = session.status || "waiting_for_frames";
  const frameCount = session.frame_count || 0;
  const latest = formatTime(session.last_frame_at);
  const delivered = formatTime(session.command_returned_at);
  const ended = formatTime(session.ended_at);
  let label = "Command delivered";
  if (status === "active") label = "Capturing burst";
  if (status === "ended") label = "Burst ended";
  const meta = [
    `${frameCount} frame${frameCount === 1 ? "" : "s"}`,
    latest ? `latest ${latest}` : delivered ? `delivered ${delivered}` : "",
    ended ? `ended ${ended}` : "",
  ].filter(Boolean).join(" · ");
  return {
    state: isCurrent && status !== "ended" ? "active" : status,
    label,
    meta,
    progress: streamProgress(session),
    sessionId: session.session_id,
    playable: Boolean((session.frames || []).length || session.latest_frame?.url),
  };
}

function renderTimelineItem(nodeId, item) {
  const progress = item.progress == null
    ? ""
    : `<div class="stream-progress"><span style="width: ${Math.max(0, Math.min(100, item.progress))}%"></span></div>`;
  const playable = item.playable
    ? `<button class="burst-play" type="button" data-burst-session="${escapeAttr(item.sessionId)}">Play</button>`
    : "";
  return `
    <div class="stream-stage ${escapeAttr(item.state)}">
      <span class="stream-dot"></span>
      <div>
        <div class="stream-stage-title">
          <strong>${escapeHtml(item.label)}</strong>
          ${playable}
        </div>
        <span>${escapeHtml(item.meta || "")}</span>
        ${progress}
      </div>
    </div>
  `;
}

function findTimelineSession(sessionId, payload) {
  const sessions = [payload.session, ...(payload.history || [])].filter(Boolean);
  return sessions.find((session) => session.session_id === sessionId);
}

function playBurst(nodeId, session) {
  const frames = session.frames?.length ? session.frames : session.latest_frame ? [session.latest_frame] : [];
  const preview = document.querySelector(`[data-stream-preview="${cssEscape(nodeId)}"]`);
  const status = document.querySelector(`[data-command-status="${cssEscape(nodeId)}"]`);
  if (!preview || !frames.length) return;
  const existing = burstPlayers.get(nodeId);
  if (existing) window.clearInterval(existing);
  let index = 0;
  preview.hidden = false;
  preview.title = `Playing ${frames.length} saved burst frame${frames.length === 1 ? "" : "s"}`;
  setStatus(status, `playing burst · 1 / ${frames.length}`);
  preview.src = `${frames[0].url}?t=${encodeURIComponent(frames[0].received_at || Date.now())}`;
  if (frames.length === 1) return;
  const timer = window.setInterval(() => {
    index += 1;
    if (index >= frames.length) {
      window.clearInterval(timer);
      burstPlayers.delete(nodeId);
      setStatus(status, `burst playback ended · ${frames.length} frames`);
      return;
    }
    const frame = frames[index];
    preview.src = `${frame.url}?t=${encodeURIComponent(frame.received_at || Date.now())}`;
    setStatus(status, `playing burst · ${index + 1} / ${frames.length}`);
  }, 220);
  burstPlayers.set(nodeId, timer);
}

function streamProgress(session) {
  if (!session.command_returned_at || !session.expected_until) return null;
  const start = Date.parse(session.command_returned_at);
  const end = Date.parse(session.expected_until);
  const now = Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  if (session.status === "ended") return 100;
  return ((now - start) / (end - start)) * 100;
}

function formatTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function escapeAttr(value) {
  return String(value).replace(/[^a-z0-9_-]/gi, "");
}

function cssEscape(value) {
  if (window.CSS?.escape) return CSS.escape(value);
  return String(value).replace(/["\\]/g, "\\$&");
}
