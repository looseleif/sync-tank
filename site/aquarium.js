(() => {
  "use strict";
  const canvas = document.querySelector("#fish-tank");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const feed = document.querySelector("#feed");
  const pause = document.querySelector("#pause");
  const status = document.querySelector("#tank-status");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let paused = reducedMotion.matches;
  let width = 440;
  let last = 0;
  let frame = 0;
  let lastMeal = 0;
  const food = [];
  const fish = Array.from({ length: 6 }, (_, i) => ({ x: 35 + i * 68, y: 19 + (i % 3) * 18, direction: i % 2 ? -1 : 1, speed: 9 + i * 2, color: ["#ffcd6d", "#f4a9bc", "#b1d985"][i % 3] }));
  const pixels = ["00001111000", "10011111100", "11111111110", "10011111100", "00001111000"];

  function draw() {
    ctx.fillStyle = "#081b1b";
    ctx.fillRect(0, 0, width, 75);
    ctx.fillStyle = "#526455";
    ctx.fillRect(0, 71, width, 4);
    for (let x = 10; x < width; x += 53) {
      ctx.fillStyle = "#458f70";
      ctx.fillRect(x, 53, 2, 18);
      ctx.fillRect(x - 4, 57, 4, 2);
      ctx.fillRect(x + 2, 62, 5, 2);
      ctx.fillStyle = "#304840";
      ctx.fillRect(x + 25, 68, 9, 3);
    }
    for (const f of fish) {
      ctx.save();
      ctx.translate(Math.round(f.x), Math.round(f.y));
      ctx.scale(f.direction, 1);
      ctx.fillStyle = f.color;
      pixels.forEach((row, y) => [...row].forEach((pixel, x) => { if (pixel === "1") ctx.fillRect(x - 5, y - 2, 1, 1); }));
      ctx.fillStyle = "#081b1b";
      ctx.fillRect(3, -1, 1, 1);
      ctx.restore();
    }
    ctx.fillStyle = "#f8df95";
    for (const bite of food) ctx.fillRect(Math.round(bite.x), Math.round(bite.y), 1, 1);
  }

  function tick(now) {
    frame = 0;
    if (paused || document.hidden) return;
    if (now - last >= 1000 / 30) {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      for (const f of fish) {
        const target = food[0];
        if (target) {
          f.direction = target.x >= f.x ? 1 : -1;
          f.y += Math.sign(target.y - f.y) * Math.min(Math.abs(target.y - f.y), dt * 16);
        }
        f.x += f.direction * f.speed * dt;
        if (f.x > width - 8) { f.x = width - 8; f.direction = -1; }
        if (f.x < 8) { f.x = 8; f.direction = 1; }
      }
      for (let i = food.length - 1; i >= 0; i--) {
        food[i].y += dt * 7;
        if (food[i].y > 69 || fish.some(f => Math.hypot(f.x - food[i].x, f.y - food[i].y) < 5)) food.splice(i, 1);
      }
      if (!food.length && lastMeal) { status.textContent = "Snack time is over. Back to exploring."; lastMeal = 0; }
      draw();
    }
    frame = requestAnimationFrame(tick);
  }

  function resume() {
    if (!paused && !document.hidden && !frame) { last = performance.now(); frame = requestAnimationFrame(tick); }
  }

  function syncControls() {
    pause.textContent = paused ? "Play" : "Pause";
    pause.setAttribute("aria-pressed", String(paused));
    feed.disabled = paused;
    status.textContent = paused ? "Aquarium paused." : "Just swimming along.";
  }

  function addFood(x) {
    if (paused || food.length >= 18) return;
    for (let i = 0; i < 6; i++) food.push({ x: Math.max(8, Math.min(width - 8, x + (i - 3) * 3)), y: 4 + i });
    lastMeal = performance.now();
    status.textContent = "A little snack for the pixel residents.";
  }

  new ResizeObserver(() => {
    const oldWidth = width;
    width = Math.max(120, Math.floor(canvas.clientWidth / 2));
    canvas.width = width;
    canvas.height = 75;
    fish.forEach(f => { f.x = Math.max(8, Math.min(width - 8, f.x * width / oldWidth)); });
    food.forEach(bite => { bite.x *= width / oldWidth; });
    draw();
  }).observe(canvas);
  feed.addEventListener("click", () => addFood(width / 2));
  canvas.addEventListener("click", event => addFood((event.clientX - canvas.getBoundingClientRect().left) / canvas.clientWidth * width));
  pause.addEventListener("click", () => {
    paused = !paused;
    if (paused) { cancelAnimationFrame(frame); frame = 0; }
    syncControls();
    resume();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { cancelAnimationFrame(frame); frame = 0; } else resume();
  });
  reducedMotion.addEventListener("change", event => {
    if (event.matches) { paused = true; cancelAnimationFrame(frame); frame = 0; syncControls(); }
  });
  pause.disabled = false;
  syncControls();
  draw();
  resume();
})();
