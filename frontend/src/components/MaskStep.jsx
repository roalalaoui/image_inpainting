import { useEffect, useRef, useState } from "react";

const COLOR = "#b5532f";

// Draws one history item onto the overlay canvas. The history is replayed for undo/redo.
function paint(ctx, s) {
  if (s.type === "clear") { ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); return; }
  if (s.type === "image") { ctx.globalCompositeOperation = "source-over"; ctx.drawImage(s.canvas, 0, 0); return; }
  ctx.globalCompositeOperation = s.tool === "eraser" ? "destination-out" : "source-over";
  ctx.strokeStyle = COLOR; ctx.lineWidth = s.size; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(s.points[0][0], s.points[0][1]);
  if (s.points.length === 1) ctx.lineTo(s.points[0][0] + 0.01, s.points[0][1]);
  for (const p of s.points.slice(1)) ctx.lineTo(p[0], p[1]);
  ctx.stroke();
}

function maskImageToOverlay(url, w, h) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      const d = ctx.getImageData(0, 0, w, h);
      for (let i = 0; i < w * h; i++) {
        const on = d.data[i * 4] > 127;
        d.data[i * 4] = 181; d.data[i * 4 + 1] = 83; d.data[i * 4 + 2] = 47; d.data[i * 4 + 3] = on ? 255 : 0;
      }
      ctx.putImageData(d, 0, 0);
      resolve(c);
    };
    img.onerror = () => reject(new Error("The sample mask could not be loaded."));
    img.src = url;
  });
}

// Converts the overlay into the mask sent to the backend: white = damaged, black = valid.
function exportMask(canvas) {
  const { width: w, height: h } = canvas;
  const src = canvas.getContext("2d").getImageData(0, 0, w, h).data;
  const out = document.createElement("canvas"); out.width = w; out.height = h;
  const octx = out.getContext("2d");
  const od = octx.createImageData(w, h);
  let white = 0;
  for (let i = 0; i < w * h; i++) {
    const on = src[i * 4 + 3] > 127;
    const v = on ? 255 : 0;
    od.data[i * 4] = od.data[i * 4 + 1] = od.data[i * 4 + 2] = v; od.data[i * 4 + 3] = 255;
    if (on) white++;
  }
  octx.putImageData(od, 0, 0);
  return new Promise((resolve) => out.toBlob((blob) => resolve({ blob, white }), "image/png"));
}

export default function MaskStep({ image, strokes, setStrokes, redo, setRedo, onBack, onNext }) {
  const { width, height } = image;
  const canvasRef = useRef(null);
  const current = useRef(null);
  const [tool, setTool] = useState("brush");
  const [size, setSize] = useState(() => Math.max(6, Math.round(Math.max(width, height) / 40)));
  const [ring, setRing] = useState(null);
  const [error, setError] = useState(null);

  // Replay history whenever it changes (initial draw, undo, redo, clear).
  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.clearRect(0, 0, width, height);
    strokes.forEach((s) => paint(ctx, s));
  }, [strokes, width, height]);

  function pos(e) {
    const r = canvasRef.current.getBoundingClientRect();
    return [(e.clientX - r.left) * (width / r.width), (e.clientY - r.top) * (height / r.height)];
  }

  function down(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    const s = { type: "path", tool, size, points: [pos(e)] };
    current.current = s;
    paint(canvasRef.current.getContext("2d"), s);
  }

  function move(e) {
    const r = canvasRef.current.getBoundingClientRect();
    setRing({ x: e.clientX - r.left, y: e.clientY - r.top });
    const s = current.current;
    if (!s) return;
    const p = pos(e);
    paint(canvasRef.current.getContext("2d"), { ...s, points: [s.points[s.points.length - 1], p] });
    s.points.push(p);
  }

  function up() {
    const s = current.current;
    current.current = null;
    if (s) { setStrokes([...strokes, s]); setRedo([]); }
  }

  const undo = () => { setRedo([...redo, strokes[strokes.length - 1]]); setStrokes(strokes.slice(0, -1)); };
  const redoOne = () => { setStrokes([...strokes, redo[redo.length - 1]]); setRedo(redo.slice(0, -1)); };
  const clear = () => { setStrokes([...strokes, { type: "clear" }]); setRedo([]); };

  async function loadSampleMask() {
    try {
      const canvas = await maskImageToOverlay(image.sampleMaskUrl, width, height);
      setStrokes([...strokes, { type: "image", canvas }]); setRedo([]); setError(null);
    } catch (e) { setError(e.message); }
  }

  async function next() {
    const { blob, white } = await exportMask(canvasRef.current);
    if (white === 0) return setError("Paint over the damaged area first. The mask is empty.");
    if (white === width * height) return setError("The mask covers the whole image. Leave some of it unpainted.");
    setError(null);
    onNext({ blob, url: URL.createObjectURL(blob), pixels: white });
  }

  const cssScale = canvasRef.current ? canvasRef.current.getBoundingClientRect().width / width : 1;

  return (
    <section className="panel">
      <h2>Paint over the damaged area</h2>
      <div className="mask-layout">
        <div className="stage" style={{ aspectRatio: `${width} / ${height}`, maxWidth: `min(100%, ${width}px)` }}>
          <img src={image.url} alt="Image to mask" draggable="false" />
          <canvas ref={canvasRef} width={width} height={height}
                  onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}
                  onPointerLeave={() => setRing(null)} aria-label="Mask drawing area" />
          {ring && <span className="ring" style={{ left: ring.x, top: ring.y, width: size * cssScale, height: size * cssScale }} />}
        </div>

        <aside className="tools">
          <div className="seg" role="group" aria-label="Tool">
            <button className={tool === "brush" ? "on" : ""} onClick={() => setTool("brush")}>Brush</button>
            <button className={tool === "eraser" ? "on" : ""} onClick={() => setTool("eraser")}>Eraser</button>
          </div>
          <label className="slider">Brush size: {size} px
            <input type="range" min="2" max={Math.max(20, Math.round(Math.max(width, height) / 4))}
                   value={size} onChange={(e) => setSize(+e.target.value)} />
          </label>
          <div className="row">
            <button className="btn ghost" onClick={undo} disabled={!strokes.length}>Undo</button>
            <button className="btn ghost" onClick={redoOne} disabled={!redo.length}>Redo</button>
            <button className="btn ghost" onClick={clear} disabled={!strokes.length}>Clear</button>
          </div>
          {image.sampleMaskUrl && <button className="btn ghost" onClick={loadSampleMask}>Use the sample mask</button>}
          <p className="hint">The terracotta area is saved as white (damaged). Everything else is black (valid).</p>
        </aside>
      </div>
      {error && <p className="alert" role="alert">{error}</p>}
      <div className="nav">
        <button className="btn ghost" onClick={onBack}>Back</button>
        <button className="btn primary" onClick={next}>Continue to methods</button>
      </div>
    </section>
  );
}
