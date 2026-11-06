import { useRef, useState } from "react";

const MAX_BYTES = 25 * 1024 * 1024;
const MAX_PIXELS = 2048 * 2048;
const SAMPLES = [
  { label: "Landscape", image: "sample1_landscape.png", mask: "sample1_landscape_mask.png" },
  { label: "Tiled wall", image: "sample2_tiles.png", mask: "sample2_tiles_mask.png" },
];

function readDimensions(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("This file could not be read as an image."));
    img.src = url;
  });
}

export default function UploadStep({ onImage, current, onContinue }) {
  const [error, setError] = useState(null);
  const [over, setOver] = useState(false);
  const inputRef = useRef(null);

  async function accept(file, sampleMaskUrl = null) {
    setError(null);
    if (!file) return;
    if (!/^image\/(png|jpe?g)$/.test(file.type)) return setError("Use a PNG, JPG or JPEG image.");
    if (file.size > MAX_BYTES) return setError("The file is larger than 25 MB.");
    const url = URL.createObjectURL(file);
    try {
      const { width, height } = await readDimensions(url);
      if (width * height > MAX_PIXELS) {
        URL.revokeObjectURL(url);
        return setError(`The image is ${width}x${height}. Please use one up to about 2048x2048 pixels.`);
      }
      onImage({ file, url, width, height, name: file.name, sampleMaskUrl });
    } catch (e) {
      URL.revokeObjectURL(url);
      setError(e.message);
    }
  }

  async function loadSample(s) {
    try {
      const blob = await (await fetch(`/samples/${s.image}`)).blob();
      await accept(new File([blob], s.image, { type: "image/png" }), `/samples/${s.mask}`);
    } catch {
      setError("The sample could not be loaded.");
    }
  }

  return (
    <section className="panel">
      <h2>Choose a photograph</h2>
      <div className={`dropzone ${over ? "over" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }}
           onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); accept(e.dataTransfer.files[0]); }}>
        <p className="lead">Drop an image here</p>
        <p className="hint">PNG, JPG or JPEG, up to about 2048 x 2048 pixels</p>
        <button className="btn primary" onClick={() => inputRef.current.click()}>Browse files</button>
        <input ref={inputRef} type="file" accept="image/png,image/jpeg" hidden
               onChange={(e) => { accept(e.target.files[0]); e.target.value = ""; }} />
      </div>
      {error && <p className="alert" role="alert">{error}</p>}

      <div className="samples">
        <span>No image at hand? Try a sample with a ready-made mask:</span>
        {SAMPLES.map((s) => (
          <button key={s.label} className="btn ghost" onClick={() => loadSample(s)}>{s.label}</button>
        ))}
      </div>

      {current && (
        <div className="preview-row">
          <figure className="print"><img src={current.url} alt="Uploaded preview" />
            <figcaption>{current.name}, {current.width} x {current.height} px</figcaption></figure>
          <button className="btn primary" onClick={onContinue}>Continue to mask</button>
        </div>
      )}
    </section>
  );
}
