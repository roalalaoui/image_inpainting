import { useState } from "react";
import JSZip from "jszip";
import CompareSlider from "./CompareSlider.jsx";

const STATUS = {
  success: { label: "Restored", cls: "ok" },
  error: { label: "Failed", cls: "err" },
  unavailable: { label: "Unavailable", cls: "off" },
};

const dataUrl = (r) => `data:image/png;base64,${r.image_base64}`;

function save(href, filename) {
  const a = document.createElement("a");
  a.href = href; a.download = filename; a.click();
}

export default function ResultsStep({ image, outcome, onBack, onRestart }) {
  const base = image.name.replace(/\.[^.]+$/, "");
  const good = outcome.results.filter((r) => r.status === "success");
  const [compareKey, setCompareKey] = useState(good[0]?.algorithm);
  const compared = good.find((r) => r.algorithm === compareKey) || good[0];

  async function downloadZip() {
    const zip = new JSZip();
    good.forEach((r) => zip.file(`${base}_${r.algorithm}.png`, r.image_base64, { base64: true }));
    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    save(url, `${base}_inpainting_results.zip`);
    URL.revokeObjectURL(url);
  }

  return (
    <section className="panel">
      <h2>Results</h2>

      <table className="status">
        <thead><tr><th>Method</th><th>Status</th><th>Time</th><th>Details</th></tr></thead>
        <tbody>
          {outcome.results.map((r) => (
            <tr key={r.algorithm}>
              <td>{r.name}</td>
              <td><span className={`badge ${STATUS[r.status].cls}`}>{STATUS[r.status].label}</span></td>
              <td>{r.time_ms != null ? `${(r.time_ms / 1000).toFixed(2)} s` : "-"}</td>
              <td className="detail">{r.message || ""}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {good.length === 0 ? (
        <p className="empty">Nothing was restored. Fix the problems listed above, or connect the algorithms in backend/algorithms, then run again.</p>
      ) : (
        <>
          <div className="gallery">
            <figure className="print"><img src={image.url} alt="Original" /><figcaption>Original</figcaption></figure>
            {good.map((r) => (
              <figure className="print" key={r.algorithm}>
                <img src={dataUrl(r)} alt={`Restored by ${r.name}`} />
                <figcaption>
                  <strong>{r.name}</strong> in {(r.time_ms / 1000).toFixed(2)} s
                  <button className="link" onClick={() => save(dataUrl(r), `${base}_${r.algorithm}.png`)}>Download</button>
                </figcaption>
              </figure>
            ))}
          </div>

          <h3>Before and after</h3>
          <div className="seg wrap" role="group" aria-label="Method to compare">
            {good.map((r) => (
              <button key={r.algorithm} className={compared.algorithm === r.algorithm ? "on" : ""}
                      onClick={() => setCompareKey(r.algorithm)}>{r.name}</button>
            ))}
          </div>
          <CompareSlider before={image.url} after={dataUrl(compared)} afterLabel={compared.name} />

          <div className="nav start">
            <button className="btn primary" onClick={downloadZip}>Download all as ZIP</button>
          </div>
        </>
      )}

      <div className="nav">
        <button className="btn ghost" onClick={onBack}>Change methods</button>
        <button className="btn ghost" onClick={onRestart}>Start with a new image</button>
      </div>
    </section>
  );
}
