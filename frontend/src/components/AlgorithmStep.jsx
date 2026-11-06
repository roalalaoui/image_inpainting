export default function AlgorithmStep({ algos, backendError, onRetry, image, mask, selected, setSelected,
                                        running, runError, onBack, onRun }) {
  const toggle = (key) =>
    setSelected(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);
  const hasUnavailable = algos.some((a) => selected.includes(a.key) && !a.available);

  return (
    <section className="panel">
      <h2>Choose the methods to run</h2>

      {backendError && (
        <div className="alert" role="alert">
          <p>{backendError}</p>
          <button className="btn ghost" onClick={onRetry}>Try again</button>
        </div>
      )}

      <div className="inputs">
        <figure className="print small"><img src={image.url} alt="Original" /><figcaption>Image</figcaption></figure>
        <figure className="print small"><img src={mask.url} alt="Mask" /><figcaption>Mask, {mask.pixels.toLocaleString()} px</figcaption></figure>
        <p className="hint">Every method receives exactly this image and this mask.</p>
      </div>

      <div className="toolbar">
        <button className="btn ghost" onClick={() => setSelected(algos.map((a) => a.key))} disabled={!algos.length}>Select all</button>
        <button className="btn ghost" onClick={() => setSelected([])} disabled={!selected.length}>Clear selection</button>
      </div>

      <div className="cards">
        {algos.map((a) => {
          const on = selected.includes(a.key);
          return (
            <label key={a.key} className={`card ${on ? "on" : ""}`}>
              <input type="checkbox" checked={on} onChange={() => toggle(a.key)} />
              <span className="card-title">{a.name}</span>
              <span className="card-cat">{a.category}</span>
              <span className="card-desc">{a.description}</span>
              <span className={`badge ${a.available ? "ok" : "off"}`}>
                {a.available ? "Connected" : "Unavailable: not connected"}
              </span>
            </label>
          );
        })}
      </div>

      {hasUnavailable && (
        <p className="hint">Methods that are not connected will be reported as unavailable. They produce no image.</p>
      )}
      {runError && <p className="alert" role="alert">{runError}</p>}

      <div className="nav">
        <button className="btn ghost" onClick={onBack} disabled={running}>Back</button>
        <button className="btn primary" onClick={onRun} disabled={!selected.length || running}>
          {running ? "Restoring, please wait..." : `Run ${selected.length || ""} ${selected.length === 1 ? "method" : "methods"}`}
        </button>
      </div>
      {running && <p className="hint">Methods run one after another on the server. Slower ones, such as K-SVD or LaMa, can take a while.</p>}
    </section>
  );
}
