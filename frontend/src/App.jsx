import { useCallback, useEffect, useState } from "react";
import { fetchAlgorithms, runInpaint } from "./api.js";
import Stepper from "./components/Stepper.jsx";
import UploadStep from "./components/UploadStep.jsx";
import MaskStep from "./components/MaskStep.jsx";
import AlgorithmStep from "./components/AlgorithmStep.jsx";
import ResultsStep from "./components/ResultsStep.jsx";

export default function App() {
  const [step, setStep] = useState(1);
  const [image, setImage] = useState(null);   // { file, url, width, height, name, sampleMaskUrl }
  const [strokes, setStrokes] = useState([]); // mask drawing history, kept when moving between steps
  const [redo, setRedo] = useState([]);
  const [mask, setMask] = useState(null);     // { blob, url, pixels }
  const [algos, setAlgos] = useState([]);
  const [backendError, setBackendError] = useState(null);
  const [selected, setSelected] = useState([]);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [outcome, setOutcome] = useState(null);

  const loadAlgos = useCallback(() => {
    fetchAlgorithms().then((a) => { setAlgos(a); setBackendError(null); }).catch((e) => setBackendError(e.message));
  }, []);
  useEffect(loadAlgos, [loadAlgos]);

  function handleImage(img) {
    setImage(img); setStrokes([]); setRedo([]); setMask(null); setOutcome(null); setRunError(null);
    setStep(2);
  }

  function handleMask(m) { setMask(m); setOutcome(null); setStep(3); }

  async function run() {
    setRunning(true); setRunError(null);
    const keys = algos.map((a) => a.key).filter((k) => selected.includes(k)); // registry order
    try {
      setOutcome(await runInpaint(image.file, mask.blob, keys));
      setStep(4);
    } catch (e) {
      setRunError(e.message);
    } finally {
      setRunning(false);
    }
  }

  const reached = { 1: true, 2: !!image, 3: !!mask, 4: !!outcome };

  return (
    <div className="app">
      <header className="masthead">
        <h1>Image Inpainting Lab</h1>
        <p>A darkroom for mending damaged photographs, with five methods on the bench.</p>
      </header>
      <Stepper step={step} reached={reached} onGo={setStep} />
      <main>
        {step === 1 && <UploadStep onImage={handleImage} current={image} onContinue={() => setStep(2)} />}
        {step === 2 && image && (
          <MaskStep image={image} strokes={strokes} setStrokes={setStrokes} redo={redo} setRedo={setRedo}
                    onBack={() => setStep(1)} onNext={handleMask} />
        )}
        {step === 3 && image && mask && (
          <AlgorithmStep algos={algos} backendError={backendError} onRetry={loadAlgos}
                         image={image} mask={mask} selected={selected} setSelected={setSelected}
                         running={running} runError={runError}
                         onBack={() => setStep(2)} onRun={run} />
        )}
        {step === 4 && image && outcome && (
          <ResultsStep image={image} outcome={outcome} onBack={() => setStep(3)}
                       onRestart={() => { setImage(null); setMask(null); setOutcome(null); setStep(1); }} />
        )}
      </main>
    </div>
  );
}
