const STEPS = ["Upload", "Mask", "Methods", "Results"];

export default function Stepper({ step, reached, onGo }) {
  return (
    <ol className="stepper" aria-label="Progress">
      {STEPS.map((label, i) => {
        const n = i + 1;
        const state = n === step ? "current" : reached[n] ? "done" : "todo";
        return (
          <li key={label} className={state}>
            <button disabled={!reached[n] || n === step} onClick={() => onGo(n)}
                    aria-current={n === step ? "step" : undefined}>
              <span className="num">{n}</span>{label}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
