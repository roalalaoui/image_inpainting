import { useState } from "react";

// Drag the divider: the left side shows the original, the right side the restored image.
export default function CompareSlider({ before, after, afterLabel }) {
  const [pos, setPos] = useState(50);
  return (
    <div className="compare">
      <img src={before} alt="Original" draggable="false" />
      <img src={after} alt={`Restored by ${afterLabel}`} draggable="false" style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
      <span className="tag left">Before</span>
      <span className="tag right">After</span>
      <span className="divider" style={{ left: `${pos}%` }} />
      <input type="range" min="0" max="100" step="0.1" value={pos} onChange={(e) => setPos(+e.target.value)}
             aria-label="Before and after position" />
    </div>
  );
}
