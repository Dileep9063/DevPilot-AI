import { diffLines } from "../diff";

const CONTEXT = 3;

export default function DiffView({ original, proposed }) {
  const lines = diffLines(original || "", proposed || "");
  const added = lines.filter((l) => l.type === "add").length;
  const removed = lines.filter((l) => l.type === "del").length;

  // Collapse long unchanged stretches, keeping CONTEXT lines around changes.
  const keep = new Array(lines.length).fill(false);
  lines.forEach((l, i) => {
    if (l.type !== "same") {
      for (let k = Math.max(0, i - CONTEXT); k <= Math.min(lines.length - 1, i + CONTEXT); k++) {
        keep[k] = true;
      }
    }
  });

  const rows = [];
  let skipped = 0;
  lines.forEach((l, i) => {
    if (keep[i] || lines.length <= 40) {
      if (skipped) {
        rows.push({ type: "gap", text: `… ${skipped} unchanged lines …` });
        skipped = 0;
      }
      rows.push(l);
    } else {
      skipped++;
    }
  });
  if (skipped) rows.push({ type: "gap", text: `… ${skipped} unchanged lines …` });

  return (
    <div className="diff">
      <div className="diff-stats">
        <span className="add">+{added}</span> <span className="del">−{removed}</span>
        {!original && <span className="tag">new file</span>}
      </div>
      <pre>
        {rows.map((r, i) => (
          <div key={i} className={`line ${r.type}`}>
            <span className="sign">
              {r.type === "add" ? "+" : r.type === "del" ? "−" : " "}
            </span>
            {r.text || " "}
          </div>
        ))}
      </pre>
    </div>
  );
}
