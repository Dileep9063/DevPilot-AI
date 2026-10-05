// Minimal line diff (LCS). Fine for source files up to a few thousand lines.
export function diffLines(oldText, newText) {
  const a = oldText === "" ? [] : oldText.split("\n");
  const b = newText === "" ? [] : newText.split("\n");

  if (a.length * b.length > 4_000_000) {
    return [
      ...a.map((t) => ({ type: "del", text: t })),
      ...b.map((t) => ({ type: "add", text: t })),
    ];
  }

  const dp = Array.from({ length: a.length + 1 }, () =>
    new Uint32Array(b.length + 1)
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      dp[i][j] =
        a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const out = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      out.push({ type: "same", text: a[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ type: "del", text: a[i++] });
    } else {
      out.push({ type: "add", text: b[j++] });
    }
  }
  while (i < a.length) out.push({ type: "del", text: a[i++] });
  while (j < b.length) out.push({ type: "add", text: b[j++] });
  return out;
}
