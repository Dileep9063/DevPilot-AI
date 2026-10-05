const LABELS = {
  completed: "Completed",
  rejected: "Rejected",
  cannot_complete: "Could not complete",
  tests_failed: "Tests failed — nothing pushed",
  push_failed: "Push failed — not on GitHub",
  pr_failed: "Pushed, but PR not created",
  error: "Error",
};

function Details({ title, children }) {
  if (!children) return null;
  return (
    <details>
      <summary>{title}</summary>
      <pre className="plain">{children}</pre>
    </details>
  );
}

export default function ResultCard({ result }) {
  const status = result.status || "error";
  const messages = (result.messages || [])
    .map((m) => `[${m.role}] ${m.content}`)
    .join("\n\n");

  return (
    <section className="card">
      <h2>
        Result <span className={`badge ${status}`}>{LABELS[status] || status}</span>
      </h2>

      {result.error && <p className="error">{result.error}</p>}

      {result.pr_url && (
        <p className="pr">
          Pull request:{" "}
          <a href={result.pr_url} target="_blank" rel="noreferrer">
            {result.pr_url}
          </a>
        </p>
      )}

      {result.final_result && <p className="summary">{result.final_result}</p>}

      <Details title="Test output">{result.test_result}</Details>
      <Details title="Debug analysis">{result.debug_result}</Details>
      <Details title="Branch">{result.branch_result}</Details>
      <Details title="Commit">{result.commit_result}</Details>
      <Details title="Push">{result.push_result}</Details>
      <Details title="Pull request">{result.pr_result}</Details>
      <Details title="Agent conversation">{messages}</Details>
    </section>
  );
}
