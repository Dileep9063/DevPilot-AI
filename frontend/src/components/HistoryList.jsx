const LABELS = {
  running: "Running",
  waiting_for_approval: "Needs approval",
  completed: "Completed",
  rejected: "Rejected",
  cannot_complete: "Could not complete",
  tests_failed: "Tests failed",
  push_failed: "Push failed",
  pr_failed: "PR not created",
  unknown: "Older run",
  error: "Error",
};

function repoName(url) {
  return (url || "").replace(/^https?:\/\/github\.com\//, "").replace(/\.git$/, "");
}

export default function HistoryList({ runs, activeId, disabled, onOpen, onRefresh }) {
  return (
    <section className="card">
      <h2>
        Recent runs
        <button type="button" className="link small" onClick={onRefresh} disabled={disabled}>
          Refresh
        </button>
      </h2>

      {runs.length === 0 && <p className="muted">No runs yet.</p>}

      <ul className="history">
        {runs.map((run) => (
          <li key={run.thread_id}>
            <button
              type="button"
              className={`row ${run.thread_id === activeId ? "active" : ""}`}
              disabled={disabled}
              onClick={() => onOpen(run)}
            >
              <span className="task">{run.task}</span>
              <span className="meta">
                {repoName(run.repo_url)} · {new Date(run.created_at).toLocaleString()}
              </span>
              <span className={`badge ${run.status}`}>{LABELS[run.status] || run.status}</span>
            </button>
            {run.pr_url && (
              <a className="prlink" href={run.pr_url} target="_blank" rel="noreferrer">
                View pull request
              </a>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
