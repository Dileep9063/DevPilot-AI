import "./App.css";
import { useCallback, useEffect, useState } from "react";
import { api, tokens } from "./api";
import AuthForm from "./components/AuthForm";
import ApprovalCard from "./components/ApprovalCard";
import ResultCard from "./components/ResultCard";
import HistoryList from "./components/HistoryList";
import GitHubConnect from "./components/GitHubConnect";

export default function App() {
  const [authed, setAuthed] = useState(() => !!tokens.access());
  const [repoUrl, setRepoUrl] = useState("");
  const [repoOptions, setRepoOptions] = useState([]);
  const [githubAppMode, setGithubAppMode] = useState(false);
  const [reposLoading, setReposLoading] = useState(false);
  const [repoLoadError, setRepoLoadError] = useState("");
  const [task, setTask] = useState("");
  const [threadId, setThreadId] = useState("");
  const [approval, setApproval] = useState(null); // pending approval_request
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState([]);
  const activeStatus = result?.status === "queued" || result?.status === "running";

  const loadRuns = useCallback(async () => {
    try {
      setRuns((await api.runs()).runs);
    } catch {
      /* history is non-critical; ignore */
    }
  }, []);

  // Load repositories already selected in the GitHub App installation.
  // Selecting one here supplies its clone URL; users do not paste it again.
  useEffect(() => {
    if (!authed) return undefined;
    let cancelled = false;
    const loadRepositories = async () => {
      try {
        const status = await api.githubStatus();
        if (cancelled) return;
        const appMode = Boolean(status.app_configured);
        setGithubAppMode(appMode);
        if (!appMode || !status.connected) {
          setRepoOptions([]);
          setRepoUrl("");
          return;
        }
        setReposLoading(true);
        setRepoLoadError("");
        const data = await api.githubRepos();
        if (cancelled) return;
        const options = data.repositories || [];
        setRepoOptions(options);
        setRepoUrl((current) =>
          options.some((item) => item.clone_url === current)
            ? current
            : (options[0]?.clone_url || "")
        );
      } catch (err) {
        if (!cancelled) setRepoLoadError(err.message);
      } finally {
        if (!cancelled) setReposLoading(false);
      }
    };
    loadRepositories();
    return () => { cancelled = true; };
  }, [authed]);

  // Poll persisted run state while a background worker processes the task.
  useEffect(() => {
    if (!threadId || !activeStatus || approval) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const detail = await api.runDetail(threadId);
        if (cancelled) return;
        if (detail.status === "waiting_for_approval" && detail.approval_request) {
          setApproval(detail.approval_request);
          setResult(null);
          loadRuns();
        } else if (detail.status !== "queued" && detail.status !== "running") {
          setApproval(null);
          setResult({ status: detail.status, pr_url: detail.pr_url, final_result: detail.summary });
          loadRuns();
        }
      } catch {
        // Keep the active run visible during transient polling failures.
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [threadId, activeStatus, approval, loadRuns]);

  useEffect(() => {
    if (!authed) return undefined;
    let cancelled = false;
    api
      .runs()
      .then((data) => {
        if (!cancelled) setRuns(data.runs);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [authed]);

  const logout = () => {
    tokens.clear();
    setAuthed(false);
    reset();
  };

  const reset = () => {
    setThreadId("");
    setApproval(null);
    setResult(null);
  };

  const handle = (data) => {
    loadRuns();
    if (data.thread_id) setThreadId(data.thread_id);
    if (data.status === "waiting_for_approval") {
      setApproval(data.approval_request);
      setResult(null);
    } else {
      setApproval(null);
      setResult(data);
    }
  };

  const guard = async (fn) => {
    setLoading(true);
    try {
      handle(await fn());
    } catch (err) {
      if (!tokens.access()) setAuthed(false);
      setApproval(null);
      setResult({ status: "error", error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const run = (e) => {
    e.preventDefault();
    setResult(null);
    setApproval(null);
    guard(() => api.runAgent(repoUrl.trim(), task.trim()));
  };

  const decide = (decision) => guard(() => api.resume(threadId, decision));

  // Re-open a past run: paused runs get their approval card back.
  const openRun = async (run) => {
    setLoading(true);
    try {
      const detail = await api.runDetail(run.thread_id);
      setThreadId(detail.thread_id);

      if (detail.status === "waiting_for_approval" && detail.approval_request) {
        setResult(null);
        setApproval(detail.approval_request);
      } else {
        setApproval(null);
        setResult({
          status: detail.status,
          pr_url: detail.pr_url,
          final_result: detail.summary,
        });
      }
    } catch (err) {
      setApproval(null);
      setResult({ status: "error", error: err.message });
    } finally {
      setLoading(false);
    }
  };

  if (!authed) return <AuthForm onAuthed={() => setAuthed(true)} />;

  const busy = loading || activeStatus;
  const waiting = !!approval;

  return (
    <main className="shell">
      <header>
        <h1>DevPilot AI</h1>
        <button className="link" onClick={logout}>Log out</button>
      </header>

      <GitHubConnect />

      <form className="card" onSubmit={run}>
        <h2>New task</h2>
        {githubAppMode ? (
          <>
            <label htmlFor="repository-select">GitHub repository</label>
            <select
              id="repository-select"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              disabled={busy || waiting || reposLoading || repoOptions.length === 0}
              required
            >
              {repoOptions.length === 0 && <option value="">No repositories available</option>}
              {repoOptions.map((repo) => (
                <option key={repo.full_name} value={repo.clone_url}>
                  {repo.full_name}{repo.private ? " (private)" : ""}
                </option>
              ))}
            </select>
            {reposLoading && <p className="muted">Loading your GitHub repositories…</p>}
            {repoLoadError && <p className="error">{repoLoadError}</p>}
            {!reposLoading && !repoLoadError && repoOptions.length === 0 && (
              <p className="muted">No repositories are available. Use “Add or change repositories” above to grant DevPilot access.</p>
            )}
          </>
        ) : (
          <input
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            placeholder="https://github.com/owner/repository"
            disabled={busy || waiting}
            required
          />
        )}
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Describe the change, e.g. “Add a logout() function to auth.py”"
          rows={5}
          disabled={busy || waiting}
          required
        />
        <div className="actions">
          <button className="primary" disabled={busy || waiting || !repoUrl.trim() || !task.trim()}>
            {activeStatus ? "Agent working…" : loading ? "Submitting…" : "Run agent"}
          </button>
          {(result || waiting) && !busy && (
            <button type="button" className="link" onClick={reset}>Start over</button>
          )}
        </div>
        {threadId && <p className="muted">Run: <code>{threadId}</code></p>}
      </form>

      {busy && <div className="card"><p className="muted">{activeStatus ? "Agent queued or running — checking progress automatically…" : "Submitting your task…"}</p></div>}
      {approval && !busy && <ApprovalCard request={approval} loading={busy} onDecide={decide} />}
      {result && !busy && <ResultCard result={result} />}

      <HistoryList
        runs={runs}
        activeId={threadId}
        disabled={busy}
        onOpen={openRun}
        onRefresh={loadRuns}
      />
    </main>
  );
}
