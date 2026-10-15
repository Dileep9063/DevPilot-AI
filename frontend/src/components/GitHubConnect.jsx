import { useEffect, useState } from "react";
import { api } from "../api";

// GitHub sends the browser back to the app with ?github=connected|error.
function readNotice() {
  const params = new URLSearchParams(window.location.search);
  const flag = params.get("github");
  if (!flag) return null;
  return flag === "connected"
    ? { kind: "ok", text: "GitHub connected. You can now run tasks on the repositories you shared." }
    : { kind: "error", text: params.get("reason") || "Could not connect GitHub." };
}

export default function GitHubConnect() {
  const [status, setStatus] = useState(null);
  const [notice] = useState(readNotice);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Remove ?github=... from the address bar once it has been read.
    if (new URLSearchParams(window.location.search).has("github")) {
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .githubStatus()
      .then((data) => {
        if (!cancelled) setStatus(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const connect = async () => {
    // Repository access updates do not trigger a redirect when GitHub App
    // OAuth-on-install is enabled and no Setup URL can be configured.
    // Keep DevPilot open in its current tab while the user edits access.
    const updatingExistingConnection = Boolean(status?.connected);
    const popup = updatingExistingConnection ? window.open("about:blank", "_blank") : null;
    setBusy(true);
    setError("");
    try {
      const { url } = await api.githubConnect();
      if (updatingExistingConnection && popup) {
        popup.opener = null;
        popup.location.href = url;
      } else {
        if (updatingExistingConnection && !popup) {
          setError("Your browser blocked the new tab. Allow pop-ups for DevPilot and try again.");
          setBusy(false);
          return;
        }
        window.location.assign(url);
      }
    } catch (err) {
      if (popup && !popup.closed) popup.close();
      setError(err.message);
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    setError("");
    try {
      await api.githubDisconnect();
      setStatus(await api.githubStatus());
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const banner = notice && <p className={`notice ${notice.kind}`}>{notice.text}</p>;

  // Server runs in single-token mode (GITHUB_TOKEN): nothing to connect.
  if (!status || !status.app_configured) return banner || null;

  if (!status.connected) {
    return (
      <section className="card">
        {banner}
        <h2>Connect GitHub</h2>
        <p className="muted">
          Choose which repositories DevPilot may open pull requests on. You can change this at any time.
        </p>
        {error && <p className="error">{error}</p>}
        <div className="actions">
          <button className="primary" onClick={connect} disabled={busy}>
            {busy ? "Opening GitHub…" : "Connect GitHub"}
          </button>
        </div>
      </section>
    );
  }

  const accounts = status.installations
    .map((i) => `${i.account || "account"}${i.repository_selection === "selected" ? " (selected repos)" : ""}`)
    .join(", ");

  return (
    <section className="card">
      {banner}
      <h2>
        GitHub connected <span className="badge completed">{accounts}</span>
      </h2>
      <p className="muted">
        Repository settings open in a new tab. When you finish changing access on GitHub, return to this tab to continue.
      </p>
      {error && <p className="error">{error}</p>}
      <div className="actions">
        <button onClick={connect} disabled={busy}>{busy ? "Opening GitHub…" : "Add or change repositories"}</button>
        <button className="link" onClick={disconnect} disabled={busy}>Disconnect</button>
      </div>
    </section>
  );
}
