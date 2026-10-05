import { useState } from "react";
import { api, tokens } from "../api";

export default function AuthForm({ onAuthed }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      if (mode === "register") await api.register(username, email, password);
      const data = await api.login(username, password);
      tokens.save(data.access, data.refresh);
      onAuthed();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="shell narrow">
      <h1>DevPilot AI</h1>
      <form className="card" onSubmit={submit}>
        <h2>{mode === "login" ? "Log in" : "Create account"}</h2>
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" required />
        {mode === "register" && (
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" type="email" />
        )}
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" type="password" required />
        {error && <p className="error">{error}</p>}
        <div className="actions">
          <button className="primary" disabled={loading}>
            {loading ? "Please wait…" : mode === "login" ? "Log in" : "Register"}
          </button>
          <button type="button" className="link" onClick={() => setMode(mode === "login" ? "register" : "login")}>
            {mode === "login" ? "Create account" : "Back to login"}
          </button>
        </div>
      </form>
    </main>
  );
}
