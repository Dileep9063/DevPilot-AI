import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [accessToken, setAccessToken] = useState(() => localStorage.getItem("devpilot_access_token") || "");

  const authenticate = async () => {
    setLoading(true);
    setResponse("");
    try {
      if (mode === "register") {
        const r = await fetch(`${API_URL}/api/auth/register/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, email, password }),
        });
        const data = await r.json();
        if (!r.ok) throw new Error(JSON.stringify(data));
      }

      const r = await fetch(`${API_URL}/api/auth/login/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Login failed");

      localStorage.setItem("devpilot_access_token", data.access);
      localStorage.setItem("devpilot_refresh_token", data.refresh);
      setAccessToken(data.access);
      setResponse("Authentication successful.");
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!message.trim() || !accessToken) return;
    setLoading(true);
    setResponse("");
    try {
      const r = await fetch(`${API_URL}/api/ai/chat/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ message }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || data.error || "Something went wrong");
      setResponse(data.response);
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("devpilot_access_token");
    localStorage.removeItem("devpilot_refresh_token");
    setAccessToken("");
    setResponse("");
  };

  if (!accessToken) {
    return (
      <div>
        <h1>DevPilot AI</h1>
        <h2>{mode === "login" ? "Login" : "Create account"}</h2>
        {mode === "register" && (
          <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" type="email" />
        )}
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" />
        <input value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" type="password" />
        <button onClick={authenticate} disabled={loading}>
          {loading ? "Please wait..." : mode === "login" ? "Login" : "Register"}
        </button>
        <button onClick={() => setMode(mode === "login" ? "register" : "login")} disabled={loading}>
          {mode === "login" ? "Create account" : "Back to login"}
        </button>
        <p>{response}</p>
      </div>
    );
  }

  return (
    <div>
      <h1>DevPilot AI</h1>
      <button onClick={logout}>Logout</button>
      <input value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Enter your task..." />
      <button onClick={sendMessage} disabled={loading}>
        {loading ? "Thinking..." : "Ask AI"}
      </button>
      <div>
        <h2>AI Response</h2>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default App;
