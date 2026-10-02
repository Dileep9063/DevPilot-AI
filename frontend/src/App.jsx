import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [task, setTask] = useState("");
  const [response, setResponse] = useState(null);

  const [loading, setLoading] = useState(false);
  const [accessToken, setAccessToken] = useState(
    () => localStorage.getItem("devpilot_access_token") || ""
  );

  const [threadId, setThreadId] = useState("");
  const [approvalRequired, setApprovalRequired] = useState(false);
  const [proposedPath, setProposedPath] = useState("");
  const [proposedContent, setProposedContent] = useState("");
  const [refreshToken, setRefreshToken] = useState(
    () => localStorage.getItem("devpilot_refresh_token") || ""
  );

  const authenticate = async () => {
    setLoading(true);
    setResponse(null);

    try {
      if (mode === "register") {
        const registerResponse = await fetch(
          `${API_URL}/api/auth/register/`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              username,
              email,
              password,
            }),
          }
        );

        const registerData = await registerResponse.json();

        if (!registerResponse.ok) {
          throw new Error(JSON.stringify(registerData));
        }
      }

      const loginResponse = await fetch(`${API_URL}/api/auth/login/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
        }),
      });

      const loginData = await loginResponse.json();

      if (!loginResponse.ok) {
        throw new Error(loginData.detail || "Login failed");
      }

      localStorage.setItem("devpilot_access_token", loginData.access);
      localStorage.setItem("devpilot_refresh_token", loginData.refresh);

      setAccessToken(loginData.access);
      setRefreshToken(loginData.refresh);
      setResponse({
        status: "success",
        message: "Authentication successful.",
      });
    } catch (error) {
      setResponse({
        status: "error",
        error: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleAgentResult = (data) => {
    setResponse(data);

    if (data.status === "waiting_for_approval") {
      setApprovalRequired(true);

      const approval = data.approval_request || {};

      setProposedPath(
        approval.path || approval.proposed_path || ""
      );

      setProposedContent(
        approval.content || approval.proposed_content || ""
      );

      return;
    }

    setApprovalRequired(false);
    setProposedPath("");
    setProposedContent("");
  };

  const refreshAccessToken = async () => {
    if (!refreshToken) return false;

    const refreshResponse = await fetch(
      `${API_URL}/api/auth/refresh/`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh: refreshToken }),
      }
    );

    if (!refreshResponse.ok) {
      logout();
      return false;
    }

    const data = await refreshResponse.json();
    localStorage.setItem("devpilot_access_token", data.access);
    setAccessToken(data.access);
    return true;
  };

  const authenticatedFetch = async (url, options = {}) => {
    let result = await fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${accessToken}`,
      },
    });

    if (result.status === 401 && refreshToken) {
      const refreshed = await refreshAccessToken();
      if (refreshed) {
        result = await fetch(url, {
          ...options,
          headers: {
            ...(options.headers || {}),
            Authorization: `Bearer ${localStorage.getItem("devpilot_access_token")}`,
          },
        });
      }
    }

    return result;
  };

  const runAgent = async () => {
    if (!task.trim() || !accessToken) {
      return;
    }

    setLoading(true);
    setResponse(null);
    setApprovalRequired(false);

    try {
      const agentResponse = await authenticatedFetch(`${API_URL}/api/agent/run/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({
          task,
          ...(threadId ? { thread_id: threadId } : {}),
        }),
      });

      const data = await agentResponse.json();

      if (!agentResponse.ok) {
        throw new Error(
          data.detail || data.error || "Agent request failed"
        );
      }

      if (data.thread_id) {
        setThreadId(data.thread_id);
      }

      handleAgentResult(data);
    } catch (error) {
      setResponse({
        status: "error",
        error: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const resumeAgent = async (decision) => {
    if (!threadId || !accessToken) {
      return;
    }

    setLoading(true);
    setResponse(null);

    try {
      const resumeResponse = await authenticatedFetch(
        `${API_URL}/api/agent/resume/`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          body: JSON.stringify({
            thread_id: threadId,
            decision,
          }),
        }
      );

      const data = await resumeResponse.json();

      if (!resumeResponse.ok) {
        throw new Error(
          data.detail || data.error || "Agent resume failed"
        );
      }

      handleAgentResult(data);
    } catch (error) {
      setResponse({
        status: "error",
        error: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    localStorage.removeItem("devpilot_access_token");
    localStorage.removeItem("devpilot_refresh_token");

    setAccessToken("");
    setRefreshToken("");
    setResponse(null);
    setTask("");
    setThreadId("");
    setApprovalRequired(false);
    setProposedPath("");
    setProposedContent("");
  };

  if (!accessToken) {
    return (
      <div>
        <h1>DevPilot AI</h1>

        <h2>
          {mode === "login" ? "Login" : "Create account"}
        </h2>

        {mode === "register" && (
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            type="email"
          />
        )}

        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Username"
        />

        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          type="password"
        />

        <button onClick={authenticate} disabled={loading}>
          {loading
            ? "Please wait..."
            : mode === "login"
            ? "Login"
            : "Register"}
        </button>

        <button
          onClick={() =>
            setMode(mode === "login" ? "register" : "login")
          }
          disabled={loading}
        >
          {mode === "login"
            ? "Create account"
            : "Back to login"}
        </button>

        {response && (
          <pre>
            {JSON.stringify(response, null, 2)}
          </pre>
        )}
      </div>
    );
  }

  return (
    <div>
      <h1>DevPilot AI</h1>

      <button onClick={logout}>Logout</button>

      <h2>AI Software Engineering Agent</h2>

      <textarea
        value={task}
        onChange={(e) => setTask(e.target.value)}
        placeholder="Example: Add a logout function to workspace/auth.py"
        rows={6}
        cols={70}
      />

      <br />

      <button onClick={runAgent} disabled={loading || !task.trim()}>
        {loading ? "Agent working..." : "Run Agent"}
      </button>

      {threadId && (
        <p>
          <strong>Thread ID:</strong> {threadId}
        </p>
      )}

      {approvalRequired && (
        <div>
          <h2>Human Approval Required</h2>

          <p>
            The agent wants to modify the following file:
          </p>

          <p>
            <strong>{proposedPath || "Unknown file"}</strong>
          </p>

          {proposedContent && (
            <pre
              style={{
                whiteSpace: "pre-wrap",
                border: "1px solid #ccc",
                padding: "10px",
              }}
            >
              {proposedContent}
            </pre>
          )}

          <button
            onClick={() => resumeAgent("approve")}
            disabled={loading}
          >
            Approve
          </button>

          <button
            onClick={() => resumeAgent("reject")}
            disabled={loading}
          >
            Reject
          </button>
        </div>
      )}

      {response && (
        <div>
          <h2>Agent Response</h2>

          <pre
            style={{
              whiteSpace: "pre-wrap",
              border: "1px solid #ccc",
              padding: "10px",
            }}
          >
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;