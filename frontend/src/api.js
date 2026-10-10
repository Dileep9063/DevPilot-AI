const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const ACCESS = "devpilot_access_token";
const REFRESH = "devpilot_refresh_token";

export const tokens = {
  access: () => localStorage.getItem(ACCESS) || "",
  refresh: () => localStorage.getItem(REFRESH) || "",
  save: (access, refresh) => {
    localStorage.setItem(ACCESS, access);
    if (refresh) localStorage.setItem(REFRESH, refresh);
  },
  clear: () => {
    localStorage.removeItem(ACCESS);
    localStorage.removeItem(REFRESH);
  },
};

async function request(method, path, body, authed = true) {
  const send = () =>
    fetch(`${API_URL}${path}`, {
      method,
      headers: {
        ...(body ? { "Content-Type": "application/json" } : {}),
        ...(authed ? { Authorization: `Bearer ${tokens.access()}` } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });

  let res = await send();

  // Access token expired: refresh once, then retry.
  if (res.status === 401 && authed && tokens.refresh()) {
    const r = await fetch(`${API_URL}/api/auth/refresh/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh: tokens.refresh() }),
    });
    if (r.ok) {
      tokens.save((await r.json()).access);
      res = await send();
    } else {
      tokens.clear();
      throw new Error("Session expired. Please log in again.");
    }
  }

  let data = {};
  try {
    data = await res.json();
  } catch {
    /* non-JSON error body */
  }

  if (!res.ok) {
    const detail =
      data.error ||
      data.detail ||
      (Object.keys(data).length ? JSON.stringify(data) : `HTTP ${res.status}`);
    throw new Error(detail);
  }
  return data;
}

const post = (path, body, authed = true) => request("POST", path, body, authed);
const get = (path) => request("GET", path);

export const api = {
  login: (username, password) =>
    post("/api/auth/login/", { username, password }, false),
  register: (username, email, password) =>
    post("/api/auth/register/", { username, email, password }, false),
  runAgent: (repo_url, task) => post("/api/agent/run/", { repo_url, task }),
  resume: (thread_id, decision) =>
    post("/api/agent/resume/", { thread_id, decision }),
  githubStatus: () => get("/api/github/status/"),
  githubConnect: () => get("/api/github/connect/"),
  githubDisconnect: () => post("/api/github/disconnect/", {}),
  runs: () => get("/api/agent/runs/"),
  runDetail: (thread_id) => get(`/api/agent/runs/${thread_id}/`),
};
