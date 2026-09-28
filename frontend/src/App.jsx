import { useState } from "react";

function App() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) {
      return;
    }

    setLoading(true);
    setResponse("");

    try {
      const result = await fetch(
        "http://127.0.0.1:8000/api/ai/chat/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
          }),
        }
      );

      const data = await result.json();

      if (!result.ok) {
        throw new Error(data.error || "Something went wrong");
      }

      setResponse(data.response);
    } catch (error) {
      setResponse(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>DevPilot AI</h1>

      <input
        type="text"
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        placeholder="Enter your task..."
      />

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