import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    const message = input.trim();

    if (!message || loading) {
      return;
    }

    // Add user message immediately
    const userMessage = {
      role: "user",
      content: message,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          message: message,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Backend returned HTTP ${response.status}`
        );
      }

      const data = await response.json();

      const botMessage = {
        role: "bot",
        content: data.reply || "No response received.",
      };

      setMessages((prev) => [
        ...prev,
        botMessage,
      ]);
    } catch (error) {
      console.error("Error fetching response:", error);

      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content:
            "Sorry, I was unable to get a response from the backend.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Allow Enter key to send message
  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-app">

      {/* Header */}
      <div className="chat-header">
        Azure AI Chat
      </div>

      {/* Chat messages */}
      <div className="chat-window">

        {messages.length === 0 && (
          <div className="chat-message bot">
            Hello! How can I help you?
          </div>
        )}

        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chat-message ${
              msg.role === "user"
                ? "user"
                : "bot"
            }`}
          >
            {msg.content}
          </div>
        ))}

        {loading && (
          <div className="chat-message bot">
            Typing...
          </div>
        )}

      </div>

      {/* Input */}
      <div className="chat-input">

        <input
          type="text"
          value={input}
          onChange={(event) =>
            setInput(event.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={loading}
        />

        <button
          onClick={sendMessage}
          disabled={
            loading || !input.trim()
          }
        >
          {loading ? "Sending..." : "Send"}
        </button>

      </div>

    </div>
  );
}

export default App;