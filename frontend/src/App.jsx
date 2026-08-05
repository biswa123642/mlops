import React, { useState } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function App() {
  const [formData, setFormData] = useState({
    tenure: 12,
    monthly_charges: 80.5,
    support_calls: 2,
    contract_type: 0,
    internet_service: 2,
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value, type } = event.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(value) : Number(value),
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setResult(null);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(
          errData.detail || `API request failed with status ${response.status}`
        );
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setError(err.message || "Unable to connect to the prediction backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="prediction-card">
        <h1>Customer Churn Prediction</h1>
        <p className="subtitle">Enter customer information to predict churn.</p>

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="tenure">Tenure (Months)</label>
            <input
              id="tenure"
              type="number"
              name="tenure"
              value={formData.tenure}
              min="0"
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="monthly_charges">Monthly Charges ($)</label>
            <input
              id="monthly_charges"
              type="number"
              name="monthly_charges"
              value={formData.monthly_charges}
              min="0"
              step="0.01"
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="support_calls">Support Calls</label>
            <input
              id="support_calls"
              type="number"
              name="support_calls"
              value={formData.support_calls}
              min="0"
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="contract_type">Contract Type</label>
            <select
              id="contract_type"
              name="contract_type"
              value={formData.contract_type}
              onChange={handleChange}
            >
              <option value={0}>Month-to-month</option>
              <option value={1}>One Year</option>
              <option value={2}>Two Year</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="internet_service">Internet Service</label>
            <select
              id="internet_service"
              name="internet_service"
              value={formData.internet_service}
              onChange={handleChange}
            >
              <option value={0}>No Internet</option>
              <option value={1}>DSL</option>
              <option value={2}>Fiber Optic</option>
            </select>
          </div>

          <button type="submit" disabled={loading}>
            {loading ? "Predicting..." : "Predict Churn"}
          </button>
        </form>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="result-card">
            <h2>Prediction Result</h2>
            <div className={`status-badge churn-${result.churn}`}>
              {result.churn
                ? "High Risk: Likely to Churn"
                : "Low Risk: Unlikely to Churn"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
