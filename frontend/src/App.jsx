import { useState } from "react";

import PredictionForm from "./components/PredictionForm";
import PredictionResult from "./components/PredictionResult";
import { predictChurn } from "./api";


function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");


  const handlePredict = async (payload) => {
    setError("");
    setResult(null);

    try {
      const response = await predictChurn(payload);
      setResult(response);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        "Prediction failed.";

      setError(message);
    }
  };


  return (
    <main className="container">
      <h1>Customer Churn Prediction</h1>

      <PredictionForm
        onPredict={handlePredict}
      />

      {error && (
        <div
          className="error"
          role="alert"
          aria-live="polite"
        >
          {error}
        </div>
      )}

      <PredictionResult
        result={result}
      />
    </main>
  );
}


export default App;