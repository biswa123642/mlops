function formatProbability(value) {
  const probability = Number(value);

  if (!Number.isFinite(probability)) {
    return "N/A";
  }

  return `${(probability * 100).toFixed(2)}%`;
}


function PredictionResult({ result }) {
  if (!result) {
    return null;
  }


  const prediction = Number(
    result.prediction?.[0]
  );

  const probability = result.probability?.[0] || [];

  const noChurnProbability = probability[0];

  const churnProbability =
    result.churn_probability?.[0] ??
    probability[1];


  const isChurning = prediction === 1;


  return (
    <section
      className={`prediction-result ${
        isChurning ? "churn-result" : "no-churn-result"
      }`}
      aria-live="polite"
    >
      <h2>Prediction Result</h2>

      <p>
        <strong>Prediction:</strong>{" "}
        {isChurning
          ? "Customer Will Churn"
          : "Customer Will Not Churn"}
      </p>

      <p>
        <strong>No Churn Probability:</strong>{" "}
        {formatProbability(noChurnProbability)}
      </p>

      <p>
        <strong>Churn Probability:</strong>{" "}
        {formatProbability(churnProbability)}
      </p>
    </section>
  );
}


export default PredictionResult;