import { useState } from "react";


const initialState = {
  tenure: "",
  monthly_charges: "",
  support_calls: "",
  contract_type: "",
  internet_service: "",
};


function PredictionForm({ onPredict }) {
  const [formData, setFormData] = useState(initialState);
  const [loading, setLoading] = useState(false);


  const handleChange = (event) => {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };


  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);

    try {
      await onPredict({
        tenure: Number(formData.tenure),
        monthly_charges: Number(formData.monthly_charges),
        support_calls: Number(formData.support_calls),
        contract_type: Number(formData.contract_type),
        internet_service: Number(formData.internet_service),
      });
    } finally {
      setLoading(false);
    }
  };


  return (
    <form
      onSubmit={handleSubmit}
      className="prediction-form"
    >
      <label htmlFor="tenure">
        Tenure in months
      </label>

      <input
        id="tenure"
        type="number"
        name="tenure"
        min="0"
        value={formData.tenure}
        onChange={handleChange}
        required
      />


      <label htmlFor="monthly_charges">
        Monthly charges
      </label>

      <input
        id="monthly_charges"
        type="number"
        name="monthly_charges"
        min="0"
        step="0.01"
        value={formData.monthly_charges}
        onChange={handleChange}
        required
      />


      <label htmlFor="support_calls">
        Support calls
      </label>

      <input
        id="support_calls"
        type="number"
        name="support_calls"
        min="0"
        value={formData.support_calls}
        onChange={handleChange}
        required
      />


      <label htmlFor="contract_type">
        Contract type
      </label>

      <input
        id="contract_type"
        type="number"
        name="contract_type"
        min="0"
        max="2"
        value={formData.contract_type}
        onChange={handleChange}
        required
      />


      <label htmlFor="internet_service">
        Internet service
      </label>

      <input
        id="internet_service"
        type="number"
        name="internet_service"
        min="0"
        max="1"
        value={formData.internet_service}
        onChange={handleChange}
        required
      />


      <button
        type="submit"
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>
    </form>
  );
}


export default PredictionForm;