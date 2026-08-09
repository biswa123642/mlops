import axios from "axios";


const api = axios.create({
  baseURL: (
    import.meta.env.VITE_API_BASE_URL ||
    "http://localhost:8000"
  ).replace(/\/$/, ""),
  headers: {
    "Content-Type": "application/json",
  },
});


export const predictChurn = async (payload) => {
  const response = await api.post(
    "/predict",
    payload
  );

  return response.data;
};


export default api;