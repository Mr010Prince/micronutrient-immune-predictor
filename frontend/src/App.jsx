import React, { useState } from "react";

const FEATURES = [
  "age",
  "gender",
  "race_ethnicity",
  "income_poverty_ratio",
  "vitamin_A",
  "vitamin_E",
  "vitamin_C",
  "vitamin_D_25OH",
  "ferritin",
  "serum_iron",
  "TIBC",
  "transferrin_saturation",
  "transferrin_receptor",
  "RBC_folate",
  "serum_total_folate",
  "blood_selenium",
];

const LABELS = {
  age: "Age (years)",
  gender: "Gender",
  race_ethnicity: "Race/Ethnicity",
  income_poverty_ratio: "Income Poverty Ratio",
  vitamin_A: "Vitamin A (µg/dL)",
  vitamin_E: "Vitamin E (mg/dL)",
  vitamin_C: "Vitamin C (mg/dL)",
  vitamin_D_25OH: "Vitamin D 25-OH (ng/mL)",
  ferritin: "Ferritin (ng/mL)",
  serum_iron: "Serum Iron (µg/dL)",
  TIBC: "Total Iron Binding Capacity (µg/dL)",
  transferrin_saturation: "Transferrin Saturation (%)",
  transferrin_receptor: "Transferrin Receptor (mg/L)",
  RBC_folate: "Red Blood Cell Folate (nmol/L)",
  serum_total_folate: "Serum Total Folate (ng/mL)",
  blood_selenium: "Blood Selenium (µg/L)",
};

const GENDER_OPTIONS = [
  { value: 1, label: "Male" },
  { value: 2, label: "Female" },
];

const RACE_OPTIONS = [
  { value: 1, label: "Mexican American" },
  { value: 2, label: "Other Hispanic" },
  { value: 3, label: "Non-Hispanic White" },
  { value: 4, label: "Non-Hispanic Black" },
  { value: 6, label: "Non-Hispanic Asian" },
  { value: 7, label: "Other/Multi-Racial" },
];

const PLACEHOLDERS = {
  age: "e.g. 35",
  income_poverty_ratio: "e.g. 3.5",
  vitamin_A: "e.g. 30",
  vitamin_E: "e.g. 12",
  vitamin_C: "e.g. 55",
  vitamin_D_25OH: "e.g. 30",
  ferritin: "e.g. 80",
  serum_iron: "e.g. 17",
  TIBC: "e.g. 320",
  transferrin_saturation: "e.g. 26",
  transferrin_receptor: "e.g. 0.8",
  RBC_folate: "e.g. 500",
  serum_total_folate: "e.g. 15",
  blood_selenium: "e.g. 120",
};

const API_URL = "http://localhost:8000/predict";

export default function App() {
  const [view, setView] = useState("landing");
  const [form, setForm] = useState(() =>
    FEATURES.reduce((acc, f) => ({ ...acc, [f]: f === "gender" || f === "race_ethnicity" ? "" : "" }), {})
  );
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = {};
    for (const f of FEATURES) {
      const v = parseFloat(form[f]);
      if (isNaN(v)) {
        setError(`Missing or invalid value for "${LABELS[f]}"`);
        setLoading(false);
        return;
      }
      payload[f] = v;
    }

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const topShap = result
    ? Object.entries(result.shap_values)
        .map(([k, v]) => ({ feature: k, value: v }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 5)
    : [];

  const maxAbs = topShap.length ? Math.max(...topShap.map((s) => Math.abs(s.value))) : 1;

  if (view === "landing") {
    return (
      <div className="container">
        <h1>Micronutrient Immune Dysfunction Predictor</h1>
        <p style={{ marginBottom: "20px" }}>
          Predict immune dysfunction risk from micronutrient and demographic inputs.
        </p>
        <button onClick={() => setView("dashboard")}>Start Prediction</button>
      </div>
    );
  }

  return (
    <div className="container">
      <button
        onClick={() => setView("landing")}
        style={{ marginBottom: "16px", background: "transparent", border: "1px solid #4a90d9", borderRadius: "4px", padding: "6px 12px", cursor: "pointer" }}
      >
        Back
      </button>

      <h1 style={{ marginBottom: "16px" }}>Immune Dysfunction Predictor</h1>

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          {FEATURES.map((f) => (
            <React.Fragment key={f}>
              <label htmlFor={f}>{LABELS[f]}</label>
              {f === "gender" ? (
                <select id={f} name={f} value={form[f]} onChange={handleChange} required>
                  <option value="">Select...</option>
                  {GENDER_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              ) : f === "race_ethnicity" ? (
                <select id={f} name={f} value={form[f]} onChange={handleChange} required>
                  <option value="">Select...</option>
                  {RACE_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={f}
                  name={f}
                  type="number"
                  step="any"
                  placeholder={PLACEHOLDERS[f] || ""}
                  value={form[f]}
                  onChange={handleChange}
                  required
                />
              )}
            </React.Fragment>
          ))}
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="results">
          <h2 style={{ fontSize: "18px", marginBottom: "16px" }}>Result</h2>

          <p>
            <strong>Prediction: </strong>
            <span
              className={
                result.prediction === 1
                  ? "prediction-badge prediction-deficient"
                  : "prediction-badge prediction-normal"
              }
            >
              {result.prediction === 1 ? "Deficient" : "Not Deficient"}
            </span>
          </p>

          <p className="probability">
            <strong>Probability: </strong>
            {(result.probability * 100).toFixed(1)}%
          </p>

          <h3 style={{ fontSize: "15px", marginBottom: "10px" }}>Top 5 SHAP values</h3>
          {topShap.map((s) => {
            const pct = (Math.abs(s.value) / maxAbs) * 100;
            const fillClass = s.value >= 0 ? "positive" : "negative";
            return (
              <div key={s.feature} className="shap-bar">
                <div className="shap-bar-label">
                  <span>{LABELS[s.feature] || s.feature}</span>
                  <span>{s.value.toFixed(4)}</span>
                </div>
                <div className="shap-bar-track">
                  <div
                    className={`shap-bar-fill ${fillClass}`}
                    style={{ width: `${pct}%` }}
                  >
                    {s.value.toFixed(4)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
