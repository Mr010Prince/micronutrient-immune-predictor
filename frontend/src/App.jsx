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

const DEMO_FEATURES = ["age", "gender", "race_ethnicity", "income_poverty_ratio"];
const BIO_FEATURES = FEATURES.filter((f) => !DEMO_FEATURES.includes(f));

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

const WHY_CARDS = [
  {
    title: "Data-Driven",
    desc: "Trained on real NHANES 2017-2018 clinical data",
  },
  {
    title: "Explainable",
    desc: "SHAP shows exactly which biomarkers drove each prediction",
  },
  {
    title: "Fast Screening",
    desc: "Instant risk assessment from lab values",
  },
];

const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000") + "/predict";

const INITIAL_FORM = () => FEATURES.reduce((acc, f) => ({ ...acc, [f]: "" }), {});

function FeatureInput({ f, form, handleChange }) {
  if (f === "gender") {
    return (
      <select id={f} name={f} value={form[f]} onChange={handleChange} required>
        <option value="" disabled>Select...</option>
        {GENDER_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    );
  }
  if (f === "race_ethnicity") {
    return (
      <select id={f} name={f} value={form[f]} onChange={handleChange} required>
        <option value="" disabled>Select...</option>
        {RACE_OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    );
  }
  return (
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
  );
}

export default function App() {
  const [view, setView] = useState("landing");
  const [form, setForm] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleReset = () => {
    setForm(INITIAL_FORM());
    setResult(null);
    setError(null);
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
      <div className="landing-wrapper">
        <nav className="nav-bar">
          <span className="nav-logo">Micronutrient Immune Predictor</span>
        </nav>

        <div className="hero">
          <h1 className="hero-headline">
            Predict Immune Dysfunction Risk From Micronutrient Data
          </h1>
          <p className="hero-subtext">
            This tool uses machine learning (XGBoost) on NHANES 2017-2018 health
            data to flag immune dysfunction risk from vitamin, mineral and
            demographic markers, with SHAP explainability showing exactly which
            biomarkers drove each prediction.
          </p>
          <button className="accent-btn" onClick={() => setView("dashboard")}>
            Start Prediction
          </button>
        </div>

        <div className="why-row">
          {WHY_CARDS.map((card, i) => (
            <div key={i} className="why-card">
              <h3 className="why-title">{card.title}</h3>
              <p className="why-desc">{card.desc}</p>
            </div>
          ))}
        </div>

        <div className="about">
          <h3>About this project</h3>
          <p>
            This is a final year Computer Engineering project at Abia State
            University focused on predicting micronutrient deficiency-linked
            immune dysfunction risk. The model uses XGBoost, evaluated against
            Logistic Regression and Random Forest, with SHAP-based
            explainability for transparent, clinically-interpretable predictions.
          </p>
        </div>

        <footer className="footer">
          Final Year Project — Dept. of Computer Engineering, Abia State University
        </footer>
      </div>
    );
  }

  return (
    <div className="dashboard-wrapper">
      <header className="db-header">
        <div className="db-logo">Micronutrient Immune Predictor</div>
        <button className="db-back" onClick={() => setView("landing")}>
          Back to Home
        </button>
      </header>

      <div className="db-stats">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#3b82f6' }} />
          <div className="stat-body">
            <div className="stat-number">16</div>
            <div className="stat-label">Biomarkers Tracked</div>
            <div className="stat-caption">NHANES 2017-2018</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#f59e0b' }} />
          <div className="stat-body">
            <div className="stat-number">6</div>
            <div className="stat-label">ML Models Compared</div>
            <div className="stat-caption">Grid Search & Bayesian Opt</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#10b981' }} />
          <div className="stat-body">
            <div className="stat-number">XGBoost</div>
            <div className="stat-label">Best Model</div>
            <div className="stat-caption">Top Test ROC-AUC</div>
          </div>
        </div>
      </div>

      <div className="db-main">
        <main className="db-panel">
          <h2 className="db-panel-title">Patient Biomarker Input</h2>
          <p className="input-note">
            All values should reflect standard lab test units as shown in each field.
          </p>

          <form onSubmit={handleSubmit}>
            <h3 className="section-heading">Demographics</h3>
            <div className="db-input-grid">
              {DEMO_FEATURES.map((f) => (
                <div className="db-field" key={f}>
                  <label htmlFor={f}>{LABELS[f]}</label>
                  <FeatureInput f={f} form={form} handleChange={handleChange} />
                </div>
              ))}
            </div>

            <h3 className="section-heading">Biomarkers</h3>
            <div className="db-input-grid">
              {BIO_FEATURES.map((f) => (
                <div className="db-field" key={f}>
                  <label htmlFor={f}>{LABELS[f]}</label>
                  <FeatureInput f={f} form={form} handleChange={handleChange} />
                </div>
              ))}
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading} className="accent-btn">
                {loading ? "Predicting..." : "Predict"}
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="reset-btn"
              >
                Reset
              </button>
            </div>
          </form>
        </main>

        {result && (
          <aside className="db-panel db-result">
            <h2 className="db-panel-title">Prediction Result</h2>
            <div className="results">
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

              {result.prediction === 1 ? (
                <p className="interpretation deficient">
                  This profile shows elevated risk markers — consult a healthcare
                  provider for proper diagnosis.
                </p>
              ) : (
                <p className="interpretation normal">
                  This profile shows low risk based on current inputs.
                </p>
              )}

              <h3 className="shap-heading">Top 5 SHAP values</h3>
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
          </aside>
        )}
      </div>

      {error && <p className="error">{error}</p>}
    </div>
  );
}
