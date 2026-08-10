import { useState } from "react";
import {
  ArrowRight,
  BadgeDollarSign,
  CalendarClock,
  Check,
  Landmark,
  LockKeyhole,
  RotateCcw,
  Sparkles,
} from "lucide-react";

const initialForm = {
  model_selection: "v4_stacked",
  credit_score: 740,
  dti: 35,
  ltv: 80,
  cltv: 80,
  mortgage_insurance_pct: 0,
  num_borrowers: 1,
  num_units: 1,
  first_time_homebuyer: false,
  occupancy_status: "P",
  property_type: "SF",
  property_state: "CA",
  msa: "0",
  channel: "R",
  loan_purpose: "P",
};

const modelOptions = [
  {
    key: "v4_stacked",
    title: "Original mixture",
    badge: "v4",
    description: "CatBoost + XGBoost blended together.",
  },
  {
    key: "v4_catboost",
    title: "Original CatBoost",
    badge: "v4",
    description: "Your direct CatBoost lending model.",
  },
  {
    key: "smart_tree",
    title: "Stable smart model",
    badge: "v5",
    description: "Tree model built to avoid pickle version issues.",
  },
  {
    key: "fast_linear",
    title: "Stable simple model",
    badge: "v5",
    description: "Fast baseline model for comparison.",
  },
];

const states = [
  "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
  "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
  "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
  "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
  "VA", "WA", "WV", "WI", "WY", "DC",
];

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function Field({ label, hint, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const update = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          model_selection: form.model_selection,
          credit_score: Number(form.credit_score),
          dti: Number(form.dti),
          ltv: Number(form.ltv),
          cltv: Number(form.cltv),
          mortgage_insurance_pct: Number(form.mortgage_insurance_pct),
          num_borrowers: Number(form.num_borrowers),
          num_units: Number(form.num_units),
        }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "Unable to create estimate");
      setResult(body);
      requestAnimationFrame(() =>
        document.querySelector(".result-card")?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        }),
      );
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark"><Landmark size={20} /></span>
          <span>LoanFit</span>
        </a>
        <div className="research-pill"><Sparkles size={14} /> Research model</div>
      </header>

      <section className="hero">
        <div>
          <p className="eyebrow">A clearer starting point</p>
          <h1>Find a mortgage shape that fits.</h1>
          <p className="hero-copy">
            Add a few borrower and property details to estimate a suitable loan
            amount and term from historical Freddie Mac lending patterns.
          </p>
        </div>
        <div className="hero-note">
          <LockKeyhole size={18} />
          <span>Your details stay in this application and are not stored.</span>
        </div>
      </section>

      <div className="workspace">
        <form className="form-card" onSubmit={submit}>
          <div className="section-heading">
            <span>00</span>
            <div>
              <h2>Choose model style</h2>
              <p>Pick how the estimate should be created.</p>
            </div>
          </div>

          <div className="model-selector" role="radiogroup" aria-label="Model selection">
            {modelOptions.map((option) => (
              <label
                key={option.key}
                className={form.model_selection === option.key ? "model-option selected" : "model-option"}
              >
                <input
                  type="radio"
                  name="model_selection"
                  value={option.key}
                  checked={form.model_selection === option.key}
                  onChange={update}
                />
                <span>
                  {option.title}
                  <em>{option.badge}</em>
                </span>
                <small>{option.description}</small>
              </label>
            ))}
          </div>

          <div className="divider" />

          <div className="section-heading">
            <span>01</span>
            <div>
              <h2>Borrower profile</h2>
              <p>The financial basics used by the estimate.</p>
            </div>
          </div>

          <div className="form-grid">
            <Field label="Credit score">
              <input name="credit_score" type="number" min="300" max="850" value={form.credit_score} onChange={update} required />
            </Field>
            <Field label="Debt-to-income" hint="Monthly debt as a percentage of income">
              <div className="input-suffix">
                <input name="dti" type="number" min="0" max="65" step="0.1" value={form.dti} onChange={update} required />
                <span>%</span>
              </div>
            </Field>
            <Field label="Number of borrowers">
              <select name="num_borrowers" value={form.num_borrowers} onChange={update}>
                {[1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}
              </select>
            </Field>
            <label className="check-field">
              <input name="first_time_homebuyer" type="checkbox" checked={form.first_time_homebuyer} onChange={update} />
              <span className="check-box"><Check size={14} /></span>
              <span>First-time homebuyer</span>
            </label>
          </div>

          <div className="divider" />
          <div className="section-heading">
            <span>02</span>
            <div>
              <h2>Property & loan</h2>
              <p>Tell us what the financing is intended for.</p>
            </div>
          </div>

          <div className="form-grid">
            <Field label="Property state">
              <select name="property_state" value={form.property_state} onChange={update}>
                {states.map((state) => <option key={state}>{state}</option>)}
              </select>
            </Field>
            <Field label="Property type">
              <select name="property_type" value={form.property_type} onChange={update}>
                <option value="SF">Single-family home</option>
                <option value="PU">Planned unit development</option>
                <option value="CO">Condominium</option>
                <option value="MH">Manufactured home</option>
                <option value="CP">Co-operative</option>
              </select>
            </Field>
            <Field label="Occupancy">
              <select name="occupancy_status" value={form.occupancy_status} onChange={update}>
                <option value="P">Primary residence</option>
                <option value="S">Second home</option>
                <option value="I">Investment property</option>
              </select>
            </Field>
            <Field label="Loan purpose">
              <select name="loan_purpose" value={form.loan_purpose} onChange={update}>
                <option value="P">Purchase</option>
                <option value="C">Cash-out refinance</option>
                <option value="N">No cash-out refinance</option>
              </select>
            </Field>
            <Field label="Loan-to-value">
              <div className="input-suffix">
                <input name="ltv" type="number" min="1" max="110" step="0.1" value={form.ltv} onChange={update} required />
                <span>%</span>
              </div>
            </Field>
            <Field label="Combined LTV">
              <div className="input-suffix">
                <input name="cltv" type="number" min="1" max="150" step="0.1" value={form.cltv} onChange={update} required />
                <span>%</span>
              </div>
            </Field>
            <Field label="Mortgage insurance">
              <div className="input-suffix">
                <input name="mortgage_insurance_pct" type="number" min="0" max="55" step="0.1" value={form.mortgage_insurance_pct} onChange={update} />
                <span>%</span>
              </div>
            </Field>
            <Field label="Property units">
              <select name="num_units" value={form.num_units} onChange={update}>
                {[1, 2, 3, 4].map((value) => <option key={value}>{value}</option>)}
              </select>
            </Field>
          </div>

          <details>
            <summary>Additional details</summary>
            <div className="form-grid details-grid">
              <Field label="Metro area code (MSA)" hint="Use 0 when unknown">
                <input name="msa" value={form.msa} onChange={update} maxLength="8" />
              </Field>
              <Field label="Origination channel">
                <select name="channel" value={form.channel} onChange={update}>
                  <option value="R">Retail</option>
                  <option value="C">Correspondent</option>
                  <option value="B">Broker</option>
                </select>
              </Field>
            </div>
          </details>

          {error && <div className="error-message">{error}</div>}
          <div className="form-actions">
            <button className="reset-button" type="button" onClick={reset}>
              <RotateCcw size={16} /> Reset
            </button>
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Estimating…" : "Create my estimate"}
              {!loading && <ArrowRight size={18} />}
            </button>
          </div>
        </form>

                {/* Dynamic Multi-Model Output Display View */}
         {/* Dynamic Multi-Model Output Display View */}
        <div className={`result-card ${result ? "has-result" : ""}`}>
          <p className="result-kicker">Risk Assessment</p>
          <h2>Estimation Profile</h2>

          {result ? (
            <div>
              {/* 1. Loan Amount */}
              <div className="result-block">
                <div className="result-icon">
                  <BadgeDollarSign size={22} />
                </div>
                <div>
                  <p>Estimated Loan Target</p>
                  <strong>{currency.format(result.estimated_loan_amount)}</strong>
                </div>
              </div>

              {/* 2. Classified Loan Term Category */}
              <div className="result-block">
                <div className="result-icon">
                  <CalendarClock size={22} />
                </div>
                <div>
                  <p>Optimal Term Shape</p>
                <strong>{result.estimated_loan_term_years} Year Fixed</strong>
                <small>({result.estimated_loan_term_months} Months)</small>
              </div>
            </div>

              <div className="result-block">
                <div className="result-icon">
                  <LockKeyhole size={22} />
                </div>
                <div>
                  <p>Estimated Interest Rate</p>
                  <strong>{result.estimated_interest_rate}%</strong>
                  <small>Model-estimated note rate</small>
                </div>
              </div>

              <div className="result-block">
                <div className="result-icon">
                  <Sparkles size={22} />
                </div>
                <div>
                  <p>Model used</p>
                  <strong className="model-used">{result.model_label}</strong>
                  <small>{result.model_family}</small>
                </div>
              </div>

              {result.model_summary && (
                <div className="result-explainer">
                  <Sparkles size={16} />
                  <p>{result.model_summary}</p>
                </div>
              )}

              {result.shap_explanation && (
                <div className="shap-panel">
                  <p className="result-kicker">SHAP feature drivers</p>
                  <small>{result.shap_explanation.method}</small>
                  {result.shap_explanation.drivers.length > 0 ? (
                    <div className="shap-list">
                      {result.shap_explanation.drivers.map((driver) => (
                        <div className="shap-row" key={driver.feature}>
                          <div>
                            <span>{driver.label}</span>
                            <small>{driver.direction}</small>
                          </div>
                          <strong className={driver.value >= 0 ? "positive" : "negative"}>
                            {driver.value >= 0 ? `+${driver.value}` : driver.value}
                          </strong>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="shap-empty">Install SHAP in the active Python environment to show feature drivers.</p>
                  )}
                </div>
              )}

              <p className="disclaimer">{result.disclaimer}</p>
            </div>
          ) : (
            <div className="empty-result">
              <div className="orb">
                <span><BadgeDollarSign size={24} /></span>
                <span><LockKeyhole size={20} /></span>
              </div>
              <p>Awaiting Input Parameters</p>
              <p>
                Fill out the metrics form parameters to execute your
                analytical models and generate portfolio predictions.
              </p>
            </div>
          )}
        </div>

      </div>

      <footer>
        <div>&copy; 2026 LoanFit Research Studio</div>
        <div>Freddie Mac Single-Family Dataset Variant</div>
      </footer>
    </main>
  );
}
