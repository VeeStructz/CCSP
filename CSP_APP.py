import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Concrete Compressive Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Dataset parameters from Table 1 (UCI Concrete Compressive Strength Dataset) 
# Mean and std used to sample realistic mix combinations via clipped normal
# distributions. Min/max define absolute physical bounds for each constituent.
CEMENT_MIN,  CEMENT_MAX,  CEMENT_MEAN,  CEMENT_STD  = 102.0,  540.0, 281.17, 104.51
SLAG_MIN,    SLAG_MAX,    SLAG_MEAN,    SLAG_STD    = 0.0,    359.4,  73.90,  86.28
FLYASH_MIN,  FLYASH_MAX,  FLYASH_MEAN,  FLYASH_STD  = 0.0,    200.1,  54.19,  64.00
WATER_MIN,   WATER_MAX,   WATER_MEAN,   WATER_STD   = 121.8,  247.0, 181.57,  21.35
SP_MIN,      SP_MAX,      SP_MEAN,      SP_STD      = 0.0,     32.2,   6.20,   5.97
COARSE_MIN,  COARSE_MAX,  COARSE_MEAN,  COARSE_STD  = 801.0, 1145.0, 972.92,  77.75
FINE_MIN,    FINE_MAX,    FINE_MEAN,    FINE_STD    = 594.0,  992.6, 773.58,  80.18

# Water-to-cement ratio bounds
# Lower bound (0.30): below this, mixes are generally unworkable without
# high-range water reducers. Upper bound (0.45): maximum permissible w/c
# ratio for structural concrete as specified for this study.
WC_MIN, WC_MAX = 0.30, 0.45

# Derived cement minimum for the optimizer: ensures that even at the maximum
# permitted w/c ratio (0.45), the resulting water content cannot fall below
# the dataset minimum (121.8 kg). i.e. cement_min = WATER_MIN / WC_MAX = 271 kg.
# Sampling cement below this threshold would force the np.clip on water to
# override the w/c constraint, producing physically inconsistent mixes.
CEMENT_OPT_MIN = int(np.ceil(WATER_MIN / WC_MAX))  # 271 kg


# ── Load Models & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model1 = joblib.load('best_model_1.pkl')
        model2 = joblib.load('best_model_2.pkl')
        scaler = joblib.load('scaler.pkl')
        return model1, model2, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please make sure .pkl files are in the same directory.")
        return None, None, None

model1, model2, scaler = load_artifacts()

# Sidebar Navigation
st.sidebar.title("🏗️ Navigation")
st.sidebar.markdown("Go to")
page = st.sidebar.radio("", ["Strength Predictor", "Mix Optimizer", "Project Info"])
st.sidebar.markdown("---")
st.sidebar.info("Developed for Concrete Mix Optimization Project using ML.")


def clipped_normal(rng, mean, std, low, high, size):
    """Sample from a normal distribution clipped to [low, high].
    Concentrates generated values around the observed dataset mean
    while remaining within absolute constituent bounds."""
    return np.clip(rng.normal(mean, std, size), low, high)


# ── PAGE 1: STRENGTH PREDICTOR ─────────────────────────────────────────────────
if page == "Strength Predictor":
    st.title("🧪 Concrete Compressive Strength Predictor")
    st.markdown("Adjust the ingredient values below to predict the Compressive Strength (MPa).")

    col1, col2, col3 = st.columns(3)

    with col1:
        cement = st.number_input("Cement (kg/m³)",             CEMENT_MIN, CEMENT_MAX, 300.0)
        slag   = st.number_input("Blast Furnace Slag (kg/m³)", SLAG_MIN,   SLAG_MAX,   100.0)
        flyash = st.number_input("Fly Ash (kg/m³)",            FLYASH_MIN, FLYASH_MAX,   0.0)

    with col2:
        water            = st.number_input("Water (kg/m³)",            WATER_MIN, WATER_MAX, 180.0)
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", SP_MIN,    SP_MAX,      5.0)
        age              = st.number_input("Age (Days)", 1, 365, 28)

    with col3:
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", COARSE_MIN, COARSE_MAX, 950.0)
        fine_agg   = st.number_input("Fine Aggregate (kg/m³)",   FINE_MIN,   FINE_MAX,   750.0)

    model_choice = st.radio(
        "Choose Model:",
        ["CatBoost (Best Model 1)", "XGBoost (Best Model 2)"],
        horizontal=True
    )
    active_model = model1 if "CatBoost" in model_choice else model2

    if st.button("Predict Strength"):
        if model1 is None:
            st.error("Models not loaded. Cannot predict.")
        else:
            input_data   = np.array([[cement, slag, flyash, water,
                                       superplasticizer, coarse_agg, fine_agg, age]])
            input_scaled = scaler.transform(input_data)
            prediction   = active_model.predict(input_scaled)[0]

            st.markdown("---")
            c1, c2 = st.columns([1, 2])

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Predicted Strength</h3>
                    <h1 style="color: #4CAF50;">{prediction:.2f} MPa</h1>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                fig, ax = plt.subplots(figsize=(6, 1.5))
                bar_color = 'green' if prediction > 40 else 'orange' if prediction > 20 else 'red'
                ax.barh(['Strength'], [prediction], color=bar_color, height=0.5)
                ax.set_xlim(0, 100)
                ax.set_xlabel('MPa')
                ax.axvline(x=prediction, color='black', linestyle='--')
                ax.axvspan(0,   20, color='red',    alpha=0.1, label='Low (<20)')
                ax.axvspan(20,  40, color='orange', alpha=0.1, label='Standard (20–40)')
                ax.axvspan(40, 100, color='green',  alpha=0.1, label='High Performance (>40)')
                ax.legend(loc='lower right', fontsize=7)
                st.pyplot(fig)


# ── PAGE 2: MIX OPTIMIZER ──────────────────────────────────────────────────────
elif page == "Mix Optimizer":
    st.title("🚀 AI Mix Optimizer")
    st.markdown("""
    **Goal:** Find the cheapest mix (lowest cement) that meets your Target Strength.
    *This module runs 5,000 simulations to find the optimal ratio.*
    """)

    target_strength = st.slider("Target Strength (MPa)", 10.0, 80.0, 40.0)

    if st.button("Optimize Mix Design"):
        if model1 is None:
            st.error("Models not loaded. Cannot optimize.")
        else:
            with st.spinner("Running AI simulations..."):
                n_simulations = 5000
                rng = np.random.default_rng()

                # Candidate mixes are sampled from clipped normal distributions
                # parameterised by the training dataset mean and standard deviation
                # of each constituent (Table 1). Water content is derived directly
                # from cement by sampling the w/c ratio uniformly within [0.30, 0.45],
                # ensuring all generated mixes satisfy the structural concrete
                # water-to-cement ratio requirement specified for this study.
                # Cement is sampled from CEMENT_OPT_MIN (271 kg) upward so that
                # water = cement x w/c always stays above the dataset water minimum
                # (121.8 kg) without the clip overriding the w/c constraint.
                cement_arr = clipped_normal(rng, CEMENT_MEAN, CEMENT_STD,
                                            CEMENT_OPT_MIN, CEMENT_MAX, n_simulations)
                wc_arr     = rng.uniform(WC_MIN, WC_MAX, n_simulations)
                water_arr  = cement_arr * wc_arr

                # 1. Generate binders first
                slag_arr   = clipped_normal(rng, SLAG_MEAN,   SLAG_STD,
                                            SLAG_MIN,   SLAG_MAX,   n_simulations)
                flyash_arr = clipped_normal(rng, FLYASH_MEAN, FLYASH_STD,
                                            FLYASH_MIN, FLYASH_MAX, n_simulations)
                
                # 2. Calculate Total Binder to enforce the Superplasticizer rule
                total_binder = cement_arr + slag_arr + flyash_arr
                
                # 3. Generate SP, but cap it at 3% of the total binder weight
                sp_raw = clipped_normal(rng, SP_MEAN, SP_STD, SP_MIN, SP_MAX, n_simulations)
                sp_capped = np.minimum(sp_raw, total_binder * 0.03)

                # 4. Build the simulation dataframe
                sim_data = pd.DataFrame({
                    'cement':           cement_arr,
                    'slag':             slag_arr,
                    'flyash':           flyash_arr,
                    'water':            water_arr,
                    'superplasticizer': sp_capped,
                    'coarseaggregate':  clipped_normal(rng, COARSE_MEAN, COARSE_STD,
                                                       COARSE_MIN, COARSE_MAX, n_simulations),
                    'fineaggregate':    clipped_normal(rng, FINE_MEAN, FINE_STD,
                                                       FINE_MIN, FINE_MAX, n_simulations),
                    'age':              np.full(n_simulations, 28),
                })

                # Predict compressive strength across all candidate mixes
                feature_cols = ['cement', 'slag', 'flyash', 'water',
                                 'superplasticizer', 'coarseaggregate',
                                 'fineaggregate', 'age']
                sim_scaled = scaler.transform(sim_data[feature_cols])
                sim_data['Predicted_MPa'] = model1.predict(sim_scaled)

                # Retain only mixes that satisfy the target strength requirement
                valid_mixes = sim_data[sim_data['Predicted_MPa'] >= target_strength]

                if valid_mixes.empty:
                    st.warning(
                        f"No suitable mix achieving {target_strength} MPa was identified "
                        "within the current simulation bounds. Try reducing the target strength."
                    )
                else:
                    # Select the mix with the lowest cement content
                    optimal_mix = valid_mixes.sort_values(by='cement').iloc[0]

                    st.success(
                        f"Optimization Complete! Found solution with "
                        f"{optimal_mix['Predicted_MPa']:.2f} MPa."
                    )

                    # Display recipe
                    st.subheader("🏆 Optimal Mix Recipe (per m³)")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Cement",     f"{optimal_mix['cement']:.1f} kg")
                    col2.metric("Water",      f"{optimal_mix['water']:.1f} kg")
                    col3.metric("Aggregates",
                                f"{optimal_mix['coarseaggregate'] + optimal_mix['fineaggregate']:.1f} kg")
                    col4.metric("Admixtures", f"{optimal_mix['superplasticizer']:.1f} kg")

                    # Pie chart
                    labels = ['Cement', 'Slag/Flyash', 'Water', 'Aggregates']
                    sizes  = [
                        optimal_mix['cement'],
                        optimal_mix['slag'] + optimal_mix['flyash'],
                        optimal_mix['water'],
                        optimal_mix['coarseaggregate'] + optimal_mix['fineaggregate'],
                    ]
                    fig, ax = plt.subplots()
                    ax.pie(
                        sizes, labels=labels, autopct='%1.1f%%',
                        startangle=90,
                        colors=['#95a5a6', '#7f8c8d', '#3498db', '#f1c40f']
                    )
                    ax.axis('equal')
                    st.pyplot(fig)


# ── PAGE 3: PROJECT INFO ───────────────────────────────────────────────────────
elif page == "Project Info":
    st.title("Project Details")
    st.markdown("""
    ### Optimization of Concrete Mix Design using ML

    **Objective:** To enhance structural performance and reduce material costs by predicting
    concrete compressive strength accurately.

    **Models Used:**
    - **CatBoost** — High-performance gradient boosting on symmetric decision trees (web deployment).
    - **XGBoost** — Scalable gradient boosting; also deployed in the native Android application.

    **Data Source:**
    - UCI Concrete Compressive Strength Dataset: 1,030 samples, 8 input features.
    """)
