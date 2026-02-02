import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Concrete Mix Optimizer",
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

# 1. Load Models & Scaler
@st.cache_resource
def load_artifacts():
    try:
        # Assuming best_model_1 is CatBoost and best_model_2 is XGBoost based on your previous output
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
page = st.sidebar.radio("Go to", ["Strength Predictor", "Mix Optimizer", "Project Info"])

st.sidebar.markdown("---")
st.sidebar.info("Developed for Concrete Mix Optimization Project using ML.")

# --- PAGE 1: STRENGTH PREDICTOR ---
if page == "Strength Predictor":
    st.title("Concrete Compressive Strength Predictor")
    st.markdown("Adjust the ingredient values below to predict the Compressive Strength (MPa).")

    # Input Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 300.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 100.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 0.0)
    
    with col2:
        water = st.number_input("Water (kg/m³)", 100.0, 250.0, 180.0)
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 40.0, 5.0)
        age = st.number_input("Age (Days)", 1, 365, 28)

    with col3:
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 950.0)
        fine_agg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 750.0)

    # Model Selection
    model_choice = st.radio("Choose Model:", ["CatBoost (Best Model 1)", "XGBoost (Best Model 2)"], horizontal=True)
    active_model = model1 if "CatBoost" in model_choice else model2

    # Prediction Logic
    if st.button("Predict Strength"):
        # Prepare input array
        input_data = np.array([[cement, slag, flyash, water, superplasticizer, coarse_agg, fine_agg, age]])
        
        # Scale input (Important!)
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = active_model.predict(input_scaled)[0]

        # Display Result
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
            # Visualization: Where does this fall?
            fig, ax = plt.subplots(figsize=(6, 1.5))
            bar_color = 'green' if prediction > 40 else 'orange' if prediction > 20 else 'red'
            
            # Simple bar chart
            ax.barh(['Strength'], [prediction], color=bar_color, height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('MPa')
            ax.axvline(x=prediction, color='black', linestyle='--', label=f'{prediction:.1f}')
            
            # Add reference zones
            ax.axvspan(0, 20, color='red', alpha=0.1, label='Low')
            ax.axvspan(20, 40, color='orange', alpha=0.1, label='Standard')
            ax.axvspan(40, 100, color='green', alpha=0.1, label='High Performance')
            
            st.pyplot(fig)

# --- PAGE 2: MIX OPTIMIZER ---
elif page == "Mix Optimizer":
    st.title("AI Mix Optimizer")
    st.markdown("""
    **Goal:** Find the cheapest mix (lowest cement) that meets your Target Strength.
    *This module runs 5,000 simulations to find the optimal ratio.*
    """)

    target_strength = st.slider("Target Strength (MPa)", 20.0, 80.0, 40.0)
    
    if st.button("Optimize Mix Design"):
        with st.spinner("Running AI simulations..."):
            # Monte Carlo Simulation
            n_simulations = 5000
            
            # Randomly generate mixes within realistic bounds
            sim_data = pd.DataFrame({
                'cement': np.random.uniform(150, 500, n_simulations),
                'slag': np.random.uniform(0, 300, n_simulations),
                'flyash': np.random.uniform(0, 200, n_simulations),
                'water': np.random.uniform(140, 220, n_simulations),
                'superplasticizer': np.random.uniform(0, 20, n_simulations),
                'coarseaggregate': np.random.uniform(800, 1100, n_simulations),
                'fineaggregate': np.random.uniform(600, 900, n_simulations),
                'age': np.full(n_simulations, 28) # Standard 28-day strength
            })
            
            # Scale and Predict
            sim_scaled = scaler.transform(sim_data)
            sim_data['Predicted_MPa'] = model1.predict(sim_scaled) # Use best model
            
            # Filter: Keep mixes that meet target strength (+/- 2 MPa buffer)
            valid_mixes = sim_data[sim_data['Predicted_MPa'] >= target_strength]
            
            if valid_mixes.empty:
                st.warning("No mix found for this high strength with standard parameters. Try increasing cement limits.")
            else:
                # Optimization Goal: Minimize Cement (Cost proxy)
                optimal_mix = valid_mixes.sort_values(by='cement').iloc[0]
                
                st.success(f"Optimization Complete! Found solution with {optimal_mix['Predicted_MPa']:.2f} MPa.")
                
                # Display Recipe
                st.subheader("🏆 Optimal Mix Recipe (per m³)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Cement", f"{optimal_mix['cement']:.1f} kg")
                col2.metric("Water", f"{optimal_mix['water']:.1f} kg")
                col3.metric("Aggregates", f"{optimal_mix['coarseaggregate'] + optimal_mix['fineaggregate']:.1f} kg")
                col4.metric("Admixtures", f"{optimal_mix['superplasticizer']:.1f} kg")
                
                # Pie Chart of Mix
                labels = ['Cement', 'Slag/Flyash', 'Water', 'Aggregates']
                sizes = [
                    optimal_mix['cement'], 
                    optimal_mix['slag'] + optimal_mix['flyash'],
                    optimal_mix['water'],
                    optimal_mix['coarseaggregate'] + optimal_mix['fineaggregate']
                ]
                
                fig, ax = plt.subplots()
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#95a5a6', '#7f8c8d', '#3498db', '#f1c40f'])
                ax.axis('equal')
                st.pyplot(fig)

# --- PAGE 3: PROJECT INFO ---
elif page == "Project Info":
    st.title("Project Details")
    st.markdown("""
    ### Optimization of Concrete Mix Design using ML
    
    **Objective:** To enhance structural performance and reduce material costs by predicting concrete strength accurately.
    
    **Models Used:**
    - **CatBoost:** High-performance gradient boosting on decision trees.
    - **XGBoost:** Scalable and accurate gradient boosting library.
    
    **Data Source:**
    - Comprehensive dataset comprising 1030 concrete samples with 8 input features.
    """)

    

