import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ACO Knapsack Dashboard",
    page_icon="🎒",
    layout="wide"
)

# --- ACO for Knapsack Logic ---
class ACO_Knapsack:
    def __init__(self, values, weights, capacity, n_ants, iterations, alpha, beta, evaporation):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.n_items = len(values)
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha      # Pheromone importance
        self.beta = beta        # Heuristic importance (Value/Weight ratio)
        self.evaporation = evaporation
        self.pheromone = np.ones(self.n_items)
        
        # Heuristic: Efficiency of each item (value per unit of weight)
        self.heuristic = self.values / (self.weights + 1e-9) 

    def run(self, progress_bar, chart_placeholder):
        best_value = 0
        best_selection = np.zeros(self.n_items)
        history = []

        for i in range(self.iterations):
            all_solutions = []
            all_values = []

            for ant in range(self.n_ants):
                selection = self._build_solution()
                total_v = np.sum(selection * self.values)
                total_w = np.sum(selection * self.weights)
                
                # Only consider valid solutions
                if total_w <= self.capacity:
                    all_solutions.append(selection)
                    all_values.append(total_v)
                    if total_v > best_value:
                        best_value = total_v
                        best_selection = selection.copy()

            # Update Pheromones (Evaporation)
            self.pheromone *= (1 - self.evaporation)
            
            # Update Pheromones (Deposit based on quality)
            for sol, val in zip(all_solutions, all_values):
                deposit = val / (best_value + 1e-9)
                self.pheromone += (sol * deposit)

            history.append(best_value)
            
            # UI Updates
            progress_bar.progress((i + 1) / self.iterations)
            chart_placeholder.line_chart(history)
            time.sleep(0.01)
            
        return best_selection, best_value, history

    def _build_solution(self):
        solution = np.zeros(self.n_items)
        current_weight = 0
        indices = np.arange(self.n_items)
        np.random.shuffle(indices) # Randomize entry order

        for idx in indices:
            if current_weight + self.weights[idx] <= self.capacity:
                # Probability calculation
                phi = self.pheromone[idx] ** self.alpha
                eta = self.heuristic[idx] ** self.beta
                
                # Decision rule (Simplified for 0/1 Knapsack)
                prob = (phi * eta) / (phi * eta + 1.0)
                if np.random.random() < prob:
                    solution[idx] = 1
                    current_weight += self.weights[idx]
        return solution

# --- Streamlit UI Layout ---
st.title("🐜 ACO Knapsack Problem Dashboard")
st.markdown("""
This dashboard uses **Ant Colony Optimization** to solve the 0/1 Knapsack Problem. 
Upload a CSV, set the capacity, and watch the ants find the best combination of items.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Algorithm Parameters")
    capacity = st.number_input("Knapsack Capacity (Max Weight)", value=100, min_value=1)
    ants = st.slider("Number of Ants", 5, 100, 20)
    iters = st.slider("Iterations", 10, 500, 100)
    
    st.subheader("Hyperparameters")
    alpha = st.slider("Alpha (Pheromone)", 0.0, 5.0, 1.0)
    beta = st.slider("Beta (Heuristic)", 0.0, 5.0, 2.0)
    evap = st.slider("Evaporation Rate", 0.1, 0.9, 0.5)

# Main Area
st.subheader("1. Load Dataset")
uploaded_file = st.file_uploader("Upload CSV (must contain 'value' and 'weight' columns)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # CLEANING: Fixes the KeyError by cleaning column names
    df.columns = df.columns.str.strip().str.lower()
    
    st.write("✅ Dataset Loaded. Columns found:", list(df.columns))
    st.dataframe(df.head(), use_container_width=True)

    if 'value' in df.columns and 'weight' in df.columns:
        if st.button("🚀 Start Optimization", type="primary"):
            
            # Execution
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            
            aco = ACO_Knapsack(
                values=df['value'].values,
                weights=df['weight'].values,
                capacity=capacity,
                n_ants=ants,
                iterations=iters,
                alpha=alpha,
                beta=beta,
                evaporation=evap
            )
            
            best_sol, max_val, history = aco.run(progress_bar, chart_placeholder)
            
            # Display Results
            st.success(f"Done! Maximum Value Found: {max_val}")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.write("### 📦 Selected Items")
                df['selected'] = best_sol.astype(bool)
                selected_df = df[df['selected'] == True]
                st.dataframe(selected_df)
            
            with res_col2:
                st.write("### 📊 Performance Summary")
                total_weight = selected_df['weight'].sum()
                st.metric("Total Value", f"{max_val}")
                st.metric("Total Weight", f"{total_weight} / {capacity}")
    else:
        st.error("Error: CSV missing required columns. Ensure columns are named 'value' and 'weight'.")
else:
    st.info("👋 Please upload a CSV to begin. Example format: item_name, value, weight")
