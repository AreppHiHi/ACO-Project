import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# PART 1: ACO ALGORITHM LOGIC
# ==========================================
def run_aco(df, item_col, val_col, wt_col, capacity, n_ants, n_iter, alpha=1.0, beta=2.0, evap=0.5):
    # Ensure data is numeric
    items = df[item_col].values
    values = pd.to_numeric(df[val_col], errors='coerce').fillna(0).values
    weights = pd.to_numeric(df[wt_col], errors='coerce').fillna(0).values
    
    n_items = len(items)
    pheromone = np.full(n_items, 0.1) 
    best_value = 0
    best_comb = np.zeros(n_items)
    history = []

    for _ in range(n_iter):
        for _ in range(n_ants):
            cur_w, cur_v = 0, 0
            temp_comb = np.zeros(n_items)
            avail = list(range(n_items))
            
            while avail:
                probs = []
                valid_idx = []
                for i in avail:
                    if cur_w + weights[i] <= capacity:
                        # Heuristic: Value/Weight ratio
                        h = values[i] / weights[i] if weights[i] > 0 else values[i]
                        p = (pheromone[i]**alpha) * (h**beta)
                        probs.append(p)
                        valid_idx.append(i)
                
                if not probs or sum(probs) == 0: break
                
                # Selection logic
                probs = np.array(probs) / sum(probs)
                pick = np.random.choice(valid_idx, p=probs)
                temp_comb[pick] = 1
                cur_w += weights[pick]
                cur_v += values[pick]
                avail.remove(pick)

            # Update Global Best
            if cur_v > best_value:
                best_value = cur_v
                best_comb = temp_comb.copy()

        # Pheromone Update
        pheromone *= (1 - evap)
        if best_comb is not None:
            for i in range(n_items):
                if best_comb[i] == 1: 
                    pheromone[i] += (best_value / 100)
        history.append(best_value)
    
    return best_value, best_comb, history

# ==========================================
# PART 2: STREAMLIT UI
# ==========================================
st.set_page_config(page_title="ACO Knapsack Evaluator", layout="wide")

st.title("🐜 ACO Dataset Solver (Fixed Version)")
st.write("Upload your CSV file, map the columns, and run the Ant Colony Optimization.")

# FILE UPLOADER
uploaded_file = st.file_uploader("Step 1: Choose your CSV file", type=["csv"])

if uploaded_file is not None:
    # Handle CSV reading with dynamic separator
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    # Clean column names
    df.columns = df.columns.str.strip()
    
    st.success("File uploaded successfully!")
    
    # SIDEBAR SETTINGS
    st.sidebar.header("Configuration")
    cols = df.columns.tolist()
    
    # Column Mapping
    item_name = st.sidebar.selectbox("Item Name Column", cols, index=0)
    val_name = st.sidebar.selectbox("Value Column", cols, index=1 if len(cols)>1 else 0)
    wt_name = st.sidebar.selectbox("Weight Column", cols, index=2 if len(cols)>2 else 0)
    
    # ACO Parameters
    cap = st.sidebar.number_input("Knapsack Capacity", min_value=1, value=100)
    ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
    iters = st.sidebar.slider("Iterations", 10, 300, 100)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10))

    # RUN BUTTON
    if st.button("🚀 Start ACO Optimization"):
        with st.spinner('Calculating optimal solution...'):
            start_t = time.time()
            # EXECUTE ALGORITHM
            res_val, res_comb, history = run_aco(df, item_name, val_name, wt_name, cap, ants, iters)
            end_t = time.time()
            
            st.divider()
            
            # --- RESULTS SECTION ---
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("✅ Optimization Result")
                st.metric("Total Profit", f"{res_val}")
                st.write(f"Computation Time: {end_t - start_t:.4f}s")
                
                # FIXED: Use .loc with boolean indexing to avoid KeyError
                selected_items = df.loc[res_comb == 1]
                st.write("**Items Selected in Knapsack:**")
                st.dataframe(selected_items)
            
            with col2:
                st.subheader("📈 Convergence Graph")
                
                fig, ax = plt.subplots()
                ax.plot(history, color='#1f77b4', linewidth=2)
                ax.set_title("Performance Trend")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Max Value Found")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.info("The graph shows how the ants 'converged' to the best solution.")
else:
    st.info("Please upload a CSV file to begin.")
