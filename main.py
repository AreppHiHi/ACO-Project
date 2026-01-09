import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- ACO LOGIC ---
def aco_knapsack(items, weights, values, capacity, n_ants=20, n_iterations=50, alpha=1.0, beta=2.0, evaporation=0.5):
    n_items = len(items)
    pheromone = np.full(n_items, 0.1) 
    best_global_value = 0
    best_global_combination = None
    history = []

    for iteration in range(n_iterations):
        all_ant_values = []
        for ant in range(n_ants):
            current_weight, current_value = 0, 0
            selected_items = np.zeros(n_items)
            available_indices = list(range(n_items))
            
            while available_indices:
                probs = []
                valid_indices = []
                for i in available_indices:
                    if current_weight + weights[i] <= capacity:
                        heuristic = values[i] / weights[i] if weights[i] > 0 else values[i]
                        p = (pheromone[i] ** alpha) * (heuristic ** beta)
                        probs.append(p)
                        valid_indices.append(i)
                
                if not probs or sum(probs) == 0: break
                
                probs = np.array(probs) / sum(probs)
                next_item = np.random.choice(valid_indices, p=probs)
                selected_items[next_item] = 1
                current_weight += weights[next_item]
                current_value += values[next_item]
                available_indices.remove(next_item)

            if current_value > best_global_value:
                best_global_value = current_value
                best_global_combination = selected_items.copy()
            all_ant_values.append(current_value)

        pheromone *= (1 - evaporation)
        if best_global_combination is not None:
            for i in range(n_items):
                if best_global_combination[i] == 1:
                    pheromone[i] += (best_global_value / 100) 
        history.append(best_global_value)
    
    return best_global_value, best_global_combination, history

# --- STREAMLIT UI ---
st.set_page_config(page_title="ACO Knapsack Evaluator", layout="wide")
st.title("🐜 ACO Knapsack Evaluation System")

# SIDEBAR
st.sidebar.header("1. Parameters")
capacity = st.sidebar.number_input("Max Capacity", min_value=1, value=100)
n_ants = st.sidebar.slider("Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)

st.sidebar.header("2. Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        # Cuba baca fail (mengendalikan pelbagai jenis pemisah/separator)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # 1. Cuci Nama Kolum (Buang space, tukar huruf kecil)
        df.columns = df.columns.str.strip().str.lower()
        
        st.success("File uploaded successfully!")
        st.write("### Data Preview")
        st.dataframe(df.head())

        # 2. Pemilihan Kolum secara Manual (Mengelakkan KeyError)
        all_cols = df.columns.tolist()
        st.info("Match your CSV columns to the system variables:")
        
        c1, c2, c3 = st.columns(3)
        with c1: it_col = st.selectbox("Item Name", all_cols, index=0)
        with c2: val_col = st.selectbox("Value/Price", all_cols, index=1 if len(all_cols)>1 else 0)
        with c3: wt_col = st.selectbox("Weight", all_cols, index=2 if len(all_cols)>2 else 0)

        # 3. Jalankan Algoritma
        if st.button("🚀 Run ACO"):
            # Tukar data ke format nombor (elak ralat string)
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
            df[wt_col] = pd.to_numeric(df[wt_col], errors='coerce').fillna(0)

            best_v, best_c, history = aco_knapsack(
                df[it_col].values, df[wt_col].values, df[val_col].values, capacity, n_ants, n_iter
            )

            # --- PAPARAN KEPUTUSAN ---
            st.divider()
            k1, k2 = st.columns([1, 1])
            with k1:
                st.subheader("Results")
                st.metric("Optimal Value", f"{best_v}")
                res_df = df[best_c == 1]
                st.write("**Items Selected:**")
                st.dataframe(res_df)
            
            with k2:
                st.subheader("Performance Graph")
                
                fig, ax = plt.subplots()
                ax.plot(history, marker='o', color='green')
                ax.set_title("Optimization History")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Best Value")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.warning("Please make sure your CSV is formatted correctly (Comma separated).")
else:
    st.info("Please upload a CSV file to begin.")
