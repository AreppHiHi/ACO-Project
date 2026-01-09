import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- FUNGSI ALGORITMA ACO ---
def aco_knapsack(items, weights, values, capacity, n_ants=10, n_iterations=50, alpha=1.0, beta=2.0, evaporation=0.5):
    n_items = len(items)
    pheromone = np.ones(n_items)  # Inisialisasi feromon
    best_value = 0
    best_combination = None
    history = []

    for iteration in range(n_iterations):
        all_ant_values = []
        for ant in range(n_ants):
            current_weight = 0
            current_value = 0
            selected_items = np.zeros(n_items)
            
            # Senarai barang yang belum dipilih
            available_indices = list(range(n_items))
            
            while available_indices:
                # Kira kebarangkalian (Pheromone ^ alpha * Heuristic ^ beta)
                probs = []
                for i in available_indices:
                    if current_weight + weights[i] <= capacity:
                        heuristic = values[i] / weights[i]
                        prob = (pheromone[i] ** alpha) * (heuristic ** beta)
                        probs.append(prob)
                    else:
                        probs.append(0)
                
                if sum(probs) == 0: break
                
                probs = np.array(probs) / sum(probs)
                next_item = np.random.choice(available_indices, p=probs)
                
                selected_items[next_item] = 1
                current_weight += weights[next_item]
                current_value += values[next_item]
                available_indices.remove(next_item)

            # Kemas kini terbaik global
            if current_value > best_value:
                best_value = current_value
                best_combination = selected_items.copy()
            
            all_ant_values.append(current_value)

        # Update Pheromone (Evaporation + Deposition)
        pheromone *= (1 - evaporation)
        if best_combination is not None:
            for i in range(n_items):
                if best_combination[i] == 1:
                    pheromone[i] += (best_value / capacity) # Deposit berdasarkan kualiti

        history.append(max(all_ant_values))
    
    return best_value, best_combination, history

# --- ANTARA MUKA STREAMLIT ---
st.set_page_config(page_title="ACO Knapsack Solver", layout="wide")
st.title("🐜 Ant Colony Optimization: Knapsack Problem")

st.sidebar.header("Konfigurasi Algoritma")
capacity = st.sidebar.number_input("Kapasiti Beg (Weight)", min_value=1, value=50)
n_ants = st.sidebar.slider("Bilangan Semut", 1, 50, 10)
n_iterations = st.sidebar.slider("Bilangan Iterasi", 10, 200, 50)
alpha = st.sidebar.slider("Pengaruh Feromon (Alpha)", 0.0, 5.0, 1.0)
beta = st.sidebar.slider("Pengaruh Heuristik (Beta)", 0.0, 5.0, 2.0)

uploaded_file = st.file_uploader("Muat naik Dataset CSV (item, value, weight)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Item")
    st.write(df)

    if st.button("Jalankan Algoritma ACO"):
        with st.spinner('Semut sedang mencari penyelesaian...'):
            start_time = time.time()
            best_val, best_comb, history = aco_knapsack(
                df['item'].values, df['weight'].values, df['value'].values, 
                capacity, n_ants, n_iterations, alpha, beta
            )
            end_time = time.time()

        # --- PAPARAN KEPUTUSAN ---
        st.success(f"Selesai dalam {end_time - start_time:.2f} saat!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nilai Maksimum (Profit)", f"{best_val}")
            selected_df = df[best_comb == 1]
            st.write("**Barang yang dipilih:**")
            st.table(selected_df)

        with col2:
            st.write("**Graf Penumpuan (Convergence Graph)**")
            fig, ax = plt.subplots()
            ax.plot(history, color='orange', linewidth=2)
            ax.set_xlabel("Iterasi")
            ax.set_ylabel("Nilai Terbaik")
            ax.set_title("Peningkatan Nilai vs Masa")
            st.pyplot(fig)
