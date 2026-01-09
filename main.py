import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- LOGIK ALGORITMA ACO ---
def run_aco(df, item_col, val_col, wt_col, capacity, n_ants, n_iter, alpha=1.0, beta=2.0, evap=0.5):
    items = df[item_col].values
    values = pd.to_numeric(df[val_col], errors='coerce').fillna(0).values
    weights = pd.to_numeric(df[wt_col], errors='coerce').fillna(0).values
    
    n_items = len(items)
    pheromone = np.full(n_items, 0.1) 
    best_value = 0
    best_comb = None
    history = []

    for _ in range(n_iter):
        current_iter_best = 0
        for _ in range(n_ants):
            cur_w, cur_v = 0, 0
            temp_comb = np.zeros(n_items)
            avail = list(range(n_items))
            
            while avail:
                probs = []
                valid_idx = []
                for i in avail:
                    if cur_w + weights[i] <= capacity:
                        h = values[i] / weights[i] if weights[i] > 0 else values[i]
                        p = (pheromone[i]**alpha) * (h**beta)
                        probs.append(p)
                        valid_idx.append(i)
                
                if not probs or sum(probs) == 0: break
                
                probs = np.array(probs) / sum(probs)
                pick = np.random.choice(valid_idx, p=probs)
                temp_comb[pick] = 1
                cur_w += weights[pick]
                cur_v += values[pick]
                avail.remove(pick)

            if cur_v > best_value:
                best_value = cur_v
                best_comb = temp_comb.copy()
            current_iter_best = max(current_iter_best, cur_v)

        pheromone *= (1 - evap)
        if best_comb is not None:
            for i in range(n_items):
                if best_comb[i] == 1: pheromone[i] += (best_value / 100)
        history.append(best_value)
    
    return best_value, best_comb, history

# --- ANTARA MUKA (UI) STREAMLIT ---
st.set_page_config(page_title="Upload Dataset & Run ACO", layout="wide")

st.title("📁 ACO Dataset Solver")
st.write("Sila muat naik fail CSV anda untuk menjalankan algoritma Ant Colony Optimization.")

# 1. BAHAGIAN UPLOAD FILE
uploaded_file = st.file_uploader("Pilih fail CSV anda (Contoh: dataset.csv)", type=["csv"])

if uploaded_file is not None:
    # Baca fail CSV yang diupload
    df = pd.read_csv(uploaded_file)
    # Bersihkan nama kolum daripada sebarang ruang (space)
    df.columns = df.columns.str.strip()
    
    st.success("Fail berjaya dimuat naik!")
    st.subheader("Data daripada CSV anda:")
    st.dataframe(df.head(10)) # Tunjukkan 10 baris pertama

    # 2. PEMILIHAN KOLUM (Map dataset user ke algorithm)
    st.sidebar.header("Konfigurasi Data")
    cols = df.columns.tolist()
    
    item_name = st.sidebar.selectbox("Pilih Kolum Nama Barang", cols)
    val_name = st.sidebar.selectbox("Pilih Kolum Nilai (Value)", cols)
    wt_name = st.sidebar.selectbox("Pilih Kolum Berat (Weight)", cols)
    
    st.sidebar.header("Parameter ACO")
    cap = st.sidebar.number_input("Kapasiti Knapsack", min_value=1, value=100)
    ants = st.sidebar.slider("Bilangan Semut", 5, 50, 15)
    iters = st.sidebar.slider("Bilangan Iterasi", 10, 200, 50)

    # 3. BUTANG RUN
    if st.button("🚀 Jalankan Algoritma ACO"):
        with st.spinner('Semut sedang menganalisis dataset anda...'):
            res_val, res_comb, history = run_aco(df, item_name, val_name, wt_name, cap, ants, iters)
            
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Keputusan Terbaik")
                st.metric("Total Profit", f"{res_val}")
                st.write("**Barang yang terpilih:**")
                st.write(df[res_comb == 1])
            
            with col2:
                st.subheader("Graf Penumpuan (Convergence)")
                
                fig, ax = plt.subplots()
                ax.plot(history, color='blue', linewidth=2)
                ax.set_xlabel("Iterasi")
                ax.set_ylabel("Nilai Terbaik")
                st.pyplot(fig)
else:
    st.info("Menunggu fail CSV dimuat naik. Sila gunakan menu di atas.")
