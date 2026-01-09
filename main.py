import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title="ACO Optimization Results", layout="wide")

# =====================================================
# ANT COLONY OPTIMIZATION (ACO) - Optimized Logic
# =====================================================
class AntColonyOptimizer:
    def __init__(self, distance_matrix, n_ants, n_iterations, alpha, beta, rho):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone = np.ones((self.n, self.n))
        self.best_solution = None
        self.best_cost = float("inf")
        self.convergence = []

    def objective(self, solution):
        # Menggunakan indexing numpy untuk kelajuan pengiraan kos
        idx1 = solution
        idx2 = solution[1:] + [solution[0]]
        return np.sum(self.D[idx1, idx2])

    def optimize(self):
        # Pre-calculate heuristic (1/distance) untuk kelajuan
        eta = 1.0 / (self.D + np.eye(self.n) * 1e-10)
        
        for _ in range(self.n_iterations):
            all_solutions = []
            all_costs = []

            for _ in range(self.n_ants):
                current = np.random.randint(self.n)
                solution = [current]
                visited = {current}

                for _ in range(self.n - 1):
                    # Kira kebarangkalian menggunakan operasi vektor
                    phi = self.pheromone[current]
                    h = eta[current]
                    
                    # Mask node yang sudah dilawati
                    mask = np.ones(self.n, dtype=bool)
                    mask[list(visited)] = False
                    
                    probs = (phi[mask] ** self.alpha) * (h[mask] ** self.beta)
                    probs /= probs.sum()
                    
                    next_node = np.random.choice(np.where(mask)[0], p=probs)
                    solution.append(next_node)
                    visited.add(next_node)
                    current = next_node

                cost = self.objective(solution)
                all_solutions.append(solution)
                all_costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

            self.convergence.append(self.best_cost)

            # Update pheromone (Vektorasi)
            self.pheromone *= (1 - self.rho)
            for sol, cost in zip(all_solutions, all_costs):
                for i in range(self.n - 1):
                    self.pheromone[sol[i], sol[i+1]] += 1.0 / cost
                self.pheromone[sol[-1], sol[0]] += 1.0 / cost

        return self.best_solution, self.best_cost, self.convergence

# =====================================================
# PENGIRAAN JARAK (VEKTORASI)
# =====================================================
@st.cache_data
def compute_distance_matrix(points):
    # Pengiraan jarak Euclidean menggunakan broadcasting numpy (sangat laju)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

# =====================================================
# UI TANPA EMOJI
# =====================================================
st.title("Ant Colony Optimization (ACO) Dashboard")
st.markdown("""
Aplikasi ini menjalankan optimasi Ant Colony Optimization (ACO) 
untuk meminimumkan kos perjalanan bagi titik data yang dimuat naik.
""")

uploaded_file = st.file_uploader("Muat naik fail CSV (lajur numerik sahaja)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error("Fail CSV memerlukan sekurang-kurangnya dua lajur numerik.")
        st.stop()

    points = numeric_df.values
    D = compute_distance_matrix(points)

    # Sidebar Parameters
    st.sidebar.header("Parameter ACO")
    n_ants = st.sidebar.slider("Bilangan Semut", 5, 50, 20)
    n_iterations = st.sidebar.slider("Iterasi", 10, 200, 50)
    alpha = st.sidebar.slider("Alpha (Pheromone)", 0.1, 5.0, 1.0)
    beta = st.sidebar.slider("Beta (Heuristik)", 0.1, 5.0, 2.0)
    rho = st.sidebar.slider("Kadar Penguapan (Rho)", 0.01, 0.9, 0.5)

    if st.button("Jalankan Optimasi"):
        np.random.seed(42)
        
        with st.spinner("Sedang memproses..."):
            optimizer = AntColonyOptimizer(D, n_ants, n_iterations, alpha, beta, rho)
            best_solution, best_cost, convergence = optimizer.optimize()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Kurva Konvergens")
            fig1, ax1 = plt.subplots()
            ax1.plot(convergence, color='blue')
            ax1.set_xlabel("Iterasi")
            ax1.set_ylabel("Kos Objektif")
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.subheader("Laluan Optimasi")
            fig2, ax2 = plt.subplots()
            route = best_solution + [best_solution[0]]
            ax2.plot(points[route, 0], points[route, 1], marker="o", linestyle="-", color='red')
            ax2.set_title(f"Kos Minimum: {best_cost:.4f}")
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Ringkasan Keputusan")
        st.write(f"Jumlah Titik: {len(points)}")
        st.write(f"Kos Paling Optimum: {best_cost:.4f}")
        st.success("Proses selesai.")
else:
    st.info("Sila muat naik fail CSV untuk memulakan.")
