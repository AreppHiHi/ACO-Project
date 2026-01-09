import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# Ant Colony Optimization (ACO)
# ==============================
class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, rho):
        self.distances = distances
        self.n_nodes = distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.pheromone = np.ones((self.n_nodes, self.n_nodes))
        self.best_cost = float("inf")
        self.best_path = None
        self.convergence = []

    def _path_cost(self, path):
        return sum(
            self.distances[path[i]][path[i + 1]]
            for i in range(len(path) - 1)
        ) + self.distances[path[-1]][path[0]]

    def _probability(self, current, visited):
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0

        heuristic = 1 / (self.distances[current] + 1e-10)
        heuristic[list(visited)] = 0

        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return prob / prob.sum()

    def run(self):
        for _ in range(self.n_iterations):
            paths, costs = [], []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n_nodes)
                path = [start]
                visited = set(path)

                while len(path) < self.n_nodes:
                    probs = self._probability(path[-1], visited)
                    next_node = np.random.choice(self.n_nodes, p=probs)
                    path.append(next_node)
                    visited.add(next_node)

                cost = self._path_cost(path)
                paths.append(path)
                costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path

            self.convergence.append(self.best_cost)
            self.pheromone *= (1 - self.rho)

            for path, cost in zip(paths, costs):
                for i in range(len(path) - 1):
                    self.pheromone[path[i]][path[i + 1]] += 1 / cost

        return self.best_path, self.best_cost, self.convergence


# ==============================
# Streamlit Dashboard
# ==============================
st.set_page_config(page_title="ACO CSV Dashboard", layout="wide")

st.title("🐜 Ant Colony Optimization with CSV Upload")
st.markdown("""
Upload a **CSV file**, tune **ACO parameters**, and visualize  
**convergence behavior, solution quality, and trade-offs** interactively.
""")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error("CSV must contain at least TWO numeric columns.")
    else:
        points = numeric_df.values

        # Distance matrix
        n_nodes = len(points)
        distances = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                distances[i][j] = np.linalg.norm(points[i] - points[j])

        # Sidebar parameters
        st.sidebar.header("ACO Parameters")
        n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
        n_iterations = st.sidebar.slider("Iterations", 20, 300, 100)
        alpha = st.sidebar.slider("Alpha (pheromone)", 0.1, 5.0, 1.0)
        beta = st.sidebar.slider("Beta (heuristic)", 0.1, 5.0, 2.0)
        rho = st.sidebar.slider("Evaporation Rate (ρ)", 0.01, 0.9, 0.5)

        if st.button("🚀 Run ACO"):
            aco = AntColony(
                distances,
                n_ants,
                n_iterations,
                alpha,
                beta,
                rho
            )

            best_path, best_cost, convergence = aco.run()

            col1, col2 = st.columns(2)

            # Convergence plot
            with col1:
                st.subheader("📉 Convergence Curve")
                fig1, ax1 = plt.subplots()
                ax1.plot(convergence)
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Best Cost")
                ax1.grid(True)
                st.pyplot(fig1)

            # Best path visualization
            with col2:
                st.subheader("🗺️ Best Solution Path")
                fig2, ax2 = plt.subplots()
                route = best_path + [best_path[0]]
                ax2.plot(
                    points[route, 0],
                    points[route, 1],
                    marker="o"
                )
                ax2.set_title(f"Best Cost = {best_cost:.4f}")
                ax2.grid(True)
                st.pyplot(fig2)

            st.subheader("📊 Performance Summary")
            st.write(f"**Nodes:** {n_nodes}")
            st.write(f"**Best Cost:** {best_cost:.4f}")
            st.write(f"**Alpha:** {alpha}, **Beta:** {beta}, **ρ:** {rho}")

            st.success("ACO optimization completed successfully!")

else:
    st.info("⬆️ Please upload a CSV file to begin.")
