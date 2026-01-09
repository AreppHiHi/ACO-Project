import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Ant Colony Optimization (ACO)
# ==============================
class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha, beta, rho):
        self.distances = distances
        self.n_cities = distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha      # pheromone importance
        self.beta = beta        # heuristic importance
        self.rho = rho          # evaporation rate

        self.pheromone = np.ones((self.n_cities, self.n_cities))
        self.best_cost = float("inf")
        self.best_path = None
        self.convergence = []

    def _path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distances[path[i]][path[i + 1]]
        cost += self.distances[path[-1]][path[0]]
        return cost

    def _probability(self, current, visited):
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0

        heuristic = 1 / (self.distances[current] + 1e-10)
        heuristic[list(visited)] = 0

        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return prob / prob.sum()

    def run(self):
        for _ in range(self.n_iterations):
            paths = []
            costs = []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n_cities)
                path = [start]
                visited = set(path)

                while len(path) < self.n_cities:
                    probs = self._probability(path[-1], visited)
                    next_city = np.random.choice(self.n_cities, p=probs)
                    path.append(next_city)
                    visited.add(next_city)

                cost = self._path_cost(path)
                paths.append(path)
                costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path

            # Save convergence data
            self.convergence.append(self.best_cost)

            # Evaporation
            self.pheromone *= (1 - self.rho)

            # Pheromone update
            for path, cost in zip(paths, costs):
                for i in range(len(path) - 1):
                    self.pheromone[path[i]][path[i + 1]] += 1 / cost

        return self.best_path, self.best_cost, self.convergence


# ==============================
# Streamlit Dashboard
# ==============================
st.set_page_config(page_title="ACO Interactive Dashboard", layout="wide")

st.title("🐜 Ant Colony Optimization (ACO) Interactive Dashboard")
st.markdown("""
This dashboard allows **dynamic exploration of ACO parameters**, visualization of
**algorithm convergence**, and analysis of **solution quality trade-offs**.
""")

# Sidebar controls
st.sidebar.header("ACO Parameters")

n_cities = st.sidebar.slider("Number of Cities", 5, 30, 15)
n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
n_iterations = st.sidebar.slider("Iterations", 20, 300, 100)
alpha = st.sidebar.slider("Alpha (Pheromone Influence)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (Heuristic Influence)", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Evaporation Rate (ρ)", 0.01, 0.9, 0.5)

# Generate problem (TSP)
np.random.seed(42)
cities = np.random.rand(n_cities, 2)

distances = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        distances[i][j] = np.linalg.norm(cities[i] - cities[j])

# Run algorithm
if st.button("🚀 Run ACO"):
    aco = AntColony(
        distances=distances,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        rho=rho
    )

    best_path, best_cost, convergence = aco.run()

    col1, col2 = st.columns(2)

    # Convergence curve
    with col1:
        st.subheader("📉 Convergence Curve")
        fig1, ax1 = plt.subplots()
        ax1.plot(convergence)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Solution Cost")
        ax1.grid(True)
        st.pyplot(fig1)

    # Best route visualization
    with col2:
        st.subheader("🗺️ Best Route Found")
        fig2, ax2 = plt.subplots()
        route = best_path + [best_path[0]]
        ax2.plot(
            cities[route, 0],
            cities[route, 1],
            marker="o"
        )
        ax2.set_title(f"Best Cost = {best_cost:.4f}")
        ax2.grid(True)
        st.pyplot(fig2)

    # Metrics summary
    st.subheader("📊 Performance Summary")
    st.write(f"**Number of Cities:** {n_cities}")
    st.write(f"**Best Solution Cost:** {best_cost:.4f}")
    st.write(f"**Alpha (α):** {alpha}, **Beta (β):** {beta}, **Evaporation (ρ):** {rho}")

    st.success("ACO optimization completed successfully!")
