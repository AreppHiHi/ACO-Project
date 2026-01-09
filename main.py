import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# ANT COLONY OPTIMIZATION (ACO) - OPTIMIZATION CORE
# =====================================================
class AntColonyOptimizer:
    """
    Objective:
    Minimize total traversal cost (TSP-style optimization)
    """

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

    # -----------------------------
    # Objective Function
    # -----------------------------
    def objective(self, solution):
        cost = 0
        for i in range(len(solution) - 1):
            cost += self.D[solution[i]][solution[i + 1]]
        cost += self.D[solution[-1]][solution[0]]
        return cost

    # -----------------------------
    # Transition Probability
    # -----------------------------
    def transition_probability(self, current, visited):
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0

        heuristic = 1 / (self.D[current] + 1e-10)
        heuristic[list(visited)] = 0

        prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return prob / prob.sum()

    # -----------------------------
    # Optimization Process
    # -----------------------------
    def optimize(self):
        for _ in range(self.n_iterations):
            solutions = []
            costs = []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n)
                solution = [start]
                visited = set(solution)

                while len(solution) < self.n:
                    probs = self.transition_probability(solution[-1], visited)
                    next_node = np.random.choice(self.n, p=probs)
                    solution.append(next_node)
                    visited.add(next_node)

                cost = self.objective(solution)
                solutions.append(solution)
                costs.append(cost)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = solution

            self.convergence.append(self.best_cost)

            # Pheromone evaporation
            self.pheromone *= (1 - self.rho)

            # Pheromone reinforcement
            for sol, cost in zip(solutions, costs):
                for i in range(len(sol) - 1):
                    self.pheromone[sol[i]][sol[i + 1]] += 1 / cost

        return self.best_solution, self.best_cost, self.convergence


# =====================================================
# STREAMLIT DASHBOARD
# =====================================================
st.set_page_config(page_title="ACO Optimization Results", layout="wide")

st.title("🐜 Ant Colony Optimization (ACO) – Full Optimization System")
st.markdown("""
This dashboard demonstrates **true optimization using Ant Colony Optimization (ACO)**.
The goal is to **minimize the total traversal cost** of data points loaded from a CSV file.
""")

# -----------------------------
# CSV UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file (numeric columns only)", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Upload a CSV file to begin optimization.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("📄 Dataset Preview")
st.dataframe(df)

numeric_df = df.select_dtypes(include=[np.number])

if numeric_df.shape[1] < 2:
    st.error("CSV must contain at least TWO numeric columns.")
    st.stop()

points = numeric_df.values
n_nodes = len(points)

# -----------------------------
# PROBLEM FORMULATION
# -----------------------------
D = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        D[i][j] = np.linalg.norm(points[i] - points[j])

# -----------------------------
# PARAMETER CONTROLS
# -----------------------------
st.sidebar.header("ACO Parameters")

n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
n_iterations = st.sidebar.slider("Iterations", 20, 300, 100)
alpha = st.sidebar.slider("Alpha (pheromone influence)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic influence)", 0.1, 5.0, 2.0)
rho = st.sidebar.slider("Evaporation rate (ρ)", 0.01, 0.9, 0.5)

# -----------------------------
# RUN OPTIMIZATION
# -----------------------------
if st.button("🚀 Run ACO Optimization"):
    optimizer = AntColonyOptimizer(D, n_ants, n_iterations, alpha, beta, rho)
    best_solution, best_cost, convergence = optimizer.optimize()

    col1, col2 = st.columns(2)

    # -----------------------------
    # RESULT VISUAL 1: CONVERGENCE
    # -----------------------------
    with col1:
        st.subheader("📉 Optimization Convergence Curve")
        fig1, ax1 = plt.subplots()
        ax1.plot(convergence)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Best Objective Value")
        ax1.grid(True)
        st.pyplot(fig1)

    # -----------------------------
    # RESULT VISUAL 2: BEST SOLUTION
    # -----------------------------
    with col2:
        st.subheader("🗺️ Optimized Solution Path")
        fig2, ax2 = plt.subplots()
        route = best_solution + [best_solution[0]]
        ax2.plot(points[route, 0], points[route, 1], marker="o")
        ax2.set_title(f"Minimum Cost = {best_cost:.4f}")
        ax2.grid(True)
        st.pyplot(fig2)

    # -----------------------------
    # NUMERICAL RESULTS
    # -----------------------------
    st.subheader("📊 Optimization Results Summary")
    st.write(f"**Total Nodes:** {n_nodes}")
    st.write(f"**Optimal Cost:** {best_cost:.4f}")
    st.write(f"**Alpha:** {alpha}, **Beta:** {beta}, **ρ:** {rho}")

    st.success("Optimization completed successfully!")
