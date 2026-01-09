import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# Ant Colony Optimization (ACO) – OPTIMIZATION ENGINE
# =====================================================
class AntColonyOptimizer:
    """
    Objective:
    Minimize total path cost over all nodes (TSP-style optimization)
    """

    def __init__(self, distance_matrix, n_ants, n_iterations, alpha, beta, rho):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha    # pheromone influence
        self.beta = beta      # heuristic influence
        self.rho = rho        # evaporation rate

        self.pheromone = np.ones((self.n, self.n))
        self.best_solution = None
        self.best_fitness = float("inf")
        self.convergence = []

    # ===============================
    # Objective (Fitness) Function
    # ===============================
    def fitness(self, solution):
        cost = 0
        for i in range(len(solution) - 1):
            cost += self.D[solution[i]][solution[i + 1]]
        cost += self.D[solution[-1]][solution[0]]
        return cost

    # ===============================
    # Probabilistic Transition Rule
    # ===============================
    def transition_prob(self, current, visited):
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0

        heuristic = 1 / (self.D[current] + 1e-10)
        heuristic[list(visited)] = 0

        probability = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return probability / probability.sum()

    # ===============================
    # Main Optimization Loop
    # ===============================
    def optimize(self):
        for _ in range(self.n_iterations):
            solutions = []
            fitness_values = []

            for _ in range(self.n_ants):
                start = np.random.randint(self.n)
                solution = [start]
                visited = set(solution)

                while len(solution) < self.n:
                    probs = self.transition_prob(solution[-1], visited)
                    next_node = np.random.choice(self.n, p=probs)
                    solution.append(next_node)
                    visited.add(next_node)

                fit = self.fitness(solution)
                solutions.append(solution)
                fitness_values.append(fit)

                # Global best update (OPTIMIZATION STEP)
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = solution

            # Track optimization progress
            self.convergence.append(self.best_fitness)

            # Evaporation
            self.pheromone *= (1 - self.rho)

            # Reinforcement (learning from good solutions)
            for sol, fit in zip(solutions, fitness_values):
                for i in range(len(sol) - 1):
                    self.pheromone[sol[i]][sol[i + 1]] += 1 / fit

        return self.best_solution, self.best_fitness, self.convergence


# =====================================================
# Streamlit Dashboard – USER INTERFACE
# =====================================================
st.set_page_config(page_title="ACO Optimization Dashboard", layout="wide")
st.title("🐜 Ant Colony Optimization (ACO) – Full Optimization System")

st.markdown("""
This system performs **true optimization** using **Ant Colony Optimization (ACO)**.
The objective is to **minimize the total traversal cost** between data points uploaded
via a CSV file.
""")

# ===============================
# CSV Upload
# ===============================
uploaded_file = st.file_uploader("📂 Upload CSV File (numeric data only)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        st.error("CSV must contain at least TWO numeric columns.")
        st.stop()

    # Convert CSV rows to nodes
    points = numeric_df.values
    n_nodes = len(points)

    # Distance matrix (problem formulation)
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            D[i][j] = np.linalg.norm(points[i] - points[j])

    # ===============================
    # Optimization Parameters
    # ===============================
    st.sidebar.header("ACO Optimization Parameters")

    n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
    n_iterations = st.sidebar.slider("Iterations", 20, 300, 100)
    alpha = st.sidebar.slider("Alpha (Pheromone Importance)", 0.1, 5.0, 1.0)
    beta = st.sidebar.slider("Beta (Heuristic Importance)", 0.1, 5.0, 2.0)
    rho = st.sidebar.slider("Evaporation Rate (ρ)", 0.01, 0.9, 0.5)

    # ===============================
    # Run Optimization
    # ===============================
    if st.button("🚀 Run ACO Optimization"):
        optimizer = AntColonyOptimizer(
            distance_matrix=D,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho
        )

        best_solution, best_fitness, convergence = optimizer.optimize()

        # ===============================
        # Visualization
        # ===============================
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Optimization Convergence")
            fig1, ax1 = plt.subplots()
            ax1.plot(convergence)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Best Objective Value")
            ax1.grid(True)
            st.pyplot(fig1)

        with col2:
            st.subheader("🗺️ Optimized Path")
            fig2, ax2 = plt.subplots()
            route = best_solution + [best_solution[0]]
            ax2.plot(
                points[route, 0],
                points[route, 1],
                marker="o"
            )
            ax2.set_title(f"Minimum Cost = {best_fitness:.4f}")
            ax2.grid(True)
            st.pyplot(fig2)

        # ===============================
        # Optimization Summary
        # ===============================
        st.subheader("📊 Optimization Results")
        st.write(f"**Number of Nodes:** {n_nodes}")
        st.write(f"**Optimal Cost:** {best_fitness:.4f}")
        st.write(f"**Alpha:** {alpha}, **Beta:** {beta}, **ρ:** {rho}")

        st.success("Optimization completed successfully!")

else:
    st.info("⬆️ Upload a CSV file to start optimization.")
