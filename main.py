import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ACO Algorithm Dashboard",
    page_icon="🐜",
    layout="wide"
)

# --- Helper Functions ---
def generate_cities(n_cities, seed=42):
    """Generates random 2D coordinates for cities."""
    np.random.seed(seed)
    return np.random.rand(n_cities, 2) * 100

def calculate_distance_matrix(cities):
    """Calculates Euclidean distance between all pairs of cities."""
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])
    # Avoid division by zero for self-distance
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix

def path_distance(path, dist_matrix):
    """Calculates total distance of a path."""
    dist = 0
    for i in range(len(path) - 1):
        dist += dist_matrix[path[i]][path[i+1]]
    dist += dist_matrix[path[-1]][path[0]] # Return to start
    return dist

# --- ACO Algorithm Class ---
class AntColonyOptimization:
    def __init__(self, cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
        self.cities = cities
        self.dist_matrix = calculate_distance_matrix(cities)
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Distance importance
        self.evaporation_rate = evaporation_rate
        self.Q = Q          # Pheromone constant
        
        # Initialize Pheromones (small constant value)
        self.pheromone = np.ones((self.n_cities, self.n_cities))

    def run(self, progress_bar, chart_placeholder, map_placeholder):
        best_path = None
        best_distance = np.inf
        history = []

        for it in range(self.n_iterations):
            all_paths = []
            all_distances = []

            # 1. Move Ants
            for ant in range(self.n_ants):
                path = self._construct_path()
                dist = path_distance(path, self.dist_matrix)
                all_paths.append(path)
                all_distances.append(dist)
                
                if dist < best_distance:
                    best_distance = dist
                    best_path = path

            # 2. Update Pheromones
            self._update_pheromones(all_paths, all_distances)

            # 3. Store History
            history.append(best_distance)

            # 4. Visualization Updates (Every few iterations to save rendering time)
            if it % 2 == 0 or it == self.n_iterations - 1:
                # Update Progress
                progress_bar.progress((it + 1) / self.n_iterations)
                
                # Update Convergence Chart
                chart_placeholder.line_chart(history)
                
                # Update Map
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(f"Iteration: {it+1} | Best Distance: {best_distance:.2f}")
                
                # Plot Cities
                ax.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=50, zorder=5)
                
                # Plot Best Path
                if best_path is not None:
                    path_coords = self.cities[best_path]
                    path_coords = np.vstack([path_coords, path_coords[0]]) # Close loop
                    ax.plot(path_coords[:, 0], path_coords[:, 1], c='blue', linewidth=2, alpha=0.7)
                
                ax.axis('off')
                map_placeholder.pyplot(fig)
                plt.close(fig)
                
                time.sleep(0.01) # Small delay for visual effect

        return best_path, best_distance, history

    def _construct_path(self):
        path = [np.random.randint(self.n_cities)]
        visited = set(path)

        for _ in range(self.n_cities - 1):
            current = path[-1]
            probabilities = self._calculate_probabilities(current, visited)
            next_city = self._roulette_wheel_selection(probabilities)
            path.append(next_city)
            visited.add(next_city)
        
        return path

    def _calculate_probabilities(self, current, visited):
        pheromone = np.power(self.pheromone[current], self.alpha)
        heuristic = np.power(1.0 / self.dist_matrix[current], self.beta)
        
        # Mask visited cities
        mask = np.ones(self.n_cities)
        mask[list(visited)] = 0
        
        probabilities = pheromone * heuristic * mask
        sum_prob = np.sum(probabilities)
        
        if sum_prob == 0:
            # Fallback if numerical issues (shouldn't happen often)
            probabilities = mask
            sum_prob = np.sum(probabilities)
            
        return probabilities / sum_prob

    def _roulette_wheel_selection(self, probabilities):
        return np.random.choice(range(self.n_cities), p=probabilities)

    def _update_pheromones(self, paths, distances):
        # Evaporation
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Deposit
        for path, dist in zip(paths, distances):
            deposit = self.Q / dist
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += deposit
                self.pheromone[path[i+1]][path[i]] += deposit # Symmetric
            # Last link back to start
            self.pheromone[path[-1]][path[0]] += deposit
            self.pheromone[path[0]][path[-1]] += deposit

# --- Streamlit Layout ---

st.title("🐜 Ant Colony Optimization (ACO) Dashboard")
st.markdown("""
This dashboard visualizes the **Ant Colony Optimization** algorithm solving the **Traveling Salesperson Problem (TSP)**. 
Adjust the parameters in the sidebar to see how they affect convergence speed and solution quality.
""")

# --- Sidebar: Parameters ---
st.sidebar.header("⚙️ Algorithm Parameters")

n_cities = st.sidebar.slider("Number of Cities", 5, 50, 15)
n_ants = st.sidebar.slider("Number of Ants", 2, 50, 10)
n_iterations = st.sidebar.slider("Iterations", 10, 200, 50)

st.sidebar.markdown("---")
st.sidebar.subheader("Hyperparameters")

alpha = st.sidebar.slider("Alpha (Pheromone Importance)", 0.0, 5.0, 1.0, help="Higher alpha makes ants follow pheromones more strictly.")
beta = st.sidebar.slider("Beta (Heuristic Importance)", 0.0, 5.0, 2.0, help="Higher beta makes ants prefer closer cities (greedy approach).")
evaporation_rate = st.sidebar.slider("Evaporation Rate", 0.01, 1.0, 0.5, help="Rate at which pheromone trails disappear.")
seed = st.sidebar.number_input("Random Seed", value=42)

# --- Main Execution Area ---

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data Setup")
    if st.button("Generate New Map"):
        seed = np.random.randint(0, 1000)
    
    cities = generate_cities(n_cities, seed)
    
    # Preview Initial Map
    fig_init, ax_init = plt.subplots(figsize=(4, 3))
    ax_init.scatter(cities[:, 0], cities[:, 1], c='gray')
    ax_init.set_title("City Locations")
    ax_init.axis('off')
    st.pyplot(fig_init)

with col2:
    st.subheader("Algorithm Performance")
    start_btn = st.button("🚀 Run Simulation", type="primary")

# Placeholders for dynamic updates
st.markdown("---")
metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    st.markdown("### 🗺️ Live Optimal Path")
    map_placeholder = st.empty()
with metric_col2:
    st.markdown("### 📉 Convergence Curve")
    chart_placeholder = st.empty()

progress_bar = st.progress(0)

if start_btn:
    # Initialize ACO
    aco = AntColonyOptimization(
        cities=cities,
        n_ants=n_ants,
        n_iterations=n_iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        Q=100
    )
    
    # Run
    best_path, best_dist, history = aco.run(progress_bar, chart_placeholder, map_placeholder)
    
    st.success(f"Simulation Complete! Best Distance Found: **{best_dist:.2f}**")
    
    # Explanation of results
    with st.expander("See Detailed Metrics"):
        st.write(f"**Initial Distance:** {history[0]:.2f}")
        st.write(f"**Improvement:** {((history[0] - best_dist)/history[0])*100:.1f}%")
