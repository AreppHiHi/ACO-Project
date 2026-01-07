import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="ACO Knapsack Solver", layout="wide")

# --- ACO for Knapsack Logic ---
class ACO_Knapsack:
    def __init__(self, values, weights, capacity, n_ants=10, iterations=50, alpha=1, beta=2, evaporation=0.5):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.n_items = len(values)
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance (Value/Weight ratio)
        self.evaporation = evaporation
        self.pheromone = np.ones(self.n_items)
        self.heuristic = values / weights  # Efficiency of each item

    def run(self):
        best_value = 0
        best_selection = None
        history = []

        placeholder = st.empty()

        for i in range(self.iterations):
            solutions = []
            values_found = []

            for ant in range(self.n_ants):
                selection = self._build_solution()
                total_v = np.sum(selection * self.values)
                total_w = np.sum(selection * self.weights)
                
                if total_w <= self.capacity:
                    solutions.append(selection)
                    values_found.append(total_v)
                    if total_v > best_value:
                        best_value = total_v
                        best_selection = selection

            # Update Pheromones
            self.pheromone *= (1 - self.evaporation)
            for sol, val in zip(solutions, values_found):
                self.pheromone += (sol * (val / best_value if best_value > 0 else 0))

            history.append(best_value)
            
        return best_selection, best_value, history

    def _build_solution(self):
        solution = np.zeros(self.n_items)
        current_weight = 0
        items_indices = np.arange(self.n_items)
        
        # Shuffle to give each ant a different starting perspective
        np.random.shuffle(items_indices)

        for idx in items_indices:
            if current_weight + self.weights[idx] <= self.capacity:
                # Probability of picking item based on pheromone and heuristic
                prob = (self.pheromone[idx]**self.alpha) * (self.heuristic[idx]**self.beta)
                # Random choice vs Probability (Simplified for Knapsack)
                if np.random.random() < (prob / (prob + 1)): 
                    solution[idx] = 1
                    current_weight += self.weights[idx]
        return solution

# --- Streamlit UI ---
st.title("🐜 ACO Knapsack Problem Solver")

with st.sidebar:
    st.header("Settings")
    capacity = st.number_input("Knapsack Capacity", value=50)
    ants = st.slider("Ants", 1, 50, 10)
    iters = st.slider("Iterations", 10, 200, 50)
    
st.subheader("1. Load Dataset")
uploaded_file = st.file_uploader("Upload Knapsack CSV (columns: item_name, value, weight)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())
    
    if st.button("🚀 Run ACO Optimization"):
        aco = ACO_Knapsack(
            values=df['value'].values, 
            weights=df['weight'].values, 
            capacity=capacity,
            n_ants=ants,
            iterations=iters
        )
        
        best_sol, max_val, history = aco.run()
        
        # Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Value Found", f"${max_val}")
            df['Selected'] = best_sol.astype(bool)
            st.write("Selected Items:", df[df['Selected'] == True])
            
        with col2:
            st.line_chart(history)
            st.caption("Convergence Curve (Max Value vs Iteration)")
else:
    st.info("Please upload a CSV file to begin. The file should have 'value' and 'weight' columns.")
