import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# --- 1. ACO ALGORITHM (CORE LOGIC) ---
def aco_knapsack(items, weights, values, capacity, n_ants=10, n_iterations=50, alpha=1.0, beta=2.0, evaporation=0.5):
    """
    Executes the Ant Colony Optimization algorithm for the Knapsack Problem.
    """
    n_items = len(items)
    # Initialize pheromones with a small positive value
    pheromone = np.full(n_items, 0.1) 
    best_value = 0
    best_combination = None
    history = []

    for iteration in range(n_iterations):
        all_ant_values = []
        
        for ant in range(n_ants):
            current_weight = 0
            current_value = 0
            selected_items = np.zeros(n_items)
            available_indices = list(range(n_items))
            
            # Ant builds a solution
            while available_indices:
                probs = []
                valid_indices = []
                
                for i in available_indices:
                    # Check if item fits in the knapsack
                    if current_weight + weights[i] <= capacity:
                        # Heuristic: Value to Weight Ratio (Greedy factor)
                        heuristic = values[i] / weights[i] if weights[i] > 0 else values[i]
                        
                        # ACO Formula: Probability = (Pheromone^alpha) * (Heuristic^beta)
                        p = (pheromone[i] ** alpha) * (heuristic ** beta)
                        probs.append(p)
                        valid_indices.append(i)
                
                # If no items fit or no valid probabilities, stop
                if not probs or sum(probs) == 0:
                    break
                
                # Normalize probabilities (Roulette Wheel Selection)
                probs = np.array(probs) / sum(probs)
                next_item = np.random.choice(valid_indices, p=probs)
                
                # Update current ant's state
                selected_items[next_item] = 1
                current_weight += weights[next_item]
                current_value += values[next_item]
                available_indices.remove(next_item)

            # Check if this ant found a new global best
            if current_value > best_value:
                best_value = current_value
                best_combination = selected_items.copy()
            
            all_ant_values.append(current_value)

        # --- PHEROMONE UPDATE PHASE ---
        # 1. Evaporation: Decrease pheromone on all paths
        pheromone *= (1 - evaporation)
        
        # 2. Deposit: Strengthen pheromone on the best path found so far
        if best_combination is not None:
            for i in range(n_items):
                if best_combination[i] == 1:
                    # Amount deposited is proportional to solution quality
                    pheromone[i] += (best_value / 100) 

        # Record history for the graph
        history.append(best_value)
    
    return best_value, best_combination, history

# --- 2. STREAMLIT USER INTERFACE (UI) ---
st.set_page_config(page_title="ACO Knapsack Lab", layout="wide")

st.title("🐜 ACO Knapsack Solver & Evaluator")
st.markdown("""
This system uses **Ant Colony Optimization (ACO)** to solve the Knapsack Problem. 
It simulates digital ants foraging for the most optimal combination of items to maximize value within a weight limit.
""")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Algorithm Configuration")

# Input parameters
capacity = st.sidebar.number_input("Knapsack Capacity (Max Weight)", min_value=1, value=50)
n_ants = st.sidebar.slider("Number of Ants", 5, 100, 20, help="More ants increase exploration but slow down processing.")
n_iterations = st.sidebar.slider("Number of Iterations", 10, 500, 100, help="How many times the colony repeats the search.")
evaporation = st.sidebar.slider("Pheromone Evaporation Rate", 0.1, 0.9, 0.5, help="High evaporation makes ants forget old paths faster.")

# Advanced Parameters (Optional)
with st.sidebar.expander("Advanced Parameters (Alpha/Beta)"):
    alpha = st.slider("Alpha (Pheromone Importance)", 0.0, 5.0, 1.0)
    beta = st.slider("Beta (Heuristic Importance)", 0.0, 5.0, 2.0)

# File Uploader
st.sidebar.header("📁 Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")

# --- MAIN EXECUTION ---
if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    # Clean column names (remove spaces, lowercase) to prevent KeyErrors
    df.columns = df.columns.str.strip().str.lower()
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Dynamic Column Mapping
    st.markdown("### Column Mapping")
    cols = df.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        item_col = st.selectbox("Select 'Item' Column:", cols, index=0)
    with c2:
        val_col = st.selectbox("Select 'Value' Column:", cols, index=1 if len(cols)>1 else 0)
    with c3:
        weight_col = st.selectbox("Select 'Weight' Column:", cols, index=2 if len(cols)>2 else 0)

    # Run Button
    if st.button("🚀 Run ACO Evaluation"):
        with st.spinner('Ants are foraging for solutions...'):
            start_time = time.time()
            
            # Execute Algorithm
            best_val, best_comb, history = aco_knapsack(
                df[item_col].values, 
                df[weight_col].values, 
                df[val_col].values, 
                capacity, n_ants, n_iterations, alpha, beta, evaporation
            )
            
            duration = time.time() - start_time

        # --- 3. RESULTS & VISUALIZATION ---
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.success("Optimization Complete!")
            st.metric("Max Profit Found", f"{best_val}")
            st.metric("Processing Time", f"{duration:.4f} seconds")
            
            # Display Selected Items
            selected_items_df = df[best_comb == 1]
            st.write(f"**Selected Items ({len(selected_items_df)}):**")
            st.dataframe(selected_items_df[[item_col, val_col, weight_col]])
            
            # Download Button
            csv_result = selected_items_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Result (CSV)",
                data=csv_result,
                file_name="aco_result.csv",
                mime="text/csv"
            )

        with res_col2:
            st.subheader("📈 Convergence Graph")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history, color='#1E88E5', linewidth=2, label='Best Value')
            ax.set_title("Performance over Iterations")
            ax.set_xlabel("Iteration Number")
            ax.set_ylabel("Best Value (Fitness)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            st.pyplot(fig)
            
            st.info("""
            **How to interpret this graph:**
            - **Steep Rise:** The ants are rapidly learning and discovering better combinations.
            - **Plateau (Flat Line):** The algorithm has converged; the ants have likely found the optimal or near-optimal solution.
            """)

else:
    # Default State (No file uploaded)
    st.info("👋 Please upload a CSV file to begin. Ensure your file has columns for Item Name, Value, and Weight.")
    
    st.markdown("**Example CSV Format:**")
    example_df = pd.DataFrame({
        'item': ['Laptop', 'Headphones', 'Coffee', 'Notebook'],
        'value': [1500, 200, 10, 5],
        'weight': [2000, 500, 200, 100]
    })
    st.code(example_df.to_csv(index=False), language='csv')
