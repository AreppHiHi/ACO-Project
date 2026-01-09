import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# PART 1: ACO ALGORITHM LOGIC
# ==========================================
def aco_knapsack(items, weights, values, capacity, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, evaporation=0.5):
    """
    Core function to run Ant Colony Optimization for Knapsack Problem.
    """
    n_items = len(items)
    # Initialize pheromone levels (small amount on all paths)
    pheromone = np.full(n_items, 0.1) 
    
    best_global_value = 0
    best_global_combination = None
    history = [] # To store best value per iteration for the graph

    for iteration in range(n_iterations):
        all_ant_values = []
        
        for ant in range(n_ants):
            current_weight = 0
            current_value = 0
            selected_items = np.zeros(n_items) # 0 = not selected, 1 = selected
            available_indices = list(range(n_items))
            
            # --- Ant constructs a solution ---
            while available_indices:
                probs = []
                valid_indices = []
                
                for i in available_indices:
                    # Check if item fits in the bag
                    if current_weight + weights[i] <= capacity:
                        # Heuristic information (Greedy factor: Value / Weight)
                        heuristic = values[i] / weights[i] if weights[i] > 0 else values[i]
                        
                        # Calculate probability using ACO formula
                        p = (pheromone[i] ** alpha) * (heuristic ** beta)
                        probs.append(p)
                        valid_indices.append(i)
                
                # If no items fit, stop
                if not probs or sum(probs) == 0:
                    break
                
                # Normalize probabilities (Roulette Wheel)
                probs = np.array(probs) / sum(probs)
                
                # Ant makes a choice
                next_item = np.random.choice(valid_indices, p=probs)
                
                # Update ant's bag
                selected_items[next_item] = 1
                current_weight += weights[next_item]
                current_value += values[next_item]
                available_indices.remove(next_item)

            # --- Check if this ant found a new best solution ---
            if current_value > best_global_value:
                best_global_value = current_value
                best_global_combination = selected_items.copy()
            
            all_ant_values.append(current_value)

        # --- Pheromone Update (Global Update) ---
        # 1. Evaporation (Trail fades over time)
        pheromone *= (1 - evaporation)
        
        # 2. Deposit (Reinforce the best path)
        if best_global_combination is not None:
            for i in range(n_items):
                if best_global_combination[i] == 1:
                    # Logic: Better value = Stronger pheromone deposit
                    pheromone[i] += (best_global_value / 100) 

        # Save history for the graph
        history.append(best_global_value)
    
    return best_global_value, best_global_combination, history

# ==========================================
# PART 2: STREAMLIT USER INTERFACE (UI)
# ==========================================
st.set_page_config(page_title="ACO Knapsack Project", layout="wide")

st.title("🐜 Ant Colony Optimization (ACO) for Knapsack Problem")
st.markdown("""
**Project Evaluation:** This system uses artificial ants to find the optimal combination of items 
to maximize value without exceeding the weight capacity.
""")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("1. Algorithm Settings")
capacity = st.sidebar.number_input("Knapsack Capacity (Max Weight)", min_value=1, value=3000)
n_ants = st.sidebar.slider("Number of Ants", 5, 100, 20)
n_iterations = st.sidebar.slider("Number of Iterations", 10, 500, 50)
evaporation = st.sidebar.slider("Evaporation Rate", 0.1, 0.9, 0.5, help="Higher value = Pheromone vanishes faster.")

st.sidebar.markdown("---")
st.sidebar.header("2. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")

# --- MAIN SECTION ---
if uploaded_file:
    # Read the file
    df = pd.read_csv(uploaded_file)
    
    # Clean column names (remove spaces, convert to lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    st.write("### Data Preview")
    st.dataframe(df.head())

    # --- DYNAMIC COLUMN MAPPING ---
    # This prevents errors if your CSV headers are named differently
    st.warning("⚠️ Please confirm the columns below match your CSV data:")
    cols = df.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        item_col = st.selectbox("Select Item Name Column", cols, index=0)
    with c2:
        val_col = st.selectbox("Select Value/Price Column", cols, index=1 if len(cols)>1 else 0)
    with c3:
        weight_col = st.selectbox("Select Weight Column", cols, index=2 if len(cols)>2 else 0)

    # --- RUN BUTTON ---
    if st.button("🚀 Run ACO Algorithm"):
        with st.spinner('The ants are exploring the search space...'):
            start_time = time.time()
            
            # Execute the function
            best_val, best_comb, history = aco_knapsack(
                items=df[item_col].values, 
                weights=df[weight_col].values, 
                values=df[val_col].values, 
                capacity=capacity, 
                n_ants=n_ants, 
                n_iterations=n_iterations,
                evaporation=evaporation
            )
            
            elapsed_time = time.time() - start_time

        # --- RESULTS DISPLAY ---
        st.success("Optimization Finished!")
        
        # Metrics
        m1, m2 = st.columns(2)
        m1.metric("Maximum Value Found", f"{best_val}")
        m2.metric("Execution Time", f"{elapsed_time:.4f} seconds")

        # Two columns for Table and Graph
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Selected Items Solution")
            # Filter the dataframe to show only selected items
            result_df = df[best_comb == 1]
            st.dataframe(result_df)
            
            # CSV Download Button
            csv_data = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Result CSV",
                data=csv_data,
                file_name="aco_solution.csv",
                mime="text/csv",
            )

        with col_right:
            st.subheader("Convergence Graph")
            # Plotting
            fig, ax = plt.subplots()
            ax.plot(history, color='blue', linewidth=2)
            ax.set_title("Optimization Progress")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Best Value Found")
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
            st.info("This graph shows how the ants improved the solution over time.")

else:
    # Instructions when no file is uploaded
    st.info("👋 Please upload a CSV file to start. The file must have columns for Item, Weight, and Value.")
    
    st.markdown("#### Example CSV Format:")
    st.code("""item,value,weight
Laptop,2000,2500
Phone,1000,500
Book,50,300""", language="csv")
