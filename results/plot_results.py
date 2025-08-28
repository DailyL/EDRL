import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mticker
# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'arial'
plt.rcParams['font.size'] = 20

def exponential_smoothing(x, alpha=0.05):
    """
    Apply exponential smoothing to a time series.
    
    Args:
        x: Array-like input data
        alpha: Smoothing factor (0 < alpha < 1)
              Small values = more smoothing, larger values = less smoothing
    
    Returns:
        Smoothed array of the same length as x
    """
    s = np.zeros_like(x)

    for idx, x_val in enumerate(x):
        if idx == 0:
            s[idx] = x[idx]
        else:
            s[idx] = alpha * x_val + (1-alpha) * s[idx-1]

    return s

def read_data(folder_list):
    """
    Read one txt file from each folder in the provided list.
    
    Args:
        folder_list: List of folder paths, each containing one txt file
        
    Returns:
        DataFrame containing combined data from all txt files
    """
    all_data = []
    
    for folder in folder_list:
        # Get the algorithm/run name from the folder name
        run_name = os.path.basename(folder)
        
        # Find the txt file in this folder
        txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        
        if not txt_files:
            print(f"Warning: No txt file found in folder {folder}")
            continue
            
        # Use the first txt file found
        file_path = os.path.join(folder, txt_files[0])
        
        try:
            # Read data, assuming tab-separated values with header
            df = pd.read_csv(file_path, sep='\t')
            
            # Add metadata columns
            df['run_name'] = run_name
            df['source_file'] = txt_files[0]
            
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Combine all dataframes
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    
    print("No data files found!")
    return pd.DataFrame()


def plot_comparison(data_by_algo, metric='Avg_Eval_ret', x_metric='Timestep', 
                   output_file='algorithm_comparison.png', confidence=0.75, 
                   smooth=True, ylim=None, alpha=0.05, color_palette=None):
    """
    Plot multiple algorithms on the same graph, with each algorithm having multiple runs.
    
    Args:
        data_by_algo: Dictionary mapping algorithm names to their dataframes
        metric: Column name for the y-axis metric to plot
        x_metric: Column name for the x-axis metric
        output_file: Filename to save the plot
        confidence: Confidence interval (0-1)
        smooth: Whether to apply exponential smoothing
        alpha: Smoothing factor (0 < alpha < 1)
        color_palette: Dictionary mapping algorithm names to specific colors
    """
    plt.figure(figsize=(12, 7))
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    # Default color palette for different algorithms
    default_colors = plt.cm.tab10.colors
    
    # Use custom color palette if provided
    if color_palette is None:
        color_palette = {}
    
    # Plot each algorithm
    for i, (algorithm, algo_data) in enumerate(data_by_algo.items()):
        # Use custom color if defined, otherwise fall back to default
        if algorithm in color_palette:
            color = color_palette[algorithm]
        else:
            color = default_colors[i % len(default_colors)]
        
        # Group by timestep
        grouped = algo_data.groupby(x_metric)[metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Calculate confidence interval
        grouped['ci'] = grouped['std'] / np.sqrt(grouped['count']) * stats.t.ppf((1 + confidence) / 2, grouped['count'] - 1)
        
        # Apply smoothing if requested
        plot_data = grouped.copy()
        if smooth:
            plot_data['mean'] = exponential_smoothing(grouped['mean'].values, alpha)
            plot_data['ci'] = exponential_smoothing(grouped['ci'].values, alpha)
        
        # Plot mean line
        plt.plot(plot_data[x_metric], plot_data['mean'], 
                color=color, linewidth=2.5, label=f'{algorithm}')
        
        # Plot confidence interval
        plt.fill_between(plot_data[x_metric], 
                        plot_data['mean'] - plot_data['ci'], 
                        plot_data['mean'] + plot_data['ci'], 
                        alpha=0.2, color=color)
    
    # Update y-axis label based on metric
    y_label = "Average Return" if "ret" in metric.lower() else "Average Cost"
    
    # Add horizontal line at y=1 for cost plots
    if "cost" in metric.lower():
        plt.axhline(y=1.0, color='black', linestyle='--', linewidth=3.5, alpha=0.7, label='Cost Threshold')
    
    plt.xlim(0, 1.5e6)
    if ylim is not None:
        plt.ylim(ylim)
    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=2.5)
    #plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    
    xticks = np.linspace(0, 1.5e6, num=7) 
    plt.xticks(xticks, [f'{x/1e6:.1f}' for x in xticks],fontsize=26, fontweight='bold')
    #plt.yticks(fontsize=26, fontweight='bold')
    plt.yticks([ytick for ytick in plt.yticks()[0] if ytick != 0], fontsize=26, fontweight='bold')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color('black')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    # Define algorithms with their corresponding folders for return plot
    algorithms_ret = {
        "EthicAR": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_379",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_12576",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_8128"
        ],
        "LSTMSAC": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSAC_SRL_Traj_Env__2025-06-12_2387",
            "/home/dianzhaoli/EDRL/experiments/LSTMSAC_SRL_Traj_Env__2025-06-12_3825",
            "/home/dianzhaoli/EDRL/experiments/LSTMSAC_SRL_Traj_Env__2025-06-12_5587"
        ],
        "SACLAG": [
            "/home/dianzhaoli/EDRL/experiments/SACLAG_SRL_Traj_Env__2025-06-13_16455",
            "/home/dianzhaoli/EDRL/experiments/SACLAG_SRL_Traj_Env__2025-06-13_16054",
            "/home/dianzhaoli/EDRL/experiments/SACLAG_SRL_Traj_Env__2025-06-13_15632"
        ],
        "EthicAR w/o PER": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_6279",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_8459",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-06-25_17253"
        ]
    }

    # Define algorithms with their corresponding folders for cost plot
    algorithms_cost = {
        "EthicAR": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_15725",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_16307",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_17863",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_9147",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_9953",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_7669"
        ],

        "SACLAG": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-04_2028",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-03_2250",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-03_4150"
        ],
        "EthicAR w/o PER": [
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-05_7436",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-05_1368",
            "/home/dianzhaoli/EDRL/experiments/LSTMSACLAG_SRL_Traj_Env__2025-05-05_494"
        ]
    }

    color_palette = {
        "EthicAR": "#699dcb",  
        "SACLAG": "#e68785",   
        "LSTMSAC": "#efc085",     
        "EthicAR w/o PER": "#6aaa81",  
    }
    
    # Read data for each algorithm (return)
    data_by_algo_ret = {}
    for algo_name, folders in algorithms_ret.items():
        algo_data = read_data(folders)
        if not algo_data.empty:
            data_by_algo_ret[algo_name] = algo_data

    # Read data for each algorithm (cost)
    data_by_algo_cost = {}
    for algo_name, folders in algorithms_cost.items():
        algo_data = read_data(folders)
        if not algo_data.empty:
            data_by_algo_cost[algo_name] = algo_data
    
    if not data_by_algo_ret and not data_by_algo_cost:
        print("No data to plot. Check the provided folder paths.")
        return
    
    os.makedirs("results", exist_ok=True)
    
    # Plot return
    if data_by_algo_ret:
        plot_comparison(data_by_algo_ret, 
                       metric='Avg_Eval_ret', 
                       output_file='results/eval_return_comparison.pdf',
                       smooth=True, 
                       alpha=0.05,
                       color_palette=color_palette)
    
    # Plot cost
    if data_by_algo_cost:
        plot_comparison(data_by_algo_cost, 
                       metric='Avg_Eval_cost_per_step', 
                       output_file='results/eval_cost_comparison.pdf',
                       smooth=True, 
                       alpha=0.05,
                       ylim=(0, 18),
                       confidence=0.5,
                       color_palette=color_palette)

if __name__ == "__main__":
    main()