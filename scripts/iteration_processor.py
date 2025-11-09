"""
This script reads 'benchmark_run.csv' (GPU data) and 'benchmark_cpu.csv'
(CPU data) and compiles all raw iteration runs into multiple outputs:

1. 'all_iterations.csv': A single, long-format CSV file.
2. 'all_iterations_by_sheet.xlsx': An Excel file with:
   - A first sheet "Average Summary" with the mean of all iterations.
   - Subsequent sheets for each 'N' size, with Metrics as columns.
3. 'summary_average.csv': A CSV file containing only the average summary table.
4. A new 'graphs/' folder containing:
   - A 'summary_performance_graph.png' (log scale) of the average summary.
   - A separate graph for each 'N' size (e.g., 'iterations_(1 << 8)^2.png')
     showing the performance over the 10 iterations.

This script calculates the MEAN (average) for the summary files.
"""

import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# --- Configuration ---
GPU_CSV_FILE = './results/gpu_benchmark.csv'
CPU_CSV_FILE = './results/cpu_benchmark.csv'
OUTPUT_CSV_FILE = './results/all_iterations.csv'
OUTPUT_EXCEL_FILE = './results/all_iterations_by_sheet.xlsx'
OUTPUT_SUMMARY_CSV_FILE = './results/summary_average.csv'
OUTPUT_GRAPH_FOLDER = './results/graphs' # New graph output folder

# Mapping for threads_per_block to a simpler name
# 64 -> 8*8, 256 -> 16*16, 1024 -> 32*32
THREADS_MAP = {
    64: "8*8",
    256: "16*16",
    1024: "32*32"
}

# Mapping for GPU task_type to a simpler name (case-insensitive)
TASK_TYPE_MAP = {
    'matrixmultnaive': 'Naive',
    'matrixmulttiled': 'Tiled'
}

# ---------------------

def process_gpu_data(filename):
    """
    Loads the GPU benchmark data with new header format:
    task_type,platform,threads_per_block,size,duration_ms
    """
    print(f"Processing GPU iterations from '{filename}'...")
    try:
        df = pd.read_csv(filename)
        
        # Verify expected columns
        expected_cols = ['task_type', 'platform', 'threads_per_block', 'size', 'duration_ms']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"GPU CSV missing expected columns. Found: {df.columns}")
            
        # Map names and create metrics
        df['task_type'] = df['task_type'].str.lower().map(TASK_TYPE_MAP)
        df['threads_per_block'] = df['threads_per_block'].map(THREADS_MAP)
        df['Metric'] = df['task_type'] + ' ' + df['threads_per_block']
        df['N'] = df['size'].str.strip().apply(lambda s: f"({s})^2")
        df['Iteration'] = df.groupby(['N', 'Metric']).cumcount() + 1
        
        return df[['N', 'Metric', 'Iteration', 'duration_ms']]
    except Exception as e:
        print(f"Error processing GPU data: {e}")
        sys.exit(1)

def process_cpu_data(filename):
    """
    Loads the CPU benchmark data with new header format:
    task_type,platform,size,duration_ms
    """
    print(f"Processing CPU iterations from '{filename}'...")
    try:
        df = pd.read_csv(filename)
        
        # Verify expected columns
        expected_cols = ['task_type', 'platform', 'size', 'duration_ms']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"CPU CSV missing expected columns. Found: {df.columns}")
            
        # Create metrics
        df['Metric'] = 'CPU'
        df['N'] = df['size'].str.strip().apply(lambda s: f"({s})^2")
        df['Iteration'] = df.groupby(['N', 'Metric']).cumcount() + 1
        
        return df[['N', 'Metric', 'Iteration', 'duration_ms']]
    except Exception as e:
        print(f"Error processing CPU data: {e}")
        sys.exit(1)

# --- NEW GRAPHING FUNCTIONS ---

def plot_summary_graph(df, output_filename):
    """
    Generates the main summary graph (log scale) from the average data.
    """
    print(f"Generating summary graph: {output_filename}")
    plt.figure(figsize=(12, 8))
    
    # Use markers and linestyles to differentiate
    markers = ['o', 's', 'v', '^', '<', '>', 'D']
    linestyles = ['-', '-', '-', '--', '--', '--', '-.']
    
    x_labels = df.index
    x_indices = range(len(x_labels))

    for i, col_name in enumerate(df.columns):
        marker = markers[i % len(markers)]
        style = linestyles[i % len(linestyles)]
        plt.plot(x_indices, df[col_name], label=col_name, marker=marker, linestyle=style)
    
    plt.yscale('log')
    plt.title('Matrix Multiplication Performance (Log Scale)', fontsize=16)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Average Execution Time (ms) - Log Scale', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.xticks(ticks=x_indices, labels=x_labels, rotation=15)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close() # Close plot to free memory

def plot_iteration_graph(df, sheet_name, output_filename):
    """
    Generates a graph for a single 'N' size, showing performance
    over all 10 iterations.
    """
    print(f"Generating iteration graph: {output_filename}")
    plt.figure(figsize=(12, 8))
    
    x_axis = df['Iteration']
    metrics = [col for col in df.columns if col != 'Iteration']
    markers = ['o', 's', 'v', '^', '<', '>', 'D']

    for i, metric in enumerate(metrics):
        plt.plot(x_axis, df[metric], label=metric, marker=markers[i % len(markers)])
    
    plt.yscale('log') # Use log scale to see all lines
    plt.title(f'Iteration Performance for {sheet_name}', fontsize=16)
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.xticks(x_axis) # Ensure ticks are 1, 2, 3... 10
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close() # Close plot to free memory

# ----------------------------

def main():
    # --- Create Graph Folder ---
    os.makedirs(OUTPUT_GRAPH_FOLDER, exist_ok=True)
    print(f"Ensured graph output folder exists: '{OUTPUT_GRAPH_FOLDER}'")

    gpu_data = process_gpu_data(GPU_CSV_FILE)
    cpu_data = process_cpu_data(CPU_CSV_FILE)
    
    # Join the GPU and CPU dataframes
    final_df = pd.concat([gpu_data, cpu_data], ignore_index=True)
    
    # --- Sort the data ---
    # First by a sort_key extracted from N, then Metric, then Iteration
    final_df['sort_key'] = final_df['N'].str.extract(r'1 << (\d+)').astype(int)
    final_df = final_df.sort_values(['sort_key', 'Metric', 'Iteration'])
    
    # --- 1. Save the long-format CSV file ---
    # We drop the temporary sort_key before saving
    csv_df = final_df.drop(columns='sort_key')
    csv_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print(f"\nSuccessfully compiled all iterations and saved to '{OUTPUT_CSV_FILE}'.")
    
    # --- 2. Generate the new Average Summary table ---
    print("Generating average summary table...")
    # Group by N, sort_key, and Metric, then calculate the mean
    summary_df_grouped = final_df.groupby(['N', 'Metric', 'sort_key'])['duration_ms'].mean()
    
    # Unstack the 'Metric' level to turn metrics into columns
    summary_df_wide = summary_df_grouped.unstack('Metric')
    
    # Reset the index to get 'N' and 'sort_key' as columns, then sort by 'sort_key'
    summary_df_final = summary_df_wide.reset_index().sort_values('sort_key')
    
    # Set 'N' as the index and drop the helper 'sort_key' column
    summary_df_final = summary_df_final.drop(columns='sort_key').set_index('N')
    
    # Re-order columns to a logical format
    all_metrics = [
        'Naive 8*8', 'Naive 16*16', 'Naive 32*32',
        'Tiled 8*8', 'Tiled 16*16', 'Tiled 32*32',
        'CPU'
    ]
    ordered_cols = [col for col in all_metrics if col in summary_df_final.columns]
    summary_df_final = summary_df_final[ordered_cols]
    
    # --- 3. Save the new Average Summary CSV file ---
    summary_df_final.to_csv(OUTPUT_SUMMARY_CSV_FILE)
    print(f"Successfully saved average summary to '{OUTPUT_SUMMARY_CSV_FILE}'.")
    
    # --- 4. Plot the Average Summary Graph ---
    summary_plot_path = os.path.join(OUTPUT_GRAPH_FOLDER, 'summary_performance_graph.png')
    plot_summary_graph(summary_df_final, summary_plot_path)

    # --- 5. Save the multi-sheet Excel file (Wide Format) ---
    print(f"Generating Excel file '{OUTPUT_EXCEL_FILE}' with separate sheets...")
    
    # Use pd.ExcelWriter to write to multiple sheets
    with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
        
        # --- Write the new "Average Summary" sheet FIRST ---
        summary_df_final.to_excel(writer, sheet_name="Average Summary")
        print("- Added 'Average Summary' sheet.")

        # --- Write the individual raw data sheets ---
        # Group the sorted DataFrame by the 'sort_key' (8, 9, 10...)
        for key, group_df in final_df.groupby('sort_key'):
            # Create a clear sheet name, like in the screenshot
            sheet_name = f"(1 << {key})^2"
            
            # --- Pivot the data for this sheet ---
            # Index: Iteration
            # Columns: Metric (Naive 8*8, CPU, etc.)
            # Values: duration_ms
            try:
                sheet_data = group_df.pivot(
                    index='Iteration', 
                    columns='Metric', 
                    values='duration_ms'
                )
            except Exception as e:
                print(f"Error pivoting data for sheet {sheet_name}: {e}")
                continue # Skip this sheet if pivoting fails

            # Re-order columns to a logical format
            # Use the same 'ordered_cols' from the summary
            sheet_data = sheet_data[ordered_cols] 

            # Reset the index so 'Iteration' becomes a column, matching screenshot
            sheet_data = sheet_data.reset_index()

            # Save this pivoted data to its own sheet
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"- Added '{sheet_name}' sheet with raw iterations.")
            
            # --- NEW: Plot this sheet's data ---
            iteration_plot_path = os.path.join(OUTPUT_GRAPH_FOLDER, f'iterations_{sheet_name}.png')
            plot_iteration_graph(sheet_data, sheet_name, iteration_plot_path)
            
    print(f"\nSuccessfully generated Excel file: '{OUTPUT_EXCEL_FILE}'")
    print(f"All graphs saved to '{OUTPUT_GRAPH_FOLDER}' folder.")

if __name__ == "__main__":
    main()