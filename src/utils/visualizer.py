import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

class ResearchVisualizer:
    """
    Standardized plots for research reports.
    """
    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({"font.family": "serif", "figure.figsize": (10, 6)})

    def save_plot(self, name: str):
        path = os.path.join(self.output_dir, f"{name}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {path}")

    def plot_equity_curve(self, strategy_ret, benchmark_ret):
        plt.figure()
        # Calculate cumulative returns
        cum_strat = (1 + strategy_ret).cumprod()
        cum_bench = (1 + benchmark_ret).cumprod()
        
        plt.plot(cum_strat, label='Stochastic Strategy', color='#2c3e50', linewidth=2)
        plt.plot(cum_bench, label='S&P 500 (Benchmark)', color='#95a5a6', linestyle='--', alpha=0.7)
        
        plt.title("Cumulative Performance (Log Scale)")
        plt.ylabel("Growth of $1 Investment")
        plt.yscale('log')
        plt.legend()
        self.save_plot("equity_curve")
        plt.show()

    def plot_dynamic_allocation(self, df_weights):
        """Stacked area chart showing Long vs Short exposure."""
        plt.figure(figsize=(12, 6))
        
        # Separate Longs and Shorts for cleaner plotting
        df_long = df_weights.clip(lower=0)
        df_short = df_weights.clip(upper=0)
        
        plt.stackplot(df_long.index, df_long.T, labels=df_long.columns, alpha=0.8)
        plt.stackplot(df_short.index, df_short.T, labels=df_short.columns, alpha=0.8)
        
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Dynamic Portfolio Allocation (Long/Short)")
        plt.ylabel("Weight Exposure")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        self.save_plot("allocation_tunnel")
        plt.show()

    def plot_attribution(self, cumulative_pnl):
        plt.figure(figsize=(12, 6))
        for col in cumulative_pnl.columns:
            plt.plot(cumulative_pnl.index, cumulative_pnl[col], label=col)
            
        plt.title("Cumulative P&L Attribution by Asset")
        plt.ylabel("Contribution to Return")
        plt.legend()
        self.save_plot("pnl_attribution")
        plt.show()