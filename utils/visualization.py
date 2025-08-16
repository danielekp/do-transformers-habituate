import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import List, Dict, Optional

def plot_habituation_analysis(list_of_activations: List[Dict], 
                            target_token: str,
                            control_magnitude: Optional[float] = None,
                            figsize: tuple = (16, 12),
                            save_path: Optional[str] = None):
    """Complete visualization utility for habituation experiment results."""
    
    # Extract magnitudes from activation data
    magnitudes = [act['magnitude'] for act in list_of_activations]
    layer_num = list_of_activations[0]['layer'] if list_of_activations else "Unknown"
    
    # Basic statistics
    n_occurrences = len(magnitudes)
    mean_mag = np.mean(magnitudes)
    std_mag = np.std(magnitudes)
    first_mag = magnitudes[0] if magnitudes else 0
    last_mag = magnitudes[-1] if magnitudes else 0
    percent_change = ((last_mag - first_mag) / first_mag) * 100 if first_mag != 0 else 0
    
    print(f"Target token: '{target_token}'")
    print(f"Layer analyzed: {layer_num}")
    print(f"Number of occurrences: {n_occurrences}")
    print(f"First magnitude: {first_mag:.4f}")
    print(f"Last magnitude: {last_mag:.4f}")
    print(f"Mean ± Std: {mean_mag:.4f} ± {std_mag:.4f}")
    print(f"Change: {percent_change:.2f}%")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'Neural Habituation Analysis: "{target_token}" in Layer {layer_num}', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Main habituation curve
    x_vals = range(1, len(magnitudes) + 1)
    axes[0, 0].plot(x_vals, magnitudes, 'o-', color='steelblue', 
                   linewidth=2, markersize=6, alpha=0.8, label='Repetitions')
    
    # Add control line if provided
    if control_magnitude is not None:
        axes[0, 0].axhline(y=control_magnitude, color='red', linestyle='--', 
                          alpha=0.7, linewidth=2, label='Control')
        print(f"Control magnitude: {control_magnitude:.4f}")
    
    axes[0, 0].set_xlabel('Occurrence Number')
    axes[0, 0].set_ylabel('MLP Output Magnitude')
    axes[0, 0].set_title('Habituation Pattern')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add trend annotation
    if len(magnitudes) > 1:
        trend_direction = "↓ Decreasing" if last_mag < first_mag else "↑ Increasing"
        axes[0, 0].text(0.02, 0.98, f'{trend_direction}\n({percent_change:+.1f}%)', 
                       transform=axes[0, 0].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='lightgray', alpha=0.8))
    
    # Plot 2: Linear trend analysis
    if len(magnitudes) > 1:
        x = np.array(x_vals)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, magnitudes)
        
        axes[0, 1].plot(x, magnitudes, 'o', alpha=0.6, color='steelblue', 
                       markersize=6, label='Data points')
        axes[0, 1].plot(x, slope * x + intercept, 'r--', linewidth=2, 
                       label=f'Trend (slope: {slope:.6f})')
        
        # Add R² and p-value
        axes[0, 1].text(0.02, 0.98, f'R² = {r_value**2:.3f}\np = {p_value:.4f}', 
                       transform=axes[0, 1].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='lightblue', alpha=0.8))
        
        print(f"\nLinear Trend Analysis:")
        print(f"Slope: {slope:.6f} ± {std_err:.6f}")
        print(f"R²: {r_value**2:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant trend: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
    else:
        axes[0, 1].text(0.5, 0.5, 'Need >1 data point\nfor trend analysis', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    axes[0, 1].set_xlabel('Occurrence Number')
    axes[0, 1].set_ylabel('MLP Output Magnitude')
    axes[0, 1].set_title('Linear Trend Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Distribution histogram
    axes[0, 2].hist(magnitudes, bins=min(10, max(3, len(magnitudes)//3)), 
                   alpha=0.7, color='lightcoral', edgecolor='darkred')
    axes[0, 2].axvline(x=mean_mag, color='blue', linestyle='-', linewidth=2, label='Mean')
    axes[0, 2].axvline(x=first_mag, color='green', linestyle='--', linewidth=2, label='First')
    axes[0, 2].axvline(x=last_mag, color='orange', linestyle='--', linewidth=2, label='Last')
    
    if control_magnitude is not None:
        axes[0, 2].axvline(x=control_magnitude, color='red', linestyle=':', 
                          linewidth=2, label='Control')
    
    axes[0, 2].set_xlabel('MLP Output Magnitude')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Magnitude Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Rolling statistics (if enough data)
    if len(magnitudes) >= 5:
        window_size = min(5, len(magnitudes) // 3)
        rolling_mean = pd.Series(magnitudes).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(magnitudes).rolling(window=window_size, center=True).std()
        
        axes[1, 0].plot(x_vals, magnitudes, 'o-', alpha=0.5, color='lightblue', 
                       label='Raw data', markersize=4)
        axes[1, 0].plot(x_vals, rolling_mean, 'r-', linewidth=3, 
                       label=f'Rolling mean (window={window_size})')
        
        # Add error bands
        axes[1, 0].fill_between(x_vals, rolling_mean - rolling_std, rolling_mean + rolling_std,
                               alpha=0.2, color='red', label='±1 std')
        
        axes[1, 0].set_xlabel('Occurrence Number')
        axes[1, 0].set_ylabel('MLP Output Magnitude')
        axes[1, 0].set_title('Rolling Statistics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Need ≥5 data points\nfor rolling statistics', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Rolling Statistics (Insufficient Data)')
    
    # Plot 5: Habituation curve with confidence intervals
    if len(magnitudes) > 2:
        # Fit exponential decay model: y = a * exp(-b * x) + c
        from scipy.optimize import curve_fit
        
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        try:
            x_data = np.array(x_vals)
            y_data = np.array(magnitudes)
            popt, pcov = curve_fit(exp_decay, x_data, y_data, 
                                  p0=[y_data[0] - y_data[-1], 0.1, y_data[-1]],
                                  maxfev=1000)
            
            x_smooth = np.linspace(1, len(magnitudes), 100)
            y_smooth = exp_decay(x_smooth, *popt)
            
            axes[1, 1].plot(x_vals, magnitudes, 'o', color='steelblue', 
                           markersize=6, label='Data')
            axes[1, 1].plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                           label=f'Exp. decay fit')
            
            # Add decay rate
            decay_rate = popt[1]
            axes[1, 1].text(0.02, 0.98, f'Decay rate: {decay_rate:.4f}', 
                           transform=axes[1, 1].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='lightyellow', alpha=0.8))
            
            print(f"\nExponential Decay Fit:")
            print(f"Decay rate (b): {decay_rate:.4f}")
            print(f"Asymptote (c): {popt[2]:.4f}")
            
        except:
            axes[1, 1].plot(x_vals, magnitudes, 'o-', color='steelblue')
            axes[1, 1].text(0.5, 0.5, 'Could not fit\nexponential model', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].plot(x_vals, magnitudes, 'o-', color='steelblue')
    
    axes[1, 1].set_xlabel('Occurrence Number')
    axes[1, 1].set_ylabel('MLP Output Magnitude')
    axes[1, 1].set_title('Exponential Decay Model')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics table
    axes[1, 2].axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Number of occurrences', f'{n_occurrences}'],
        ['Layer analyzed', f'{layer_num}'],
        ['First magnitude', f'{first_mag:.4f}'],
        ['Last magnitude', f'{last_mag:.4f}'],
        ['Mean magnitude', f'{mean_mag:.4f}'],
        ['Std deviation', f'{std_mag:.4f}'],
        ['Percent change', f'{percent_change:.2f}%'],
    ]
    
    if len(magnitudes) > 1:
        summary_data.extend([
            ['Linear slope', f'{slope:.6f}'],
            ['R-squared', f'{r_value**2:.4f}'],
            ['p-value', f'{p_value:.6f}'],
            ['Significant?', 'Yes' if p_value < 0.05 else 'No']
        ])
    
    if control_magnitude is not None:
        summary_data.append(['Control magnitude', f'{control_magnitude:.4f}'])
    
    # Create table
    table = axes[1, 2].table(cellText=summary_data[1:], 
                            colLabels=summary_data[0],
                            cellLoc='left',
                            loc='center',
                            bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        table[(i, 0)].set_facecolor('#E6E6FA')
        table[(i, 1)].set_facecolor('#F0F8FF')
    
    axes[1, 2].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    
    # Return analysis results
    analysis_results = {
        'n_occurrences': n_occurrences,
        'layer': layer_num,
        'magnitudes': magnitudes,
        'mean_magnitude': mean_mag,
        'std_magnitude': std_mag,
        'first_magnitude': first_mag,
        'last_magnitude': last_mag,
        'percent_change': percent_change
    }
    
    if len(magnitudes) > 1:
        analysis_results.update({
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        })
    
    return analysis_results
