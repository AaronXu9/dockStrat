from typing import Any, List, Dict, Optional, Tuple, Union
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import ast
import statsmodels.formula.api as smf
import warnings
from statsmodels.stats.contingency_tables import mcnemar, StratifiedTable
import statsmodels.formula.api as smf
from pygam import LinearGAM, s, f
from statsmodels.stats.multitest import multipletests


class DockingAnalysisBase:
    """Base kclass for all docking analysis tools."""
    
    def __init__(self, df_combined):
        """
        Initialize the analysis base class.
        
        Parameters:
        -----------
        df_combined : pandas.DataFrame
            Combined dataframe with results from all methods
        """
        self.df_combined = df_combined
    
    def get_best_rmsd_per_protein(self, method=None, rmsd_threshold=2.0):
        """
        Get best RMSD per protein for one or all methods.
        
        Parameters:
        -----------
        method : str, optional
            Method name to filter results. If None, returns all methods.
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful
            
        Returns:
        --------
        dict or pandas.DataFrame
            Dictionary mapping methods to DataFrames with best RMSDs per protein,
            or a single DataFrame if method is specified
        """
        if method is not None:
            # Filter data for the specified method
            df_method = self.df_combined[self.df_combined['method'] == method]
            
            # Get best RMSD per protein
            best_rmsd_df = df_method.groupby('protein').apply(
                lambda x: x.loc[x['rmsd'].idxmin()] if not x['rmsd'].isna().all() else x.iloc[0]
            ).reset_index(drop=True)
            
            # Add success column
            best_rmsd_df[f'rmsd_≤_{rmsd_threshold}å'] = best_rmsd_df['rmsd'] <= rmsd_threshold
            
            return best_rmsd_df
        else:
            # Get all methods
            methods = self.df_combined['method'].unique()
            
            # Get best RMSD per protein for each method
            best_rmsd_by_method = {}
            for m in methods:
                df_m = self.df_combined[self.df_combined['method'] == m]
                best_rmsd = df_m.groupby('protein').apply(
                    lambda x: x.loc[x['rmsd'].idxmin()] if not x['rmsd'].isna().all() else x.iloc[0]
                ).reset_index(drop=True)
                
                # Add success column
                best_rmsd[f'rmsd_≤_{rmsd_threshold}å'] = best_rmsd['rmsd'] <= rmsd_threshold
                
                best_rmsd_by_method[m] = best_rmsd
                
            return best_rmsd_by_method
    
    def create_success_matrix(self, methods, rmsd_threshold=2.0):
        """
        Create a matrix showing success/failure across multiple methods.
        
        Parameters:
        -----------
        methods : list
            List of method names to include in the analysis
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with protein IDs and success status for each method
        """
        # Get best RMSD per protein for each method
        best_rmsd_by_method = {}
        for method in methods:
            best_rmsd_by_method[method] = self.get_best_rmsd_per_protein(method, rmsd_threshold)
        
        # Find common proteins across all methods
        common_proteins = set(best_rmsd_by_method[methods[0]]['protein'])
        for method in methods[1:]:
            common_proteins = common_proteins.intersection(
                set(best_rmsd_by_method[method]['protein'])
            )
        common_proteins = list(common_proteins)
        
        # Create success matrix
        success_matrix = pd.DataFrame({'protein': common_proteins})
        
        # Add success columns for each method
        for method in methods:
            method_success = {}
            for protein in common_proteins:
                success = best_rmsd_by_method[method][
                    best_rmsd_by_method[method]['protein'] == protein
                ][f'rmsd_≤_{rmsd_threshold}å'].iloc[0]
                method_success[protein] = success
            success_matrix[f'{method}_success'] = success_matrix['protein'].map(method_success)
        
        # Calculate total successes for each protein
        success_matrix['total_successes'] = success_matrix[
            [f'{method}_success' for method in methods]
        ].sum(axis=1)
        
        return success_matrix

    def _prepare_rmsd_for_plotting(self, df, rmsd_col='rmsd', clip_value=None, 
                                use_log=False, handle_outliers='clip'):
        """
        Helper function to prepare RMSD values for visualization by handling outliers.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing RMSD values
        rmsd_col : str
            Name of the RMSD column
        clip_value : float, optional
            Maximum RMSD value to show (if None, will calculate based on percentiles)
        use_log : bool
            Whether to add log-transformed RMSD values
        handle_outliers : str
            Method to handle outliers: 'clip', 'log', 'remove', or 'none'
            
        Returns:
        --------
        DataFrame, dict
            Processed DataFrame and dictionary with outlier information
        """
        df_plot = df.copy()
        
        # Calculate stats about outliers
        Q1 = df_plot[rmsd_col].quantile(0.25)
        Q3 = df_plot[rmsd_col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        
        # Set default clip value if not provided
        if clip_value is None:
            clip_value = min(np.percentile(df_plot[rmsd_col], 95) * 1.5, 20.0)
        
        outlier_info = {
            'original_max': df_plot[rmsd_col].max(),
            'display_max': clip_value,
            'n_outliers': (df_plot[rmsd_col] > clip_value).sum(),
            'outlier_percent': (df_plot[rmsd_col] > clip_value).mean() * 100,
            'method': handle_outliers
        }
        
        # Handle outliers based on specified method
        if handle_outliers == 'clip':
            df_plot[f'{rmsd_col}_plot'] = np.clip(df_plot[rmsd_col], 0, clip_value)
        elif handle_outliers == 'log':
            df_plot[f'{rmsd_col}_plot'] = np.log10(df_plot[rmsd_col])
            outlier_info['log_threshold'] = np.log10(clip_value)
        elif handle_outliers == 'remove':
            df_plot = df_plot[df_plot[rmsd_col] <= clip_value].copy()
        else:  # 'none'
            df_plot[f'{rmsd_col}_plot'] = df_plot[rmsd_col]
        
        return df_plot, outlier_info

class PoseBustersAnalysis(DockingAnalysisBase):
    """Analysis of PoseBusters validity metrics and their impact on docking success."""
    
    # Define constants for PoseBusters metrics
    BUST_TEST_COLUMNS = [
        # chemical validity and consistency
        "mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded", "sanitization",
        "molecular_formula", "molecular_bonds", "tetrahedral_chirality", 
        "double_bond_stereochemistry",
        # intramolecular validity
        "bond_lengths", "bond_angles", "internal_steric_clash",
        "aromatic_ring_flatness", "double_bond_flatness",
        # intermolecular validity
        "minimum_distance_to_protein", "minimum_distance_to_organic_cofactors",
        "minimum_distance_to_inorganic_cofactors", "volume_overlap_with_protein",
        "volume_overlap_with_organic_cofactors", "volume_overlap_with_inorganic_cofactors"
    ]
    
    # Create category mapping for visualization
    CATEGORY_MAPPING = {
        # chemical validity and consistency
        "mol_pred_loaded": "Chemical Validity",
        "mol_true_loaded": "Chemical Validity", 
        "mol_cond_loaded": "Chemical Validity",
        "sanitization": "Chemical Validity",
        "molecular_formula": "Chemical Validity",
        "molecular_bonds": "Chemical Validity",
        "tetrahedral_chirality": "Chemical Validity", 
        "double_bond_stereochemistry": "Chemical Validity",
        # intramolecular validity
        "bond_lengths": "Intramolecular Validity",
        "bond_angles": "Intramolecular Validity",
        "internal_steric_clash": "Intramolecular Validity",
        "aromatic_ring_flatness": "Intramolecular Validity",
        "double_bond_flatness": "Intramolecular Validity",
        # intermolecular validity
        "minimum_distance_to_protein": "Intermolecular Validity",
        "minimum_distance_to_organic_cofactors": "Intermolecular Validity",
        "minimum_distance_to_inorganic_cofactors": "Intermolecular Validity",
        "volume_overlap_with_protein": "Intermolecular Validity",
        "volume_overlap_with_organic_cofactors": "Intermolecular Validity",
        "volume_overlap_with_inorganic_cofactors": "Intermolecular Validity"
    }
    
    def __init__(self, df_combined):
        """
        Initialize the PoseBusters analysis class.
        
        Parameters:
        -----------
        df_combined : pandas.DataFrame
            Combined dataframe with results including PoseBusters metrics for all methods
        """
        super().__init__(df_combined)
        
        # Check if PoseBusters columns are available
        self.available_bust_columns = [
            col for col in self.BUST_TEST_COLUMNS if col in df_combined.columns
        ]
        
        if len(self.available_bust_columns) == 0:
            print("Warning: No PoseBusters metrics found in the dataset.")
    
    def analyze_single_method(self, method, rmsd_threshold=2.0, imbalance_threshold=0.05, plot=True):
        """
        Analyze how PoseBusters validity metrics affect success rate of a single docking method.
        
        Parameters:
        -----------
        method : str
            Name of the docking method to analyze
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful, default is 2.0 Å
        imbalance_threshold : float, optional
            Threshold below which a metric is considered severely imbalanced
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing analysis results by validity category
        """
        from scipy import stats
        if len(self.available_bust_columns) == 0:
            print("No PoseBusters metrics found in the dataset. Cannot perform analysis.")
            return None
        
        # Get best RMSD per protein for the method
        best_rmsd_df = self.get_best_rmsd_per_protein(method, rmsd_threshold)
        
        # Separate successful and unsuccessful cases
        successful = best_rmsd_df[best_rmsd_df[f'rmsd_≤_{rmsd_threshold}å']]
        unsuccessful = best_rmsd_df[~best_rmsd_df[f'rmsd_≤_{rmsd_threshold}å']]
        
        success_rate = len(successful) / len(best_rmsd_df) * 100 if len(best_rmsd_df) > 0 else 0
        print(f"Method: {method}")
        print(f"Total proteins: {len(best_rmsd_df)}")
        print(f"Success rate (RMSD ≤ {rmsd_threshold}Å): {success_rate:.2f}%")
        
        # Analyze metrics
        posebusters_stats = {}
        
        for column in self.available_bust_columns:
            if column not in best_rmsd_df.columns:
                continue
                
            # Calculate class balance
            column_values = best_rmsd_df[column].fillna(False)
            true_count = column_values.sum()
            false_count = len(column_values) - true_count
            true_ratio = true_count / len(column_values) if len(column_values) > 0 else 0
            false_ratio = false_count / len(column_values) if len(column_values) > 0 else 0
            
            # Check for severe imbalance
            is_imbalanced = min(true_ratio, false_ratio) < imbalance_threshold
            
            # Calculate pass rates
            success_pass_rate = successful[column].fillna(False).mean() * 100 if len(successful) > 0 else 0
            fail_pass_rate = unsuccessful[column].fillna(False).mean() * 100 if len(unsuccessful) > 0 else 0
            
            # Calculate success rates for poses that pass and fail this metric
            pass_this_metric = best_rmsd_df[best_rmsd_df[column].fillna(False)]
            fail_this_metric = best_rmsd_df[~best_rmsd_df[column].fillna(False)]
            
            # Get counts for each category
            pass_count = len(pass_this_metric)
            fail_count = len(fail_this_metric)
            
            # Calculate success rates with handling for empty groups
            success_rate_when_pass = pass_this_metric[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if pass_count > 0 else 0
            success_rate_when_fail = fail_this_metric[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if fail_count > 0 else 0
            success_rate_difference = success_rate_when_pass - success_rate_when_fail
            
            # Fisher's exact test
            contingency = [
                [pass_this_metric[f'rmsd_≤_{rmsd_threshold}å'].sum(), 
                 pass_count - pass_this_metric[f'rmsd_≤_{rmsd_threshold}å'].sum()],
                [fail_this_metric[f'rmsd_≤_{rmsd_threshold}å'].sum(), 
                 fail_count - fail_this_metric[f'rmsd_≤_{rmsd_threshold}å'].sum()]
            ]
            
            try:
                _, p_value = stats.fisher_exact(contingency)
                significant = p_value < 0.05
                
                # If imbalanced, adjust significance interpretation
                if is_imbalanced:
                    significance_reliable = False
                else:
                    significance_reliable = True
            except Exception as e:
                print(f"Error performing Fisher's exact test for {column}: {e}")
                p_value, significant, significance_reliable = None, False, False
            
            posebusters_stats[column] = {
                'success_pass_rate': success_pass_rate,
                'fail_pass_rate': fail_pass_rate,
                'metric_difference': success_pass_rate - fail_pass_rate,
                'success_rate_when_pass': success_rate_when_pass,
                'success_rate_when_fail': success_rate_when_fail,
                'success_rate_difference': success_rate_difference,
                'p_value': p_value,
                'significant': significant,
                'significance_reliable': significance_reliable,
                'category': self.CATEGORY_MAPPING.get(column, "Other"),
                'pass_count': pass_count,
                'fail_count': fail_count,
                'is_imbalanced': is_imbalanced,
                'balance_ratio': min(true_ratio, false_ratio) / max(true_ratio, false_ratio) if max(true_ratio, false_ratio) > 0 else 0
            }
        
        # Organize by category
        metrics_by_category = {}
        for metric, stats in posebusters_stats.items():
            category = stats['category']
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)
        
        if plot:
            self._plot_single_method_results(method, rmsd_threshold, posebusters_stats, metrics_by_category)
        
        # Summary of significant metrics
        significant_metrics = {k: v for k, v in posebusters_stats.items() if v.get('significant', False)}
        
        print("\nMetrics with significant impact on success rate:")
        for category in ["Chemical Validity", "Intramolecular Validity", "Intermolecular Validity"]:
            category_metrics = [m for m, stats in significant_metrics.items() 
                              if stats['category'] == category]
            
            if category_metrics:
                print(f"\n{category}:")
                for metric in category_metrics:
                    stats = posebusters_stats[metric]
                    confidence_note = " (LOW CONFIDENCE due to imbalance)" if stats['is_imbalanced'] else ""
                    print(f"- {metric}: Success rate {stats['success_rate_when_pass']:.1f}% when pass vs. "
                          f"{stats['success_rate_when_fail']:.1f}% when fail (diff: {stats['success_rate_difference']:.1f}%, "
                          f"p={stats['p_value']:.4f}, pass/fail counts: {stats['pass_count']}/{stats['fail_count']}){confidence_note}")
            else:
                print(f"\n{category}: No significant metrics found")
        
        # Print summary of imbalanced metrics
        imbalanced_metrics = {k: v for k, v in posebusters_stats.items() if v.get('is_imbalanced', False)}
        if imbalanced_metrics:
            print("\nNote: The following metrics have severe class imbalance (one class < 5% of data):")
            for metric, stats in imbalanced_metrics.items():
                min_class = "pass" if stats['pass_count'] < stats['fail_count'] else "fail"
                min_count = min(stats['pass_count'], stats['fail_count'])
                total = stats['pass_count'] + stats['fail_count']
                percent = (min_count / total) * 100 if total > 0 else 0
                print(f"- {metric}: Only {min_count} samples ({percent:.2f}%) {min_class} this metric")
        
        return {
            'method': method,
            'overall_success_rate': success_rate,
            'metrics_stats': posebusters_stats,
            'metrics_by_category': metrics_by_category,
            'significant_metrics': significant_metrics,
            'imbalanced_metrics': imbalanced_metrics
        }
    
    def _plot_single_method_results(self, method, rmsd_threshold, posebusters_stats, metrics_by_category):
        """
        Helper function to plot single method results.
        """
        # Visualize results by category
        for category, metrics in metrics_by_category.items():
            if not metrics:
                continue
            
            # Sort metrics by success rate difference
            metrics.sort(key=lambda x: posebusters_stats[x]['success_rate_difference'], reverse=True)
            
            # Create figure
            plt.figure(figsize=(12, max(6, len(metrics) * 0.5)))
            
            # Prepare data for the plot
            labels = []
            differences = []
            significant = []
            is_imbalanced = []
            
            for m in metrics:
                stat = posebusters_stats[m]
                # Add sample size to label for context
                labels.append(f"{m.replace('_', ' ').title()} (P:{stat['pass_count']}/F:{stat['fail_count']})")
                differences.append(stat['success_rate_difference'])
                significant.append(stat['significant'])
                is_imbalanced.append(stat['is_imbalanced'])
            
            # Create bar chart with lighter color for imbalanced metrics
            colors = []
            for i, diff in enumerate(differences):
                if is_imbalanced[i]:
                    # Use lighter colors for imbalanced metrics
                    colors.append('lightgreen' if diff > 0 else 'lightcoral')
                else:
                    colors.append('green' if diff > 0 else 'red')
            
            bars = plt.barh(labels, differences, color=colors)
            
            # Add significance markers with different symbol for imbalanced metrics
            for i, (sig, imb) in enumerate(zip(significant, is_imbalanced)):
                if sig:
                    marker = 'o' if imb else '*'
                    plt.text(0, i, marker, ha='center', fontsize=14, fontweight='bold')
            
            # Add reference line and labels
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.xlabel('Difference in Success Rate (% points) when metric passes vs. fails')
            plt.title(f'{category} Metrics Impact on Success for {method}')
            plt.grid(axis='x', alpha=0.3)
            
            # Add legend for imbalanced metrics
            legend_elements = [
                plt.Rectangle((0,0),1,1, color='green', label='Balanced'),
                plt.Rectangle((0,0),1,1, color='lightgreen', label='Imbalanced (<5% in one class)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
                          markersize=10, label='Significant'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                          markersize=10, label='Significant but unreliable')
            ]
            plt.legend(handles=legend_elements, loc='lower right')
            
            # Add data labels with dynamic placement to avoid collision
            for bar, imb, val in zip(bars, is_imbalanced, differences):
                width = bar.get_width()
                # Y position: center of the bar
                y = bar.get_y() + bar.get_height() / 2
                confidence = '' if not imb else ' (low confidence)'
                # Choose offset and alignment based on bar direction and size
                if width >= 0:
                    offset = 1 if abs(width) < 1 else 3
                    ha = 'left'
                else:
                    offset = -1 if abs(width) < 1 else -3
                    ha = 'right'
                # X position: end of bar plus offset
                x = bar.get_x() + width + offset
                plt.text(x, y, f'{width:.1f}%{confidence}', ha=ha, va='center', color='black')

            # Expand horizontal margins so negative labels don’t get clipped
            plt.gca().margins(x=0.1)
            plt.subplots_adjust(left=0.25)

            plt.tight_layout()
            plt.show()
    
    def analyze_universal(self, methods, rmsd_threshold=2.0, plot=True):
        """
        Analyze PoseBusters metrics for cases where all methods succeed or all methods fail.
        
        Parameters:
        -----------
        methods : list
            List of method names to analyze
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful, default is 2.0 Å
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and visualizations
        """
        if len(self.available_bust_columns) == 0:
            print("No PoseBusters metrics found in the dataset. Cannot perform analysis.")
            return None
        
        # Create success matrix
        success_matrix = self.create_success_matrix(methods, rmsd_threshold)
        
        # Get best RMSD per protein for each method
        best_rmsd_by_method = {method: self.get_best_rmsd_per_protein(method, rmsd_threshold) 
                              for method in methods}
        
        # Identify proteins where all methods succeed or all methods fail
        all_success_proteins = success_matrix[
            success_matrix['total_successes'] == len(methods)
        ]['protein'].tolist()
        
        all_failure_proteins = success_matrix[
            success_matrix['total_successes'] == 0
        ]['protein'].tolist()
        
        print(f"Analyzing PoseBusters metrics for {len(all_success_proteins)} cases where all methods succeed")
        print(f"and {len(all_failure_proteins)} cases where all methods fail")
        
        # Check if we have enough data to analyze
        if len(all_success_proteins) < 5 or len(all_failure_proteins) < 5:
            print("Warning: Small sample size may affect statistical significance.")
        
        # Get PoseBusters results for these proteins
        all_success_results = pd.DataFrame()
        all_failure_results = pd.DataFrame()
        
        # For each method, collect the PoseBusters data
        for method in methods:
            method_success = best_rmsd_by_method[method][
                best_rmsd_by_method[method]['protein'].isin(all_success_proteins)
            ]
            method_failure = best_rmsd_by_method[method][
                best_rmsd_by_method[method]['protein'].isin(all_failure_proteins)
            ]
            
            all_success_results = pd.concat([all_success_results, method_success])
            all_failure_results = pd.concat([all_failure_results, method_failure])
        
        # Analyze PoseBusters metrics
        posebusters_stats = {}
        
        for column in self.available_bust_columns:
            if column not in all_success_results.columns or column not in all_failure_results.columns:
                continue
                
            # Calculate pass rates
            success_pass_rate = all_success_results[column].fillna(False).mean() * 100 if len(all_success_results) > 0 else 0
            failure_pass_rate = all_failure_results[column].fillna(False).mean() * 100 if len(all_failure_results) > 0 else 0
            
            # Fisher's exact test
            contingency = [
                [all_success_results[column].fillna(False).sum(), 
                 len(all_success_results) - all_success_results[column].fillna(False).sum()],
                [all_failure_results[column].fillna(False).sum(), 
                 len(all_failure_results) - all_failure_results[column].fillna(False).sum()]
            ]
            
            try:
                _, p_value = stats.fisher_exact(contingency)
                significant = p_value < 0.05
            except:
                p_value, significant = None, False
            
            posebusters_stats[column] = {
                'success_pass_rate': success_pass_rate,
                'failure_pass_rate': failure_pass_rate,
                'difference': success_pass_rate - failure_pass_rate,
                'p_value': p_value,
                'significant': significant,
                'category': self.CATEGORY_MAPPING.get(column, "Other"),
                'success_count': all_success_results[column].fillna(False).sum(),
                'failure_count': all_failure_results[column].fillna(False).sum(),
            }
        
        # Organize metrics by category
        metrics_by_category = {}
        for metric, stats in posebusters_stats.items():
            category = stats['category']
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)
        
        if plot and posebusters_stats:
            self._plot_universal_results(posebusters_stats, metrics_by_category)
        
        # Summary of significant differences
        significant_metrics = {k: v for k, v in posebusters_stats.items() if v.get('significant', False)}
        
        if significant_metrics:
            print("\nSignificant differences in PoseBusters metrics:")
            for metric, stats in significant_metrics.items():
                print(f"{metric}: Success = {stats['success_pass_rate']:.1f}%, "
                      f"Failure = {stats['failure_pass_rate']:.1f}%, "
                      f"difference = {stats['difference']:.1f}%, "
                      f"p-value = {stats['p_value']:.4f}")
        else:
            print("\nNo statistically significant differences found in PoseBusters metrics.")
        
        # Insights and conclusions
        if significant_metrics:
            print("\nInsights by category:")
            
            # Group insights by category
            insights_by_category = {}
            for metric, stats in significant_metrics.items():
                category = stats['category']
                if category not in insights_by_category:
                    insights_by_category[category] = []
                
                if stats['difference'] > 0:
                    insights_by_category[category].append(
                        f"Higher pass rate for {metric} is associated with universal success " +
                        f"({stats['success_pass_rate']:.1f}% vs {stats['failure_pass_rate']:.1f}%)"
                    )
                else:
                    insights_by_category[category].append(
                        f"Lower pass rate for {metric} is associated with universal success " +
                        f"({stats['success_pass_rate']:.1f}% vs {stats['failure_pass_rate']:.1f}%)"
                    )
            
            for category, insights in insights_by_category.items():
                print(f"\n{category}:")
                for insight in insights:
                    print(f"- {insight}")
        
        return {
            'posebusters_stats': posebusters_stats,
            'all_success_proteins': all_success_proteins,
            'all_failure_proteins': all_failure_proteins,
            'all_success_results': all_success_results,
            'all_failure_results': all_failure_results,
            'significant_metrics': significant_metrics,
            'categories': metrics_by_category,
            'success_matrix': success_matrix
        }
    
    def _plot_universal_results(self, posebusters_stats, metrics_by_category):
        """Helper function to plot universal success vs. failure results."""
        # Create visualization by category
        for category, metrics in metrics_by_category.items():
            if not metrics:
                continue
                
            plt.figure(figsize=(12, max(6, len(metrics) * 0.5)))
            metrics.sort(key=lambda x: posebusters_stats[x]['difference'], reverse=True)
            
            success_rates = [posebusters_stats[m]['success_pass_rate'] for m in metrics]
            failure_rates = [posebusters_stats[m]['failure_pass_rate'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, success_rates, width, label='All methods succeed', color='green', alpha=0.7)
            bars2 = plt.bar(x + width/2, failure_rates, width, label='All methods fail', color='red', alpha=0.7)
            
            # Add significance markers
            for i, metric in enumerate(metrics):
                if posebusters_stats[metric]['significant']:
                    plt.text(i, max(success_rates[i], failure_rates[i]) + 3, '*', 
                             ha='center', fontsize=14, fontweight='bold')
            
            plt.xlabel('PoseBusters Metric')
            plt.ylabel('Pass Rate (%)')
            plt.title(f'{category} Metrics: Universal Success vs. Universal Failure')
            plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add data labels
            for i, v in enumerate(success_rates):
                plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
            for i, v in enumerate(failure_rates):
                plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
                
            plt.tight_layout()
            plt.show()
        
        # Create summary heatmap of differences
        plt.figure(figsize=(10, max(8, len(posebusters_stats) * 0.4)))
        
        # Prepare data for heatmap
        metric_labels = []
        difference_values = []
        significance = []
        categories = []
        
        for metric, stats in posebusters_stats.items():
            metric_labels.append(metric.replace('_', ' ').title())
            difference_values.append(stats['difference'])
            significance.append(stats['significant'])
            categories.append(stats['category'])
        
        # Create a dataframe for the heatmap
        heatmap_df = pd.DataFrame({
            'Metric': metric_labels,
            'Difference': difference_values,
            'Significant': significance,
            'Category': categories
        })
        
        # Sort by category and then by difference
        heatmap_df = heatmap_df.sort_values(['Category', 'Difference'], ascending=[True, False])
        
        # Create the heatmap
        heatmap = sns.heatmap(
            heatmap_df.pivot_table(
                index='Metric',
                values='Difference',
                aggfunc='first'
            ).sort_index(),
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Difference in Pass Rate (%) (Success - Failure)'}
        )
        
        # Add asterisks for significant differences
        for i, sig in enumerate(heatmap_df['Significant']):
            if sig:
                plt.text(1.05, i + 0.5, '*', fontsize=14, fontweight='bold')
        
        plt.title('Difference in PoseBusters Pass Rates: Universal Success vs. Universal Failure')
        plt.tight_layout()
        plt.show()
    
    def analyze_comparative(self, method1, method2, rmsd_threshold=2.0, plot=True):
        """
        Analyze PoseBusters metrics for cases where one method succeeds and the other fails,
        using McNemar's test for paired data.
        
        Parameters:
        -----------
        method1 : str
            Name of the first method to compare
        method2 : str
            Name of the second method to compare
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful, default is 2.0 Å
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and visualizations
        """
        if len(self.available_bust_columns) == 0:
            print("No PoseBusters metrics found in the dataset. Cannot perform analysis.")
            return None
        
        # Get best RMSD per protein for each method
        best_rmsd_method1 = self.get_best_rmsd_per_protein(method1, rmsd_threshold)
        best_rmsd_method2 = self.get_best_rmsd_per_protein(method2, rmsd_threshold)
        
        # Find common proteins
        common_proteins = set(best_rmsd_method1['protein']).intersection(set(best_rmsd_method2['protein']))
        
        # Create a dataframe to track success/failure for each protein
        success_matrix = pd.DataFrame({'protein': list(common_proteins)})
        method1_success = {}
        method2_success = {}
        
        for protein in common_proteins:
            method1_success[protein] = best_rmsd_method1[
                best_rmsd_method1['protein'] == protein
            ][f'rmsd_≤_{rmsd_threshold}å'].iloc[0]
            
            method2_success[protein] = best_rmsd_method2[
                best_rmsd_method2['protein'] == protein
            ][f'rmsd_≤_{rmsd_threshold}å'].iloc[0]
        
        success_matrix[f'{method1}_success'] = success_matrix['protein'].map(method1_success)
        success_matrix[f'{method2}_success'] = success_matrix['protein'].map(method2_success)
        
        # Identify proteins where one method succeeds and the other fails
        method1_only_success = success_matrix[
            (success_matrix[f'{method1}_success']) & (~success_matrix[f'{method2}_success'])
        ]['protein'].tolist()
        
        method2_only_success = success_matrix[
            (~success_matrix[f'{method1}_success']) & (success_matrix[f'{method2}_success'])
        ]['protein'].tolist()
        
        # Extract results for the differential success cases
        method1_only_results = best_rmsd_method1[best_rmsd_method1['protein'].isin(method1_only_success)]
        method2_only_results = best_rmsd_method2[best_rmsd_method2['protein'].isin(method2_only_success)]
        
        print(f"Analyzing PoseBusters metrics for {len(method1_only_success)} cases where {method1} succeeds but {method2} fails")
        print(f"and {len(method2_only_success)} cases where {method2} succeeds but {method1} fails")
        
        # Create a merged dataset for paired analysis
        merged_results = pd.merge(
            best_rmsd_method1[['protein'] + self.available_bust_columns], 
            best_rmsd_method2[['protein'] + self.available_bust_columns],
            on='protein',
            suffixes=(f'_{method1}', f'_{method2}')
        )
        
        # Analyze PoseBusters metrics
        posebusters_stats = {}
        
        for column in self.available_bust_columns:
            if column not in method1_only_results.columns or column not in method2_only_results.columns:
                continue
                
            # Calculate pass rates
            method1_pass_rate = method1_only_results[column].fillna(False).mean() * 100
            method2_pass_rate = method2_only_results[column].fillna(False).mean() * 100
            
            # Create the column names with method suffixes
            col1 = f"{column}_{method1}"
            col2 = f"{column}_{method2}"
            
            # Create a contingency table for McNemar's test
            # Format: [[both pass, method1 pass & method2 fail], [method1 fail & method2 pass, both fail]]
            contingency = np.zeros((2, 2), dtype=int)
            
            # Fill the contingency table
            both_pass = ((merged_results[col1].fillna(False)) & (merged_results[col2].fillna(False))).sum()
            m1_pass_m2_fail = ((merged_results[col1].fillna(False)) & (~merged_results[col2].fillna(False))).sum()
            m1_fail_m2_pass = ((~merged_results[col1].fillna(False)) & (merged_results[col2].fillna(False))).sum()
            both_fail = ((~merged_results[col1].fillna(False)) & (~merged_results[col2].fillna(False))).sum()
            
            contingency[0, 0] = both_pass
            contingency[0, 1] = m1_pass_m2_fail
            contingency[1, 0] = m1_fail_m2_pass
            contingency[1, 1] = both_fail
            
            # Calculate McNemar's test
            try:
                # Use exact=True for small sample sizes
                result = mcnemar(contingency, exact=True, correction=True)
                statistic = result.statistic
                p_value = result.pvalue
                significant = p_value < 0.05
            except:
                statistic, p_value, significant = None, None, False
            
            # Calculate odds ratio as effect size measure
            if m1_pass_m2_fail == 0 or m1_fail_m2_pass == 0:
                # Apply correction to avoid division by zero
                odds_ratio_value = ((m1_pass_m2_fail + 0.5) / (m1_fail_m2_pass + 0.5))
            else:
                odds_ratio_value = (m1_pass_m2_fail / m1_fail_m2_pass)
            
            posebusters_stats[column] = {
                f'{method1}_pass_rate': method1_pass_rate,
                f'{method2}_pass_rate': method2_pass_rate,
                'difference': method1_pass_rate - method2_pass_rate,
                'mcnemar_statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'category': self.CATEGORY_MAPPING.get(column, "Other"),
                'odds_ratio': odds_ratio_value,
                'contingency_table': contingency,
                'both_pass': both_pass,
                'm1_pass_m2_fail': m1_pass_m2_fail,
                'm1_fail_m2_pass': m1_fail_m2_pass,
                'both_fail': both_fail
            }
        
        # Organize metrics by category
        metrics_by_category = {}
        for metric, stats in posebusters_stats.items():
            category = stats['category']
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)
        
        if plot:
            self._plot_comparative_results(method1, method2, posebusters_stats, metrics_by_category)
        
        # Summary of significant differences
        significant_metrics = {k: v for k, v in posebusters_stats.items() if v.get('significant', False)}
        
        if significant_metrics:
            print("\nSignificant differences in PoseBusters metrics (McNemar's test):")
            for metric, stats in significant_metrics.items():
                print(f"{metric}: {method1} = {stats[f'{method1}_pass_rate']:.1f}%, "
                      f"{method2} = {stats[f'{method2}_pass_rate']:.1f}%, "
                      f"difference = {stats['difference']:.1f}%, "
                      f"p-value = {stats['p_value']:.4f}, "
                      f"odds ratio = {stats['odds_ratio']:.2f}")
                print(f"  Contingency: Both pass: {stats['both_pass']}, "
                      f"{method1} only: {stats['m1_pass_m2_fail']}, "
                      f"{method2} only: {stats['m1_fail_m2_pass']}, "
                      f"Both fail: {stats['both_fail']}")
        else:
            print("\nNo statistically significant differences found in PoseBusters metrics.")
        
        # Insights and conclusions
        if significant_metrics:
            print("\nInsights:")
            
            # Group insights by category
            insights_by_category = {}
            for metric, stats in significant_metrics.items():
                category = stats['category']
                if category not in insights_by_category:
                    insights_by_category[category] = []
                
                if stats['difference'] > 0:
                    insights_by_category[category].append(
                        f"{method1} performs better on {metric} validation "
                        f"({stats[f'{method1}_pass_rate']:.1f}% vs {stats[f'{method2}_pass_rate']:.1f}%)"
                    )
                else:
                    insights_by_category[category].append(
                        f"{method2} performs better on {metric} validation "
                        f"({stats[f'{method2}_pass_rate']:.1f}% vs {stats[f'{method1}_pass_rate']:.1f}%)"
                    )
            
            for category, insights in insights_by_category.items():
                print(f"\n{category}:")
                for insight in insights:
                    print(f"- {insight}")
        
        return {
            'posebusters_stats': posebusters_stats,
            'method1_only_proteins': method1_only_success,
            'method2_only_proteins': method2_only_success,
            'method1_only_results': method1_only_results,
            'method2_only_results': method2_only_results,
            'significant_metrics': significant_metrics,
            'categories': metrics_by_category,
            'merged_results': merged_results
        }
    
    def _plot_comparative_results(self, method1, method2, posebusters_stats, metrics_by_category):
        """Helper function to plot comparative analysis results."""
        # Create visualization by category
        for category, metrics in metrics_by_category.items():
            if not metrics:
                continue
                
            plt.figure(figsize=(12, max(6, len(metrics) * 0.5)))
            metrics.sort(key=lambda x: posebusters_stats[x]['difference'], reverse=True)
            
            method1_rates = [posebusters_stats[m][f'{method1}_pass_rate'] for m in metrics]
            method2_rates = [posebusters_stats[m][f'{method2}_pass_rate'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, method1_rates, width, label=f'{method1} only success', color='blue', alpha=0.7)
            bars2 = plt.bar(x + width/2, method2_rates, width, label=f'{method2} only success', color='orange', alpha=0.7)
            
            # Add significance markers
            for i, metric in enumerate(metrics):
                if posebusters_stats[metric]['significant']:
                    plt.text(i, max(method1_rates[i], method2_rates[i]) + 3, '*', 
                             ha='center', fontsize=14, fontweight='bold')
            
            plt.xlabel('PoseBusters Metric')
            plt.ylabel('Pass Rate (%)')
            plt.title(f'{category} Metrics: {method1} vs {method2} Unique Successes')
            plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add data labels
            for i, v in enumerate(method1_rates):
                plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
            for i, v in enumerate(method2_rates):
                plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
                
            plt.tight_layout()
            plt.show()
        
        # Create heatmap for difference in pass rates
        plt.figure(figsize=(12, 8))
        
        # Organize data for heatmap
        categories = list(metrics_by_category.keys())
        max_metrics = max(len(metrics) for metrics in metrics_by_category.values())
        
        # Create data matrix for heatmap (difference in pass rates)
        heatmap_data = np.zeros((len(categories), max_metrics))
        heatmap_data.fill(np.nan)  # Fill with NaN to hide empty cells
        
        # X and Y tick labels
        y_labels = []
        x_labels = []
        
        for i, category in enumerate(categories):
            metrics = metrics_by_category[category]
            metrics.sort(key=lambda x: abs(posebusters_stats[x]['difference']), reverse=True)
            
            for j, metric in enumerate(metrics):
                heatmap_data[i, j] = posebusters_stats[metric]['difference']
                
                # Add metric label if this is the first category
                if i == 0:
                    x_labels.append(metric.replace('_', ' ').title())
                    
            # Add category label
            y_labels.append(category)
            
            # Fill unused cells with NaN
            for j in range(len(metrics), max_metrics):
                heatmap_data[i, j] = np.nan
        
        # Define custom colormap (blue for positive difference, orange for negative)
        colors = ["orange", "white", "blue"]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=n_bins)
        
        # Create heatmap
        ax = sns.heatmap(
            heatmap_data, 
            cmap=cmap,
            center=0,
            annot=True, 
            fmt=".1f",
            linewidths=.5, 
            yticklabels=y_labels,
            mask=np.isnan(heatmap_data),  # Mask NaN values
            cbar_kws={'label': f'Difference in Pass Rate (%) ({method1} - {method2})'}
        )
        
        # Set x-tick labels
        x_ticks = np.arange(0.5, max_metrics)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        plt.title(f'Difference in PoseBusters Pass Rates: {method1} vs {method2}')
        plt.tight_layout()
        plt.show()

    def analyze_method_group_comparative(self, group1_methods, group2_methods, group1_label=None, group2_label=None, 
                                        rmsd_threshold=2.0, plot=True):
        """
        Compare PoseBusters metrics between two groups of methods (e.g., ML-based vs. physics-based).
        
        Parameters:
        -----------
        group1_methods : list
            List of method names in the first group
        group2_methods : list
            List of method names in the second group
        group1_label : str, optional
            Label for the first group (e.g., "ML-based"). If None, uses "Group 1"
        group2_label : str, optional
            Label for the second group (e.g., "Physics-based"). If None, uses "Group 2"
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful, default is 2.0 Å
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing analysis results and visualizations
        """
        if len(self.available_bust_columns) == 0:
            print("No PoseBusters metrics found in the dataset. Cannot perform analysis.")
            return None
            
        # Set default labels if not provided
        group1_label = group1_label or "Group 1"
        group2_label = group2_label or "Group 2"
        
        print(f"Comparing {len(group1_methods)} {group1_label} methods vs. {len(group2_methods)} {group2_label} methods")
        print(f"{group1_label} methods: {', '.join(group1_methods)}")
        print(f"{group2_label} methods: {', '.join(group2_methods)}")
        
        # Get best RMSD per protein for each method
        group1_best_rmsd = {}
        for method in group1_methods:
            group1_best_rmsd[method] = self.get_best_rmsd_per_protein(method, rmsd_threshold)
            
        group2_best_rmsd = {}
        for method in group2_methods:
            group2_best_rmsd[method] = self.get_best_rmsd_per_protein(method, rmsd_threshold)
        
        # Create dataframes combining all methods in each group
        group1_combined = pd.concat(group1_best_rmsd.values())
        group2_combined = pd.concat(group2_best_rmsd.values())
        
        # Find common proteins across all methods
        all_methods = group1_methods + group2_methods
        all_proteins = set(self.df_combined['protein'].unique())
        
        # Analyze PoseBusters metrics across the groups
        posebusters_stats = {}
        
        for column in self.available_bust_columns:
            # Calculate pass rates for each group
            group1_pass_rate = group1_combined[column].fillna(False).mean() * 100 if len(group1_combined) > 0 else 0
            group2_pass_rate = group2_combined[column].fillna(False).mean() * 100 if len(group2_combined) > 0 else 0
            
            # Calculate success rates (% of proteins with RMSD ≤ threshold) when passing this metric
            group1_pass = group1_combined[group1_combined[column].fillna(False)]
            group1_fail = group1_combined[~group1_combined[column].fillna(False)]
            group2_pass = group2_combined[group2_combined[column].fillna(False)]
            group2_fail = group2_combined[~group2_combined[column].fillna(False)]
            
            g1_success_when_pass = group1_pass[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if len(group1_pass) > 0 else 0
            g1_success_when_fail = group1_fail[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if len(group1_fail) > 0 else 0
            g2_success_when_pass = group2_pass[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if len(group2_pass) > 0 else 0
            g2_success_when_fail = group2_fail[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100 if len(group2_fail) > 0 else 0
            
            # Calculate proportions and sample sizes for context
            g1_pass_proportion = len(group1_pass) / len(group1_combined) * 100 if len(group1_combined) > 0 else 0
            g2_pass_proportion = len(group2_pass) / len(group2_combined) * 100 if len(group2_combined) > 0 else 0
            
            # ── Quantile binning for continuous columns ─────────────────────
            if pd.api.types.is_numeric_dtype(self.df_combined[column]):
                # 5-quantile bins; duplicates='drop' avoids ValueError when <5 unique values
                self.df_combined[f'{column}_qbin'] = pd.qcut(
                    self.df_combined[column], q=5, duplicates='drop')
                column_to_use = f'{column}_qbin'
            else:
                column_to_use = column

            # Chi-square test for pass rate difference between groups
            try:
                contingency = [
                    [group1_combined[column].fillna(False).sum(), 
                     len(group1_combined) - group1_combined[column].fillna(False).sum()],
                    [group2_combined[column].fillna(False).sum(), 
                     len(group2_combined) - group2_combined[column].fillna(False).sum()]
                ]
                
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                significant = p_value < 0.05
            except:
                chi2, p_value, significant = None, None, False
            
            # Store the results
            posebusters_stats[column] = {
                f'{group1_label}_pass_rate': group1_pass_rate,
                f'{group2_label}_pass_rate': group2_pass_rate,
                'difference': group1_pass_rate - group2_pass_rate,
                f'{group1_label}_success_when_pass': g1_success_when_pass,
                f'{group1_label}_success_when_fail': g1_success_when_fail,
                f'{group2_label}_success_when_pass': g2_success_when_pass,
                f'{group2_label}_success_when_fail': g2_success_when_fail,
                f'{group1_label}_pass_benefit': g1_success_when_pass - g1_success_when_fail,
                f'{group2_label}_pass_benefit': g2_success_when_pass - g2_success_when_fail,
                'benefit_difference': (g1_success_when_pass - g1_success_when_fail) - 
                                     (g2_success_when_pass - g2_success_when_fail),
                'chi2': chi2,
                'p_value': p_value,
                'significant': significant,
                'category': self.CATEGORY_MAPPING.get(column, "Other"),
                f'{group1_label}_n_pass': len(group1_pass),
                f'{group1_label}_n_fail': len(group1_fail),
                f'{group2_label}_n_pass': len(group2_pass),
                f'{group2_label}_n_fail': len(group2_fail),
                f'{group1_label}_pass_proportion': g1_pass_proportion,
                f'{group2_label}_pass_proportion': g2_pass_proportion,
            }
        
        # Organize metrics by category
        metrics_by_category = {}
        for metric, stats in posebusters_stats.items():
            category = stats['category']
            if category not in metrics_by_category:
                metrics_by_category[category] = []
            metrics_by_category[category].append(metric)
        
        if plot:
            self._plot_group_comparative_results(group1_label, group2_label, posebusters_stats, metrics_by_category)
        
        # Summary of significant differences
        significant_metrics = {k: v for k, v in posebusters_stats.items() if v.get('significant', False)}
        
        if significant_metrics:
            print("\nSignificant differences in PoseBusters metrics between method groups:")
            for metric, stats in significant_metrics.items():
                print(f"{metric}: {group1_label} = {stats[f'{group1_label}_pass_rate']:.1f}%, "
                      f"{group2_label} = {stats[f'{group2_label}_pass_rate']:.1f}%, "
                      f"difference = {stats['difference']:.1f}%, "
                      f"p-value = {stats['p_value']:.4f}")
        else:
            print("\nNo statistically significant differences found in PoseBusters metrics between method groups.")
        
        # Compare how each metric affects success rates within groups
        print("\nHow PoseBusters metrics affect success rates within groups:")
        
        for category in metrics_by_category:
            print(f"\n{category}:")
            for metric in metrics_by_category[category]:
                stats = posebusters_stats[metric]
                g1_benefit = stats[f'{group1_label}_pass_benefit']
                g2_benefit = stats[f'{group2_label}_pass_benefit']
                
                print(f"- {metric}:")
                print(f"  {group1_label}: Success rate {stats[f'{group1_label}_success_when_pass']:.1f}% when pass vs. "
                      f"{stats[f'{group1_label}_success_when_fail']:.1f}% when fail (benefit: {g1_benefit:.1f}%)")
                print(f"  {group2_label}: Success rate {stats[f'{group2_label}_success_when_pass']:.1f}% when pass vs. "
                      f"{stats[f'{group2_label}_success_when_fail']:.1f}% when fail (benefit: {g2_benefit:.1f}%)")
                
                if abs(g1_benefit - g2_benefit) > 5:  # Threshold for notable difference
                    print(f"  Note: {group1_label if g1_benefit > g2_benefit else group2_label} benefits more from passing this metric "
                          f"(difference: {abs(g1_benefit - g2_benefit):.1f}%)")
        
        return {
            'posebusters_stats': posebusters_stats,
            'metrics_by_category': metrics_by_category,
            'significant_metrics': significant_metrics,
            'group1_combined': group1_combined,
            'group2_combined': group2_combined,
            'group1_label': group1_label,
            'group2_label': group2_label
        }
    
    def _plot_group_comparative_results(self, group1_label, group2_label, posebusters_stats, metrics_by_category):
        """Helper function to plot group comparative analysis results."""
        # Create visualization by category
        for category, metrics in metrics_by_category.items():
            if not metrics:
                continue
                
            plt.figure(figsize=(12, max(6, len(metrics) * 0.5)))
            metrics.sort(key=lambda x: posebusters_stats[x]['difference'], reverse=True)
            
            group1_rates = [posebusters_stats[m][f'{group1_label}_pass_rate'] for m in metrics]
            group2_rates = [posebusters_stats[m][f'{group2_label}_pass_rate'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = plt.bar(x - width/2, group1_rates, width, label=f'{group1_label}', color='blue', alpha=0.7)
            bars2 = plt.bar(x + width/2, group2_rates, width, label=f'{group2_label}', color='orange', alpha=0.7)
            
            # Add significance markers
            for i, metric in enumerate(metrics):
                if posebusters_stats[metric]['significant']:
                    plt.text(i, max(group1_rates[i], group2_rates[i]) + 3, '*', 
                             ha='center', fontsize=14, fontweight='bold')
            
            plt.xlabel('PoseBusters Metric')
            plt.ylabel('Pass Rate (%)')
            plt.title(f'{category} Metrics: {group1_label} vs {group2_label}')
            plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Add data labels
            for i, v in enumerate(group1_rates):
                plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
            for i, v in enumerate(group2_rates):
                plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
                
            plt.tight_layout()
            plt.show()
        
        # Plot the benefit comparison (how much each group benefits from passing vs failing metrics)
        plt.figure(figsize=(12, max(6, len(metrics) * 0.5)))
        metrics.sort(key=lambda x: abs(posebusters_stats[x]['benefit_difference']), reverse=True)
        
        group1_benefits = [posebusters_stats[m][f'{group1_label}_pass_benefit'] for m in metrics]
        group2_benefits = [posebusters_stats[m][f'{group2_label}_pass_benefit'] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, group1_benefits, width, label=f'{group1_label} benefit', color='blue', alpha=0.7)
        bars2 = plt.bar(x + width/2, group2_benefits, width, label=f'{group2_label} benefit', color='orange', alpha=0.7)
        
        plt.xlabel('PoseBusters Metric')
        plt.ylabel('Success Rate Benefit (%) from Passing Metric')
        plt.title(f'Benefit of Passing PoseBusters Metrics: {group1_label} vs {group2_label}')
        plt.xticks(x, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        # Create summary heatmap of differences
        plt.figure(figsize=(10, max(8, len(posebusters_stats) * 0.4)))
        # Prepare data for heatmap
        metric_labels = []
        difference_values = [] 
        significance = []
        categories = []
        for metric, stats in posebusters_stats.items():
            metric_labels.append(metric.replace('_', ' ').title())
            difference_values.append(stats['difference'])
            significance.append(stats['significant'])
            categories.append(stats['category'])
        # Create a dataframe for the heatmap
        heatmap_df = pd.DataFrame({
            'Metric': metric_labels,
            'Difference': difference_values,
            'Significant': significance,
            'Category': categories
        })
        # Sort by category and then by difference
        heatmap_df = heatmap_df.sort_values(['Category', 'Difference'], ascending=[True, False])    
        # Create the heatmap    
        heatmap = sns.heatmap(
            heatmap_df.pivot_table(
                index='Metric',
                values='Difference',
                aggfunc='first'
            ).sort_index(),
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Difference in Pass Rate (%) (Group 1 - Group 2)'}
        )
        # Add asterisks for significant differences
        for i, sig in enumerate(heatmap_df['Significant']):
            if sig:
                plt.text(1.05, i + 0.5, '*', fontsize=14, fontweight='bold')
        plt.title(f'Difference in PoseBusters Pass Rates: {group1_label} vs {group2_label}')
        plt.tight_layout()
        plt.show()
        
    def mixed_effect_analysis(self, filter_name: str, method_groups: Dict[str, List[str]], 
                            rmsd_threshold: float = 2.0, outcome_type: str = 'rmsd', 
                            plot: bool = True) -> Dict:
        """
        Perform mixed effects analysis on a PoseBusters filter, examining its effect on RMSD
        or success rate across different method groups while accounting for protein-level variation.
        
        Parameters:
        -----------
        filter_name : str
            Name of the PoseBusters filter to analyze (must be one of the available bust columns)
        method_groups : dict
            Dictionary mapping method group labels to lists of method names
            e.g., {'ML-based': ['diffdock', 'equibind'], 'Physics-based': ['vina', 'gnina']}
        rmsd_threshold : float
            RMSD threshold for determining success (used if outcome_type is 'success')
        outcome_type : str
            Type of outcome to model: 'rmsd' (continuous) or 'success' (binary)
        plot : bool
            Whether to generate visualization plots
        
        Returns:
        --------
        dict
            Dictionary containing analysis results including fitted model and statistics
        """
        
        # Check if filter exists in the dataset
        if filter_name not in self.available_bust_columns:
            print(f"Filter '{filter_name}' not found in available PoseBusters metrics.")
            return {'error': f"Filter '{filter_name}' not available"}
        
        # Validate method groups
        if not isinstance(method_groups, dict) or len(method_groups) == 0:
            return {'error': "method_groups must be a non-empty dictionary"}
        
        all_methods = sum(method_groups.values(), [])
        if len(all_methods) == 0:
            return {'error': "No methods specified in method_groups"}
        
        # Create combined dataset for analysis
        data_list = []
        for group_name, methods in method_groups.items():
            for method in methods:
                # Get best RMSD per protein for this method
                method_df = self.get_best_rmsd_per_protein(method, rmsd_threshold)
                
                # Skip if no data for this method
                if method_df is None or len(method_df) == 0:
                    continue
                    
                # Add method group label
                method_df['method_group'] = group_name
                method_df['method_name'] = method
                
                data_list.append(method_df)
        
        if not data_list:
            return {'error': "No valid data found for any of the specified methods"}
        
        # Combine all data
        combined_df = pd.concat(data_list, ignore_index=True)
        
        # Clean data - drop rows with NaN in the filter column
        analysis_df = combined_df.dropna(subset=[filter_name, 'rmsd', 'protein', 'method_group'])
        
        if len(analysis_df) < 10:  # Require minimum sample size
            return {'error': f"Insufficient data for analysis after filtering missing values: {len(analysis_df)} rows"}
        
        # Prepare outcome variable
        if outcome_type == 'success':
            analysis_df['outcome'] = (analysis_df['rmsd'] <= rmsd_threshold).astype(int)
        else:  # 'rmsd'
            # Optionally winsorize extreme RMSD values to prevent undue influence
            analysis_df['outcome'] = np.clip(analysis_df['rmsd'], None, 15.0)
        
        # Fit the mixed-effects model
        try:
            # Different model specifications depending on the outcome type
            if outcome_type == 'success':
                # For binary outcome, use logistic regression
                formula = f"outcome ~ {filter_name} * C(method_group)"
                model = smf.mixedlm(formula, analysis_df, groups=analysis_df['protein']).fit()
            else:
                # For continuous outcome, use linear mixed model
                formula = f"outcome ~ {filter_name} * C(method_group)"
                model = smf.mixedlm(formula, analysis_df, groups=analysis_df['protein']).fit()
            
            # Print model summary
            print(model.summary())
            
            # Test for significance of the interaction effect
            # Fit model without interaction term for comparison
            formula_no_interact = f"outcome ~ {filter_name} + C(method_group)"
            model_no_interact = smf.mixedlm(formula_no_interact, analysis_df, 
                                        groups=analysis_df['protein']).fit()
            
            # Likelihood ratio test
            lr_stat = 2 * (model.llf - model_no_interact.llf)
            df_diff = len(model.params) - len(model_no_interact.params)
            p_interaction = stats.chi2.sf(lr_stat, df_diff)
            
            print(f"\nInteraction test ({filter_name} × method_group):")
            print(f"LR statistic: {lr_stat:.3f}, df: {df_diff}, p-value: {p_interaction:.6f}")
            
            # Determine if the filter effect is significant
            filter_p_value = model.pvalues.get(filter_name, np.nan)
            filter_significant = filter_p_value < 0.05 if not np.isnan(filter_p_value) else False
            
            # Determine if the filter effect varies by method group (interaction)
            interaction_significant = p_interaction < 0.05
            
            # Create plots if requested
            if plot:
                self._plot_posebusters_mixed_effects(analysis_df, filter_name, model, 
                                                outcome_type, rmsd_threshold, 
                                                p_interaction=p_interaction)
            
            # Return analysis results
            results = {
                'model': model,
                'model_no_interact': model_no_interact,
                'lr_stat': lr_stat,
                'p_interaction': p_interaction,
                'filter_p_value': filter_p_value,
                'filter_significant': filter_significant,
                'interaction_significant': interaction_significant,
                'outcome_type': outcome_type,
                'filter_name': filter_name,
                'formula': formula,
                'analysis_df': analysis_df
            }
            
            return results
        
        except Exception as e:
            print(f"Error fitting mixed effects model: {str(e)}")
            return {'error': str(e)}

    def _plot_posebusters_mixed_effects(self, df, filter_name, model, outcome_type, rmsd_threshold, p_interaction=None):
        """
        Helper function to plot mixed effects analysis results for PoseBusters filters.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the analysis data
        filter_name : str
            Name of the PoseBusters filter analyzed
        model : statsmodels.regression.mixed_linear_model.MixedLMResults
            Fitted mixed effects model
        outcome_type : str
            Type of outcome modeled: 'rmsd' (continuous) or 'success' (binary)
        rmsd_threshold : float
            RMSD threshold used for success determination
        p_interaction : float, optional
            P-value for interaction term from likelihood ratio test
        """
        # Create figure with 2 subplots: raw data and predicted values
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Plot raw data - scatter plot with method groups in different colors
        method_groups = df['method_group'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(method_groups)))
        
        for i, group in enumerate(method_groups):
            group_data = df[df['method_group'] == group]
            
            # For binary filter, jitter the x-values slightly for better visibility
            if np.all(np.isin(group_data[filter_name].unique(), [0, 1])):
                x = group_data[filter_name] + np.random.uniform(-0.1, 0.1, size=len(group_data))
            else:
                x = group_data[filter_name]
                
            axes[0].scatter(x, group_data['outcome'], 
                        alpha=0.5, label=group, color=colors[i])
        
        # Add reference line for success threshold if outcome is RMSD
        if outcome_type == 'rmsd':
            axes[0].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                        label=f'RMSD = {rmsd_threshold}Å')
        
        axes[0].set_xlabel(f'PoseBusters Filter: {filter_name}')
        axes[0].set_ylabel('RMSD (Å)' if outcome_type == 'rmsd' else 'Success (1=yes, 0=no)')
        axes[0].set_title('Raw Data')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Plot predicted values based on the model
        # Create prediction grid
        if np.all(np.isin(df[filter_name].unique(), [0, 1])):
            # Binary filter - just use 0 and 1
            pred_x = np.array([0, 1])
        else:
            # Continuous filter - create range
            pred_x = np.linspace(df[filter_name].min(), df[filter_name].max(), 100)
        
        # Generate predictions for each method group
        for i, group in enumerate(method_groups):
            # Create prediction data
            pred_data = pd.DataFrame({
                filter_name: pred_x,
                'method_group': group,
                'protein': df['protein'].iloc[0]  # Just need a valid protein ID
            })
            
            # Add other necessary columns with default values
            for col in df.columns:
                if col not in pred_data and col != 'outcome' and col != 'rmsd':
                    if col in model.params:
                        pred_data[col] = df[col].iloc[0]
            
            # Make predictions
            try:
                pred_data['predicted'] = model.predict(pred_data)
                
                # Plot predictions
                axes[1].plot(pred_x, pred_data['predicted'], 
                            label=f"{group} (pred)", color=colors[i], linewidth=2)
                
                # Add confidence intervals if possible
                try:
                    preds = model.get_prediction(pred_data)
                    ci = preds.conf_int()
                    axes[1].fill_between(pred_x, ci[:, 0], ci[:, 1], 
                                        alpha=0.2, color=colors[i])
                except:
                    # Skip confidence intervals if not available
                    pass
                    
            except Exception as e:
                print(f"Error generating predictions for {group}: {str(e)}")
        
        # Add reference line for success threshold if outcome is RMSD
        if outcome_type == 'rmsd':
            axes[1].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                        label=f'RMSD = {rmsd_threshold}Å')
        
        axes[1].set_xlabel(f'PoseBusters Filter: {filter_name}')
        axes[1].set_ylabel('Predicted RMSD (Å)' if outcome_type == 'rmsd' else 'Predicted Success Probability')
        
        # Add interaction p-value to the title if provided
        title = 'Model Predictions'
        if p_interaction is not None:
            title += f' (Interaction p={p_interaction:.4f})'
        axes[1].set_title(title)
        
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Mixed Effects Analysis of {filter_name} on {"RMSD" if outcome_type == "rmsd" else "Success Rate"}', 
                    fontsize=14, y=1.05)
        plt.subplots_adjust(top=0.85)
        plt.show()

class PropertyAnalysis(DockingAnalysisBase):
    """Analysis of various property types and their impact on docking success."""
        # Define constants for PoseBusters metrics
    
    PLINDER_TEST_COLUMNS = [
        # binary 
        "ligand_is_covalent", "ligand_is_ion", "ligand_is_cofactor", "ligand_is_artifact",
        # discrete
        "system_num_protein_chains",
        "ligand_num_rot_bonds", "ligand_num_hbd", "ligand_num_hba", "ligand_num_rings",
        # continuous
        "entry_resolution", "entry_validation_molprobity", 
        "system_num_pocket_residues", "system_num_interactions",
        "ligand_molecular_weight", "ligand_crippen_clogp", 
        "ligand_num_interacting_residues", "ligand_num_neighboring_residues", "ligand_num_interactions",
    ]
    
    # Create category mapping for visualization
    CATEGORY_MAPPING = {
        "ligand_is_covalent": "Ligand Properties",
        "ligand_is_ion": "Ligand Properties",
        "ligand_is_cofactor": "Ligand Properties",
        "ligand_is_artifacrt": "Ligand Properties",
        "system_num_protein_chains": "System Properties",
        "ligand_num_rot_bonds": "Ligand Properties",    
        "ligand_num_hbd": "Ligand Properties",
        "ligand_num_hba": "Ligand Properties",
        "ligand_num_rings": "Ligand Properties",
        "entry_resolution": "Entry Properties",
        "entry_validation_molprobity": "Entry Properties",
        "system_num_pocket_residues": "System Properties",
        "system_num_interactions": "System Properties",
        "ligand_molecular_weight": "Ligand Properties",
        "ligand_crippen_clogp": "Ligand Properties",
        "ligand_num_interacting_residues": "Ligand Properties",
        "ligand_num_neighboring_residues": "Ligand Properties",
        "ligand_num_interactions": "Ligand Properties",
        "ligand_is_artifact": "Ligand Properties"     
    }
    
    def __init__(self, df_combined):
        """
        Initialize the property analysis class.
        
        Parameters:
        -----------
        df_combined : pandas.DataFrame
            Combined dataframe with results from all methods including properties to analyze
        """
        super().__init__(df_combined)
        
    def analyze_property_single_method(self, method: str, property_name: str, 
                                       property_type: str = 'continuous', 
                                       rmsd_threshold: float = 2.0,
                                       bins: List[float] = None,
                                       plot: bool = True) -> Dict:
        """
        Analyze how a specific property affects success rate of a single docking method.
        
        Parameters:
        -----------
        method : str
            Name of the docking method to analyze
        property_name : str
            Name of the property column to analyze
        property_type : str, optional
            Type of property: 'continuous', 'discrete', or 'binary'
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful, default is 2.0 Å
        bins : list, optional
            Custom bins for categorizing continuous properties
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing analysis results for the property
        """
        # Get best RMSD per protein for the method
        best_rmsd_df = self.get_best_rmsd_per_protein(method, rmsd_threshold)
        
        if property_name not in best_rmsd_df.columns:
            print(f"Property '{property_name}' not found in the dataset for method '{method}'")
            return None
        
        # Clean data - drop rows with NaN in the property column
        analysis_df = best_rmsd_df.dropna(subset=[property_name])
        
        # Check if we have enough data
        if len(analysis_df) < 5:
            print(f"Not enough data for analysis: only {len(analysis_df)} samples with valid {property_name}")
            return None
        
        # Analyze based on property type
        if property_type == 'continuous':
            return self._analyze_continuous_property(analysis_df, property_name, rmsd_threshold, bins, plot, method)
        elif property_type == 'discrete':
            return self._analyze_discrete_property(analysis_df, property_name, rmsd_threshold, plot, method)
        elif property_type == 'binary':
            return self._analyze_binary_property(analysis_df, property_name, rmsd_threshold, plot, method)
        else:
            print(f"Invalid property type: {property_type}. Must be 'continuous', 'discrete', or 'binary'")
            return None
    
    def _analyze_continuous_property(self, df, property_name, rmsd_threshold, bins, plot, method_name=None):
        """Helper function to analyze continuous properties."""
        # Calculate correlations
        correlation, p_value = stats.pearsonr(df[property_name], df['rmsd'])
        spearman_corr, spearman_p = stats.spearmanr(df[property_name], df['rmsd'])

        # Create property categories for binning
        if bins is None:
            # Auto-create 5 bins
            min_val = df[property_name].min()
            max_val = df[property_name].max()
            range_val = max_val - min_val
            step = range_val / 5
            bins = [min_val, min_val+step, min_val+2*step, min_val+3*step, min_val+4*step, float('inf')]
            bin_labels = [f'≤{min_val+step:.1f}', f'{min_val+step:.1f}-{min_val+2*step:.1f}',
                         f'{min_val+2*step:.1f}-{min_val+3*step:.1f}',
                         f'{min_val+3*step:.1f}-{min_val+4*step:.1f}', f'>{min_val+4*step:.1f}']
        else:
            if len(bins) < 3:
                print("Bins list must have at least 3 elements")
                return None
            bin_labels = [f'≤{bins[1]}']
            for i in range(1, len(bins)-2):
                bin_labels.append(f'{bins[i]}-{bins[i+1]}')
            bin_labels.append(f'>{bins[-2]}')

        property_category = f'{property_name}_category'
        df[property_category] = pd.cut(df[property_name], bins=bins, labels=bin_labels)

        # Print key statistics
        title = f'Impact of {property_name} on Docking RMSD' + (f' for {method_name}' if method_name else '')
        print(f"\n{title}")
        print(f"Pearson correlation: {correlation:.4f}, p-value: {p_value:.6f}")
        print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")

        # Calculate success rates by category
        success_by_category = df.groupby(property_category).agg(
            total_count=('rmsd', 'count'),
            success_count=(f'rmsd_≤_{rmsd_threshold}å', 'sum'),
            mean_rmsd=('rmsd', 'mean')
        ).reset_index()
        success_by_category['success_rate'] = success_by_category['success_count'] / success_by_category['total_count'] * 100

        # ANOVA to test if categories have significantly different RMSDs
        categories = df[property_category].dropna().unique()
        # Decide whether classic ANOVA is appropriate --------------------------
        # If the bin counts are highly unbalanced or Levene’s test rejects
        # homoscedasticity, fall back to a more robust alternative
        if len(categories) > 1:  # Need at least 2 categories for ANOVA
            groups = [df[df[property_category] == cat]['rmsd'] for cat in categories]
            group_sizes = np.array([len(g) for g in groups])
            unbalanced = (group_sizes.max() / group_sizes.min() > 4) if len(group_sizes) > 1 else False

            # Welch‑ANOVA (unequal variances) via scipy if groups are unbalanced
            use_welch = unbalanced
            if use_welch:
                try:
                    welch_f, welch_p = stats.f_oneway(*groups, axis=0)  # Welch in SciPy ≥1.11
                except TypeError:
                    welch_f, welch_p = None, None
            else:
                welch_f, welch_p = None, None

            valid_groups = [group for group in groups if len(group) > 0]
            if len(valid_groups) > 1:
                if use_welch and welch_p is not None:
                    f_stat, anova_p = welch_f, welch_p
                else:
                    f_stat, anova_p = stats.f_oneway(*valid_groups)
                anova_significant = anova_p < 0.05
            else:
                f_stat, anova_p, anova_significant = None, None, False
        else:
            f_stat, anova_p, anova_significant = None, None, False
            welch_f, welch_p = None, None

        # Kruskal–Wallis as a distribution‑free fallback
        if len(categories) > 1:
            valid_groups = [df[df[property_category] == cat]['rmsd'] for cat in categories if len(df[df[property_category] == cat]['rmsd']) > 0]
            if len(valid_groups) > 1:
                kw_H, kw_p = stats.kruskal(*valid_groups)
            else:
                kw_H, kw_p = None, None
        else:
            kw_H, kw_p = None, None

        # Visualizations
        if plot:
            self._plot_continuous_property_analysis(
                df, property_name, property_category,
                rmsd_threshold, success_by_category, title,
                anova_p=anova_p, welch_p=welch_p, kw_p=kw_p, anova_significant=anova_significant
            )

        # Return results
        results = {
            'property_type': 'continuous',
            'property_name': property_name,
            'pearson_correlation': correlation,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'success_by_category': success_by_category,
            'anova_f_statistic': f_stat,
            'anova_p_value': anova_p,
            'anova_significant': anova_significant if anova_p is not None else False,
            'categories': list(categories),
            'welch_p_value': welch_p,
            'kruskal_p_value': kw_p,
        }

        # Print category-specific success rates
        print("\nSuccess rates by category:")
        print(success_by_category[[property_category, 'total_count', 'success_rate', 'mean_rmsd']])

        if anova_p is not None:
            print(f"\nANOVA test: F={f_stat:.3f}, p={anova_p:.6f}")
            if anova_significant:
                print(f"Significant differences in RMSD across {property_name} categories")
        if welch_p is not None:
            print(f"Welch ANOVA p={welch_p:.6f} (used unbalanced‑sizes variant)")
        if kw_p is not None:
            print(f"Kruskal–Wallis H={kw_H:.3f}, p={kw_p:.6f}")

        return results
    
    def _plot_continuous_property_analysis(self, df, property_name, property_category, rmsd_threshold,
                                          success_by_category, title,
                                          anova_p=None, welch_p=None, kw_p=None, anova_significant=None):
        """Helper function to plot continuous property analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16)

        # 1. Scatter plot with regression line
        df_plot, outlier_info = self._prepare_rmsd_for_plotting(
            df, rmsd_col=f'rmsd', clip_value=15.0, handle_outliers='clip'
        )
        sns.scatterplot(data=df_plot, x=property_name, y='rmsd_plot', alpha=0.6, ax=axes[0])

        # Add annotation if outliers were handled
        if outlier_info['n_outliers'] > 0:
            axes[0].annotate(
                f"*{outlier_info['n_outliers']} outliers > {outlier_info['display_max']:.1f}Å not shown "
                f"({outlier_info['outlier_percent']:.1f}% of data)",
                xy=(0.5, 0.97), xycoords='axes fraction',
                ha='center', fontsize=9, color='red'
            )
        sns.regplot(data=df_plot, x=property_name, y='rmsd_plot', lowess=True, scatter=False, color='red', ax=axes[0])
        axes[0].axhline(y=rmsd_threshold, color='green', linestyle='--',
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0].set_title('Correlation Plot', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Boxplot by category
        sns.boxplot(data=df_plot, x=property_category, y='rmsd', ax=axes[1])
        axes[1].axhline(y=rmsd_threshold, color='green', linestyle='--',
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1].set_ylabel('RMSD (Å)', fontsize=12)
        axes[1].set_title('RMSD Distribution by Category', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)

        # 3. Success rate by category
        bar_plot = sns.barplot(x=property_category, y='success_rate', data=success_by_category, ax=axes[2])
        axes[2].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[2].set_ylabel('Success Rate (%)', fontsize=12)
        # Add statistical test label to subplot title
        stat_label = ''
        if anova_significant is not None and anova_significant and anova_p is not None:
            stat_label = f" (ANOVA p={anova_p:.3g})"
        elif welch_p is not None:
            stat_label = f" (Welch p={welch_p:.3g})"
        elif kw_p is not None:
            stat_label = f" (KW p={kw_p:.3g})"
        axes[2].set_title(f'Success Rate (≤ {rmsd_threshold}Å) by Category' + stat_label, fontsize=14)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)

        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.patches):
            count = success_by_category['total_count'].iloc[i]
            axes[2].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'n={count}',
                ha='center',
                fontsize=9
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def _analyze_discrete_property(self, df, property_name, rmsd_threshold, plot, method_name=None):
        """Helper function to analyze discrete properties."""
        # Ensure discrete property is treated as categorical
        df[property_name] = df[property_name].astype('category')
        
        # Calculate non-parametric correlation
        spearman_corr, spearman_p = stats.spearmanr(df[property_name].cat.codes, df['rmsd'])
        
        title = f'Impact of {property_name} on Docking RMSD' + (f' for {method_name}' if method_name else '')
        print(f"\n{title}")
        print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")
        
        # Calculate success rates by discrete value
        success_by_value = df.groupby(property_name).agg(
            total_count=('rmsd', 'count'),
            success_count=(f'rmsd_≤_{rmsd_threshold}å', 'sum'),
            mean_rmsd=('rmsd', 'mean')
        ).reset_index()
        
        success_by_value['success_rate'] = success_by_value['success_count'] / success_by_value['total_count'] * 100
        
        # ANOVA to test if discrete values have significantly different RMSDs
        values = df[property_name].unique()
        if len(values) > 1:  # Need at least 2 values for ANOVA
            groups = [df[df[property_name] == val]['rmsd'] for val in values]
            valid_groups = [group for group in groups if len(group) > 0]
            if len(valid_groups) > 1:
                f_stat, anova_p = stats.f_oneway(*valid_groups)
                anova_significant = anova_p < 0.05
            else:
                f_stat, anova_p, anova_significant = None, None, False
        else:
            f_stat, anova_p, anova_significant = None, None, False
        
        # Visualizations
        if plot:
            self._plot_discrete_property_analysis(df, property_name, rmsd_threshold, success_by_value, title)
        
        # Print category-specific success rates
        print("\nSuccess rates by discrete value:")
        print(success_by_value[[property_name, 'total_count', 'success_rate', 'mean_rmsd']])
        
        if anova_p is not None:
            print(f"\nANOVA test: F={f_stat:.3f}, p={anova_p:.6f}")
            if anova_significant:
                print(f"Significant differences in RMSD across {property_name} values")
        
        # Return results
        return {
            'property_type': 'discrete',
            'property_name': property_name,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'success_by_value': success_by_value,
            'anova_f_statistic': f_stat,
            'anova_p_value': anova_p,
            'anova_significant': anova_significant if anova_p is not None else False,
            'values': list(values)
        }
    
    def _plot_discrete_property_analysis(self, df, property_name, rmsd_threshold, success_by_value, title):
        """Helper function to plot discrete property analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16)
        
        # 1. Box plot by discrete values
        # 1. Box plot
        sns.boxplot(data=df, x=property_name, y='rmsd', ax=axes[0], color='blue')
        axes[0].axhline(y=rmsd_threshold, color='green', linestyle='--', 
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[0].set_title('RMSD Distribution (Box Plot)', fontsize=14)
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Violin plot
        sns.violinplot(data=df, x=property_name, y='rmsd', ax=axes[1])
        axes[1].axhline(y=rmsd_threshold, color='green', linestyle='--', 
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[1].set_title('RMSD Distribution (Violin Plot)', fontsize=14)
        axes[1].set_xlabel(property_name, fontsize=12)
        axes[1].set_ylabel('RMSD (Å)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. Success rate bar plot
        bar_plot = sns.barplot(x=property_name, y='success_rate', data=success_by_value, ax=axes[2])
        axes[2].set_title('Success Rate by Value', fontsize=14)
        axes[2].set_xlabel(property_name, fontsize=12)
        axes[2].set_ylabel(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) %', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.patches):
            count = success_by_value['total_count'].iloc[i]
            axes[2].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'n={count}',
                ha='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def _analyze_binary_property(self, df, property_name, rmsd_threshold, plot, method_name=None):
        """Helper function to analyze binary properties."""
        # Ensure the property is treated as boolean
        if df[property_name].dtype != bool:
            # Convert to boolean if it's not already
            if df[property_name].dtype == 'object':
                df[property_name] = df[property_name].astype(str).str.lower() == 'true'
            else:
                df[property_name] = df[property_name].astype(bool)
        
        # Split data by binary property
        true_group = df[df[property_name]]
        false_group = df[~df[property_name]]
        
        # Check if we have data in both groups
        if len(true_group) == 0 or len(false_group) == 0:
            print(f"Warning: Property {property_name} has all True or all False values. Cannot perform comparison.")
            return None
        
        # Calculate success rates for each group
        true_success_rate = true_group[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100
        false_success_rate = false_group[f'rmsd_≤_{rmsd_threshold}å'].mean() * 100
        
        # Mann-Whitney U test for RMSD difference between groups
        u_stat, p_value = stats.mannwhitneyu(true_group['rmsd'], false_group['rmsd'], alternative='two-sided')
        
        # Fisher's exact test for success/failure association
        success_fail_table = pd.crosstab(df[property_name], df[f'rmsd_≤_{rmsd_threshold}å'])
        _, fisher_p = stats.fisher_exact(success_fail_table)
        
        # Calculate effect size (Cohen's d) for RMSD
        mean_true = true_group['rmsd'].mean()
        mean_false = false_group['rmsd'].mean()
        std_pooled = np.sqrt((true_group['rmsd'].var() * (len(true_group) - 1) + 
                             false_group['rmsd'].var() * (len(false_group) - 1)) / 
                            (len(true_group) + len(false_group) - 2))
        cohen_d = abs(mean_true - mean_false) / std_pooled
        
        title = f'Impact of {property_name} on Docking RMSD' + (f' for {method_name}' if method_name else '')
        print(f"\n{title}")
        print(f"Mann-Whitney U test p-value: {p_value:.6f}")
        print(f"Fisher's exact test p-value: {fisher_p:.6f}")
        print(f"Effect size (Cohen's d): {cohen_d:.3f}")
        
        print(f"\nTrue group: {len(true_group)} samples, mean RMSD = {mean_true:.2f}, success rate = {true_success_rate:.1f}%")
        print(f"False group: {len(false_group)} samples, mean RMSD = {mean_false:.2f}, success rate = {false_success_rate:.1f}%")
        
        # Visualizations
        if plot:
            self._plot_binary_property_analysis(df, property_name, rmsd_threshold, title)
        
        # Return results
        return {
            'property_type': 'binary',
            'property_name': property_name,
            'true_group_count': len(true_group),
            'false_group_count': len(false_group),
            'true_mean_rmsd': mean_true,
            'false_mean_rmsd': mean_false,
            'true_success_rate': true_success_rate,
            'false_success_rate': false_success_rate,
            'rmsd_difference': mean_true - mean_false,
            'success_rate_difference': true_success_rate - false_success_rate,
            'mannwhitney_u': u_stat,
            'mannwhitney_p': p_value,
            'fisher_p': fisher_p,
            'cohen_d': cohen_d,
            'significant_rmsd': p_value < 0.05,
            'significant_success': fisher_p < 0.05
        }
    
    def _plot_binary_property_analysis(self, df, property_name, rmsd_threshold, title):
        """Helper function to plot binary property analysis results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16)
        
        # 1. Box plot by binary value
        df_plot, outlier_info = self._prepare_rmsd_for_plotting(
            df, rmsd_col=f'rmsd', clip_value=15.0, handle_outliers='clip'
        )
        sns.boxplot(x=property_name, y='rmsd', data=df_plot, ax=axes[0])
        axes[0].axhline(y=rmsd_threshold, color='green', linestyle='--', 
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[0].set_title('RMSD Distribution', fontsize=14)
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # 2. Histogram comparison
        sns.histplot(data=df, x='rmsd', hue=property_name, element='step', 
                    stat='density', common_norm=False, bins=20, ax=axes[1])
        axes[1].axvline(x=rmsd_threshold, color='green', linestyle='--', 
                       label=f'RMSD = {rmsd_threshold}Å')
        axes[1].set_title('RMSD Distribution by Group', fontsize=14)
        axes[1].set_xlabel('RMSD (Å)', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Success rate comparison
        success_by_value = df.groupby(property_name).agg(
            total_count=('rmsd', 'count'),
            success_count=(f'rmsd_≤_{rmsd_threshold}å', 'sum')
        ).reset_index()
        success_by_value['success_rate'] = success_by_value['success_count'] / success_by_value['total_count'] * 100
        
        bar_plot = sns.barplot(x=property_name, y='success_rate', data=success_by_value, ax=axes[2])
        axes[2].set_title('Success Rate by Group', fontsize=14)
        axes[2].set_xlabel(property_name, fontsize=12)
        axes[2].set_ylabel(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) %', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.patches):
            count = success_by_value['total_count'].iloc[i]
            axes[2].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'n={count}\n{bar.get_height():.1f}%',
                ha='center',
                fontsize=9
            )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def analyze_property_comparative(self, method1: str, method2: str, property_name: str,
                                    property_type: str = 'continuous',
                                    rmsd_threshold: float = 2.0,
                                    bins: List[float] = None,
                                    plot: bool = True) -> Dict:
        """
        Compare how a property affects success between two different methods.
        
        Parameters:
        -----------
        method1 : str
            Name of the first method to compare
        method2 : str
            Name of the second method to compare
        property_name : str
            Name of the property column to analyze
        property_type : str, optional
            Type of property: 'continuous', 'discrete', or 'binary'
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful
        bins : list, optional
            Custom bins for categorizing continuous properties
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing comparative analysis results
        """
        # Get best RMSD per protein for each method
        best_rmsd_method1 = self.get_best_rmsd_per_protein(method1, rmsd_threshold)
        best_rmsd_method2 = self.get_best_rmsd_per_protein(method2, rmsd_threshold)
        
        # Check if property exists in both methods
        if property_name not in best_rmsd_method1.columns or property_name not in best_rmsd_method2.columns:
            print(f"Property '{property_name}' not found for one or both methods")
            return None
        
        # Find common proteins between methods
        common_proteins = set(best_rmsd_method1['protein']).intersection(set(best_rmsd_method2['protein']))
        
        if len(common_proteins) < 5:
            print(f"Not enough common proteins between methods: only {len(common_proteins)} found")
            return None
        
        # Filter to common proteins
        df1 = best_rmsd_method1[best_rmsd_method1['protein'].isin(common_proteins)]
        df2 = best_rmsd_method2[best_rmsd_method2['protein'].isin(common_proteins)]
        
        # Create merged dataframe for comparison
        merged_df = pd.merge(
            df1[['protein', property_name, 'rmsd', f'rmsd_≤_{rmsd_threshold}å']],
            df2[['protein', property_name, 'rmsd', f'rmsd_≤_{rmsd_threshold}å']],
            on='protein',
            suffixes=(f'_{method1}', f'_{method2}')
        )
        
        print(f"Analyzing {property_name} effects on {method1} vs {method2}")
        print(f"Common proteins: {len(common_proteins)}")
        
        # Analyze based on property type
        if property_type == 'continuous':
            return self._analyze_continuous_property_comparative(
                merged_df, property_name, method1, method2, rmsd_threshold, bins, plot)
        elif property_type == 'discrete':
            return self._analyze_discrete_property_comparative(
                merged_df, property_name, method1, method2, rmsd_threshold, plot)
        elif property_type == 'binary':
            return self._analyze_binary_property_comparative(
                merged_df, property_name, method1, method2, rmsd_threshold, plot)
        else:
            print(f"Invalid property type: {property_type}. Must be 'continuous', 'discrete', or 'binary'")
            return None
    
    def _analyze_continuous_property_comparative(self, merged_df, property_name, method1, method2, 
                                               rmsd_threshold, bins, plot):
        """Helper function to analyze continuous properties comparatively."""
        # Extract property columns - they should be the same between methods since they're protein properties
        property_col = f'{property_name}_{method1}'  # Use the column from method1
        
        # Calculate correlations for each method
        corr1, p_val1 = stats.pearsonr(merged_df[property_col], merged_df[f'rmsd_{method1}'])
        corr2, p_val2 = stats.pearsonr(merged_df[property_col], merged_df[f'rmsd_{method2}'])
        
        # Create property categories
        if bins is None:
            # Auto-create 5 bins
            min_val = merged_df[property_col].min()
            max_val = merged_df[property_col].max()
            range_val = max_val - min_val
            step = range_val / 5
            bins = [min_val, min_val+step, min_val+2*step, min_val+3*step, min_val+4*step, float('inf')]
            bin_labels = [f'≤{min_val+step:.1f}', f'{min_val+step:.1f}-{min_val+2*step:.1f}', 
                         f'{min_val+2*step:.1f}-{min_val+3*step:.1f}', 
                         f'{min_val+3*step:.1f}-{min_val+4*step:.1f}', f'>{min_val+4*step:.1f}']
        else:
            if len(bins) < 3:
                print("Bins list must have at least 3 elements")
                return None
            bin_labels = [f'≤{bins[1]}']
            for i in range(1, len(bins)-2):
                bin_labels.append(f'{bins[i]}-{bins[i+1]}')
            bin_labels.append(f'>{bins[-2]}')
        
        property_category = f'{property_name}_category'
        merged_df[property_category] = pd.cut(merged_df[property_col], bins=bins, labels=bin_labels)
        
        # Print correlations
        print(f"\nPearson correlations with {property_name}:")
        print(f"{method1}: r={corr1:.4f}, p={p_val1:.6f}")
        print(f"{method2}: r={corr2:.4f}, p={p_val2:.6f}")
        
        # Check if correlation differences are significant using Fisher's z-transformation
        z1 = np.arctanh(corr1)
        z2 = np.arctanh(corr2)
        se_diff_z = np.sqrt(1/(len(merged_df)-3) + 1/(len(merged_df)-3))
        z_diff = (z1 - z2) / se_diff_z
        p_diff = 2 * (1 - stats.norm.cdf(np.abs(z_diff)))
        
        print(f"Correlation difference significance: z={z_diff:.4f}, p={p_diff:.6f}")
        
        # Calculate success rates by category for each method
        success_by_category = merged_df.groupby(property_category).agg(
            total_count=(property_col, 'count'),
            success_count1=(f'rmsd_≤_{rmsd_threshold}å_{method1}', 'sum'),
            success_count2=(f'rmsd_≤_{rmsd_threshold}å_{method2}', 'sum'),
            mean_rmsd1=(f'rmsd_{method1}', 'mean'),
            mean_rmsd2=(f'rmsd_{method2}', 'mean')
        ).reset_index()
        
        success_by_category['success_rate1'] = success_by_category['success_count1'] / success_by_category['total_count'] * 100
        success_by_category['success_rate2'] = success_by_category['success_count2'] / success_by_category['total_count'] * 100
        success_by_category['success_rate_diff'] = success_by_category['success_rate1'] - success_by_category['success_rate2']
        
        # Visualizations
        if plot:
            self._plot_continuous_property_comparative(merged_df, property_name, property_category, property_col,
                                                      method1, method2, rmsd_threshold, success_by_category)
        
        # Print success rates by category
        print("\nSuccess rates by category:")
        print(success_by_category[[property_category, 'total_count', 'success_rate1', 'success_rate2', 
                                  'success_rate_diff', 'mean_rmsd1', 'mean_rmsd2']])
        
        # Return results
        return {
            'property_type': 'continuous',
            'property_name': property_name,
            'method1': method1,
            'method2': method2,
            'corr1': corr1,
            'p_val1': p_val1,
            'corr2': corr2,
            'p_val2': p_val2,
            'correlation_diff_z': z_diff,
            'correlation_diff_p': p_diff,
            'success_by_category': success_by_category,
            'merged_df': merged_df
        }
    
    def _plot_continuous_property_comparative(self, merged_df, property_name, property_category, property_col,
                                             method1, method2, rmsd_threshold, success_by_category):
        """Helper function to plot comparative continuous property analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Comparative Impact of {property_name} on {method1} vs {method2}', fontsize=16)
        
        # 1. Scatter plots with regression lines
        sns.scatterplot(data=merged_df, x=property_col, y=f'rmsd_{method1}', 
                       alpha=0.6, ax=axes[0, 0], label=method1)
        sns.regplot(data=merged_df, x=property_col, y=f'rmsd_{method1}', 
                   lowess=True, scatter=False, color='blue', ax=axes[0, 0])
        
        sns.scatterplot(data=merged_df, x=property_col, y=f'rmsd_{method2}', 
                       alpha=0.6, ax=axes[0, 0], label=method2)
        sns.regplot(data=merged_df, x=property_col, y=f'rmsd_{method2}', 
                   lowess=True, scatter=False, color='orange', ax=axes[0, 0])

        axes[0, 0].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                          label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 0].set_xlabel(property_name, fontsize=12)
        axes[0, 0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 0].set_title('Correlation Plot', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Boxplots by category for both methods
        plot_data = pd.melt(merged_df, 
                           id_vars=[property_category],
                           value_vars=[f'rmsd_{method1}', f'rmsd_{method2}'],
                           var_name='method', value_name='rmsd')
        plot_data['method'] = plot_data['method'].str.replace('rmsd_', '')
        
        sns.boxplot(data=plot_data, x=property_category, y='rmsd', hue='method', ax=axes[0, 1])
        axes[0, 1].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                          label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[0, 1].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 1].set_title('RMSD Distribution by Category', fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(title='Method')
        
        # 3. Success rates by category for both methods
        melted_success = pd.melt(success_by_category,
                                id_vars=[property_category, 'total_count'],
                                value_vars=['success_rate1', 'success_rate2'],
                                var_name='method', value_name='success_rate')
        melted_success['method'] = melted_success['method'].replace({
            'success_rate1': method1, 'success_rate2': method2
        })
        
        bar_plot = sns.barplot(data=melted_success, x=property_category, y='success_rate', 
                              hue='method', ax=axes[1, 0])
        axes[1, 0].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=12)
        axes[1, 0].set_title(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) by Category', fontsize=14)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.containers[0].patches):  # Only add to the first set of bars
            count = success_by_category['total_count'].iloc[i % len(success_by_category)]
            axes[1, 0].text(
                i,
                5,
                f'n={count}',
                ha='center',
                fontsize=9
            )
        
        # 4. Success rate difference (method1 - method2)
        sns.barplot(data=success_by_category, x=property_category, y='success_rate_diff', ax=axes[1, 1])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 1].set_ylabel(f'Success Rate Difference (%) ({method1} - {method2})', fontsize=12)
        axes[1, 1].set_title('Difference in Success Rates by Category', fontsize=14)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add data labels
        for i, p in enumerate(axes[1, 1].patches):
            axes[1, 1].annotate(f'{p.get_height():.1f}%',
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha = 'center', va = 'center',
                              xytext = (0, 10 if p.get_height() >= 0 else -10),
                              textcoords = 'offset points')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    def _analyze_discrete_property_comparative(self, merged_df, property_name, method1, method2, rmsd_threshold, plot):
        """Helper function to analyze discrete properties comparatively."""
        # Extract property columns
        property_col = f'{property_name}_{method1}'  # Use column from method1
        
        # Ensure discrete property is treated as categorical
        merged_df[property_col] = merged_df[property_col].astype('category')
        
        # Calculate success rates by discrete value for each method
        success_by_value = merged_df.groupby(property_col).agg(
            total_count=(property_col, 'count'),
            success_count1=(f'rmsd_≤_{rmsd_threshold}å_{method1}', 'sum'),
            success_count2=(f'rmsd_≤_{rmsd_threshold}å_{method2}', 'sum'),
            mean_rmsd1=(f'rmsd_{method1}', 'mean'),
            mean_rmsd2=(f'rmsd_{method2}', 'mean')
        ).reset_index()
        
        success_by_value['success_rate1'] = success_by_value['success_count1'] / success_by_value['total_count'] * 100
        success_by_value['success_rate2'] = success_by_value['success_count2'] / success_by_value['total_count'] * 100
        success_by_value['success_rate_diff'] = success_by_value['success_rate1'] - success_by_value['success_rate2']
        
        # Calculate Spearman correlations for each method
        try:
            spearman1, p_val1 = stats.spearmanr(merged_df[property_col].cat.codes, merged_df[f'rmsd_{method1}'])
            spearman2, p_val2 = stats.spearmanr(merged_df[property_col].cat.codes, merged_df[f'rmsd_{method2}'])
            
            print(f"\nSpearman correlations with {property_name}:")
            print(f"{method1}: rho={spearman1:.4f}, p={p_val1:.6f}")
            print(f"{method2}: rho={spearman2:.4f}, p={p_val2:.6f}")
        except:
            spearman1, p_val1, spearman2, p_val2 = None, None, None, None
            print(f"Could not calculate Spearman correlations for {property_name}")
        
        # Visualizations
        if plot:
            self._plot_discrete_property_comparative(merged_df, property_name, property_col,
                                                   method1, method2, rmsd_threshold, success_by_value)
        
        # Print success rates by value
        print("\nSuccess rates by discrete value:")
        print(success_by_value[[property_col, 'total_count', 'success_rate1', 'success_rate2', 
                              'success_rate_diff', 'mean_rmsd1', 'mean_rmsd2']])
        
        # Return results
        return {
            'property_type': 'discrete',
            'property_name': property_name,
            'method1': method1,
            'method2': method2,
            'spearman1': spearman1,
            'p_val1': p_val1,
            'spearman2': spearman2,
            'p_val2': p_val2,
            'success_by_value': success_by_value,
            'merged_df': merged_df
        }
    
    def _plot_discrete_property_comparative(self, merged_df, property_name, property_col,
                                          method1, method2, rmsd_threshold, success_by_value):
        """Helper function to plot comparative discrete property analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Comparative Impact of {property_name} on {method1} vs {method2}', fontsize=16)
        
        # 1. Box plots by value for both methods
        plot_data = pd.melt(merged_df, 
                           id_vars=[property_col],
                           value_vars=[f'rmsd_{method1}', f'rmsd_{method2}'],
                           var_name='method', value_name='rmsd')
        plot_data['method'] = plot_data['method'].str.replace('rmsd_', '')
        
        sns.boxplot(data=plot_data, x=property_col, y='rmsd', hue='method', ax=axes[0, 0])
        axes[0, 0].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                          label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 0].set_title('RMSD Distribution (Box Plot)', fontsize=14)
        axes[0, 0].set_xlabel(property_name, fontsize=12)
        axes[0, 0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(title='Method')
        
        # 2. Violin plots by value for both methods
        sns.violinplot(data=plot_data, x=property_col, y='rmsd', hue='method', 
                      split=True, inner="quartile", ax=axes[0, 1])
        axes[0, 1].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                          label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 1].set_title('RMSD Distribution (Violin Plot)', fontsize=14)
        axes[0, 1].set_xlabel(property_name, fontsize=12)
        axes[0, 1].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(title='Method')
        
        # 3. Success rates by value for both methods
        melted_success = pd.melt(success_by_value,
                                id_vars=[property_col, 'total_count'],
                                value_vars=['success_rate1', 'success_rate2'],
                                var_name='method', value_name='success_rate')
        melted_success['method'] = melted_success['method'].replace({
            'success_rate1': method1, 'success_rate2': method2
        })
        
        bar_plot = sns.barplot(data=melted_success, x=property_col, y='success_rate', 
                              hue='method', ax=axes[1, 0])
        axes[1, 0].set_xlabel(property_name, fontsize=12)
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=12)
        axes[1, 0].set_title(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) by Value', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.containers[0].patches):  # Only add to the first set of bars
            count = success_by_value['total_count'].iloc[i % len(success_by_value)]
            axes[1, 0].text(
                i,
                5,
                f'n={count}',
                ha='center',
                fontsize=9
            )
        
        # 4. Success rate difference (method1 - method2)
        sns.barplot(data=success_by_value, x=property_col, y='success_rate_diff', ax=axes[1, 1])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel(property_name, fontsize=12)
        axes[1, 1].set_ylabel(f'Success Rate Difference (%) ({method1} - {method2})', fontsize=12)
        axes[1, 1].set_title('Difference in Success Rates by Value', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add data labels
        for i, p in enumerate(axes[1, 1].patches):
            axes[1, 1].annotate(f'{p.get_height():.1f}%',
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha = 'center', va = 'center',
                              xytext = (0, 10 if p.get_height() >= 0 else -10),
                              textcoords = 'offset points')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def _plot_binary_property_comparative(self, merged_df, property_name, property_col,
                                        method1, method2, rmsd_threshold, summary_df, results):
        """Helper function to plot comparative binary property analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Comparative Impact of {property_name} on {method1} vs {method2}', fontsize=16)
        
        # 1. RMSD boxplots by binary value and method
        plot_data = pd.melt(merged_df, 
                        id_vars=[property_col],
                        value_vars=[f'rmsd_{method1}', f'rmsd_{method2}'],
                        var_name='method', value_name='rmsd')
        plot_data['method'] = plot_data['method'].str.replace('rmsd_', '')
        
        sns.boxplot(data=plot_data, x=property_col, y='rmsd', hue='method', ax=axes[0, 0])
        axes[0, 0].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                        label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 0].set_title('RMSD Distribution by Group', fontsize=14)
        axes[0, 0].set_xlabel(property_name, fontsize=12)
        axes[0, 0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(title='Method')
        
        # 2. Success rate by binary value and method
        bar_plot = sns.barplot(data=summary_df, x='Value', y='Success_Rate', hue='Method', ax=axes[0, 1])
        axes[0, 1].set_title('Success Rate by Group', fontsize=14)
        axes[0, 1].set_xlabel(property_name, fontsize=12)
        axes[0, 1].set_ylabel(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) %', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add data labels with sample sizes
        for i, bar in enumerate(bar_plot.patches):
            count = summary_df['Count'].iloc[i % 2]  # Same count for both methods
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{bar.get_height():.1f}%',
                ha='center',
                fontsize=9
            )
        
        # Add sample size annotations
        axes[0, 1].text(0, 5, f"n={results['true_count']}", ha='center', fontsize=9)
        axes[0, 1].text(1, 5, f"n={results['false_count']}", ha='center', fontsize=9)
        
        # 3. Success rate difference between methods for each binary value
        diff_data = pd.DataFrame([
            {'Value': True, 'Diff': results['true_success_diff'], 
            'Significant': results['true_significant'] if 'true_significant' in results else False},
            {'Value': False, 'Diff': results['false_success_diff'], 
            'Significant': results['false_significant'] if 'false_significant' in results else False}
        ])
        
        diff_plot = sns.barplot(data=diff_data, x='Value', y='Diff', ax=axes[1, 0])
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_xlabel(property_name, fontsize=12)
        axes[1, 0].set_ylabel(f'Success Rate Diff (%) ({method1} - {method2})', fontsize=12)
        axes[1, 0].set_title('Method Difference by Property Value', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add significance markers
        for i, is_sig in enumerate(diff_data['Significant']):
            if is_sig:
                axes[1, 0].text(
                    i, 
                    diff_data['Diff'].iloc[i] + (5 if diff_data['Diff'].iloc[i] >= 0 else -5), 
                    '*', 
                    ha='center', 
                    fontsize=14,
                    fontweight='bold'
                )
        
        # 4. Property impact difference
        impact_data = pd.DataFrame([
            {'Method': method1, 'Impact': results['method1_impact']},
            {'Method': method2, 'Impact': results['method2_impact']}
        ])
        
        sns.barplot(data=impact_data, x='Method', y='Impact', ax=axes[1, 1])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Method', fontsize=12)
        axes[1, 1].set_ylabel(f'Property Impact (%) (True - False)', fontsize=12)
        axes[1, 1].set_title(f'Impact of {property_name} on Each Method', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add data label for impact difference
        plt.figtext(
            0.75, 0.25, 
            f"Difference in impact: {results['impact_diff']:.1f}%", 
            ha="center", 
            fontsize=12, 
            bbox={"facecolor":"orange", "alpha":0.2, "pad":5}
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def analyze_property_universal(self, methods: List[str], property_name: str,
                                property_type: str = 'continuous',
                                rmsd_threshold: float = 2.0,
                                bins: List[float] = None,
                                plot: bool = True) -> Dict:
        """
        Analyze how a property affects docking success across all specified methods.
        
        Parameters:
        -----------
        methods : list
            List of method names to analyze
        property_name : str
            Name of the property column to analyze
        property_type : str, optional
            Type of property: 'continuous', 'discrete', or 'binary'
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful
        bins : list, optional
            Custom bins for categorizing continuous properties
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing universal analysis results
        """
        # Create success matrix to identify cases where all methods succeed or fail
        success_matrix = self.create_success_matrix(methods, rmsd_threshold)
        
        # Get best RMSD per protein for each method
        best_rmsd_by_method = {method: self.get_best_rmsd_per_protein(method, rmsd_threshold) 
                            for method in methods}
        
        # Check if property exists in any method's data
        property_exists = False
        for method, df in best_rmsd_by_method.items():
            if property_name in df.columns:
                property_exists = True
                break
                
        if not property_exists:
            print(f"Property '{property_name}' not found in any method's dataset")
            return None
        
        # Identify proteins where all methods succeed or all methods fail
        all_success_proteins = success_matrix[
            success_matrix['total_successes'] == len(methods)
        ]['protein'].tolist()
        
        all_failure_proteins = success_matrix[
            success_matrix['total_successes'] == 0
        ]['protein'].tolist()
        
        print(f"Analyzing {property_name} for {len(all_success_proteins)} cases where all methods succeed")
        print(f"and {len(all_failure_proteins)} cases where all methods fail")
        
        # Create a combined dataset for universal success/failure cases
        all_success_data = []
        all_failure_data = []
        
        for method in methods:
            method_df = best_rmsd_by_method[method]
            
            if property_name not in method_df.columns:
                continue
            
            # Filter for universal success/failure proteins
            method_success = method_df[method_df['protein'].isin(all_success_proteins)].copy()
            method_failure = method_df[method_df['protein'].isin(all_failure_proteins)].copy()
            
            # Add method name
            method_success['method'] = method
            method_failure['method'] = method
            
            all_success_data.append(method_success)
            all_failure_data.append(method_failure)
        
        # Combine data
        # Combine data - ADD reset_index HERE to fix the error
        success_df = pd.concat(all_success_data).reset_index(drop=True) if all_success_data else pd.DataFrame()
        failure_df = pd.concat(all_failure_data).reset_index(drop=True) if all_failure_data else pd.DataFrame()
        
        # Ensure we have data to analyze
        if len(success_df) < 5 or len(failure_df) < 5:
            print("Not enough data for universal analysis")
            return {
                'success_df': success_df,
                'failure_df': failure_df,
                'property_name': property_name
            }
        
        # Analyze based on property type
        if property_type == 'continuous':
            return self._analyze_continuous_property_universal(
                success_df, failure_df, property_name, rmsd_threshold, bins, plot)
        elif property_type == 'discrete':
            return self._analyze_discrete_property_universal(
                success_df, failure_df, property_name, rmsd_threshold, plot)
        elif property_type == 'binary':
            return self._analyze_binary_property_universal(
                success_df, failure_df, property_name, rmsd_threshold, plot)
        else:
            print(f"Invalid property type: {property_type}. Must be 'continuous', 'discrete', or 'binary'")
            return None

    def _analyze_continuous_property_universal(self, success_df, failure_df, property_name, 
                                            rmsd_threshold, bins, plot):
        """Helper function to analyze continuous properties universally."""
        # Calculate statistics for both groups
        success_mean = success_df[property_name].mean()
        failure_mean = failure_df[property_name].mean()
        success_std = success_df[property_name].std()
        failure_std = failure_df[property_name].std()
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(
            success_df[property_name].dropna(), 
            failure_df[property_name].dropna(), 
            alternative='two-sided'
        )
        
        # Calculate Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(success_df) - 1) * success_std**2 + 
            (len(failure_df) - 1) * failure_std**2) /
            (len(success_df) + len(failure_df) - 2)
        )
        
        if pooled_std > 0:
            cohen_d = abs(success_mean - failure_mean) / pooled_std
        else:
            cohen_d = np.nan
            
        # Create categories for analysis
        if bins is None:
            # Combine data to create bins
            all_values = pd.concat([success_df[property_name], failure_df[property_name]])
            min_val = all_values.min()
            max_val = all_values.max()
            range_val = max_val - min_val
            step = range_val / 5
            bins = [min_val, min_val+step, min_val+2*step, min_val+3*step, min_val+4*step, float('inf')]
            bin_labels = [f'≤{min_val+step:.1f}', f'{min_val+step:.1f}-{min_val+2*step:.1f}', 
                        f'{min_val+2*step:.1f}-{min_val+3*step:.1f}', 
                        f'{min_val+3*step:.1f}-{min_val+4*step:.1f}', f'>{min_val+4*step:.1f}']
        else:
            if len(bins) < 3:
                print("Bins list must have at least 3 elements")
                return None
            bin_labels = [f'≤{bins[1]}']
            for i in range(1, len(bins)-2):
                bin_labels.append(f'{bins[i]}-{bins[i+1]}')
            bin_labels.append(f'>{bins[-2]}')
        
        # Add categories to dataframes
        property_category = f'{property_name}_category'
        success_df[property_category] = pd.cut(success_df[property_name], bins=bins, labels=bin_labels)
        failure_df[property_category] = pd.cut(failure_df[property_name], bins=bins, labels=bin_labels)
        
        # Calculate distribution across categories
        success_by_category = success_df.groupby(property_category).size().reset_index(name='success_count')
        failure_by_category = failure_df.groupby(property_category).size().reset_index(name='failure_count')
        
        # Merge categorical data
        # Merge categorical data without touching the category column
        category_dist = pd.merge(
            success_by_category,
            failure_by_category,
            on=property_category,
            how='outer'
        )

        # Only fill NaNs in the count columns
        category_dist[['success_count','failure_count']] = (
            category_dist[['success_count','failure_count']]
            .fillna(0)
            .astype(int)
        )

        # Now compute percentages
        category_dist['success_percent'] = (
            category_dist['success_count'] / category_dist['success_count'].sum() * 100
        )
        category_dist['failure_percent'] = (
            category_dist['failure_count'] / category_dist['failure_count'].sum() * 100
        )
        category_dist['percent_diff'] = (
            category_dist['success_percent'] - category_dist['failure_percent']
        )
        # Chi-square test for categorical distribution difference
        observed = np.vstack([
            category_dist['success_count'].values,
            category_dist['failure_count'].values
        ]).T
        
        try:
            chi2, chi2_p, _, _ = stats.chi2_contingency(observed)
            chi2_significant = chi2_p < 0.05
        except:
            chi2, chi2_p, chi2_significant = None, None, False
        
        # Print results
        print(f"\nUniversal analysis of continuous property '{property_name}':")
        print(f"Success cases: n={len(success_df)}, mean={success_mean:.3f}, std={success_std:.3f}")
        print(f"Failure cases: n={len(failure_df)}, mean={failure_mean:.3f}, std={failure_std:.3f}")
        print(f"Mann-Whitney U test: p={p_value:.6f}")
        print(f"Effect size (Cohen's d): {cohen_d:.3f}")
        
        if chi2_p is not None:
            print(f"Chi-square test for distribution: p={chi2_p:.6f}")
        
        # Visualizations
        if plot:
            self._plot_continuous_property_universal(
                success_df, failure_df, property_name, property_category, category_dist, rmsd_threshold)
        
        # Return results
        return {
            'property_type': 'continuous',
            'property_name': property_name,
            'success_mean': success_mean,
            'failure_mean': failure_mean,
            'success_std': success_std,
            'failure_std': failure_std,
            'mann_whitney_u': u_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'chi2': chi2,
            'chi2_p': chi2_p,
            'chi2_significant': chi2_significant,
            'category_distribution': category_dist,
            'significant': p_value < 0.05,
            'success_df': success_df,
            'failure_df': failure_df
        }

    def _plot_continuous_property_universal(self, success_df, failure_df, property_name, 
                                        property_category, category_dist, rmsd_threshold):
        """Helper function to plot universal continuous property analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Universal Analysis of {property_name} Impact on Docking Success', fontsize=16)
        
        # 1. Boxplot comparison: stack success and failure into one DataFrame
        plot_data = pd.concat([
            success_df[[property_name]].assign(Outcome='Universal Success'),
            failure_df[[property_name]].assign(Outcome='Universal Failure')
        ], ignore_index=True)
        
        sns.boxplot(data=plot_data, x='Outcome', y=property_name, ax=axes[0, 0])
        axes[0, 0].set_title(f'{property_name} Distribution by Outcome', fontsize=14)
        axes[0, 0].set_xlabel('Universal Outcome', fontsize=12)
        axes[0, 0].set_ylabel(property_name, fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add sample size annotations
        axes[0, 0].text(0, plot_data[property_name].min(), f"n={len(success_df)}", 
                    ha='center', fontsize=10)
        axes[0, 0].text(1, plot_data[property_name].min(), f"n={len(failure_df)}", 
                    ha='center', fontsize=10)
        
        # 2. Density plot
        sns.kdeplot(data=success_df, x=property_name, ax=axes[0, 1], label='Universal Success', common_norm=False)
        sns.kdeplot(data=failure_df, x=property_name, ax=axes[0, 1], label='Universal Failure', common_norm=False)
        axes[0, 1].set_title(f'{property_name} Density Distribution', fontsize=14)
        axes[0, 1].set_xlabel(property_name, fontsize=12)
        axes[0, 1].set_ylabel('Density', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Category distribution comparison
        category_plot_data = pd.melt(category_dist, 
                                    id_vars=[property_category],
                                    value_vars=['success_percent', 'failure_percent'],
                                    var_name='Outcome', value_name='Percentage')
        category_plot_data['Outcome'] = category_plot_data['Outcome'].map({
            'success_percent': 'Universal Success', 
            'failure_percent': 'Universal Failure'
        })
        
        sns.barplot(data=category_plot_data, x=property_category, y='Percentage', 
                hue='Outcome', ax=axes[1, 0])
        axes[1, 0].set_title(f'Distribution by {property_name} Category', fontsize=14)
        axes[1, 0].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 0].set_ylabel('Percentage of Cases (%)', fontsize=12)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add sample sizes
        total_success = category_dist['success_count'].sum()
        total_failure = category_dist['failure_count'].sum()
        plt.figtext(0.5, 0.02, f"Total: {total_success} universal success cases, {total_failure} universal failure cases", 
                ha='center', fontsize=10)
        
        # 4. Percentage difference
        diff_plot = sns.barplot(data=category_dist, x=property_category, y='percent_diff', ax=axes[1, 1])
        axes[1, 1].set_title('Success vs. Failure Distribution Difference', fontsize=14)
        axes[1, 1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 1].set_ylabel('Difference (Success % - Failure %)', fontsize=12)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add data labels
        for i, p in enumerate(diff_plot.patches):
            axes[1, 1].annotate(f'{p.get_height():.1f}%',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10 if p.get_height() >= 0 else -10),
                            textcoords='offset points')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def _analyze_discrete_property_universal(self, success_df, failure_df, property_name, 
                                             rmsd_threshold, plot):
        """Helper function to analyze discrete properties universally."""
        import pandas as pd
        import numpy as np
        from scipy import stats

        # Ensure discrete property is treated as categorical
        success_df[property_name] = success_df[property_name].astype('category')
        failure_df[property_name] = failure_df[property_name].astype('category')

        # Count occurrences in each category
        success_counts = (
            success_df[property_name]
            .value_counts()
            .rename_axis(property_name)
            .reset_index(name='success_count')
        )
        failure_counts = (
            failure_df[property_name]
            .value_counts()
            .rename_axis(property_name)
            .reset_index(name='failure_count')
        )

        # Determine all categories
        all_categories = sorted(
            set(success_counts[property_name]).union(failure_counts[property_name])
        )

        # Create full distribution DataFrame
        value_dist = pd.DataFrame({property_name: all_categories})
        value_dist = value_dist.merge(success_counts, on=property_name, how='left')
        value_dist = value_dist.merge(failure_counts, on=property_name, how='left')

        # Fill missing counts and convert to int
        value_dist['success_count'] = value_dist['success_count'].fillna(0).astype(int)
        value_dist['failure_count'] = value_dist['failure_count'].fillna(0).astype(int)

        # Compute percentages
        total_success = value_dist['success_count'].sum()
        total_failure = value_dist['failure_count'].sum()
        if total_success > 0:
            value_dist['success_percent'] = value_dist['success_count'] / total_success * 100
        else:
            value_dist['success_percent'] = 0
        if total_failure > 0:
            value_dist['failure_percent'] = value_dist['failure_count'] / total_failure * 100
        else:
            value_dist['failure_percent'] = 0
        value_dist['percent_diff'] = (
            value_dist['success_percent'] - value_dist['failure_percent']
        )

        # Prepare contingency table for chi-square
        observed = np.vstack([
            value_dist['success_count'].values,
            value_dist['failure_count'].values
        ])

        try:
            chi2, chi2_p, _, _ = stats.chi2_contingency(observed)
            chi2_significant = chi2_p < 0.05
        except Exception:
            chi2, chi2_p, chi2_significant = None, None, False

        # Print summary
        print(f"\nUniversal analysis of discrete property '{property_name}':")
        print(f"Categories: {all_categories}")
        print(f"Counts (success vs failure):\n{value_dist[[property_name, 'success_count', 'failure_count']]}")
        if chi2_p is not None:
            print(f"Chi-square test: statistic={chi2:.3f}, p-value={chi2_p:.6f}")
            if chi2_significant:
                print("Significant distribution difference between success and failure groups.")

        # Plot if requested
        if plot:
            self._plot_discrete_property_universal(
                success_df, failure_df, property_name, value_dist, rmsd_threshold
            )

        # Return results
        return {
            'property_type': 'discrete',
            'property_name': property_name,
            'value_distribution': value_dist,
            'chi2': chi2,
            'chi2_p': chi2_p,
            'chi2_significant': chi2_significant,
            'success_df': success_df,
            'failure_df': failure_df
        }

    def _plot_discrete_property_universal(self, success_df, failure_df, property_name, 
                                        value_dist, rmsd_threshold):
        """Helper function to plot universal discrete property analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Universal Analysis of {property_name} Impact on Docking Success', fontsize=16)
        
        # 1. Value count distribution comparison
        value_plot_data = pd.melt(value_dist, 
                                id_vars=[property_name],
                                value_vars=['success_percent', 'failure_percent'],
                                var_name='Outcome', value_name='Percentage')
        value_plot_data['Outcome'] = value_plot_data['Outcome'].map({
            'success_percent': 'Universal Success', 
            'failure_percent': 'Universal Failure'
        })
        
        sns.barplot(data=value_plot_data, x=property_name, y='Percentage', 
                hue='Outcome', ax=axes[0])
        axes[0].set_title(f'Distribution by {property_name} Value', fontsize=14)
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('Percentage of Cases (%)', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add sample sizes
        total_success = value_dist['success_count'].sum()
        total_failure = value_dist['failure_count'].sum()
        plt.figtext(0.5, 0.02, f"Total: {total_success} universal success cases, {total_failure} universal failure cases", 
                ha='center', fontsize=10)
        
        # 2. Percentage difference
        diff_plot = sns.barplot(data=value_dist, x=property_name, y='percent_diff', ax=axes[1])
        axes[1].set_title('Success vs. Failure Distribution Difference', fontsize=14)
        axes[1].set_xlabel(property_name, fontsize=12)
        axes[1].set_ylabel('Difference (Success % - Failure %)', fontsize=12)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Add data labels
        for i, p in enumerate(diff_plot.patches):
            axes[1].annotate(f'{p.get_height():.1f}%',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 10 if p.get_height() >= 0 else -10),
                            textcoords='offset points')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def _analyze_binary_property_universal(self, success_df, failure_df, property_name, 
                                        rmsd_threshold, plot):
        """Helper function to analyze binary properties universally."""
        # Ensure the property is treated as boolean
        if success_df[property_name].dtype != bool:
            # Convert to boolean if it's not already
            if success_df[property_name].dtype == 'object':
                success_df[property_name] = success_df[property_name].astype(str).str.lower() == 'true'
            else:
                success_df[property_name] = success_df[property_name].astype(bool)
                
        if failure_df[property_name].dtype != bool:
            # Convert to boolean if it's not already
            if failure_df[property_name].dtype == 'object':
                failure_df[property_name] = failure_df[property_name].astype(str).str.lower() == 'true'
            else:
                failure_df[property_name] = failure_df[property_name].astype(bool)
        
        # Calculate proportions for each group
        success_true_count = success_df[property_name].sum()
        success_false_count = len(success_df) - success_true_count
        failure_true_count = failure_df[property_name].sum()
        failure_false_count = len(failure_df) - failure_true_count
        
        success_true_percent = success_true_count / len(success_df) * 100 if len(success_df) > 0 else 0
        success_false_percent = success_false_count / len(success_df) * 100 if len(success_df) > 0 else 0
        failure_true_percent = failure_true_count / len(failure_df) * 100 if len(failure_df) > 0 else 0
        failure_false_percent = failure_false_count / len(failure_df) * 100 if len(failure_df) > 0 else 0
        
        # Create contingency table for chi-square test
        contingency = np.array([[success_true_count, success_false_count], 
                            [failure_true_count, failure_false_count]])
        
        # Fisher's exact test (better for small sample sizes)
        try:
            _, fisher_p = stats.fisher_exact(contingency)
            significant = fisher_p < 0.05
        except:
            fisher_p, significant = None, False
        
        # Calculate odds ratio
        if success_false_count > 0 and failure_true_count > 0:
            odds_ratio = (success_true_count / success_false_count) / (failure_true_count / failure_false_count)
        else:
            odds_ratio = float('inf') if (success_true_count > 0 and failure_false_count > 0) else 0
        
        # Print results
        print(f"\nUniversal analysis of binary property '{property_name}':")
        print(f"Success cases: n={len(success_df)}, True={success_true_percent:.1f}%, False={success_false_percent:.1f}%")
        print(f"Failure cases: n={len(failure_df)}, True={failure_true_percent:.1f}%, False={failure_false_percent:.1f}%")
        print(f"Difference in 'True' proportion: {success_true_percent - failure_true_percent:.1f}%")
        print(f"Fisher's exact test: p={fisher_p:.6f}")
        print(f"Odds ratio: {odds_ratio:.2f}")
        
        # Create summary dataframe for plotting
        summary_df = pd.DataFrame([
            {'Value': True, 'Outcome': 'Success', 'Percent': success_true_percent, 'Count': success_true_count},
            {'Value': False, 'Outcome': 'Success', 'Percent': success_false_percent, 'Count': success_false_count},
            {'Value': True, 'Outcome': 'Failure', 'Percent': failure_true_percent, 'Count': failure_true_count},
            {'Value': False, 'Outcome': 'Failure', 'Percent': failure_false_percent, 'Count': failure_false_count}
        ])
        
        # Visualizations
        if plot:
            self._plot_binary_property_universal(summary_df, property_name, odds_ratio, fisher_p, significant)
        
        # Return results
        return {
            'property_type': 'binary',
            'property_name': property_name,
            'success_true_count': success_true_count,
            'success_false_count': success_false_count,
            'failure_true_count': failure_true_count,
            'failure_false_count': failure_false_count,
            'success_true_percent': success_true_percent,
            'success_false_percent': success_false_percent,
            'failure_true_percent': failure_true_percent,
            'failure_false_percent': failure_false_percent,
            'true_diff': success_true_percent - failure_true_percent,
            'fisher_p': fisher_p,
            'significant': significant,
            'odds_ratio': odds_ratio,
            'summary_df': summary_df,
            'success_df': success_df,
            'failure_df': failure_df
        }

    def _plot_binary_property_universal(self, summary_df, property_name, odds_ratio, p_value, significant):
        """Helper function to plot universal binary property analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Universal Analysis of {property_name} Impact on Docking Success', fontsize=16)
        
        # 1. Percentage distribution
        sns.barplot(data=summary_df, x='Value', y='Percent', hue='Outcome', ax=axes[0])
        axes[0].set_title('Proportion by Outcome', fontsize=14)
        axes[0].set_xlabel(property_name, fontsize=12)
        axes[0].set_ylabel('Percentage (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        for bar, count in zip(axes[0].patches, summary_df['Count']):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'n={count}',
                ha='center',
                fontsize=9
            )
        
        # 2. Mosaic plot (alternative visualization)
        mosaic_data = pd.crosstab(
            summary_df['Value'], summary_df['Outcome'], 
            values=summary_df['Count'], aggfunc='sum', normalize='all'
        )
        
        # Create a heatmap for the mosaic effect
        sns.heatmap(mosaic_data, annot=True, fmt='.1%', cmap='viridis', ax=axes[1], cbar=False)
        axes[1].set_title(f'Outcome Distribution by {property_name}', fontsize=14)
        
        # Add significance information
        if significant:
            significance_text = f"Significant (p={p_value:.4f}, OR={odds_ratio:.2f})"
            color = 'green'
        else:
            significance_text = f"Not significant (p={p_value:.4f}, OR={odds_ratio:.2f})"
            color = 'red'
        
        plt.figtext(
            0.5, 0.01, significance_text,
            ha='center', color=color, fontsize=12,
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.15)
        plt.title(f'Binary Property Analysis for {property_name}')
        plt.show()
    
    def plot_significant_property_differences(self, universal_results, p_threshold=0.05, top_n=None, 
                                            sort_by='effect_size', figsize=(12, 8)):
        """
        Plot significant differences in properties between universal success and failure cases.
        
        Parameters:
        -----------
        universal_results : dict or list
            Results from analyze_property_universal method, can be a single result dict
            or a list of result dicts from multiple property analyses
        p_threshold : float, optional
            P-value threshold for significance, default is 0.05
        top_n : int, optional
            Show only the top N most significant properties (by effect size or p-value)
        sort_by : str, optional
            How to sort the properties: 'effect_size' or 'p_value'
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure object
        """
        # Handle both single result dict and list of result dicts
        if not isinstance(universal_results, list):
            universal_results = [universal_results]
        
        # Extract significant properties from results
        significant_props = []
        
        for result in universal_results:
            if 'property_name' not in result:
                continue
                
            property_name = result['property_name']
            property_type = result.get('property_type', 'unknown')
            is_significant = result.get('significant', False)
            
            # Skip non-significant properties
            if not is_significant:
                continue
                
            # Get p-value
            p_value = None
            if property_type == 'continuous':
                p_value = result.get('p_value')
                effect_size = result.get('cohen_d', 0)
                success_stat = result.get('success_mean', 0)
                failure_stat = result.get('failure_mean', 0)
                difference = success_stat - failure_stat
            elif property_type == 'discrete':
                p_value = result.get('chi2_p')
                effect_size = 0.3  # Placeholder - chi2 doesn't directly give an effect size
                # Get the category with the largest difference
                if 'value_distribution' in result:
                    max_diff_row = result['value_distribution']['percent_diff'].abs().idxmax()
                    difference = result['value_distribution']['percent_diff'].iloc[max_diff_row]
                    success_stat = result['value_distribution']['success_percent'].iloc[max_diff_row]
                    failure_stat = result['value_distribution']['failure_percent'].iloc[max_diff_row]
                else:
                    difference, success_stat, failure_stat = 0, 0, 0
            elif property_type == 'binary':
                p_value = result.get('fisher_p')
                effect_size = result.get('odds_ratio', 1)
                # Convert odds ratio to a symmetric effect size measure
                if effect_size > 1:
                    effect_size = effect_size  # Keep as is for OR > 1
                else:
                    effect_size = 1/effect_size  # Invert for OR < 1 to get comparable magnitudes
                success_stat = result.get('success_true_percent', 0)
                failure_stat = result.get('failure_true_percent', 0)
                difference = success_stat - failure_stat
            
            if p_value is not None and p_value <= p_threshold:
                significant_props.append({
                    'property_name': property_name,
                    'property_type': property_type,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'success_stat': success_stat,
                    'failure_stat': failure_stat,
                    'difference': difference
                })
        
        # If no significant properties found
        if not significant_props:
            print("No significant property differences found.")
            return None
        
        # Sort properties based on the specified criterion
        if sort_by == 'effect_size':
            significant_props.sort(key=lambda x: abs(x['effect_size']), reverse=True)
        else:  # 'p_value'
            significant_props.sort(key=lambda x: x['p_value'])
        
        # Limit to top N if specified
        if top_n is not None and top_n > 0:
            significant_props = significant_props[:min(top_n, len(significant_props))]
        
        # Create figure for visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Significant Property Differences: Universal Success vs. Failure', fontsize=16)
        
        # Get property names and types for plotting
        prop_names = [p['property_name'] for p in significant_props]
        prop_types = [p['property_type'] for p in significant_props]
        
        # Plot 1: Difference values for each property
        differences = [p['difference'] for p in significant_props]
        colors = ['royalblue' if d > 0 else 'firebrick' for d in differences]
        
        bars = axes[0].barh(prop_names, differences, color=colors)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Difference (Success - Failure)', fontsize=12)
        axes[0].set_ylabel('Property', fontsize=12)
        axes[0].set_title('Absolute Differences', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Add p-value annotations - adjusted to prevent overlap
        for i, prop in enumerate(significant_props):
            # Determine if bar extends to the right or left
            diff = differences[i]
            if diff >= 0:
                # For positive differences, place p-value to the left of the bar
                x_pos = diff / 2  # Center of the bar
                axes[0].text(
                    x_pos, i, 
                    f"p={prop['p_value']:.4f}", 
                    ha='center', va='center', 
                    fontsize=9,
                    color='white' if abs(diff) > 10 else 'black',
                    bbox=dict(facecolor='none', alpha=0.8, boxstyle='round,pad=0.3')
                )
            else:
                # For negative differences, place p-value to the right of the bar
                x_pos = diff / 2  # Center of the bar
                axes[0].text(
                    x_pos, i, 
                    f"p={prop['p_value']:.4f}", 
                    ha='center', va='center', 
                    fontsize=9,
                    color='white' if abs(diff) > 10 else 'black',
                    bbox=dict(facecolor='none', alpha=0.8, boxstyle='round,pad=0.3')
                )
        
        # Plot 2: Side-by-side comparison of values
        success_stats = [p['success_stat'] for p in significant_props]
        failure_stats = [p['failure_stat'] for p in significant_props]
        
        x = np.arange(len(prop_names))
        width = 0.35
        
        axes[1].barh(x - width/2, success_stats, width, label='Universal Success', color='green', alpha=0.7)
        axes[1].barh(x + width/2, failure_stats, width, label='Universal Failure', color='red', alpha=0.7)
        
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(prop_names)
        axes[1].set_xlabel('Property Value', fontsize=12)
        axes[1].set_title('Success vs. Failure Values', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add effect size annotations
        for i, prop in enumerate(significant_props):
            if prop['property_type'] == 'continuous':
                effect_label = f"d={prop['effect_size']:.2f}"
            elif prop['property_type'] == 'binary':
                effect_label = f"OR={prop['effect_size']:.2f}"
            else:
                effect_label = ""
                
            if effect_label:
                axes[1].text(
                    max(success_stats[i], failure_stats[i]) + 5, i,
                    effect_label,
                    va='center', fontsize=9,
                    bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.2')
                )
        
        # Add property type indicators
        for i, prop_type in enumerate(prop_types):
            marker = "●" if prop_type == 'continuous' else "■" if prop_type == 'discrete' else "▲"
            axes[1].text(
                -5, i, marker,
                va='center', ha='right', fontsize=12,
                color='blue' if prop_type == 'continuous' else 
                    'purple' if prop_type == 'discrete' else 'orange'
            )
        
        plt.figtext(
            0.5, 0.01,
            "● Continuous  ■ Discrete  ▲ Binary",
            ha='center', fontsize=10
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        return fig

    def plot_property_pvalue_comparison(self, property_name, property_type, universal_result, 
                                    method_results, figsize=(10, 5), log_scale=True):
        """
        Plot p-value comparison between universal analysis and individual methods.
        Uses improved layout to minimize wasted space.
        """
        # Extract universal p-value based on property type
        if property_type == 'continuous':
            universal_pvalue = universal_result.get('p_value')
            pvalue_type = 'Mann-Whitney U Test'
        elif property_type == 'discrete':
            universal_pvalue = universal_result.get('chi2_p') 
            pvalue_type = 'Chi-Square Test'
        elif property_type == 'binary':
            universal_pvalue = universal_result.get('fisher_p')
            pvalue_type = 'Fisher\'s Exact Test'
        else:
            print(f"Invalid property type: {property_type}")
            return None
        
        # Extract method p-values
        method_pvalues = {}
        method_names = list(method_results.keys())
        
        for method, result in method_results.items():
            if property_type == 'continuous':
                method_pvalues[method] = result.get('pearson_p_value', float('nan'))
            elif property_type == 'discrete':
                method_pvalues[method] = result.get('anova_p_value', float('nan'))
            elif property_type == 'binary':
                method_pvalues[method] = result.get('fisher_p', float('nan'))
                if method_pvalues[method] is None:
                    method_pvalues[method] = result.get('mannwhitney_p', float('nan'))
        
        # Use GridSpec for better space usage
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 10)  # 10 columns to allow custom widths
        
        # Create axes with proper ratios (bar plot gets 7 columns, heatmap gets 3)
        ax1 = fig.add_subplot(gs[0, :7])  # Bar chart takes 70% of width
        ax2 = fig.add_subplot(gs[0, 7:])  # Heatmap takes 30% of width
        
        # Compact title at the top
        fig.text(0.5, 0.95, f'{pvalue_type} for {property_name}', 
                ha='center', fontsize=13, fontweight='bold')
        
        # 1. Bar plot of p-values - with better spacing
        all_methods = ['Universal'] + method_names
        all_pvalues = [universal_pvalue] + [method_pvalues[m] for m in method_names]
        
        # Set colors based on significance
        colors = ['darkblue' if p <= 0.05 else 'lightblue' for p in all_pvalues]
        
        # Create bars with proper positions
        y_pos = np.arange(len(all_methods))
        bars = ax1.barh(y_pos, all_pvalues, height=0.6, color=colors)
        
        # Add significance threshold line
        ax1.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        
        # Apply log scale if requested - better for visualizing small p-values
        if log_scale:
            ax1.set_xscale('log')
            ax1.set_xlim([0.0001, 1.0])  # Reasonable limits for p-values in log scale
        
        # Add method names
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(all_methods)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add p-value annotations - more compact
        for i, bar in enumerate(bars):
            height = bar.get_width()
            if height < 0.001:
                text = f"p<0.001"
            else:
                text = f"p={height:.3f}"
            
            # Position text better
            x_pos = max(height * 1.1, 0.0001 if log_scale else 0.01)
            ax1.text(
                x_pos, bar.get_y() + bar.get_height()/2,
                text,
                ha='left', va='center', fontsize=9
            )
        
        # 2. Visual significance comparison - more compact heatmap
        # Create a matrix showing which methods/universal found the property significant
        significance_data = pd.DataFrame({
            'Analysis': all_methods,
            'Significant': [p <= 0.05 for p in all_pvalues]
        })
        
        # Count agreements/disagreements with universal result
        universal_significant = universal_pvalue <= 0.05
        agreement_count = sum(1 for m in method_names 
                            if method_pvalues.get(m, float('nan')) <= 0.05 == universal_significant)
        
        # Create compact heatmap
        sns.heatmap(
            significance_data.set_index('Analysis')['Significant'].to_frame().T,
            cmap=['#FFCCCC', '#CCFFCC'],  # Lighter colors
            cbar=False,
            linewidths=1,
            ax=ax2
        )
        
        # Remove x-axis label which is redundant
        ax2.set_xlabel('')
        
        # Add compact agreement annotation
        ax2.set_title(f'Agreement: {agreement_count}/{len(method_names)}', 
                    fontsize=11)
        
        # Remove unnecessary y-axis labels
        ax2.set_ylabel('')
        ax2.set_yticklabels([''])
        
        # Optimize layout
        plt.subplots_adjust(top=0.9, bottom=0.12, wspace=0.4, left=0.12, right=0.95)
        
        # Add property type indicator in a compact way
        prop_marker = {"continuous": "●", "discrete": "■", "binary": "▲"}
        prop_color = {"continuous": "blue", "discrete": "purple", "binary": "orange"}
        
        plt.figtext(
            0.01, 0.01,
            f"{prop_marker[property_type]}",
            color=prop_color[property_type],
            fontsize=12
        )
        
        return fig

    def plot_multi_property_pvalue_comparison(self, property_data, figsize=None):
        """
        Compare p-values across multiple properties between universal and individual method analyses.
        
        Parameters:
        -----------
        property_data : list of dicts
            List of dictionaries, each containing:
            - 'name': property name
            - 'type': property type ('continuous', 'discrete', or 'binary')
            - 'universal_result': result from universal analysis
            - 'method_results': dictionary mapping methods to their results
        figsize : tuple, optional
            Figure size (width, height). If None, will be calculated based on the number of properties.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        # Calculate appropriate figure size if not provided
        if figsize is None:
            figsize = (12, 2 + len(property_data) * 1.5)
        
        # Create figure
        fig, axes = plt.subplots(len(property_data), 1, figsize=figsize)
        if len(property_data) == 1:
            axes = [axes]  # Make iterable if only one property
        
        fig.suptitle('P-Value Comparison: Universal vs. Individual Methods', fontsize=16)
        
        # Plot each property
        for i, prop_info in enumerate(property_data):
            property_name = prop_info['name']
            property_type = prop_info['type']
            universal_result = prop_info['universal_result']
            method_results = prop_info['method_results']
            
            # Get universal p-value
            if property_type == 'continuous':
                universal_pvalue = universal_result.get('p_value', float('nan'))
                pvalue_type = 'Mann-Whitney'
            elif property_type == 'discrete':
                universal_pvalue = universal_result.get('chi2_p', float('nan'))
                pvalue_type = 'Chi-Square'
            elif property_type == 'binary':
                universal_pvalue = universal_result.get('fisher_p', float('nan'))
                pvalue_type = 'Fisher'
                
            # Get method p-values
            method_names = list(method_results.keys())
            method_pvalues = {}
            
            for method, result in method_results.items():
                if property_type == 'continuous':
                    method_pvalues[method] = result.get('pearson_p_value', float('nan'))
                elif property_type == 'discrete':
                    method_pvalues[method] = result.get('anova_p_value', float('nan'))
                elif property_type == 'binary':
                    method_pvalues[method] = result.get('fisher_p', float('nan'))
                    if method_pvalues[method] is None:
                        method_pvalues[method] = result.get('mannwhitney_p', float('nan'))
            
            # Plot bar chart of p-values
            bars = axes[i].barh(
                ['Universal'] + method_names, 
                [universal_pvalue] + [method_pvalues[m] for m in method_names],
                color=['darkblue'] + ['lightblue'] * len(method_names)
            )
            
            # Add significance threshold line
            axes[i].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
            axes[i].set_xlabel('p-value', fontsize=10)
            axes[i].set_title(f'{property_name} ({pvalue_type} test)', fontsize=12)
            axes[i].set_xscale('log')
            axes[i].grid(axis='x', alpha=0.3)
            
            # Add property type indicators
            prop_type_marker = "●" if property_type == 'continuous' else "■" if property_type == 'discrete' else "▲"
            axes[i].text(
                0.001, -0.2, prop_type_marker,
                transform=axes[i].transData,
                fontsize=14,
                color='blue' if property_type == 'continuous' else 
                    'purple' if property_type == 'discrete' else 'orange'
            )
            
            # Add p-value annotations
            for j, bar in enumerate(bars):
                width = bar.get_width()
                if width < 0.001:
                    text = "p<0.001"
                else:
                    text = f"p={width:.3f}"
                    
                # Determine if significant (green) or not (red)
                color = 'green' if width <= 0.05 else 'red'
                
                axes[i].text(
                    max(width * 1.1, 0.001),  # Ensure visibility for very small p-values
                    bar.get_y() + bar.get_height()/2.,
                    text,
                    va='center', 
                    fontsize=9,
                    color=color
                )
        
        # Add legend for property types
        plt.figtext(
            0.02, 0.01,
            "● Continuous  ■ Discrete  ▲ Binary",
            fontsize=10
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.05)
        
        return fig

    def analyze_property_group_comparative(self, group1_methods: List[str], group2_methods: List[str],
                                        property_name: str, property_type: str = 'continuous',
                                        group1_label: str = None, group2_label: str = None,
                                        rmsd_threshold: float = 2.0, bins: List[float] = None,
                                        plot: bool = True) -> Dict:
        """
        Compare how a property affects docking success between two groups of methods.
        
        Parameters:
        -----------
        group1_methods : list
            List of method names in the first group (e.g., physics-based methods)
        group2_methods : list
            List of method names in the second group (e.g., ML-based methods)
        property_name : str
            Name of the property to analyze
        property_type : str, optional
            Type of property: 'continuous', 'discrete', or 'binary'
        group1_label : str, optional
            Label for the first group. If None, uses "Group 1"
        group2_label : str, optional
            Label for the second group. If None, uses "Group 2"
        rmsd_threshold : float, optional
            RMSD threshold for considering a docking successful
        bins : list, optional
            Custom bins for categorizing continuous properties
        plot : bool, optional
            Whether to generate visualization plots
            
        Returns:
        --------
        dict
            Dictionary containing comparative analysis results
        """
        # Set default labels if not provided
        group1_label = group1_label or "Group 1"
        group2_label = group2_label or "Group 2"
        
        print(f"Comparing {len(group1_methods)} {group1_label} methods vs {len(group2_methods)} {group2_label} methods")
        print(f"{group1_label} methods: {', '.join(group1_methods)}")
        print(f"{group2_label} methods: {', '.join(group2_methods)}")
        
        # Get best RMSD per protein for each method
        group1_best_rmsd = {}
        for method in group1_methods:
            group1_best_rmsd[method] = self.get_best_rmsd_per_protein(method, rmsd_threshold)
            
        group2_best_rmsd = {}
        for method in group2_methods:
            group2_best_rmsd[method] = self.get_best_rmsd_per_protein(method, rmsd_threshold)
        
        # Create combined dataframes for each group
        group1_combined = pd.concat(group1_best_rmsd.values()).reset_index(drop=True)
        group2_combined = pd.concat(group2_best_rmsd.values()).reset_index(drop=True)
        
        # Check if property exists in both groups
        if property_name not in group1_combined.columns or property_name not in group2_combined.columns:
            print(f"Property '{property_name}' not found in one or both groups")
            return None
        
        # Clean data - drop rows with NaN in the property column
        group1_analysis = group1_combined.dropna(subset=[property_name])
        group2_analysis = group2_combined.dropna(subset=[property_name])
        
        # Analyze based on property type
        if property_type == 'continuous':
            return self._analyze_continuous_property_group_comparative(
                group1_analysis, group2_analysis, property_name, 
                group1_label, group2_label, rmsd_threshold, bins, plot)
        elif property_type == 'discrete':
            return self._analyze_discrete_property_group_comparative(
                group1_analysis, group2_analysis, property_name,
                group1_label, group2_label, rmsd_threshold, plot)
        elif property_type == 'binary':
            return self._analyze_binary_property_group_comparative(
                group1_analysis, group2_analysis, property_name,
                group1_label, group2_label, rmsd_threshold, plot)
        else:
            print(f"Invalid property type: {property_type}. Must be 'continuous', 'discrete', or 'binary'")
            return None

    def _analyze_continuous_property_group_comparative(self, group1_df, group2_df, property_name,
                                                    group1_label, group2_label, rmsd_threshold, bins, plot):
        """Helper function to analyze continuous properties between method groups."""
        # Calculate correlations for each group
        corr1, p_val1 = stats.pearsonr(group1_df[property_name], group1_df['rmsd'])
        corr2, p_val2 = stats.pearsonr(group2_df[property_name], group2_df['rmsd'])
        
        # Create property categories
        if bins is None:
            # Combine data to create bins
            all_values = pd.concat([group1_df[property_name], group2_df[property_name]])
            min_val = all_values.min()
            max_val = all_values.max()
            range_val = max_val - min_val
            step = range_val / 5
            bins = [min_val, min_val+step, min_val+2*step, min_val+3*step, min_val+4*step, float('inf')]
            bin_labels = [f'≤{min_val+step:.1f}', f'{min_val+step:.1f}-{min_val+2*step:.1f}',
                        f'{min_val+2*step:.1f}-{min_val+3*step:.1f}',
                        f'{min_val+3*step:.1f}-{min_val+4*step:.1f}', f'>{min_val+4*step:.1f}']
        else:
            bin_labels = [f'≤{bins[1]}']
            for i in range(1, len(bins)-2):
                bin_labels.append(f'{bins[i]}-{bins[i+1]}')
            bin_labels.append(f'>{bins[-2]}')
        
        property_category = f'{property_name}_category'
        group1_df[property_category] = pd.cut(group1_df[property_name], bins=bins, labels=bin_labels)
        group2_df[property_category] = pd.cut(group2_df[property_name], bins=bins, labels=bin_labels)
        
        # Print correlations
        print(f"\nPearson correlations with {property_name}:")
        print(f"{group1_label}: r={corr1:.4f}, p={p_val1:.6f}")
        print(f"{group2_label}: r={corr2:.4f}, p={p_val2:.6f}")
        
        # Check if correlation differences are significant 
        z1 = np.arctanh(corr1)
        z2 = np.arctanh(corr2)
        se_diff_z = np.sqrt(1/(len(group1_df)-3) + 1/(len(group2_df)-3))
        z_diff = (z1 - z2) / se_diff_z
        p_diff = 2 * (1 - stats.norm.cdf(np.abs(z_diff)))
        
        print(f"Correlation difference significance: z={z_diff:.4f}, p={p_diff:.6f}")
        
        # Calculate success rates by category for each group
        g1_success_by_category = group1_df.groupby(property_category).agg(
            total_count=(property_category, 'count'),
            success_count=(f'rmsd_≤_{rmsd_threshold}å', 'sum'),
            mean_rmsd=('rmsd', 'mean')
        ).reset_index()
        
        g2_success_by_category = group2_df.groupby(property_category).agg(
            total_count=(property_category, 'count'),
            success_count=(f'rmsd_≤_{rmsd_threshold}å', 'sum'),
            mean_rmsd=('rmsd', 'mean')
        ).reset_index()
        
        # Calculate success rates
        g1_success_by_category['success_rate'] = g1_success_by_category['success_count'] / g1_success_by_category['total_count'] * 100
        g2_success_by_category['success_rate'] = g2_success_by_category['success_count'] / g2_success_by_category['total_count'] * 100
        
        # Merge results from both groups for comparison
        combined_categories = pd.merge(
            g1_success_by_category, 
            g2_success_by_category,
            on=property_category, 
            suffixes=(f'_{group1_label}', f'_{group2_label}'),
            how='outer'
        )

        # Fill NA values only in numeric columns, not in the categorical column
        numeric_columns = combined_categories.columns.difference([property_category])
        combined_categories[numeric_columns] = combined_categories[numeric_columns].fillna(0)

        # Add difference column
        combined_categories['success_rate_diff'] = (
            combined_categories[f'success_rate_{group1_label}'] - 
            combined_categories[f'success_rate_{group2_label}']
        )
        
        # Add difference column
        combined_categories['success_rate_diff'] = (
            combined_categories[f'success_rate_{group1_label}'] - 
            combined_categories[f'success_rate_{group2_label}']
        )
        
        # Visualizations
        if plot:
            self._plot_continuous_property_group_comparative(
                group1_df, group2_df, property_name, property_category,
                group1_label, group2_label, rmsd_threshold, combined_categories
            )
        
        # Return results
        return {
            'property_type': 'continuous',
            'property_name': property_name,
            'group1_label': group1_label,
            'group2_label': group2_label,
            'corr1': corr1,
            'p_val1': p_val1,
            'corr2': corr2,
            'p_val2': p_val2,
            'correlation_diff_z': z_diff,
            'correlation_diff_p': p_diff,
            'combined_categories': combined_categories,
            'group1_df': group1_df,
            'group2_df': group2_df
        }

    def _plot_continuous_property_group_comparative(self, group1_df, group2_df, property_name, property_category,
                                                group1_label, group2_label, rmsd_threshold, combined_categories):
        """Helper function to plot continuous property effects between method groups."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Comparative Impact of {property_name} on {group1_label} vs {group2_label} Methods', fontsize=16)
        
        # 1. Scatter plots with regression lines for both groups
        g1_plot, g1_outlier_info = self._prepare_rmsd_for_plotting(
            group1_df, rmsd_col='rmsd', clip_value=15.0, handle_outliers='clip'
        )
        g2_plot, g2_outlier_info = self._prepare_rmsd_for_plotting(
            group2_df, rmsd_col='rmsd', clip_value=15.0, handle_outliers='clip'
        )
        
        sns.scatterplot(data=g1_plot, x=property_name, y='rmsd_plot', 
                    alpha=0.5, ax=axes[0, 0], label=group1_label, color='blue')
        sns.regplot(data=g1_plot, x=property_name, y='rmsd_plot', 
                scatter=False, color='blue', ax=axes[0, 0])
        
        sns.scatterplot(data=g2_plot, x=property_name, y='rmsd_plot',
                    alpha=0.5, ax=axes[0, 0], label=group2_label, color='orange')
        sns.regplot(data=g2_plot, x=property_name, y='rmsd_plot',
                scatter=False, color='orange', ax=axes[0, 0])
        
        axes[0, 0].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                        label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 0].set_xlabel(property_name, fontsize=12)
        axes[0, 0].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 0].set_title('Correlation with RMSD', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add outlier annotations if needed
        if g1_outlier_info['n_outliers'] > 0 or g2_outlier_info['n_outliers'] > 0:
            outlier_text = []
            if g1_outlier_info['n_outliers'] > 0:
                outlier_text.append(f"{g1_outlier_info['n_outliers']} {group1_label} outliers clipped")
            if g2_outlier_info['n_outliers'] > 0:
                outlier_text.append(f"{g2_outlier_info['n_outliers']} {group2_label} outliers clipped")
            
            axes[0, 0].annotate(
                '\n'.join(outlier_text),
                xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=9, color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        
        # 2. Boxplots by category for both groups
        combined_data = pd.concat([
            g1_plot[[property_category, 'rmsd']].assign(Group=group1_label),
            g2_plot[[property_category, 'rmsd']].assign(Group=group2_label)
        ])
        
        sns.boxplot(data=combined_data, x=property_category, y='rmsd', hue='Group', ax=axes[0, 1])
        axes[0, 1].axhline(y=rmsd_threshold, color='red', linestyle='--', 
                        label=f'RMSD = {rmsd_threshold}Å')
        axes[0, 1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[0, 1].set_ylabel('RMSD (Å)', fontsize=12)
        axes[0, 1].set_title('RMSD Distribution by Category', fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(title='Method Group')
        
        # 3. Success rates by category for both groups
        success_data = pd.DataFrame({
            property_category: combined_categories[property_category],
            group1_label: combined_categories[f'success_rate_{group1_label}'],
            group2_label: combined_categories[f'success_rate_{group2_label}']
        })
        
        success_data_melted = pd.melt(
            success_data, 
            id_vars=[property_category],
            var_name='Group', 
            value_name='Success Rate (%)'
        )
        
        bar_plot = sns.barplot(
            data=success_data_melted, 
            x=property_category, 
            y='Success Rate (%)', 
            hue='Group', 
            ax=axes[1, 0]
        )
        
        axes[1, 0].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 0].set_ylabel('Success Rate (%)', fontsize=12)
        axes[1, 0].set_title(f'Success Rate (RMSD ≤ {rmsd_threshold}Å) by Category', fontsize=14)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add sample size annotations
        for i, cat in enumerate(combined_categories[property_category]):
            g1_count = combined_categories[f'total_count_{group1_label}'].iloc[i]
            g2_count = combined_categories[f'total_count_{group2_label}'].iloc[i]
            
            axes[1, 0].annotate(
                f"n={int(g1_count)}",
                xy=(i-0.2, 5), 
                ha='center', fontsize=8
            )
            axes[1, 0].annotate(
                f"n={int(g2_count)}",
                xy=(i+0.2, 5), 
                ha='center', fontsize=8
            )
        
        # 4. Success rate difference between groups
        diff_bars = sns.barplot(
            data=combined_categories, 
            x=property_category, 
            y='success_rate_diff', 
            ax=axes[1, 1],
            color='green'
        )
        
        # Color negative bars differently
        for i, bar in enumerate(diff_bars.patches):
            if bar.get_height() < 0:
                bar.set_color('red')
        
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel(f'{property_name} Category', fontsize=12)
        axes[1, 1].set_ylabel(f'Success Rate Difference (%) ({group1_label} - {group2_label})', fontsize=12)
        axes[1, 1].set_title('Difference in Success Rates by Category', fontsize=14)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add data labels
        for i, bar in enumerate(diff_bars.patches):
            height = bar.get_height()
            color = 'black' if abs(height) < 15 else 'white'
            axes[1, 1].annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top',
                color=color, fontweight='bold', fontsize=9
            )
        
        # Add correlation values to the plot for reference
        corr1 = stats.pearsonr(group1_df[property_name], group1_df['rmsd'])[0]
        corr2 = stats.pearsonr(group2_df[property_name], group2_df['rmsd'])[0]
        
        plt.figtext(
            0.02, 0.02, 
            f"{group1_label} correlation: r={corr1:.3f}\n{group2_label} correlation: r={corr2:.3f}",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
        
        return fig

    def comparative_mixed_effect(self, df_mixed: pd.DataFrame, property: str,
                                method_col: str = 'Method_Type', rmsd_col: str = 'rmsd',
                                figsize=(10, 6)):
        """
        Perform a mixed effects analysis on the provided dataframe. 
        """

        model = smf.mixedlm(
            f"rmsd ~ {property} * {method_col}",
            data=df_mixed,
            groups=df_mixed["protein"]
        ).fit()

        print(model.summary())

    def analyze_continous_property_mixed_effects(
        self,
        group1_df: pd.DataFrame,
        group2_df: pd.DataFrame,
        property_name: str,
        group1_label: str,
        group2_label: str,
        rmsd_threshold: float = 2.0,
        n_bins: int = 5,
        winsor: Optional[float] = 15.0,   # clip extreme RMSD values (optional)
        plot: bool = True,
        best_rmsd: bool = True
    ) -> Dict:
        """
        Mixed-effects comparative analysis for a continuous property.
        Builds quantile bins, fits a GLMM: success ~ family * bin + (1|protein),
        and returns the full & reduced models plus LRT interaction p-value.
        """
        # 1) Tag family & compute success
        if best_rmsd:
            group1_df = group1_df.groupby('protein').apply(
                lambda x: x.loc[x['rmsd'].idxmin()] if not x['rmsd'].isna().all() else x.iloc[0]
            ).reset_index(drop=True)

            group2_df = group2_df.groupby('protein').apply(
                lambda x: x.loc[x['rmsd'].idxmin()] if not x['rmsd'].isna().all() else x.iloc[0]
            ).reset_index(drop=True)

        g1 = group1_df.copy();  g1['family'] = group1_label
        g2 = group2_df.copy();  g2['family'] = group2_label
        df = pd.concat([g1, g2], ignore_index=True)
        # df['success'] = (df['rmsd'] <= rmsd_threshold).astype(int)# … after creating df and before model fit …
        
        # Optional: winsorise or log-transform to curb heavy tails
        if winsor is not None:
            df['rmsd_clipped'] = np.clip(df['rmsd'], None, winsor)
            outcome_col = 'rmsd_clipped'
        else:
            outcome_col = 'rmsd'

        # 2) quantile bins for the predictor
        bin_col = f"{property_name}_qbin"
        df[bin_col] = pd.qcut(df[property_name], q=n_bins, duplicates='drop')
        needed = [outcome_col, 'family', bin_col, 'protein']
        df = df.dropna(subset=needed).reset_index(drop=True)

        # Remove empty bin levels (optional but safe)
        if pd.api.types.is_categorical_dtype(df[bin_col]):
            df[bin_col] = df[bin_col].cat.remove_unused_categories()

        # 3) full linear mixed model with interaction
        formula_full  = f"{outcome_col} ~ C(family) * C({bin_col})"
        m_full = smf.mixedlm(formula_full, df,
                            groups=df['protein']).fit()

        # 4) reduced model without interaction
        formula_noint = f"{outcome_col} ~ C(family) + C({bin_col})"
        m_noint = smf.mixedlm(formula_noint, df,
                            groups=df['protein']).fit()

        # 5) LRT for interaction
        lr_stat = 2 * (m_full.llf - m_noint.llf)
        df_diff = m_full.params.size - m_noint.params.size   # degrees of freedom diff
        p_inter = stats.chi2.sf(lr_stat, df_diff)

        # 6) optional plot
        if plot:
            self._plot_family_interaction(
                df, property_name, bin_col,
                y=outcome_col,
                title=(
                    f"{property_name} vs RMSD\n"
                    f"{group1_label} (blue) vs {group2_label} (orange)\n"
                    f"interaction p = {p_inter:.3g}"
                )
            )

        # 7) return
        return {
            "model_full": m_full,
            "model_noint": m_noint,
            "lr_stat": lr_stat,
            "p_interaction": p_inter,
            "data": df,
            "outcome": outcome_col
        }

    def _plot_family_interaction(
        self,
        df: pd.DataFrame,
        property_name: str,
        bin_col: str,
        y: Optional[str] = None,              # None → use 'success'
        title: Optional[str] = None,
        p_inter: Optional[float] = None       # optional annotation
    ):
        """
        Line/point plot of physics vs ML (or arbitrary families) across
        quantile bins of `property_name`.

        Parameters
        ----------
        df : tidy DataFrame  (must have columns: family, protein, `bin_col`, y)
        property_name : str  (for axis label)
        bin_col : str        (categorical bins created earlier)
        y : outcome column
            - None  → assumes 'success' (binary 0/1)
            - str   → continuous outcome column name
        title : str
        p_inter : float
            If provided, printed in the title as "interaction p = …".
        """
        if y is None:
            y = "success"

        fig, ax = plt.subplots(figsize=(8, 5))

        # numeric success‐rate with Wilson CI
        if df[y].dropna().isin([0, 1]).all():
            tmp = (
                df
                .groupby([bin_col, 'family'])[y]
                .agg(['sum', 'count'])
                .reset_index()
            )
            tmp['rate'] = tmp['sum'] / tmp['count']
            # Wilson CI
            ci_low, ci_up = zip(*[
                proportion_confint(c, n, method="wilson")
                for c, n in zip(tmp['sum'], tmp['count'])
            ])
            tmp['ci_low'] = ci_low
            tmp['ci_up']  = ci_up

            sns.pointplot(
                data=tmp,
                x=bin_col, y='rate', hue='family',
                errorbar=None,  # we draw CIs manually
                ax=ax, dodge=0.3, join=True
            )
            # manual error bars
            for (_, row) in tmp.iterrows():
                x_pos = (
                    list(tmp[bin_col].cat.categories).index(row[bin_col])
                    + (-0.1 if row['family'] == tmp['family'].unique()[0] else 0.1)
                )
                ax.errorbar(
                    x=x_pos, y=row['rate'],
                    yerr=[[row['rate'] - row['ci_low']], [row['ci_up'] - row['rate']]],
                    fmt='none', ecolor='black', capsize=3, lw=1
                )
            ax.set_ylabel('Success rate')
        else:
            # Continuous outcome – plot mean ± 95 % CI (t-SE)
            tmp = (
                df
                .groupby([bin_col, 'family'])[y]
                .agg(['mean', 'std', 'count'])
                .reset_index()
            )
            se = tmp['std'] / np.sqrt(tmp['count'])
            tmp['ci_low'] = tmp['mean'] - 1.96 * se
            tmp['ci_up']  = tmp['mean'] + 1.96 * se

            sns.pointplot(
                data=tmp,
                x=bin_col, y='mean', hue='family',
                errorbar=None, dodge=0.3, join=True, ax=ax
            )
            for (_, row) in tmp.iterrows():
                x_pos = (
                    list(tmp[bin_col].cat.categories).index(row[bin_col])
                    + (-0.1 if row['family'] == tmp['family'].unique()[0] else 0.1)
                )
                ax.errorbar(
                    x=x_pos, y=row['mean'],
                    yerr=[[row['mean'] - row['ci_low']], [row['ci_up'] - row['mean']]],
                    fmt='none', ecolor='black', capsize=3, lw=1
                )
            ax.set_ylabel(y)

        ax.set_xlabel(f'{property_name} (quantile bins)')
        ax.set_title(
            title if title
            else f'{property_name}: family interaction'
        )
        if p_inter is not None:
            ax.set_title(
                (title or f'{property_name}: family interaction')
                + f'\ninteraction p = {p_inter:.3g}'
            )

        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=.3)
        ax.legend(title='Family')
        plt.tight_layout()
        plt.show()

    def analyze_comparative_fixed_effects(
        self,
        df_mixed: pd.DataFrame,
        property: str,
        method_col: str = 'Method_Type',
        rmsd_col: str = 'rmsd',
        figsize=(10, 6)
    ):
        """
        Perform a fixed effects analysis on the provided dataframe. 
        """

        model = smf.mixedlm(
            f"rmsd ~ {property} * Method_Type",
            data=df_mixed,
            groups=df_mixed["protein"]
        ).fit()

        print(model.summary())

    def analyze_property_gam(self, df, property_name, outcome_col="rmsd", plot=True):
        """
        Performs Generalized Additive Model (GAM) analysis for comparing how a continuous
        property affects outcomes across different method families.
        
        Parameters:
        -----------
        df : DataFrame
            Input DataFrame with columns for outcome, property, family, and protein
        property_name : str
            Name of the continuous predictor column
        outcome_col : str
            Name of the outcome variable (default: "rmsd")
        plot : bool
            Whether to generate partial dependence plots (default: True)
            
        Returns:
        --------
        dict
            Analysis results including fitted models and statistics
        """
        results = {}
        
        # Basic validation
        required_cols = [outcome_col, property_name, 'family', 'protein']
        if not all(col in df.columns for col in required_cols):
            return {"error": f"Missing required columns. Need: {required_cols}"}
        
        if not pd.api.types.is_numeric_dtype(df[property_name]):
            return {"error": f"Property '{property_name}' must be numeric"}
        
        # Prepare data
        df_gam = df.copy()
        df_gam['family'] = pd.Categorical(df_gam['family'])
        df_gam = df_gam.dropna(subset=required_cols).reset_index(drop=True)
        
        if len(df_gam) < 10 or df_gam[property_name].nunique() < 2 or df_gam['family'].nunique() < 2:
            return {"error": "Insufficient data for GAM analysis"}
        
        # Define model terms
        gam_terms_full = (
            f('family', coding='treatment') + s(property_name) +
            s(property_name, by='family') + s('protein', bs='re')
        )
        gam_terms_noint = (
            f('family', coding='treatment') + s(property_name) + s('protein', bs='re')
        )
    
        # Fit models
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                model_full = LinearGAM(gam_terms_full).fit(df_gam, df_gam[outcome_col])
                model_noint = LinearGAM(gam_terms_noint).fit(df_gam, df_gam[outcome_col])
            
            # LRT for interaction
            lr_stat = 2 * (model_full.loglikelihood_ - model_noint.loglikelihood_)
            df_diff = model_full.statistics_['edof'] - model_noint.statistics_['edof']
            p_interaction = stats.chi2.sf(lr_stat, df_diff) if df_diff > 1e-6 else np.nan
            
            results.update({
                "model_full": model_full,
                "model_noint": model_noint,
                "lr_stat": lr_stat,
                "df_diff": df_diff,
                "p_interaction": p_interaction,
            })
            
            # Generate plot if requested
            if plot:
                results["plot"] = self.plot_gam_partial_dependencies(
                    model_full, outcome_col, property_name, p_interaction
                )
                
        except Exception as e:
            return {"error": f"GAM analysis failed: {str(e)}"}
        
        return results
    
    def plot_gam_partial_dependencies(self, model_gam, outcome_col, property_name, p_interaction=None):
        """
        Generate partial dependence plots for a GAM model.
        
        Parameters:
        -----------
        model_gam : LinearGAM
            Fitted GAM model
        outcome_col : str
            Name of the outcome variable
        property_name : str
            Name of the property analyzed
        p_interaction : float, optional
            P-value of the interaction term
            
        Returns:
        --------
        matplotlib.figure.Figure or None
        """
        try:
            # Count non-intercept terms to plot
            num_terms = sum(1 for term in model_gam.terms if not term.isintercept)
            if num_terms == 0:
                return None
                
            # Create figure with appropriate size
            fig, axes = plt.subplots(
                1, num_terms, 
                figsize=(min(6 * num_terms, 20), 5), 
                squeeze=False
            )
            
            # Plot terms
            model_gam.plot(ax=axes[0], align_ylabels=True)
            
            # Add title
            p_str = f"{p_interaction:.3g}" if p_interaction is not None else "N/A"
            fig.suptitle(
                f"GAM: {outcome_col} ~ family * s({property_name})\n"
                f"Interaction p-value: {p_str}",
                fontsize=14
            )
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            return fig
            
        except Exception as e:
            print(f"GAM plotting failed: {str(e)}")
            return None

    def complementary_success_analysis(
        self,
        method1: Union[str, List[str]],
        method2: Union[str, List[str]],
        rmsd_threshold: float = 2.0,
        best_pose: bool = True,
        plot: bool = True
    ) -> Dict:
        """
        Identify proteins where one method (or method-set) succeeds (RMSD<=thr)
        and the other fails (RMSD>thr), and vice-versa.  Returns counts plus the
        McNemar test for paired success/failure.

        Parameters
        ----------
        method1, method2 : str or list[str]
            Single method name or list of method names that form 'side A' and 'side B'.
        rmsd_threshold   : float
            Success cutoff in Å.
        best_pose        : bool
            If True, use the best RMSD among poses for each method–protein pair.
        plot             : bool
            If True, draws an UpSet-like bar plot of the four success/fail patterns.

        Returns
        -------
        dict with keys
        - table              2×2 numpy array  [[both succ, m1succ_m2fail],
                                                [m1fail_m2succ, both fail]]
        - mcnemar_p          exact McNemar p-value
        - list_*             lists of protein IDs in each quadrant
        """
        # -------- 1. normalise inputs to lists
        m1 = [method1] if isinstance(method1, str) else list(method1)
        m2 = [method2] if isinstance(method2, str) else list(method2)

        # -------- 2. best-RMSD table for every requested method
        def best_rmsd_for(methods):
            dfs = [
                self.get_best_rmsd_per_protein(m, rmsd_threshold=rmsd_threshold)
                for m in methods
            ]
            df = pd.concat(dfs)
            if best_pose:
                df = (
                    df
                    .sort_values('rmsd')
                    .groupby('protein', as_index=False)
                    .first()        # keep best
                )
            df = df[['protein', 'rmsd']]
            df['succ'] = df['rmsd'] <= rmsd_threshold
            return df

        df1 = best_rmsd_for(m1).rename(columns={'succ': 'succ1'})
        df2 = best_rmsd_for(m2).rename(columns={'succ': 'succ2'})

        # -------- 3. join on protein
        merged = df1.merge(df2, on='protein', how='inner')

        # success/fail quadrants
        both_succ      = merged[(merged.succ1) & (merged.succ2)]['protein'].tolist()
        m1succ_m2fail  = merged[(merged.succ1) & (~merged.succ2)]['protein'].tolist()
        m1fail_m2succ  = merged[(~merged.succ1) & (merged.succ2)]['protein'].tolist()
        both_fail      = merged[(~merged.succ1) & (~merged.succ2)]['protein'].tolist()

        table = np.array([
            [len(both_succ), len(m1succ_m2fail)],
            [len(m1fail_m2succ), len(both_fail)]
        ])

        # -------- 4. McNemar (paired) test
        mc = mcnemar(table, exact=True)
        p_mc = mc.pvalue

        # -------- 5. optional plot
        if plot:
            self._plot_complementary_quadrants(
                table,
                labels=(f"{'+'.join(m1)}", f"{'+'.join(m2)}"),
                p_mcnemar=p_mc
            )

        return dict(
            table        = table,
            mcnemar_p           = p_mc,
            list_both_success   = both_succ,
            list_m1succ_m2fail  = m1succ_m2fail,
            list_m1fail_m2succ  = m1fail_m2succ,
            list_both_fail      = both_fail
        )
    def compare_property_distributions_in_complementary_cases(self, 
                                                            complementary_results, 
                                                            properties, 
                                                            method_labels=None, 
                                                            property_types=None, 
                                                            plot=True,
                                                            test_significance=True,
                                                            figsize=None):
        """
        Compare the distribution of properties across complementary success/failure cases.
        
        Parameters:
        -----------
        complementary_results : dict
            Results from complementary_success_analysis() method
        properties : str or list
            Property name(s) to analyze
        method_labels : tuple, optional
            Labels for methods (e.g., 'ML', 'Physics')
        property_types : dict, optional
            Dictionary mapping property names to types ('continuous', 'discrete', 'binary')
            If None, will try to infer types automatically
        plot : bool, optional
            Whether to generate visualization plots
        test_significance : bool, optional
            Whether to perform statistical tests for distribution differences
        figsize : tuple, optional
            Figure size (width, height) in inches
            
        Returns:
        --------
        dict
            Dictionary containing analysis results for each property
        """
        # Convert single property to list for consistent handling
        if isinstance(properties, str):
            properties = [properties]
        
        # Extract protein lists for each category
        protein_groups = {
            'both_success': complementary_results['list_both_success'],
            'method1_only': complementary_results['list_m1succ_m2fail'],
            'method2_only': complementary_results['list_m1fail_m2succ'],
            'both_failure': complementary_results['list_both_fail']
        }
        
        # Set method labels if not provided
        if method_labels is None:
            method_labels = ('Method 1', 'Method 2')
        
        # Get the unique proteins from all groups
        all_proteins = set()
        for group in protein_groups.values():
            all_proteins.update(group)
        
        # Create a property lookup dictionary for each protein
        property_data = {}
        for prop in properties:
            # Find property data - filter to unique protein entries
            prop_df = self.df_combined.drop_duplicates(subset=['protein'])
            if prop in prop_df.columns:
                property_data[prop] = dict(zip(prop_df['protein'], prop_df[prop]))
            else:
                print(f"Warning: Property '{prop}' not found in the dataset")
        
        # Determine property types if not provided
        if property_types is None:
            property_types = {}
            for prop in properties:
                if prop not in property_data:
                    continue
                
                # Get a sample of values
                values = list(property_data[prop].values())
                if len(values) == 0:
                    continue
                    
                # Infer type
                if pd.api.types.is_bool_dtype(pd.Series(values)):
                    property_types[prop] = 'binary'
                elif pd.api.types.is_numeric_dtype(pd.Series(values)) and len(set(values)) > 10:
                    property_types[prop] = 'continuous'
                else:
                    property_types[prop] = 'discrete'
        
        # Create a dataframe for analysis
        analysis_data = []
        for group_name, protein_list in protein_groups.items():
            for protein in protein_list:
                row = {'protein': protein, 'group': group_name}
                
                # Add property values
                for prop in properties:
                    if prop in property_data and protein in property_data[prop]:
                        row[prop] = property_data[prop][protein]
                
                analysis_data.append(row)
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Create readable group names
        group_map = {
            'both_success': f'Both Success',
            'method1_only': f'{method_labels[0]} Only',
            'method2_only': f'{method_labels[1]} Only',
            'both_failure': f'Both Failure'
        }
        analysis_df['group_label'] = analysis_df['group'].map(group_map)
        
        # Results dictionary
        results = {'summary_stats': {}, 'test_results': {}, 'plots': {}}
        
        # Analyze each property
        for prop in properties:
            if prop not in property_data:
                continue
                
            prop_type = property_types.get(prop, 'unknown')
            
            # Calculate summary statistics
            summary = analysis_df.groupby('group')[prop].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
            results['summary_stats'][prop] = summary
            
            # Statistical tests based on property type
            if test_significance:
                if prop_type == 'continuous':
                    # ANOVA or Kruskal-Wallis
                    try:
                        f_stat, anova_p = stats.f_oneway(
                            *[analysis_df[analysis_df['group'] == g][prop].dropna() for g in protein_groups.keys()]
                        )
                        kw_stat, kw_p = stats.kruskal(
                            *[analysis_df[analysis_df['group'] == g][prop].dropna() for g in protein_groups.keys()
                            if len(analysis_df[analysis_df['group'] == g][prop].dropna()) > 0]
                        )
                        results['test_results'][prop] = {
                            'anova': {'f_stat': f_stat, 'p_value': anova_p},
                            'kruskal_wallis': {'stat': kw_stat, 'p_value': kw_p}
                        }
                    except:
                        pass
                elif prop_type == 'discrete' or prop_type == 'binary':
                    # Chi-square test
                    try:
                        contingency = pd.crosstab(analysis_df['group'], analysis_df[prop])
                        chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
                        results['test_results'][prop] = {
                            'chi_square': {'stat': chi2, 'p_value': chi2_p}
                        }
                    except:
                        pass
            
            # Create plots
            if plot:
                plot_title = f"Distribution of {prop} Across Complementary Cases"
                
                if figsize is None:
                    figsize = (12, 8)
                    
                fig = plt.figure(figsize=figsize)
                
                if prop_type == 'continuous':
                    # Create a 2x2 grid with different visualizations
                    gs = fig.add_gridspec(2, 2)
                    
                    # 1. Box plot
                    ax1 = fig.add_subplot(gs[0, 0])
                    sns.boxplot(x='group_label', y=prop, data=analysis_df, ax=ax1)
                    ax1.set_title('Box Plot')
                    ax1.set_xlabel('')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # 2. Violin plot
                    ax2 = fig.add_subplot(gs[0, 1])
                    sns.violinplot(x='group_label', y=prop, data=analysis_df, ax=ax2)
                    ax2.set_title('Violin Plot')
                    ax2.set_xlabel('')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    # 3. KDE plot
                    ax3 = fig.add_subplot(gs[1, 0])
                    for group in analysis_df['group'].unique():
                        group_data = analysis_df[analysis_df['group'] == group][prop].dropna()
                        if len(group_data) > 1:  # Need at least 2 points for KDE
                            sns.kdeplot(group_data, label=group_map[group], ax=ax3)
                    ax3.set_title('Density Plot')
                    ax3.set_xlabel(prop)
                    ax3.legend()
                    
                    # 4. CDF plot
                    ax4 = fig.add_subplot(gs[1, 1])
                    for group in analysis_df['group'].unique():
                        group_data = analysis_df[analysis_df['group'] == group][prop].dropna()
                        if len(group_data) > 0:
                            sorted_data = np.sort(group_data)
                            y = np.arange(1, len(sorted_data)+1) / len(sorted_data)
                            ax4.step(sorted_data, y, label=group_map[group])
                    ax4.set_title('Cumulative Distribution')
                    ax4.set_xlabel(prop)
                    ax4.set_ylabel('Cumulative Probability')
                    ax4.legend()
                    
                    # Add statistical test results if available
                    if prop in results.get('test_results', {}):
                        kw_p = results['test_results'][prop].get('kruskal_wallis', {}).get('p_value')
                        if kw_p is not None:
                            plt.figtext(0.5, 0.01, f"Kruskal-Wallis p-value: {kw_p:.4f}", 
                                        ha='center', fontsize=10, 
                                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                    
                elif prop_type == 'discrete' or prop_type == 'binary':
                    gs = fig.add_gridspec(1, 2)
                    
                    # 1. Count plot
                    ax1 = fig.add_subplot(gs[0, 0])
                    sns.countplot(x=prop, hue='group_label', data=analysis_df, ax=ax1)
                    ax1.set_title('Count by Group')
                    ax1.legend(title='Group')
                    
                    # 2. Stacked percentage
                    ax2 = fig.add_subplot(gs[0, 1])
                    prop_counts = pd.crosstab(analysis_df[prop], analysis_df['group_label'], normalize='columns')
                    prop_counts.plot(kind='bar', stacked=True, ax=ax2)
                    ax2.set_title('Percentage by Group')
                    ax2.set_ylabel('Percentage')
                    ax2.legend(title='Group')
                    
                    # Add chi-square test result if available
                    if prop in results.get('test_results', {}):
                        chi2_p = results['test_results'][prop].get('chi_square', {}).get('p_value')
                        if chi2_p is not None:
                            plt.figtext(0.5, 0.01, f"Chi-square p-value: {chi2_p:.4f}", 
                                    ha='center', fontsize=10,
                                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                
                # Common settings
                plt.suptitle(plot_title, fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                results['plots'][prop] = fig
                
        return results
    def _plot_complementary_quadrants(self, table, labels=('A','B'), p_mcnemar=None):
        """Tiny helper to visualise the 4-cell table."""
        import matplotlib.pyplot as plt

        both_succ, a_succ_b_fail = table[0]
        b_succ_a_fail, both_fail = table[1]

        fig, ax = plt.subplots(figsize=(4,4))
        ax.bar([0,1], [both_succ, both_fail], color='grey', alpha=.4,label='both')
        ax.bar([0], [a_succ_b_fail], bottom=[both_succ], color='steelblue', label=f'{labels[0]} only')
        ax.bar([1], [b_succ_a_fail], bottom=[both_fail], color='orange', label=f'{labels[1]} only')
        ax.set_xticks([0,1]); ax.set_xticklabels(['success','fail'])
        ax.set_ylabel('proteins')
        ax.set_title(f'Complementary outcomes\nMcNemar p = {p_mcnemar:.3g}')
        ax.legend()
        plt.tight_layout(); plt.show()

    def stratified_mcnemar_analysis(
        self,
        methods_group1: Union[str, List[str]],
        methods_group2: Union[str, List[str]],
        property_name: str,
        rmsd_threshold: float = 2.0,
        n_strata: int = 5,
        bins: Optional[List[float]] = None,
        labels: Optional[List[Any]] = None,
        exact: bool = True,
        plot: bool = True
    ) -> Dict[str, Any]:
        """
        Perform a stratified McNemar test comparing two methods (or method groups) across strata of a continuous or discrete property.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least columns ['protein','method','rmsd', property_name]
        methods_group1 : str or list of str
            Method name or list of method names for group 1 (e.g., 'diffdock')
        methods_group2 : str or list of str
            Method name or list of method names for group 2 (e.g., 'vina')
        property_name : str
            Column name of the stratification property (continuous or categorical)
        rmsd_threshold : float, optional
            Threshold for calling success (<=) vs failure (>). Default 2.0 Å.
        n_strata : int, optional
            Number of strata to cut a continuous property into. Ignored if `bins` provided.
        bins : list of float, optional
            Explicit bin edges to use for stratification. If None, quantile-based bins are used.
        labels : list, optional
            Labels for the strata. If None, generated automatically.
        exact : bool, optional
            Whether to use the exact McNemar test (True) or approximate (False).

        Returns
        -------
        results : dict
            - 'per_stratum': dict mapping stratum label -> {'table': np.ndarray, 'pvalue': float}
            - 'cmh': {'oddsratio': float, 'pvalue': float}
            - 'strata_labels': list of stratum labels in order
        """
        # normalize method lists
        g1 = [methods_group1] if isinstance(methods_group1, str) else list(methods_group1)
        g2 = [methods_group2] if isinstance(methods_group2, str) else list(methods_group2)

        # filter and compute best per-protein success flags
        def best_success(df_sub, methods):
            df_f = df_sub[df_sub['method'].isin(methods)]
            # best RMSD per protein
            best = df_f.groupby('protein')['rmsd'].min().reset_index()
            best['succ'] = best['rmsd'] <= rmsd_threshold
            return best[['protein','succ']]

        best1 = best_success(self.df_combined, g1).rename(columns={'succ': 'succ1'})
        best2 = best_success(self.df_combined, g2).rename(columns={'succ': 'succ2'})
        merged = pd.merge(best1, best2, on='protein')

        # stratify property
        # attach property to merged
        prop = self.df_combined.drop_duplicates(subset=['protein']).set_index('protein')[property_name]
        merged[property_name] = merged['protein'].map(prop)

        if bins is None:
            # quantile-based bins, drop duplicate edges if any
            labels = labels or [f"Stratum {i+1}" for i in range(n_strata)]
            merged['stratum'] = pd.qcut(
                merged[property_name],
                q=n_strata,
                labels=labels,
                duplicates='drop'      # allow dropping bins with identical edges
            )
        else:
            labels = labels or None
            merged['stratum'] = pd.cut(merged[property_name], bins=bins, labels=labels)

        results: Dict[str, Any] = {'per_stratum': {}, 'cmh': {}, 'strata_labels': []}

        # build per-stratum tables and perform McNemar
        tables = []
        for stratum, group in merged.groupby('stratum'):
            tab = np.zeros((2,2), int)
            # both success
            tab[0,0] = int(((group['succ1'] == True) & (group['succ2'] == True)).sum())
            # g1 only
            tab[0,1] = int(((group['succ1'] == True) & (group['succ2'] == False)).sum())
            # g2 only
            tab[1,0] = int(((group['succ1'] == False)& (group['succ2'] == True)).sum())
            # both fail
            tab[1,1] = int(((group['succ1'] == False)& (group['succ2'] == False)).sum())
            # mcnemar test
            res = mcnemar(tab, exact=exact)
            results['per_stratum'][stratum] = {'table': tab, 'pvalue': res.pvalue}
            results['strata_labels'].append(stratum)
            tables.append(tab)

        # Cochran‑Mantel‑Haenszel pooled test  +  homogeneity check
        st  = StratifiedTable(tables)
        cmh = st.test_null_odds()          # pooled association
        homo = st.test_equal_odds()        # OR homogeneity across strata

        results['cmh'] = {
            'statistic': cmh.statistic,        # χ² value
            'pvalue': cmh.pvalue,
            'pooled_or': st.oddsratio_pooled   # common odds‑ratio estimate
        }
        results['homogeneity'] = {
            'statistic': homo.statistic,
            'pvalue': homo.pvalue
        }
        
        if plot:
            self.plot_stratified_mcnemar_results(results)
        return results
    
    def plot_stratified_mcnemar_results(self, results):
        """
        Plot stratified McNemar results: per-stratum p-values and odds ratios with 95% CIs.
        
        Parameters
        ----------
        results : dict
            Output from stratified_mcnemar_analysis(), containing:
            - 'per_stratum': dict of {stratum: {'table': np.ndarray, 'pvalue': float}}
            - 'strata_labels': list of stratum labels in order
        """
        
        strata = results.get('strata_labels', [])
        per = results.get('per_stratum', {})
        
        # Extract p-values
        pvals = [per[s]['pvalue'] for s in strata]
        
        # Compute odds ratios and CIs per stratum
        or_list, ci_low, ci_up = [], [], []
        for s in strata:
            table = per[s]['table']
            a, b = table[0]
            c, d = table[1]
            # compute OR and 95% CI via log method if possible
            if a > 0 and b > 0 and c > 0 and d > 0:
                or_val = (a * d) / (b * c)
                se_log = np.sqrt(1/a + 1/b + 1/c + 1/d)
                z = 1.96
                ci_l = np.exp(np.log(or_val) - z * se_log)
                ci_u = np.exp(np.log(or_val) + z * se_log)
            else:
                or_val, ci_l, ci_u = np.nan, np.nan, np.nan
            or_list.append(or_val)
            ci_low.append(ci_l)
            ci_up.append(ci_u)
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(strata))
        
        # Bar plot of p-values (log scale)
        ax1.bar(x, pvals, color='skyblue', label='McNemar p-value')
        ax1.axhline(0.05, color='red', linestyle='--', label='p = 0.05')
        ax1.set_yscale('log')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strata, rotation=45, ha='right')
        ax1.set_ylabel('McNemar p-value (log scale)')
        
        # Line + errorbars for odds ratios on twin axis
        ax2 = ax1.twinx()
        ax2.errorbar(
            x, or_list,
            yerr=[np.array(or_list) - np.array(ci_low), np.array(ci_up) - np.array(or_list)],
            fmt='o-', color='navy', label='Odds ratio'
        )
        ax2.set_yscale('log')
        ax2.set_ylabel('Odds Ratio (log scale)')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.title('Stratified McNemar Results')
        plt.tight_layout()

        # --- annotation of pooled OR and homogeneity --------------------
        cmh_info = results.get('cmh', {})
        homo_info = results.get('homogeneity', {})
        cmh_text = (
            f"CMH pooled OR = {cmh_info.get('pooled_or', np.nan):.2g}\n"
            f"CMH p = {cmh_info.get('pvalue', np.nan):.2g}"
        )
        homo_text = (
            f"Homogeneity p = {homo_info.get('pvalue', np.nan):.2g}"
            if 'pvalue' in homo_info else ''
        )
        ax1.annotate(
            cmh_text + ("\n" + homo_text if homo_text else ''),
            xy=(0.02, 0.95), xycoords='axes fraction',
            va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8),
            fontsize=9
        )

        plt.show()