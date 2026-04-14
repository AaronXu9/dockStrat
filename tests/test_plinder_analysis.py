import unittest
import pandas as pd
import numpy as np
from notebooks.plinder_analysis_utils import DockingAnalysisBase, PoseBustersAnalysis

class TestDockingAnalysisBase(unittest.TestCase):
    """Tests for the DockingAnalysisBase class."""
    
    def setUp(self):
        """Set up test data."""
        # Create mock data with 3 methods and 5 proteins
        data = []
        methods = ['method1', 'method2', 'method3']
        proteins = ['protein1', 'protein2', 'protein3', 'protein4', 'protein5']
        
        # Method 1: Good at proteins 1,2,3
        # Method 2: Good at proteins 3,4
        # Method 3: Good at proteins 1,5
        rmsd_values = {
            'method1': {'protein1': 1.5, 'protein2': 1.8, 'protein3': 1.9, 'protein4': 3.2, 'protein5': 4.0},
            'method2': {'protein1': 2.5, 'protein2': 3.8, 'protein3': 1.7, 'protein4': 1.6, 'protein5': 2.2},
            'method3': {'protein1': 1.7, 'protein2': 4.8, 'protein3': 2.2, 'protein4': 3.6, 'protein5': 1.4}
        }
        
        # Create dataframe rows
        for method in methods:
            for protein in proteins:
                data.append({
                    'method': method,
                    'protein': protein,
                    'rmsd': rmsd_values[method][protein]
                })
        
        self.df_test = pd.DataFrame(data)
        self.analyzer = DockingAnalysisBase(self.df_test)
    
    def test_get_best_rmsd_per_protein_single_method(self):
        """Test getting best RMSD per protein for a single method."""
        result = self.analyzer.get_best_rmsd_per_protein(method='method1', rmsd_threshold=2.0)
        
        # Check that we get the right number of results
        self.assertEqual(len(result), 5)
        
        # Check that success column is correctly calculated
        self.assertTrue(result[result['protein'] == 'protein1']['rmsd_≤_2.0å'].iloc[0])
        self.assertTrue(result[result['protein'] == 'protein2']['rmsd_≤_2.0å'].iloc[0])
        self.assertTrue(result[result['protein'] == 'protein3']['rmsd_≤_2.0å'].iloc[0])
        self.assertFalse(result[result['protein'] == 'protein4']['rmsd_≤_2.0å'].iloc[0])
        self.assertFalse(result[result['protein'] == 'protein5']['rmsd_≤_2.0å'].iloc[0])
    
    def test_get_best_rmsd_per_protein_all_methods(self):
        """Test getting best RMSD per protein for all methods."""
        result = self.analyzer.get_best_rmsd_per_protein(method=None)
        
        # Check that we get results for all methods
        self.assertEqual(len(result), 3)
        self.assertIn('method1', result)
        self.assertIn('method2', result)
        self.assertIn('method3', result)
        
        # Check some specific values
        self.assertTrue(result['method1'][result['method1']['protein'] == 'protein1']['rmsd_≤_2.0å'].iloc[0])
        self.assertFalse(result['method1'][result['method1']['protein'] == 'protein4']['rmsd_≤_2.0å'].iloc[0])
    
    def test_create_success_matrix(self):
        """Test creating a success matrix."""
        methods = ['method1', 'method2', 'method3']
        matrix = self.analyzer.create_success_matrix(methods, rmsd_threshold=2.0)
        
        # Check matrix dimensions
        self.assertEqual(len(matrix), 5)  # 5 proteins
        
        # Check column names
        expected_columns = ['protein', 'method1_success', 'method2_success', 
                           'method3_success', 'total_successes']
        for col in expected_columns:
            self.assertIn(col, matrix.columns)
        
        # Check some values
        self.assertEqual(matrix[matrix['protein'] == 'protein1']['total_successes'].iloc[0], 2)  # Methods 1,3 succeed
        self.assertEqual(matrix[matrix['protein'] == 'protein3']['total_successes'].iloc[0], 2)  # Methods 1,2 succeed
        self.assertEqual(matrix[matrix['protein'] == 'protein2']['total_successes'].iloc[0], 1)  # Only Method 1 succeeds


class TestPoseBustersAnalysis(unittest.TestCase):
    """Tests for the PoseBustersAnalysis class."""
    
    def setUp(self):
        """Set up test data with PoseBusters metrics."""
        # Create mock data with 2 methods, 5 proteins, and PoseBusters metrics
        data = []
        methods = ['method1', 'method2']
        proteins = ['protein1', 'protein2', 'protein3', 'protein4', 'protein5']
        
        # Define RMSD values 
        rmsd_values = {
            'method1': {'protein1': 1.5, 'protein2': 1.8, 'protein3': 3.2, 'protein4': 3.5, 'protein5': 1.7},
            'method2': {'protein1': 2.5, 'protein2': 1.6, 'protein3': 1.8, 'protein4': 3.6, 'protein5': 4.0}
        }
        
        # Define PoseBusters metrics - create patterns that correlate with success
        # bond_lengths: Strong correlation with method1 success
        # internal_steric_clash: Strong correlation with method2 success
        # minimum_distance_to_protein: Weak correlation with both
        
        metrics = {
            'method1': {
                'bond_lengths': {'protein1': True, 'protein2': True, 'protein3': False, 'protein4': False, 'protein5': True},
                'internal_steric_clash': {'protein1': True, 'protein2': False, 'protein3': True, 'protein4': False, 'protein5': True},
                'minimum_distance_to_protein': {'protein1': True, 'protein2': True, 'protein3': True, 'protein4': False, 'protein5': True}
            },
            'method2': {
                'bond_lengths': {'protein1': False, 'protein2': True, 'protein3': True, 'protein4': True, 'protein5': False},
                'internal_steric_clash': {'protein1': False, 'protein2': True, 'protein3': True, 'protein4': False, 'protein5': False},
                'minimum_distance_to_protein': {'protein1': True, 'protein2': True, 'protein3': True, 'protein4': True, 'protein5': False}
            }
        }
        
        # Create dataframe rows
        for method in methods:
            for protein in proteins:
                data.append({
                    'method': method,
                    'protein': protein,
                    'rmsd': rmsd_values[method][protein],
                    'bond_lengths': metrics[method]['bond_lengths'][protein],
                    'internal_steric_clash': metrics[method]['internal_steric_clash'][protein],
                    'minimum_distance_to_protein': metrics[method]['minimum_distance_to_protein'][protein]
                })
        
        self.df_test = pd.DataFrame(data)
        self.analyzer = PoseBustersAnalysis(self.df_test)
    
    def test_analyze_single_method(self):
        """Test analyzing a single method."""
        result = self.analyzer.analyze_single_method(method='method1', rmsd_threshold=2.0, plot=False)
        
        # Check that the result contains expected fields
        self.assertIn('method', result)
        self.assertIn('overall_success_rate', result)
        self.assertIn('metrics_stats', result)
        self.assertIn('significant_metrics', result)
        
        # Check metrics stats structure
        metrics_stats = result['metrics_stats']
        self.assertIn('bond_lengths', metrics_stats)
        self.assertIn('success_rate_when_pass', metrics_stats['bond_lengths'])
        
        # With our test data, bond_lengths should be significantly correlated with method1 success
        self.assertIn('bond_lengths', result['significant_metrics'])
    
    def test_analyze_universal(self):
        """Test analyzing universal patterns."""
        result = self.analyzer.analyze_universal(methods=['method1', 'method2'], rmsd_threshold=2.0, plot=False)
        
        # Check that the result contains expected fields
        self.assertIn('posebusters_stats', result)
        self.assertIn('all_success_proteins', result)
        self.assertIn('all_failure_proteins', result)
        
        # Check the proteins lists
        # protein2 should succeed with both methods
        self.assertIn('protein2', result['all_success_proteins'])
        # protein4 should fail with both methods
        self.assertIn('protein4', result['all_failure_proteins'])
    
    def test_analyze_comparative(self):
        """Test comparative analysis between methods."""
        result = self.analyzer.analyze_comparative(
            method1='method1', method2='method2', rmsd_threshold=2.0, plot=False)
        
        # Check that the result contains expected fields
        self.assertIn('posebusters_stats', result)
        self.assertIn('method1_only_proteins', result)
        self.assertIn('method2_only_proteins', result)
        
        # Based on our test data:
        # protein1 and protein5 should succeed only with method1
        self.assertIn('protein1', result['method1_only_proteins'])
        self.assertIn('protein5', result['method1_only_proteins'])
        
        # protein3 should succeed only with method2
        self.assertIn('protein3', result['method2_only_proteins'])


if __name__ == '__main__':
    unittest.main()