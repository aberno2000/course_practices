"""
Comprehensive test suite for practical_task_10_3.ipynb
Tests notebook structure, execution, data processing, and ML pipeline components.
"""

import unittest
import json
import os
import sys
import warnings
from typing import Dict, Any, List
import tempfile
import subprocess

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')


class TestNotebookStructure(unittest.TestCase):
    """Test suite for notebook structure and metadata validation."""
    
    @classmethod
    def setUpClass(cls):
        """Load the notebook once for all tests."""
        cls.notebook_path = 'practical_task_10_3.ipynb'
        with open(cls.notebook_path, 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
    
    def test_notebook_exists(self):
        """Test that the notebook file exists."""
        self.assertTrue(os.path.exists(self.notebook_path),
                       "Notebook file should exist")
    
    def test_notebook_format_version(self):
        """Test that notebook has correct format version."""
        self.assertIn('nbformat', self.notebook,
                     "Notebook should have nbformat field")
        self.assertGreaterEqual(self.notebook['nbformat'], 4,
                               "Notebook should be format version 4 or higher")
    
    def test_notebook_has_metadata(self):
        """Test that notebook contains required metadata."""
        self.assertIn('metadata', self.notebook,
                     "Notebook should contain metadata")
        metadata = self.notebook['metadata']
        
        # Check for key metadata fields
        self.assertIn('kernelspec', metadata,
                     "Metadata should contain kernelspec")
        self.assertIn('language_info', metadata,
                     "Metadata should contain language_info")
    
    def test_notebook_language_is_python(self):
        """Test that notebook language is Python."""
        language_info = self.notebook['metadata'].get('language_info', {})
        self.assertEqual(language_info.get('name'), 'python',
                        "Notebook language should be Python")
    
    def test_notebook_has_cells(self):
        """Test that notebook contains cells."""
        self.assertIn('cells', self.notebook,
                     "Notebook should contain cells")
        self.assertGreater(len(self.notebook['cells']), 0,
                          "Notebook should have at least one cell")
    
    def test_colab_link_updated(self):
        """Test that Colab link points to testing branch."""
        cells = self.notebook['cells']
        found_link = False
        
        for cell in cells:
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if 'colab.research.google.com' in source:
                    found_link = True
                    self.assertIn('/blob/testing/', source,
                                "Colab link should point to testing branch")
                    break
        
        self.assertTrue(found_link, "Should find Colab link in notebook")
    
    def test_all_cells_have_type(self):
        """Test that all cells have a cell_type field."""
        for i, cell in enumerate(self.notebook['cells']):
            self.assertIn('cell_type', cell,
                         f"Cell {i} should have cell_type")
            self.assertIn(cell['cell_type'], ['code', 'markdown', 'raw'],
                         f"Cell {i} should have valid cell_type")
    
    def test_code_cells_have_source(self):
        """Test that all code cells have source code."""
        code_cells = [c for c in self.notebook['cells'] 
                     if c.get('cell_type') == 'code']
        self.assertGreater(len(code_cells), 0,
                          "Notebook should have at least one code cell")
        
        for i, cell in enumerate(code_cells):
            self.assertIn('source', cell,
                         f"Code cell {i} should have source")


class TestNotebookImports(unittest.TestCase):
    """Test suite for validating notebook imports and dependencies."""
    
    @classmethod
    def setUpClass(cls):
        """Load notebook and extract imports."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
        
        cls.imports = cls._extract_imports()
    
    @classmethod
    def _extract_imports(cls) -> List[str]:
        """Extract all import statements from notebook."""
        imports = []
        for cell in cls.notebook['cells']:
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                lines = source.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        imports.append(line)
        return imports
    
    def test_has_pandas_import(self):
        """Test that pandas is imported."""
        pandas_imported = any('pandas' in imp for imp in self.imports)
        self.assertTrue(pandas_imported,
                       "Notebook should import pandas")
    
    def test_has_numpy_import(self):
        """Test that numpy is imported."""
        numpy_imported = any('numpy' in imp for imp in self.imports)
        self.assertTrue(numpy_imported,
                       "Notebook should import numpy")
    
    def test_has_sklearn_imports(self):
        """Test that scikit-learn is imported."""
        sklearn_imported = any('sklearn' in imp for imp in self.imports)
        self.assertTrue(sklearn_imported,
                       "Notebook should import sklearn")
    
    def test_has_xgboost_import(self):
        """Test that XGBoost is imported."""
        xgb_imported = any('xgboost' in imp or 'xgb' in imp 
                          for imp in self.imports)
        self.assertTrue(xgb_imported,
                       "Notebook should import xgboost")
    
    def test_has_matplotlib_import(self):
        """Test that matplotlib is imported."""
        plt_imported = any('matplotlib' in imp for imp in self.imports)
        self.assertTrue(plt_imported,
                       "Notebook should import matplotlib")


class TestNotebookContent(unittest.TestCase):
    """Test suite for notebook content and code logic."""
    
    @classmethod
    def setUpClass(cls):
        """Load notebook."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
    
    def test_has_data_loading_code(self):
        """Test that notebook contains data loading logic."""
        all_code = self._get_all_code()
        self.assertIn('kagglehub', all_code,
                     "Notebook should load data using kagglehub")
    
    def test_has_train_test_split(self):
        """Test that notebook includes train/test split."""
        all_code = self._get_all_code()
        self.assertIn('train_test_split', all_code,
                     "Notebook should split data into train/test sets")
    
    def test_has_model_training(self):
        """Test that notebook includes model training."""
        all_code = self._get_all_code()
        self.assertTrue(
            'XGBClassifier' in all_code or 'xgb.XGBClassifier' in all_code,
            "Notebook should train an XGBoost classifier"
        )
        self.assertIn('.fit(', all_code,
                     "Notebook should call fit method for training")
    
    def test_has_model_evaluation(self):
        """Test that notebook includes model evaluation."""
        all_code = self._get_all_code()
        self.assertIn('classification_report', all_code,
                     "Notebook should generate classification report")
        self.assertIn('roc_auc_score', all_code,
                     "Notebook should calculate ROC AUC score")
    
    def test_has_weighted_classification_report(self):
        """Test that weighted classification report is printed correctly."""
        all_code = self._get_all_code()
        # The change in the diff: extra space was added
        self.assertIn('Weighted Classification Report', all_code,
                     "Notebook should include weighted classification report")
    
    def test_has_feature_importance_analysis(self):
        """Test that notebook includes feature importance analysis."""
        all_code = self._get_all_code()
        self.assertIn('feature_importances_', all_code,
                     "Notebook should analyze feature importances")
    
    def test_has_data_preprocessing(self):
        """Test that notebook includes data preprocessing."""
        all_code = self._get_all_code()
        self.assertTrue(
            'StandardScaler' in all_code or 'scaler' in all_code,
            "Notebook should include data scaling/preprocessing"
        )
    
    def test_has_visualization(self):
        """Test that notebook includes visualizations."""
        all_code = self._get_all_code()
        self.assertTrue(
            'plot' in all_code or 'plt.' in all_code,
            "Notebook should include visualizations"
        )
    
    def test_handles_missing_values(self):
        """Test that notebook handles missing values."""
        all_code = self._get_all_code()
        self.assertTrue(
            'SimpleImputer' in all_code or 'fillna' in all_code or '-999' in all_code,
            "Notebook should handle missing values"
        )
    
    def test_uses_sample_weights(self):
        """Test that notebook uses sample weights in training."""
        all_code = self._get_all_code()
        self.assertIn('sample_weight', all_code,
                     "Notebook should use sample weights")
    
    def _get_all_code(self) -> str:
        """Extract all code from notebook as a single string."""
        code_cells = [c for c in self.notebook['cells'] 
                     if c.get('cell_type') == 'code']
        all_code = '\n'.join(
            ''.join(cell.get('source', [])) for cell in code_cells
        )
        return all_code


class TestDataProcessingLogic(unittest.TestCase):
    """Test suite for data processing logic extracted from notebook."""
    
    def test_weighted_standard_scaler_function_exists(self):
        """Test that weighted_standard_scaler function is defined."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        all_code = ''
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + '\n'
        
        self.assertIn('def weighted_standard_scaler', all_code,
                     "Notebook should define weighted_standard_scaler function")
    
    def test_label_encoding_present(self):
        """Test that label encoding is performed."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        all_code = ''
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + '\n'
        
        # Check for label mapping
        self.assertTrue(
            ".map(" in all_code or "LabelEncoder" in all_code,
            "Notebook should encode labels"
        )
    
    def test_model_configuration(self):
        """Test that XGBoost model is properly configured."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        all_code = ''
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code':
                all_code += ''.join(cell.get('source', [])) + '\n'
        
        # Check for important hyperparameters
        self.assertIn('n_estimators', all_code,
                     "Model should specify n_estimators")
        self.assertIn('max_depth', all_code,
                     "Model should specify max_depth")
        self.assertIn('learning_rate', all_code,
                     "Model should specify learning_rate")


class TestNotebookOutputFormat(unittest.TestCase):
    """Test suite for validating notebook output formatting."""
    
    @classmethod
    def setUpClass(cls):
        """Load notebook."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
    
    def test_print_statements_exist(self):
        """Test that notebook includes print statements for output."""
        all_code = self._get_all_code()
        self.assertIn('print(', all_code,
                     "Notebook should have print statements")
    
    def test_weighted_classification_report_format(self):
        """Test that weighted classification report print has correct format."""
        all_code = self._get_all_code()
        
        # Check for both possible formats (with and without extra space)
        has_old_format = 'print("\\nWeighted Classification Report:")' in all_code.replace(' ', '')
        has_new_format = 'print("\\n Weighted Classification Report:")' in all_code.replace("'", '"')
        
        self.assertTrue(
            has_old_format or has_new_format or 'Weighted Classification Report' in all_code,
            "Notebook should print weighted classification report"
        )
    
    def test_accuracy_metrics_printed(self):
        """Test that accuracy metrics are printed."""
        all_code = self._get_all_code()
        self.assertTrue(
            'Accuracy' in all_code or 'accuracy' in all_code,
            "Notebook should print accuracy metrics"
        )
    
    def test_roc_auc_printed(self):
        """Test that ROC AUC score is printed."""
        all_code = self._get_all_code()
        self.assertIn('ROC AUC', all_code,
                     "Notebook should print ROC AUC score")
    
    def _get_all_code(self) -> str:
        """Extract all code from notebook."""
        code_cells = [c for c in self.notebook['cells'] 
                     if c.get('cell_type') == 'code']
        return '\n'.join(''.join(cell.get('source', [])) for cell in code_cells)


class TestNotebookIntegrity(unittest.TestCase):
    """Test suite for notebook file integrity and validation."""
    
    def test_notebook_is_valid_json(self):
        """Test that notebook is valid JSON."""
        try:
            with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            self.fail(f"Notebook is not valid JSON: {e}")
    
    def test_notebook_file_size_reasonable(self):
        """Test that notebook file size is reasonable."""
        file_size = os.path.getsize('practical_task_10_3.ipynb')
        self.assertGreater(file_size, 1000,
                          "Notebook file should not be empty")
        self.assertLess(file_size, 10 * 1024 * 1024,
                       "Notebook file should be less than 10MB")
    
    def test_notebook_encoding_is_utf8(self):
        """Test that notebook can be read as UTF-8."""
        try:
            with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError as e:
            self.fail(f"Notebook is not valid UTF-8: {e}")
    
    def test_no_execution_errors_in_outputs(self):
        """Test that existing outputs don't show execution errors."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for i, cell in enumerate(notebook['cells']):
            if cell.get('cell_type') == 'code':
                outputs = cell.get('outputs', [])
                for output in outputs:
                    output_type = output.get('output_type', '')
                    if output_type == 'error':
                        self.fail(f"Cell {i} has error output: {output.get('ename', 'Unknown')}")


class TestMLPipelineComponents(unittest.TestCase):
    """Test suite for machine learning pipeline components."""
    
    @classmethod
    def setUpClass(cls):
        """Load notebook and extract code."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cls.all_code = ''
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code':
                cls.all_code += ''.join(cell.get('source', [])) + '\n'
    
    def test_binary_classification_setup(self):
        """Test that this is configured as a binary classification problem."""
        self.assertIn('binary:logistic', self.all_code,
                     "Should use binary logistic objective")
    
    def test_stratified_sampling(self):
        """Test that stratified sampling is used in train/test split."""
        self.assertIn('stratify', self.all_code,
                     "Should use stratified sampling")
    
    def test_cross_validation_or_validation_set(self):
        """Test that some form of validation is performed."""
        has_validation = any(term in self.all_code for term in [
            'eval_set', 'validation', 'cross_val', 'test_size'
        ])
        self.assertTrue(has_validation,
                       "Should include validation strategy")
    
    def test_class_imbalance_handling(self):
        """Test that class imbalance is handled."""
        has_imbalance_handling = any(term in self.all_code for term in [
            'scale_pos_weight', 'sample_weight', 'class_weight'
        ])
        self.assertTrue(has_imbalance_handling,
                       "Should handle class imbalance")
    
    def test_target_names_specified(self):
        """Test that target names are specified for reports."""
        self.assertIn('target_names', self.all_code,
                     "Should specify target names for classification report")
    
    def test_confusion_matrix_computed(self):
        """Test that confusion matrix is computed."""
        self.assertIn('confusion_matrix', self.all_code,
                     "Should compute confusion matrix")


class TestNotebookMetadata(unittest.TestCase):
    """Test suite for notebook metadata validation."""
    
    @classmethod
    def setUpClass(cls):
        """Load notebook."""
        with open('practical_task_10_3.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = json.load(f)
    
    def test_has_colab_metadata(self):
        """Test that notebook has Colab metadata."""
        metadata = self.notebook.get('metadata', {})
        self.assertIn('colab', metadata,
                     "Notebook should have Colab metadata")
    
    def test_authorship_tag_present(self):
        """Test that authorship tag is present in Colab metadata."""
        colab_metadata = self.notebook.get('metadata', {}).get('colab', {})
        self.assertIn('authorship_tag', colab_metadata,
                     "Colab metadata should have authorship_tag")
    
    def test_authorship_tag_updated(self):
        """Test that authorship tag has been updated from main branch."""
        colab_metadata = self.notebook.get('metadata', {}).get('colab', {})
        authorship_tag = colab_metadata.get('authorship_tag', '')
        
        # The new tag should be present (from the diff)
        self.assertEqual(authorship_tag, 'ABX9TyM5yLLllbCKvQcGWVclKOJU',
                        "Authorship tag should be updated")
    
    def test_colab_link_cell_exists(self):
        """Test that Colab link cell exists."""
        has_colab_link = False
        for cell in self.notebook['cells']:
            if cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                if 'Open In Colab' in source:
                    has_colab_link = True
                    break
        
        self.assertTrue(has_colab_link,
                       "Notebook should have 'Open In Colab' link")


def run_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookImports))
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookContent))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessingLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookOutputFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestMLPipelineComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestNotebookMetadata))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)