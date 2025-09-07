#!/usr/bin/env python3
"""
Grid search script for GFN hyperparameter optimization.
This script trains GFN models with different parameter combinations and evaluates their performance.
"""

import os
import subprocess
import json
import pandas as pd
import itertools
import time
import sys
from datetime import datetime
import argparse
import re
from pathlib import Path
from grid_search_config import BASE_CONFIG, FULL_PARAM_GRID, QUICK_PARAM_GRID, FOCUSED_PARAM_GRID

# Force unbuffered output for better logging
import builtins

def flush_print(*args, **kwargs):
    """Print with forced flush for better logging in nohup."""
    builtins.print(*args, **kwargs)
    sys.stdout.flush()

# Try to configure line buffering (Python 3.7+)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    # Fallback for older Python versions
    pass

# Override print function
print = flush_print


class GridSearchRunner:
    def __init__(self, base_config, param_grid, cuda_device=0, force_rerun=False):
        """
        Initialize grid search runner.
        
        Args:
            base_config: Base configuration dictionary
            param_grid: Dictionary of parameter ranges to search
            cuda_device: CUDA device to use
            force_rerun: If True, rerun even if results exist
        """
        self.base_config = base_config
        self.param_grid = param_grid
        self.cuda_device = cuda_device
        self.force_rerun = force_rerun
        self.results = []
        self.results_file = f"grid_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
    def generate_param_combinations(self):
        """Generate all parameter combinations from grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_combinations.append(param_dict)
            
        return param_combinations
    
    def check_training_exists(self, params):
        """Check if training results already exist."""
        log_dir = self.get_log_dir(params)
        pool_file = os.path.join(log_dir, f'pool_{self.base_config["n_episodes"]-1}.json')
        
        if os.path.exists(log_dir) and os.path.exists(pool_file):
            print(f"Training results already exist at: {log_dir}")
            print(f"Pool file found: {pool_file}")
            return True
        return False
    
    def check_combination_completed(self, params):
        """Check if this parameter combination has been completed and saved in results."""
        # Check if we have a previous results file that contains this combination
        import glob
        
        # Look for existing result files
        result_files = glob.glob("grid_search_results_*.csv")
        if not result_files:
            return False
            
        for result_file in result_files:
            try:
                df = pd.read_csv(result_file)
                # Check if this exact parameter combination exists
                match_conditions = []
                for key, value in params.items():
                    if key in df.columns:
                        match_conditions.append(df[key] == value)
                
                if len(match_conditions) > 0:  # 明确检查长度
                    matching_rows = df[pd.concat(match_conditions, axis=1).all(axis=1)]
                    if len(matching_rows) > 0:
                        print(f"Parameter combination already completed in {result_file}")
                        return True
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
                continue
        
        return False
    
    def run_training(self, params):
        """Run training with given parameters."""
        # Check if results already exist (unless force_rerun is True)
        if not self.force_rerun and self.check_training_exists(params):
            print("Skipping training - results already exist")
            return True
            
        # Merge base config with current params
        config = {**self.base_config, **params}
        
        # Build command
        cmd = [
            'pdm', 'run', 'python', 'train_gfn.py',
            '--seed', str(config['seed']),
            '--instrument', config['instrument'],
            '--pool_capacity', str(config['pool_capacity']),
            '--log_freq', str(config['log_freq']),
            '--update_freq', str(config['update_freq']),
            '--n_episodes', str(config['n_episodes']),
            '--encoder_type', config['encoder_type'],
            '--entropy_coef', str(config['entropy_coef']),
            '--entropy_temperature', str(config['entropy_temperature']),
            '--mask_dropout_prob', str(config['mask_dropout_prob']),
            '--ssl_weight', str(config['ssl_weight']),
            '--nov_weight', str(config['nov_weight']),
            '--weight_decay_type', config['weight_decay_type'],
            '--final_weight_ratio', str(config['final_weight_ratio'])
        ]
        
        # Set CUDA device
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device)
        
        print(f"Running training with params: {params}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run training
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            if result.returncode != 0:
                print(f"Training failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return None
            print("Training completed successfully")
            return True
        except subprocess.TimeoutExpired:
            print("Training timed out after 2 hours")
            return None
        except Exception as e:
            print(f"Training failed with exception: {e}")
            return None
    
    def get_log_dir(self, params):
        """Get the log directory for given parameters."""
        config = {**self.base_config, **params}
        log_dir = os.path.join(
            'data/gfn_logs',
            f'pool_{config["pool_capacity"]}',
            f'gfn_{config["encoder_type"]}_{config["instrument"]}_{config["pool_capacity"]}_{config["seed"]}-{config["entropy_coef"]}-{config["entropy_temperature"]}-{config["mask_dropout_prob"]}-{config["ssl_weight"]}-{config["nov_weight"]}-{config["weight_decay_type"]}-{config["final_weight_ratio"]}'
        )
        return log_dir
    
    def run_evaluation(self, params):
        """Run evaluation with given parameters."""
        log_dir = self.get_log_dir(params)
        pool_file = os.path.join(log_dir, f'pool_{self.base_config["n_episodes"]-1}.json')
        
        if not os.path.exists(pool_file):
            print(f"Pool file not found: {pool_file}")
            return None
            
        # Build evaluation command
        cmd = [
            'pdm', 'run', 'python', 'run_adaptive_combination.py',
            '--expressions_file', pool_file,
            '--chunk_size', '400',
            '--window', 'inf',
            '--cuda', str(self.cuda_device),
            '--n_factors', '20',
            '--train_end_year', '2020',
            '--seed', str(self.base_config['seed'])
        ]
        
        print(f"Running evaluation with pool file: {pool_file}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 30 min timeout
            if result.returncode != 0:
                print(f"Evaluation failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return None
                
            # Parse results from stdout
            return self.parse_evaluation_results(result.stdout)
        except subprocess.TimeoutExpired:
            print("Evaluation timed out after 30 minutes")
            return None
        except Exception as e:
            print(f"Evaluation failed with exception: {e}")
            return None
    
    def parse_evaluation_results(self, stdout):
        """Parse evaluation results from stdout."""
        try:
            # Look for the final performance metrics table
            lines = stdout.split('\n')
            
            # Find the section with performance metrics
            test_line = None
            validation_line = None
            in_metrics_section = False
            in_parseable_section = False
            
            for line in lines:
                line = line.strip()
                if "Final Performance Metrics" in line:
                    in_metrics_section = True
                    continue
                elif "Parseable Format" in line:
                    in_parseable_section = True
                    continue
                
                # Prefer parseable format if available
                if in_parseable_section:
                    if line.startswith('Test'):
                        test_line = line
                        break
                elif in_metrics_section:
                    if line.startswith('Validation'):
                        validation_line = line
                    elif line.startswith('Test'):
                        test_line = line
                        # Don't break here in case there's a parseable section later
            
            if test_line is None:
                print("Could not find Test results in output")
                print("Available lines:")
                for i, line in enumerate(lines):
                    if "Test" in line or "Validation" in line or "Final" in line:
                        print(f"Line {i}: {line}")
                return None
            
            # Parse the test line - handle potential formatting issues
            # Remove any ellipsis or special characters
            test_line_clean = test_line.replace('...', '').replace('[2 rows x 9 columns]', '')
            parts = test_line_clean.split()
            
            # Filter out non-numeric parts except the first one (which should be "Test")
            numeric_parts = []
            for i, part in enumerate(parts):
                if i == 0:  # Skip "Test" label
                    continue
                try:
                    float(part)
                    numeric_parts.append(part)
                except ValueError:
                    continue
            
            if len(numeric_parts) < 9:
                print(f"Not enough numeric values found in test line: {test_line}")
                print(f"Numeric parts found: {numeric_parts}")
                
                # Try alternative parsing - look for individual metric values in the output
                return self.parse_alternative_format(stdout)
            
            results = {
                'ic': float(numeric_parts[0]),
                'ic_std': float(numeric_parts[1]), 
                'icir': float(numeric_parts[2]),
                'ric': float(numeric_parts[3]),
                'ric_std': float(numeric_parts[4]),
                'ricir': float(numeric_parts[5]),
                'ret': float(numeric_parts[6]),
                'ret_std': float(numeric_parts[7]),
                'retir': float(numeric_parts[8])
            }
            
            print(f"Parsed results: {results}")
            return results
            
        except Exception as e:
            print(f"Error parsing evaluation results: {e}")
            print(f"stdout was: {stdout}")
            return None
    
    def parse_alternative_format(self, stdout):
        """Alternative parsing method for when standard parsing fails."""
        try:
            import re
            
            # Look for patterns like "Test        0.0659  0.1637  0.4028  0.0841"
            # but handle multi-line format
            lines = stdout.split('\n')
            
            # Find all lines that contain numeric data after "Test"
            test_data = []
            for line in lines:
                if 'Test' in line and not 'Pre-calculating' in line:
                    # Extract all floating point numbers from this line
                    numbers = re.findall(r'-?\d+\.\d+', line)
                    test_data.extend(numbers)
            
            if len(test_data) >= 9:
                results = {
                    'ic': float(test_data[0]),
                    'ic_std': float(test_data[1]), 
                    'icir': float(test_data[2]),
                    'ric': float(test_data[3]),
                    'ric_std': float(test_data[4]),
                    'ricir': float(test_data[5]),
                    'ret': float(test_data[6]),
                    'ret_std': float(test_data[7]),
                    'retir': float(test_data[8])
                }
                print(f"Alternative parsing successful: {results}")
                return results
            else:
                print(f"Alternative parsing failed - not enough numbers: {test_data}")
                return None
                
        except Exception as e:
            print(f"Alternative parsing failed: {e}")
            return None
    
    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save")
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"Results saved to {self.results_file}")
        
        # Also save a summary sorted by performance
        print("\n=== TOP RESULTS BY RICIR ===")
        top_results = df.nlargest(5, 'ricir')[['entropy_coef', 'entropy_temperature', 'mask_dropout_prob', 'ssl_weight', 'nov_weight', 'ic', 'icir', 'ric', 'ricir']]
        print(top_results.to_string(index=False))
        
        print("\n=== TOP RESULTS BY IC ===")
        top_results = df.nlargest(5, 'ic')[['entropy_coef', 'entropy_temperature', 'mask_dropout_prob', 'ssl_weight', 'nov_weight', 'ic', 'icir', 'ric', 'ricir']]
        print(top_results.to_string(index=False))
    
    def run_grid_search(self):
        """Run the complete grid search."""
        param_combinations = self.generate_param_combinations()
        total_combinations = len(param_combinations)
        
        print(f"Starting grid search with {total_combinations} parameter combinations")
        print(f"Results will be saved to: {self.results_file}")
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\n{'='*60}")
            print(f"Running combination {i}/{total_combinations}")
            print(f"Parameters: {params}")
            print(f"{'='*60}")
            
            # Check if this combination is already completed (unless force_rerun is True)
            if not self.force_rerun and self.check_combination_completed(params):
                print("Skipping combination - already completed in previous run")
                continue
            
            start_time = time.time()
            
            # Run training
            train_success = self.run_training(params)
            if train_success is None:
                print("Training failed, skipping evaluation")
                continue
            
            # Run evaluation
            eval_results = self.run_evaluation(params)
            if eval_results is None:
                print("Evaluation failed, skipping this combination")
                continue
            
            # Store results
            result_record = {**params, **eval_results}
            result_record['combination_id'] = i
            result_record['duration_minutes'] = (time.time() - start_time) / 60
            self.results.append(result_record)
            
            print(f"Combination {i} completed in {result_record['duration_minutes']:.1f} minutes")
            print(f"Results: IC={eval_results['ic']:.4f}, ICIR={eval_results['icir']:.4f}, RIC={eval_results['ric']:.4f}, RICIR={eval_results['ricir']:.4f}")
            
            # Save intermediate results
            self.save_results()
        
        print(f"\nGrid search completed! Total combinations: {len(self.results)}")
        self.save_results()


def main():
    parser = argparse.ArgumentParser(description='Grid search for GFN hyperparameters')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device to use')
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer episodes')
    parser.add_argument('--focused', action='store_true', help='Run focused search with custom parameter grid')
    parser.add_argument('--force', action='store_true', help='Force rerun even if results exist')
    args = parser.parse_args()
    
    # Base configuration (from grid_search_config.py)
    base_config = BASE_CONFIG.copy()
    if args.quick:
        base_config['n_episodes'] = 1000  # Reduce for quick testing
    
    # Parameter grid to search
    if args.quick:
        param_grid = QUICK_PARAM_GRID
        print("Using QUICK parameter grid")
    elif args.focused:
        param_grid = FOCUSED_PARAM_GRID
        print("Using FOCUSED parameter grid")
    else:
        param_grid = FULL_PARAM_GRID
        print("Using FULL parameter grid")
    
    print("Grid Search Configuration:")
    print(f"Base config: {base_config}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {len(list(itertools.product(*param_grid.values())))}")
    
    # Create and run grid search
    runner = GridSearchRunner(base_config, param_grid, args.cuda, args.force)
    runner.run_grid_search()


if __name__ == '__main__':
    main()
