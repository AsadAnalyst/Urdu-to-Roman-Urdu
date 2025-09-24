import os
import json
import argparse
import shutil
from typing import Dict, List
import itertools

from train import Trainer
from evaluate import Evaluator
from utils import log_experiment


class ExperimentRunner:
    """Class to run multiple experiments with different hyperparameters"""
    
    def __init__(self, base_config_path: str = "config.json"):
        # Load base configuration
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = json.load(f)
        
        # Create experiments directory
        self.experiments_dir = "experiments"
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        print(f"Experiment runner initialized")
        print(f"Base config loaded from: {base_config_path}")
        print(f"Experiments will be saved to: {self.experiments_dir}")
    
    def create_experiment_config(self, experiment_name: str, modifications: Dict) -> str:
        """Create a modified config file for an experiment"""
        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Copy base config and apply modifications
        experiment_config = self.base_config.copy()
        
        # Apply modifications recursively
        for key_path, value in modifications.items():
            keys = key_path.split('.')
            current = experiment_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        
        # Update paths to be experiment-specific
        experiment_config['paths']['models_dir'] = os.path.join(exp_dir, "models")
        experiment_config['paths']['logs_dir'] = os.path.join(exp_dir, "logs")
        
        # Create directories
        os.makedirs(experiment_config['paths']['models_dir'], exist_ok=True)
        os.makedirs(experiment_config['paths']['logs_dir'], exist_ok=True)
        
        # Save experiment config
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)
        
        return config_path
    
    def run_single_experiment(self, experiment_name: str, modifications: Dict) -> Dict:
        """Run a single experiment with given modifications"""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"Modifications: {modifications}")
        print(f"{'='*60}")
        
        # Create experiment config
        config_path = self.create_experiment_config(experiment_name, modifications)
        
        try:
            # Initialize trainer with experiment config
            trainer = Trainer(config_path)
            
            # Run training
            training_results = trainer.train()
            
            # Generate sample translations
            trainer.generate_sample_translations()
            
            # Run evaluation if model was trained
            exp_dir = os.path.join(self.experiments_dir, experiment_name)
            model_path = os.path.join(exp_dir, "models", "best_model.pt")
            
            if os.path.exists(model_path):
                print(f"\nRunning evaluation for {experiment_name}...")
                evaluator = Evaluator(model_path, config_path)
                
                # Evaluate on test set (limited samples for speed)
                eval_results = evaluator.run_complete_evaluation(
                    output_dir=os.path.join(exp_dir, "evaluation"),
                    max_samples=1000  # Limit for faster evaluation
                )
                
                # Combine training and evaluation results
                combined_results = {
                    'experiment_name': experiment_name,
                    'modifications': modifications,
                    'training': training_results,
                    'evaluation': eval_results
                }
            else:
                combined_results = {
                    'experiment_name': experiment_name,
                    'modifications': modifications,
                    'training': training_results,
                    'evaluation': None,
                    'error': 'Model file not found'
                }
            
            # Save experiment results
            results_path = os.path.join(exp_dir, "experiment_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nExperiment {experiment_name} completed successfully!")
            print(f"Results saved to: {results_path}")
            
            return combined_results
            
        except Exception as e:
            error_results = {
                'experiment_name': experiment_name,
                'modifications': modifications,
                'error': str(e),
                'status': 'failed'
            }
            
            # Save error results
            exp_dir = os.path.join(self.experiments_dir, experiment_name)
            error_path = os.path.join(exp_dir, "error_log.json")
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_results, f, indent=2, ensure_ascii=False)
            
            print(f"\nExperiment {experiment_name} failed with error: {e}")
            return error_results
    
    def run_predefined_experiments(self) -> List[Dict]:
        """Run the three predefined experiments from the requirements"""
        print("Running predefined experiments...")
        
        # Experiment 1: Embedding Dimension Variations
        exp1_modifications = [
            {'model.embedding_dim': 128},
            {'model.embedding_dim': 256},
            {'model.embedding_dim': 512}
        ]
        
        # Experiment 2: Dropout Rate Variations
        exp2_modifications = [
            {'model.dropout': 0.1},
            {'model.dropout': 0.3},
            {'model.dropout': 0.5}
        ]
        
        # Experiment 3: Hidden Size Variations
        exp3_modifications = [
            {'model.hidden_size': 256},
            {'model.hidden_size': 512},
            {'model.hidden_size': 768}
        ]
        
        all_experiments = [
            ("exp1_embedding_128", exp1_modifications[0]),
            ("exp1_embedding_256", exp1_modifications[1]),
            ("exp1_embedding_512", exp1_modifications[2]),
            ("exp2_dropout_01", exp2_modifications[0]),
            ("exp2_dropout_03", exp2_modifications[1]),
            ("exp2_dropout_05", exp2_modifications[2]),
            ("exp3_hidden_256", exp3_modifications[0]),
            ("exp3_hidden_512", exp3_modifications[1]),
            ("exp3_hidden_768", exp3_modifications[2])
        ]
        
        # Run experiments
        results = []
        for exp_name, modifications in all_experiments:
            result = self.run_single_experiment(exp_name, modifications)
            results.append(result)
        
        return results
    
    def generate_comparison_report(self, results: List[Dict]) -> str:
        """Generate a comparison report of all experiments"""
        print("\nGenerating comparison report...")
        
        report_lines = []
        report_lines.append("# Urdu to Roman Urdu Transliteration Experiments Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary table
        report_lines.append("## Experiment Summary")
        report_lines.append("")
        report_lines.append("| Experiment | Status | BLEU | Perplexity | CER (%) | Exact Match (%) |")
        report_lines.append("|------------|--------|------|------------|---------|----------------|")
        
        successful_results = []
        
        for result in results:
            exp_name = result.get('experiment_name', 'Unknown')
            
            if 'error' in result:
                report_lines.append(f"| {exp_name} | Failed | - | - | - | - |")
            elif result.get('evaluation'):
                eval_data = result['evaluation']
                bleu = eval_data.get('bleu', 0.0)
                perplexity = eval_data.get('perplexity', 0.0)
                cer = eval_data.get('character_error_rate', 0.0)
                exact_match = eval_data.get('exact_match_accuracy', 0.0)
                
                report_lines.append(f"| {exp_name} | Success | {bleu:.2f} | {perplexity:.2f} | {cer:.2f} | {exact_match:.2f} |")
                successful_results.append(result)
            else:
                report_lines.append(f"| {exp_name} | Training Only | - | - | - | - |")
        
        report_lines.append("")
        
        # Detailed results for successful experiments
        if successful_results:
            report_lines.append("## Detailed Results")
            report_lines.append("")
            
            for result in successful_results:
                exp_name = result['experiment_name']
                modifications = result['modifications']
                training_data = result.get('training', {})
                eval_data = result.get('evaluation', {})
                
                report_lines.append(f"### {exp_name}")
                report_lines.append("")
                report_lines.append(f"**Modifications:** {modifications}")
                report_lines.append("")
                
                if training_data:
                    report_lines.append("**Training Results:**")
                    report_lines.append(f"- Final Train Loss: {training_data.get('final_train_loss', 'N/A'):.4f}")
                    report_lines.append(f"- Final Val Loss: {training_data.get('final_val_loss', 'N/A'):.4f}")
                    report_lines.append(f"- Best Val Loss: {training_data.get('best_val_loss', 'N/A'):.4f}")
                    report_lines.append(f"- Epochs Trained: {training_data.get('epochs_trained', 'N/A')}")
                    report_lines.append("")
                
                if eval_data:
                    report_lines.append("**Evaluation Results:**")
                    report_lines.append(f"- BLEU Score: {eval_data.get('bleu', 0.0):.2f}")
                    report_lines.append(f"- BLEU-1: {eval_data.get('bleu_1', 0.0):.2f}")
                    report_lines.append(f"- BLEU-2: {eval_data.get('bleu_2', 0.0):.2f}")
                    report_lines.append(f"- BLEU-3: {eval_data.get('bleu_3', 0.0):.2f}")
                    report_lines.append(f"- BLEU-4: {eval_data.get('bleu_4', 0.0):.2f}")
                    report_lines.append(f"- Perplexity: {eval_data.get('perplexity', 0.0):.2f}")
                    report_lines.append(f"- Character Error Rate: {eval_data.get('character_error_rate', 0.0):.2f}%")
                    report_lines.append(f"- Exact Match Accuracy: {eval_data.get('exact_match_accuracy', 0.0):.2f}%")
                    report_lines.append(f"- Average Edit Distance: {eval_data.get('average_edit_distance', 0.0):.2f}")
                    report_lines.append("")
        
        # Best performing model
        if successful_results:
            best_bleu = max(successful_results, key=lambda x: x.get('evaluation', {}).get('bleu', 0))
            best_cer = min(successful_results, key=lambda x: x.get('evaluation', {}).get('character_error_rate', 100))
            
            report_lines.append("## Best Performing Models")
            report_lines.append("")
            report_lines.append(f"**Best BLEU Score:** {best_bleu['experiment_name']} "
                               f"(BLEU: {best_bleu.get('evaluation', {}).get('bleu', 0):.2f})")
            report_lines.append(f"**Best Character Error Rate:** {best_cer['experiment_name']} "
                               f"(CER: {best_cer.get('evaluation', {}).get('character_error_rate', 100):.2f}%)")
            report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(self.experiments_dir, "experiments_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save as JSON
        summary_data = {
            'experiments': results,
            'summary': {
                'total_experiments': len(results),
                'successful_experiments': len(successful_results),
                'failed_experiments': len(results) - len(successful_results)
            }
        }
        
        json_path = os.path.join(self.experiments_dir, "experiments_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comparison report saved to: {report_path}")
        print(f"Summary data saved to: {json_path}")
        
        return report_content


def main():
    parser = argparse.ArgumentParser(description="Run Urdu to Roman Urdu Experiments")
    parser.add_argument("--config", default="config.json", help="Path to base config file")
    parser.add_argument("--run_predefined", action="store_true", 
                       help="Run the three predefined experiments")
    parser.add_argument("--experiment_name", help="Name of a single experiment to run")
    parser.add_argument("--modifications", help="JSON string of modifications for single experiment")
    
    args = parser.parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.config)
    
    try:
        if args.run_predefined:
            # Run all predefined experiments
            print("Running all predefined experiments...")
            results = runner.run_predefined_experiments()
            
            # Generate comparison report
            runner.generate_comparison_report(results)
            
            print(f"\nAll experiments completed!")
            print(f"Results available in: {runner.experiments_dir}")
            
        elif args.experiment_name and args.modifications:
            # Run single experiment
            modifications = json.loads(args.modifications)
            result = runner.run_single_experiment(args.experiment_name, modifications)
            print(f"\nSingle experiment {args.experiment_name} completed!")
            
        else:
            print("Please specify either --run_predefined or provide --experiment_name and --modifications")
            parser.print_help()
    
    except Exception as e:
        print(f"Experiment runner failed with error: {e}")
        raise


if __name__ == "__main__":
    main()