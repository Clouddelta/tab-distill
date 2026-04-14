"""
Batch script to run tabarena_single_mulindex.py for multiple task IDs

Usage:
    # Use default task IDs with default settings (1 sample per task)
    python run_batch_tasks_mulindex.py
    
    # Process specific task IDs (positional arguments)
    python run_batch_tasks_mulindex.py 363698 363671 363625
    
    # Specify number of samples to process per task
    python run_batch_tasks_mulindex.py 363698 363671 --num-samples 2
    
    # Specify output directory
    python run_batch_tasks_mulindex.py 363698 --num-samples 2 --output-dir my_results
    
    # Process all default tasks with custom settings
    python run_batch_tasks_mulindex.py --num-samples 5 --output-dir my_results
"""
import sys
import traceback
import time
import argparse
from pathlib import Path

# Import the process_task function
from tabarena_single_mulindex import process_task

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / 'interaction_1_14_2026_500'


def run_batch_tasks(task_ids, num_samples=0, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Run multiple tasks in sequence
    
    Args:
        task_ids: List of OpenML task IDs to process
        num_samples: Number of training samples to process per task. If 0 or None, process all.
        output_dir: Output directory for saved files
    """
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (BASE_DIR / output_dir).resolve()
    results = {}
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(task_ids)} tasks")
    print(f"Number of samples per task: {num_samples if num_samples else 'ALL'}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    for idx, task_id in enumerate(task_ids, 1):
        print(f"\n{'='*70}")
        print(f"Task {idx}/{len(task_ids)}: Processing Task ID {task_id}")
        print(f"{'='*70}")
        
        task_start = time.time()
        try:
            result_data, summary_data = process_task(
                task_id=task_id,
                num_samples=num_samples,
                output_dir=str(output_dir)
            )
            
            task_time = time.time() - task_start
            results[task_id] = {
                'success': True,
                'time': task_time,
                'result': result_data,
                'summary': summary_data
            }
            print(f"\n✓ Task {task_id} completed successfully in {task_time:.2f} seconds")
            
        except Exception as e:
            task_time = time.time() - task_start
            results[task_id] = {
                'success': False,
                'time': task_time,
                'error': str(e)
            }
            print(f"\n✗ Task {task_id} failed after {task_time:.2f} seconds")
            print(f"Error: {e}")
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total tasks: {len(task_ids)}")
    successful = sum(1 for r in results.values() if r['success'])
    failed = sum(1 for r in results.values() if not r['success'])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if failed > 0:
        print(f"\nFailed tasks:")
        for task_id, result in results.items():
            if not result['success']:
                print(f"  Task {task_id}: {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    # Define your task IDs here (default list)
    TASK_IDS = [
        363621,  # blood-transfusion-service-center: binary classification, blood donation return prediction
        363629,  # diabetes: binary classification, diabetes onset prediction
        363698,  # QSAR_fish_toxicity: regression, chemical toxicity prediction
        363685,  # maternal_health_risk: multiclass classification, maternal health risk levels
        363625,  # concrete_compressive_strength: regression, concrete strength prediction
        363671,  # Fitness_Club: binary classification, customer churn / subscription behavior
        363612,  # airfoil_self_noise: regression, airfoil noise level prediction
        363615,  # Another-Dataset-on-used-Fiat-500: regression, used car price prediction
        363674,  # hazelnut-spread-contaminant-detection: binary classification, food contamination detection
        363700,  # seismic-bumps: binary classification, seismic event bump prediction
    ]
    
    # Parse command line arguments using argparse
    parser = argparse.ArgumentParser(
        description='Batch process TabArena tasks with TabPFN and spectral explain',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default task IDs with 1 sample per task
  python run_batch_tasks_mulindex.py
  
  # Process specific task IDs (positional arguments)
  python run_batch_tasks_mulindex.py 363698 363671 363625
  
  # Process with 5 samples per task (use default task IDs)
  python run_batch_tasks_mulindex.py --num-samples 5
  
  # Process specific tasks with custom settings
  python run_batch_tasks_mulindex.py 363698 363671 --num-samples 2 --output-dir my_results
  
  # Process all default tasks with all samples (may take very long!)
  python run_batch_tasks_mulindex.py --num-samples 0
        """
    )
    
    parser.add_argument(
        'task_ids',
        type=int,
        nargs='*',  # Zero or more positional arguments
        default=None,
        help='OpenML task IDs to process (positional arguments). If not provided, uses default list.'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=0,
        help='Number of training samples to process per task. Use 0 to process all samples. (default: 1)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Output directory for saved interaction results (default: ICML_last_minute/interaction_1_14_2026_500)'
    )
    
    args = parser.parse_args()
    
    # Determine which task IDs to use
    # If positional task_ids provided, use them; otherwise use default TASK_IDS
    task_ids = args.task_ids if args.task_ids else TASK_IDS
    num_samples = args.num_samples
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (BASE_DIR / output_dir).resolve()
    
    # Print configuration
    print(f"\n{'='*70}")
    print("BATCH TASK CONFIGURATION")
    print(f"{'='*70}")
    print(f"Task IDs: {task_ids}")
    print(f"Number of samples per task: {num_samples} ({'ALL samples' if num_samples == 0 else f'{num_samples} sample(s)'})")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Run batch processing
    run_batch_tasks(task_ids, num_samples=num_samples, output_dir=output_dir)

