"""
Batch script to run tabarena_single.py for multiple task IDs
"""
import sys
import traceback
import time
from pathlib import Path

# Import the process_task function
from src.tabarena_single_mulindex import process_task


def run_batch_tasks(task_ids, num_samples=0, output_dir='interaction_12_14_2025'):
    """
    Run multiple tasks in sequence
    
    Args:
        task_ids: List of OpenML task IDs to process
        num_samples: Number of training samples to process per task. If 0 or None, process all.
        output_dir: Output directory for saved files
    """
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
                output_dir=output_dir
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
    # Define your task IDs here
    task_ids = [
        359950,
        359956,
        359959,
        363242,
        363615,
        363625,
        363675,
        363698
    ]
    
    # Allow task IDs to be passed as command line arguments
    if len(sys.argv) > 1:
        task_ids = [int(tid) for tid in sys.argv[1:]]
    
    # Allow num_samples to be specified as second argument
    # Default 0 means "use all samples"
    num_samples = 0
    if len(sys.argv) > 2:
        try:
            num_samples = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid num_samples '{sys.argv[2]}', using default (0=ALL): {num_samples}")
    
    # Run batch processing
    # Default output dir for multi-index run
    output_dir = 'interaction_12_14_2025'
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Run batch processing
    run_batch_tasks(task_ids, num_samples=num_samples, output_dir=output_dir)

