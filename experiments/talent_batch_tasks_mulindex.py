"""
Batch script to run talent_single_mulindex.py for multiple Talent datasets
"""
import sys
import traceback
import time
from pathlib import Path

# Import the process_talent_dataset function
from talent_single_mulindex import process_talent_dataset


def get_all_talent_datasets(data_dir="Talent_data"):
    """
    Automatically discover all datasets in Talent_data folder.
    
    Args:
        data_dir: Root directory containing dataset folders
        
    Returns:
        list: List of dataset names (folder names that contain info.json)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist!")
        return []
    
    datasets = []
    for item in data_path.iterdir():
        if item.is_dir():
            # Check if it has info.json file
            if (item / "info.json").exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def run_batch_talent_datasets(
    dataset_names=None,
    num_samples=0,
    index_types=None,
    output_dir='talent_interaction_result',
    data_dir="Talent_data"
):
    """
    Run multiple Talent datasets in sequence.
    
    Args:
        dataset_names: List of dataset names to process. If None, process all datasets found in data_dir.
        num_samples: Number of training samples to process per dataset. If 0 or None, process all (capped at 1000).
        index_types: List of interaction index types to compute. If None, uses all default types.
        output_dir: Output directory for saved files (default: 'talent_interaction_result')
        data_dir: Root directory containing dataset folders (default: 'Talent_data')
    """
    # Auto-discover datasets if not provided
    if dataset_names is None:
        dataset_names = get_all_talent_datasets(data_dir)
        print(f"Auto-discovered {len(dataset_names)} datasets in {data_dir}")
    
    if not dataset_names:
        print("Error: No datasets found to process!")
        return {}
    
    results = {}
    start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(dataset_names)} Talent datasets")
    print(f"Number of samples per dataset: {num_samples if num_samples else 'ALL (capped at 1000)'}")
    print(f"Index types: {index_types if index_types else 'ALL (fbii, fsii, stii, bii, sii, fourier, mobius)'}")
    print(f"Output directory: {output_dir}")
    print(f"Data directory: {data_dir}")
    print(f"{'='*70}\n")
    
    for idx, dataset_name in enumerate(dataset_names, 1):
        print(f"\n{'='*70}")
        print(f"Dataset {idx}/{len(dataset_names)}: Processing {dataset_name}")
        print(f"{'='*70}")
        
        dataset_start = time.time()
        try:
            result_data, summary_data = process_talent_dataset(
                dataset_name=dataset_name,
                num_samples=num_samples,
                index_types=index_types,
                output_dir=output_dir,
                data_dir=data_dir
            )
            
            dataset_time = time.time() - dataset_start
            results[dataset_name] = {
                'success': True,
                'time': dataset_time,
                'result': result_data,
                'summary': summary_data
            }
            print(f"\n✓ Dataset {dataset_name} completed successfully in {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes)")
            
        except Exception as e:
            dataset_time = time.time() - dataset_start
            results[dataset_name] = {
                'success': False,
                'time': dataset_time,
                'error': str(e)
            }
            print(f"\n✗ Dataset {dataset_name} failed after {dataset_time:.2f} seconds ({dataset_time/60:.2f} minutes)")
            print(f"Error: {e}")
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets: {len(dataset_names)}")
    successful = sum(1 for r in results.values() if r['success'])
    failed = sum(1 for r in results.values() if not r['success'])
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if successful > 0:
        avg_time = sum(r['time'] for r in results.values() if r['success']) / successful
        print(f"Average time per successful dataset: {avg_time:.2f} seconds ({avg_time/60:.2f} minutes)")
    
    if failed > 0:
        print(f"\nFailed datasets:")
        for dataset_name, result in results.items():
            if not result['success']:
                print(f"  {dataset_name}: {result.get('error', 'Unknown error')}")
    
    if successful > 0:
        print(f"\nSuccessful datasets:")
        for dataset_name, result in results.items():
            if result['success']:
                print(f"  {dataset_name}: {result['time']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Default: process all datasets found in Talent_data
    # You can also specify specific datasets
    dataset_names = None  # Set to None to auto-discover all datasets
    
    # Alternatively, specify datasets explicitly:
    # dataset_names = [
    #     "abalone",
    #     "analcatdata_supreme",
    #     "chscase_foot",
    #     "combined_cycle_power_plant",
    #     "Data_Science_Salaries",
    #     "debutanizer",
    #     "delta_ailerons",
    #     "delta_elevators",
    #     "Goodreads-Computer-Books",
    #     "qsar_aquatic_toxicity",
    # ]
    
    # Allow dataset names to be passed as command line arguments
    if len(sys.argv) > 1:
        # If first argument is "all", use auto-discovery
        if sys.argv[1].lower() == "all":
            dataset_names = None
        else:
            dataset_names = sys.argv[1:]
    
    # Allow num_samples to be specified as argument (after "all" or dataset names)
    # Default 0 means "use all samples" (capped at 1000 in process_talent_dataset)
    # Parse num_samples: it should be the last argument if it's a number
    num_samples = 0
    if len(sys.argv) > 1:
        # Check if last argument is a number
        try:
            last_arg = sys.argv[-1]
            if last_arg.isdigit() or (last_arg.startswith('-') and last_arg[1:].isdigit()):
                num_samples = int(last_arg)
                # Remove it from dataset_names if it was parsed
                if dataset_names and len(dataset_names) > 0 and dataset_names[-1] == last_arg:
                    dataset_names = dataset_names[:-1]
        except (ValueError, IndexError):
            pass
    
    # Allow index_types to be specified
    # For simplicity, use None (all types) unless specified via environment or hardcoded
    index_types = None  # Uses all default types: ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"]
    # Or specify specific types:
    # index_types = ["fbii", "fsii"]
    
    # Default output dir for Talent multi-index run
    output_dir = 'talent_interaction_result'
    
    # Run batch processing
    run_batch_talent_datasets(
        dataset_names=dataset_names,
        num_samples=num_samples,
        index_types=index_types,
        output_dir=output_dir,
        data_dir="data/Talent_data"
    )

