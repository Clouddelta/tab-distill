
### Installation

Setup using uv (requires [installing uv](https://docs.astral.sh/uv/getting-started/installation/) then run a script using `uv run <script>`).
- Note: relies heavily on and makes small modifications to the [spex](https://github.com/basics-lab/spectral-explain) library

### Organization
- source code with useful importable functions is in `src` folder
- main experiments to run are under `experiments` folder
- analysis scripts are in the `notebooks` folder

### Dataset
Talent Datasets, N < 10000, p < 10, regression task: 10 datasets in total.

### Usage

Produce SPEX interaction search results for Talent Benchmark datasets, which will be saved as `interactions_summary_{task_name}_{index}.pkl` in `talent_interaction_result` folder: `uv run talent_batch_tasks_mulindex.py`

Compute SPEX interaction search results, which will be saved as `interactions_summary_{task_id}_{index}.pkl` in `interaction_12_14_2025` folder: `uv run run_batch_tasks_mulindex.py`

Then compare the performance of different index choices. Use `compare_index_performance.py` for all tasks in `interaction_12_14_2025` folder, e.g.: `uv run compare_index_performance.py 363698 10 --max-interaction-order 4 --output comparison.png --no-show`


## Parameters

- `--output` or `-o`: Output path for saving model comparison plot
- `N_interactions`: Number of interactions to use
- `--max-interaction-order`: Interaction order (2, 3, or 4)
  - `2`: 2-way interactions only
  - `3`: 2-way + 3-way interactions
  - `4`: 2-way + 3-way + 4-way interactions
  - Default: `2`
- `--interaction-dir`: Directory containing interaction pickle files (default: `./interaction_12_14_2025`)
- `--no-show`: Do not display matplotlib windows (useful when saving plots)
- `--indices`: Specify indices to compare (e.g., `--indices bii fbii sii`). Default: all available indices

