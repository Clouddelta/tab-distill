
### Installation

Setup using uv (requires [installing uv](https://docs.astral.sh/uv/getting-started/installation/) then run a script using `uv run <script>`).
- Note: relies heavily on and makes small modifications to the [spex](https://github.com/basics-lab/spectral-explain) library

### Organization
- source code with useful importable functions is in `src` folder
- main experiments to run are under `experiments` folder
- analysis scripts are in the `notebooks` folder
- hyperparameter loops are in the `scripts` folder`

### Dataset
Source: TabArena benchmark 
Selection Rule: N < 10000, p < 10
Total: 10 datasets
Task types: Regression & Classification

| Task ID | Dataset Name                          | Task Type                 | Problem Description                            |
| ------: | ------------------------------------- | ------------------------- | ---------------------------------------------- |
|  363621 | blood-transfusion-service-center      | Binary Classification     | Predict whether a blood donor will return      |
|  363629 | diabetes                              | Binary Classification     | Predict diabetes onset                         |
|  363698 | QSAR_fish_toxicity                    | Regression                | Predict chemical toxicity to fish              |
|  363685 | maternal_health_risk                  | Multiclass Classification | Predict maternal health risk level             |
|  363625 | concrete_compressive_strength         | Regression                | Predict concrete compressive strength          |
|  363671 | Fitness_Club                          | Binary Classification     | Predict customer churn / subscription behavior |
|  363612 | airfoil_self_noise                    | Regression                | Predict airfoil noise level                    |
|  363615 | Another-Dataset-on-used-Fiat-500      | Regression                | Predict used car prices                        |
|  363674 | hazelnut-spread-contaminant-detection | Binary Classification     | Detect food contamination                      |
|  363700 | seismic-bumps                         | Binary Classification     | Predict seismic event bumps                    |

  
Talent Datasets, N < 10000, p < 10, regression task: 10 datasets in total.

### Usage

Produce SPEX interaction search results for Talent Benchmark datasets, which will be saved as `interactions_summary_{task_name}_{index}.pkl` in `talent_interaction_result` folder: `uv run talent_batch_tasks_mulindex.py`

[Jingyun: may need to integrate this function with the next line] Compute SPEX interaction search results for **NEW** TabArena Benchmark datasets, which will be saved as `interactions_summary_{task_id}_{index}.pkl` in `my_results` folder: `uv run_batch_tasks_mulindex.py 363671 --num-samples 0 --output-dir my_results`

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




