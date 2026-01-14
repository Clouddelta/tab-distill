from imodelsx import submit_utils
from os.path import dirname, join, expanduser
import os.path
from src.config import path_to_repo

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'output-dir': ['/home/chansingh/mntv1/jingyun/results/interaction_01_10_2025'],
    'index-type': ["fbii", "fsii", "stii", "bii", "sii", "fourier", "mobius"],
    'task-id': [
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
    ],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
# specify amlt resources
amlt_kwargs = {
    'amlt_file': join(path_to_repo, 'scripts', 'launch.yaml'),
    
    
    # 'sku': '40G2-A100',
    # 'sku': '40G1-A100',
    # 'sku': 'G2-A100',
    # 'target___name': 'msroctovc',

    'sku': '40G1-A100',
    'target___name': 'palisades26',

    # 'sku': '10C3', # 4 cpus
    # 'target___name': 'msrresrchvc',
    
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),

    'env': {
        'HF_TOKEN': f'{open(expanduser("~/.HF_TOKEN"), "r").read().strip()}',
        'TABPFN_DISABLE_TELEMETRY': '1',
        # 'PYTHONDONTWRITEBYTECODE': '1',
        # 'PYTHONPYCACHEPREFIX': '/tmp/aiscuser/pycache',
        # 'XDG_CACHE_HOME': '/tmp/aiscuser/xdg-cache',
    },
}
submit_utils.run_args_list(
    # args_list[1:4],
    args_list,
    script_name=join(path_to_repo, 'experiments', '00_run_tabarena.py'),
    # actually_run=False,

    # by default loops over jobs in serial
    # n_cpus=64,  # Uncomment to parallelize over cpus
    # gpu_ids=[0, 1, 2, 3],  # Uncomment to run individual jobs over each gpu
    # gpu_ids=[0],  # Uncomment to run all jobs on a single gpu
    # gpu_ids=[[0, 1], [2, 3]], # Uncomment to run jobs on [0, 1] and [2, 3] gpus respectively
    # gpu_ids=[[0, 1, 2, 3]],  # Run job on all gpus together

    # uncomment this to run jobs on cluster (need to run this script from the scripts directory)
    # amlt_kwargs=amlt_kwargs,
    cmd_python='pwd; .venv/bin/python',
)
