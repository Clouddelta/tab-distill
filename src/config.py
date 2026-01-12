import os
import openml
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists('/home/chansingh'):
    cache_dir = '/home/chansingh/mntv1/jingyun/joblib_cache'
    cache_dir_openml = '/home/chansingh/mntv1/jingyun/openml_cache'
    # cache_dir_tabpfn = '~/.cache/tabpfn'
    cache_dir_tabpfn = '/home/chansingh/mntv1/jingyun/tabpfn_cache'
elif os.path.exists('/mntv1'):
    cache_dir = '/mntv1/jingyun/joblib_cache'
    cache_dir_openml = '/mntv1/jingyun/openml_cache'
    # cache_dir_tabpfn = '~/.cache/tabpfn'
    cache_dir_tabpfn = '/mntv1/jingyun/tabpfn_cache'
else:
    raise ValueError("Cannot determine cache directory!")
os.makedirs(cache_dir_openml, exist_ok=True)

if __name__ == "__main__":
    print(path_to_repo)