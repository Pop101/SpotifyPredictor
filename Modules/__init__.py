import os
import hashlib

from Modules.data_preprocessor import fix_featureset
from Modules.derive_datasets import generate_all_datasets
from Modules.learning import perform_ml

# On module load: Generate Feature Importance 
# if the dataset has changed at all

def hash_matches_saved(hash):
    if not os.path.exists("Data/hash.txt"): return False
    return open("Data/hash.txt").read() == hash

def add_to_hash(paths:str or list, existing_hash=None):
    if not existing_hash:
        existing_hash = hashlib.new('md5')
    
    # Force path to be collection
    if isinstance(paths, str) or not (hasattr(paths, '__iter__') or hasattr(paths, '__next__')):
        paths = [paths]
    
    # Iterate over each path, adding to hash
    # no matter what it is
    for path in paths:
        if os.path.isfile(path):
            existing_hash = add_file_to_hash(path, existing_hash=existing_hash)
        elif os.path.isdir(path):
            existing_hash = add_folder_to_hash(path, existing_hash=existing_hash)
    return existing_hash

def add_file_to_hash(file_path:str, existing_hash=None):
    if not existing_hash:
        existing_hash = hashlib.new('md5')
    
    with open(file_path, 'rb') as file:
        existing_hash.update(file.read())
    print(file_path, existing_hash.hexdigest())
    
    return existing_hash

def add_folder_to_hash(root_path, existing_hash=None):
    if not existing_hash:
        existing_hash = hashlib.new('md5')
    
    # Recurse over directory structure
    # Pray that os.listdir returns the same order each time
    for path in os.listdir(root_path):
        add_to_hash(path)
        
    return existing_hash

# Only add non-derived datasets to hash. If they change,
# All derived datasets change.
# Also, if any modules changed, it's likely that some derived
# data changes as well. While this makes coding incredibly annoying,
# Re-derive
def calculate_hash():
    return add_to_hash([
        "Data/SpotifyFeatures.csv",
        "Modules/derive_datasets.py",
        "Modules/data_preprocessor.py",
        "Modules/learning.py"
    ]).hexdigest()

if not hash_matches_saved(calculate_hash()):
    print("Hash of data and modules differs from saved hash")
    
    # Preprocess all datasets
    print("Preprocessing")
    fix_featureset()
    
    # Perform machine learning
    print("Performing machine learning")
    perform_ml()
    
    # Generate all derived datasets
    print("Generating derived data")
    generate_all_datasets()
    
    # Recalculate and save the hash
    with open("Data/hash.txt", "w") as f:
        f.write(calculate_hash())
        