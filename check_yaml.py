import os
import yaml
from pathlib import Path

def check_yaml_paths(yaml_path):
    # Check if the YAML file itself
    if not os.path.exists(yaml_path):
        print(f"ERROR: YAML file not found at {yaml_path}")
        return None
    
    # Load YAML file
    try:
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            print(f"Successfully loaded YAML file: {yaml_path}")
    except Exception as e:
        print(f"ERROR: Failed to load YAML file: {e}")
        return None
    
    base_dir = os.path.dirname(os.path.abspath(yaml_path))

    results = {
        "yaml_file": yaml_path,
        "exists": True,
        "paths": {}
    }
    
    path_keys = ['train', 'val', 'test']
    
    # Special handling for YOLO dataset format
    if isinstance(data.get('train'), str):
        # Direct path format
        for key in path_keys:
            if key in data:
                # Handle both absolute and relative paths
                path = data[key]
                abs_path = path if os.path.isabs(path) else os.path.join(base_dir, path)
                
                results['paths'][key] = {
                    'specified_path': path,
                    'absolute_path': abs_path,
                    'exists': os.path.exists(abs_path)
                }
    
    # Check for the more structured format with paths under 'path'
    if 'path' in data:
        results['paths']['main_path'] = {
            'specified_path': data['path'],
            'absolute_path': os.path.abspath(data['path']),
            'exists': os.path.exists(data['path'])
        }
    
    # Check for train/val/test image and label directories
    for key in path_keys:
        # Check for images
        img_path = os.path.join(base_dir, 'images', key) if 'path' in data else None
        if img_path:
            results['paths'][f'{key}_images'] = {
                'specified_path': f"images/{key}",
                'absolute_path': img_path,
                'exists': os.path.exists(img_path)
            }
        
        # Check specific paths structure for YOLO
        yolo_img_path = os.path.join(base_dir, key, 'images')
        results['paths'][f'{key}_yolo_images'] = {
            'specified_path': f"{key}/images",
            'absolute_path': yolo_img_path,
            'exists': os.path.exists(yolo_img_path)
        }
        
        # Check for labels
        yolo_labels_path = os.path.join(base_dir, key, 'labels')
        results['paths'][f'{key}_yolo_labels'] = {
            'specified_path': f"{key}/labels",
            'absolute_path': yolo_labels_path,
            'exists': os.path.exists(yolo_labels_path)
        }
    
    # Print summary of path checks
    print("\nPath Check Results:")
    print("-" * 80)
    
    for path_key, path_info in results['paths'].items():
        status = "EXISTS" if path_info['exists'] else "MISSING"
        print(f"{path_key.ljust(20)}: {status}")
        print(f"  Absolute Path: {path_info['absolute_path']}")
    
    print("\nMissing Paths (need to be created):")
    missing_paths = [info['absolute_path'] for info in results['paths'].values() if not info['exists']]
    
    if missing_paths:
        for path in missing_paths:
            print(f"- {path}")
    else:
        print("All paths exist!")
    
    return results

if __name__ == "__main__":
    #Path to change
    yaml_file = r"C:\Users\Himanshu Gupta\Desktop\PROJECT\datasets\MIO-TCD-Localization\YOLO_dataset\dataset.yaml"

    # Run the check
    results = check_yaml_paths(yaml_file)
    
    # Offer to create missing directories
    if results:
        missing_paths = [info['absolute_path'] for info in results['paths'].values() if not info['exists']]
        
        if missing_paths:
            create_dirs = input("\nWould you like to create the missing directories? (y/n): ").lower()
            
            if create_dirs == 'y':
                for path in missing_paths:
                    try:
                        os.makedirs(path, exist_ok=True)
                        print(f"Created directory: {path}")
                    except Exception as e:
                        print(f"Failed to create {path}: {e}")
                
                # Verify creation
                print("\nVerifying created directories:")
                for path in missing_paths:
                    status = "Created" if os.path.exists(path) else "Still missing"
                    print(f"{path}: {status}")