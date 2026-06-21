import os
import shutil
import re
import zipfile

def setup_dataset_windows():
    zip_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'archive.zip')
    extract_target = r'your_extracted_dataset_folder_path'
    
    print(f"Currently looking for archive at: {zip_path}")
    if not os.path.exists(zip_path):
        print("Error: Cannot find archive.zip in the default Windows Downloads folder!")
        return

    print(f"STEP 1: Extracting data...")
    os.makedirs(extract_target, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            try:
                member.filename = member.filename.encode('cp437').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass 
            
            target_path = os.path.join(extract_target, member.filename)
            
            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

    print("STEP 2: Cleaning up junk directories (character that are out of Chinese unicode range)...")
    base_dirs = [
        os.path.join(extract_target, 'CASIA-HWDB_Train', 'Train'),
        os.path.join(extract_target, 'CASIA-HWDB_Test', 'Test')
    ]
    
    hanzi_pattern = re.compile(r'[\u4e00-\u9fff]')
    deleted_count = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.isdir(folder_path) and not hanzi_pattern.search(folder_name):
                try:
                    shutil.rmtree(folder_path)
                    deleted_count += 1
                except Exception as e:
                    pass
                
    print(f"COMPLETE! Successfully extracted file and cleaned up {deleted_count} junk directories.")
    print(f"All clean data is now located at: {extract_target}")

if __name__ == "__main__":
    setup_dataset_windows()