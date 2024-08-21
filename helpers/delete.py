import os
from pathlib import Path
import re

class RemoveImages:
    to_delete = [3, 4]
    
    def __init__(self, directory_path):
        self.directory_path = Path(directory_path)
        self.directory_path = self.directory_path.parent / 'satellite'
        print(f"Directory path: {self.directory_path}")
    
    
    def execute(self):
        deleted_files = []
        for file in self.directory_path.iterdir():
            if file.is_file() and self._is_image(file) and self._should_delete(file):
                file.unlink()
                deleted_files.append(file.name)
        
        return deleted_files
    
    def _is_image(self, file):
        image_extensions = ('.tiff', '.tif')
        return file.suffix.lower() in image_extensions
    
    def _should_delete(self, file):
        for num in self.to_delete:
            pattern = rf'(?<!\d){num}(?!\d)'
            if re.search(pattern, file.stem):
                return True
        return False

# Example usage:
# remover = RemoveImages("/path/to/image/directory")
# deleted = remover.execute()
# print(f"Deleted files: {deleted}")