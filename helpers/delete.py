import os
from pathlib import Path
import re

class RemoveImages:
    to_delete = [
    765, 794, 1042, 1113, 77, 99, 100, 103, 267, 269, 270, 271, 290,
    292, 293, 294, 317, 318, 319, 337, 340, 361, 364, 385, 386, 389,
    411, 412, 416, 417, 438, 439, 443, 473, 474, 478, 482, 484, 507,
    508, 509, 510, 515, 544, 547, 552, 554, 577, 583, 584, 630, 641, 
    672, 675,  702, 703, 729, 751, 862, 9, 64, 79, 101, 65, 106, 124, 
    1180, 1181, 1182, 1183, 1184, 465, 535, 536, 568, 572, 603, 668, 31,
    726, 68, 98, 118, 136, 139, 144, 158, 159, 160, 161, 166, 175, 177,
    178, 179, 180, 198, 199, 200, 201, 202, 203, 219, 220, 221, 222, 223, 
    224, 244, 246, 247, 248, 249, 1188, 1233, 1270, 1271, 1272, 1273, 1274,
    1292, 1294, 1295, 1325, 1339, 1359, 1370, 1378, 1379, 1380
    , 1394, 1418, 1420, 1421, 1423, 1427, 1428, 1430, 1431, 1432, 1439,1442,
    1443, 1445, 1449, 1454, 1458, 1464, 1466, 1468, 1470, 1471, 1472, 1473,
    1474, 1477, 1478, 1480, 1481, 1482, 1485, 1486, 381, 406, 409, 461, 530, 
    534, 569, 570, 571, 635, 775, 778, 1177, 1179, 1195, 1215, 1217, 1218,
    1237, 1256, 1275, 1276, 1281, 1299, 1300, 1318, 1320, 1330, 1331, 1341, 
    1342, 1349, 1351, 1360, 1371, 1384, 1386, 1398, 1411, 1412, 1416, 1417,
    1447, 966, 991, 1003, 1005, 1008, 1010, 1037, 1051, 1052, 1053, 1055, 
    1070, 1071, 1072, 1073, 1074, 1075, 1091, 1104, 1106, 1123, 1124, 1126, 
    1127, 1134, 1135, 1136, 1145, 1151, 1152, 1153, 1170, 128, 129, 131, 152, 
    239, 241, 265, 266, 313, 314, 335, 384, 410, 537, 538, 573, 574, 609, 671,
    700, 701, 728, 762, 773, 785, 798, 846, 847, 1187, 1252, 1255, 1359, 
    
    ]

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
        # Check if the file name contains any of the numbers to delete
        for num in self.to_delete:
            pattern = rf'(?<!\d){num}(?!\d)'
            if re.search(pattern, file.stem):
                return True
        
        return False

# Example usage:
# remover = RemoveImages("/path/to/image/directory")
# deleted = remover.execute()
# print(f"Deleted files: {deleted}")