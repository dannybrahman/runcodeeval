"""
JSONL file reader utility
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import logging


class JSONLReader:
    """Utility for reading JSONL files"""
    
    @staticmethod
    def read_file(file_path: Path) -> List[Dict[str, Any]]:
        """Read a JSONL file and return list of parsed objects"""
        logger = logging.getLogger(__name__)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        data = []
        line_num = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            data.append(obj)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at {file_path}:{line_num}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        logger.debug(f"Loaded {len(data)} objects from {file_path}")
        return data