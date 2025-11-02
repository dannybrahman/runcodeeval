"""Collection settings and configuration"""

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CollectionSettings:
    """Settings for the solution collection process"""
    
    # Dataset settings
    categories: Optional[List[str]] = None  # None means all categories
    
    # Session settings
    session_name: Optional[str] = None
    
    # Execution settings
    batch_size: int = 10
    max_concurrent_per_model: int = 3
    max_total_concurrent: int = 20
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Progress settings
    progress_save_interval: int = 30  # seconds
    
    # Validation settings
    validate_syntax: bool = True
    validate_function_names: bool = True
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'CollectionSettings':
        """Create settings from environment variables"""
        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_concurrent_per_model=int(os.getenv("MAX_CONCURRENT_PER_MODEL", "3")),
            max_total_concurrent=int(os.getenv("MAX_TOTAL_CONCURRENT", "20")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE")
        )
    
    @classmethod
    def quick_test(cls) -> 'CollectionSettings':
        """Settings for quick testing with small batch"""
        return cls(
            batch_size=2,
            max_concurrent_per_model=1,
            max_total_concurrent=5
        )
    
    @classmethod
    def production(cls) -> 'CollectionSettings':
        """Settings optimized for production runs"""
        return cls(
            batch_size=20,
            max_concurrent_per_model=5,
            max_total_concurrent=30,
            progress_save_interval=60
        )
    
    def get_categories_list(self) -> Optional[List[str]]:
        """Get categories as a list, parsing from string if needed"""
        if self.categories is None:
            return None
        
        if isinstance(self.categories, str):
            # Parse comma-separated string
            return [cat.strip() for cat in self.categories.split(",") if cat.strip()]
        
        return self.categories
    
    def validate(self) -> List[str]:
        """Validate settings and return list of warnings/errors"""
        warnings = []
        
        if self.batch_size < 1:
            warnings.append("batch_size must be at least 1")
        
        if self.max_concurrent_per_model < 1:
            warnings.append("max_concurrent_per_model must be at least 1")
        
        if self.max_total_concurrent < self.max_concurrent_per_model:
            warnings.append("max_total_concurrent should be >= max_concurrent_per_model")
        
        if self.max_retries < 0:
            warnings.append("max_retries must be non-negative")
        
        if self.retry_delay < 0:
            warnings.append("retry_delay must be non-negative")
        
        if self.progress_save_interval < 10:
            warnings.append("progress_save_interval should be at least 10 seconds")
        
        return warnings