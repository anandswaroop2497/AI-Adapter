"""
Base model handler interface that all model handlers must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseModelHandler(ABC):
    """
    Abstract base class for all model handlers.
    Each model handler should inherit from this class and implement required methods.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the model handler.
        
        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self.session = None
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """
        Return list of task names supported by this model.
        
        Returns:
            List of task name strings (e.g., ["person_detection", "person_counting"])
        """
        pass
    
    @abstractmethod
    def infer(self, task: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference for the specified task.
        
        Args:
            task: Task name (must be in supported_tasks)
            input_data: Input data dictionary containing the request data
            
        Returns:
            Result dictionary with inference results
        """
        pass
    
    def validate_task(self, task: str) -> bool:
        """
        Check if a task is supported by this handler.
        
        Args:
            task: Task name to validate
            
        Returns:
            True if supported, False otherwise
        """
        return task in self.get_supported_tasks()
