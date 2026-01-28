"""
Augmentation registry for managing available augmentations.
"""

from typing import Dict, Type, Any
from src.augmentation.base_augmentor import BaseAugmentor


class AugmentationRegistry:
    """
    Registry to manage all available augmentations.
    
    This class uses the registry pattern to allow dynamic registration
    and retrieval of augmentation classes.
    """
    
    _registry: Dict[str, Type[BaseAugmentor]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register augmentation classes.
        
        Parameters
        ----------
        name : str
            Name to register the augmentation under
            
        Returns
        -------
        function
            Decorator function
            
        Example
        -------
        >>> @AugmentationRegistry.register('color_jitter')
        >>> class ColorJitterAugmentor(BaseAugmentor):
        >>>     ...
        """
        def decorator(augmentor_class: Type[BaseAugmentor]):
            cls._registry[name] = augmentor_class
            return augmentor_class
        return decorator
    
    @classmethod
    def get_augmentor(cls, name: str, config: Dict[str, Any]) -> BaseAugmentor:
        """
        Retrieve and instantiate an augmentor by name.
        
        Parameters
        ----------
        name : str
            Name of the augmentation
        config : dict
            Configuration for the augmentation
            
        Returns
        -------
        BaseAugmentor
            Instance of the requested augmentor
            
        Raises
        ------
        ValueError
            If the augmentation name is not found in the registry
        """
        if name not in cls._registry:
            raise ValueError(
                f"Augmentation '{name}' not found in registry. "
                f"Available augmentations: {cls.list_available()}"
            )
        return cls._registry[name](config)
    
    @classmethod
    def list_available(cls) -> list:
        """
        List all registered augmentations.
        
        Returns
        -------
        list
            List of registered augmentation names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an augmentation is registered.
        
        Parameters
        ----------
        name : str
            Name of the augmentation
            
        Returns
        -------
        bool
            True if registered, False otherwise
        """
        return name in cls._registry
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister an augmentation.
        
        Parameters
        ----------
        name : str
            Name of the augmentation to unregister
        """
        if name in cls._registry:
            del cls._registry[name]
