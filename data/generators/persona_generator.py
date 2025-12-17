"""
Simple persona generator for synthetic data generation.
"""

import random
from typing import Dict, List


class PersonaGenerator:
    """Generates user personas for synthetic reminder scenarios."""
    
    def __init__(self):
        self.personas = [
            {"age": 70, "cognitive_level": "mild", "name": "Alice"},
            {"age": 75, "cognitive_level": "moderate", "name": "Bob"}, 
            {"age": 68, "cognitive_level": "clear", "name": "Carol"},
            {"age": 82, "cognitive_level": "high", "name": "David"},
            {"age": 73, "cognitive_level": "mild", "name": "Eva"}
        ]
    
    def generate_persona(self) -> Dict:
        """Generate a random persona."""
        return random.choice(self.personas)
    
    def get_personas(self) -> List[Dict]:
        """Get all available personas."""
        return self.personas
