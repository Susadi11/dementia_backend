"""
Synthetic Reminder Data Generator for Context-Aware Smart Reminder System

Generates realistic reminder-response scenarios combining:
1. Pitt Corpus speech patterns
2. Synthetic reminder contexts  
3. Both text and voice modalities
4. Dementia progression levels

This creates training data specifically for reminder system interactions.
"""

import random
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.conversational_ai.nlp.nlp_engine import NLPEngine
from src.features.conversational_ai.feature_extractor import FeatureExtractor
from data.generators.persona_generator import PersonaGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReminderSyntheticDataGenerator:
    """
    Generates synthetic reminder-response data for training models.
    
    Combines real Pitt Corpus patterns with synthetic reminder scenarios
    to create realistic training data for context-aware reminder systems.
    """
    
    def __init__(self, pitt_data_path: str = "output/pitt_text_features.csv"):
        """Initialize with Pitt Corpus data and generators."""
        self.pitt_data_path = Path(pitt_data_path)
        self.nlp_engine = NLPEngine()
        self.feature_extractor = FeatureExtractor()
        self.persona_generator = PersonaGenerator()
        
        # Load Pitt Corpus patterns
        self.pitt_patterns = self._load_pitt_patterns()
        
        # Reminder categories and templates
        self.reminder_categories = self._init_reminder_categories()
        self.response_templates = self._init_response_templates()
        
    def _load_pitt_patterns(self) -> Dict:
        """Load and analyze Pitt Corpus patterns."""
        if not self.pitt_data_path.exists():
            logger.warning(f"Pitt data not found at {self.pitt_data_path}")
            return {"control": [], "dementia": []}
        
        patterns = {"control": [], "dementia": []}
        
        with open(self.pitt_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = "dementia" if int(row['dementia_label']) == 1 else "control"
                patterns[label].append({
                    'hesitation_pauses': float(row['hesitation_pauses']),
                    'semantic_incoherence': float(row['semantic_incoherence']),
                    'low_confidence_answers': float(row['low_confidence_answers']),
                    'repeated_questions': int(row['repeated_questions']),
                    'self_correction': int(row['self_correction']),
                    'task': row['task'],
                    'participant_id': row['participant_id']
                })
        
        logger.info(f"Loaded {len(patterns['control'])} control and {len(patterns['dementia'])} dementia patterns")
        return patterns
    
    def _init_reminder_categories(self) -> Dict:
        """Initialize reminder categories with realistic scenarios."""
        return {
            "medication": {
                "templates": [
                    "Take your {medication_name} ({description})",
                    "Time for your {time_of_day} medication",
                    "Don't forget your {medication_type} pills",
                    "Your {medication_name} is due now",
                    "Please take your prescribed {medication_name}"
                ],
                "contexts": {
                    "medication_name": ["Aricept", "Donepezil", "Memantine", "blood pressure pills", "vitamins", "aspirin"],
                    "description": ["blue pill", "white tablet", "small capsule", "morning dose", "evening dose"],
                    "time_of_day": ["morning", "afternoon", "evening", "bedtime"],
                    "medication_type": ["memory", "blood pressure", "heart", "diabetes", "pain"]
                }
            },
            "meal": {
                "templates": [
                    "Time for {meal_type}",
                    "Your {meal_type} is ready",
                    "Don't forget to eat {meal_type}",
                    "It's time to have some food",
                    "Please come for {meal_type}"
                ],
                "contexts": {
                    "meal_type": ["breakfast", "lunch", "dinner", "snack", "your meal"]
                }
            },
            "appointment": {
                "templates": [
                    "Your {appointment_type} appointment is {time_reference}",
                    "Don't forget your appointment with Dr. {doctor_name}",
                    "Time to get ready for your {appointment_type} visit",
                    "Your medical appointment is coming up",
                    "Remember your {time_reference} appointment"
                ],
                "contexts": {
                    "appointment_type": ["doctor", "dentist", "therapy", "medical", "checkup"],
                    "time_reference": ["today", "in 1 hour", "this afternoon", "soon", "at 2 PM"],
                    "doctor_name": ["Smith", "Johnson", "Williams", "Brown", "Jones"]
                }
            },
            "hygiene": {
                "templates": [
                    "Time to {hygiene_activity}",
                    "Don't forget to {hygiene_activity}",
                    "Please {hygiene_activity} now",
                    "It's time for your {hygiene_routine}",
                    "Remember to {hygiene_activity} before bed"
                ],
                "contexts": {
                    "hygiene_activity": ["brush your teeth", "take a shower", "wash your hands", "comb your hair"],
                    "hygiene_routine": ["morning routine", "evening routine", "daily hygiene"]
                }
            },
            "safety": {
                "templates": [
                    "Please check if you {safety_action}",
                    "Did you remember to {safety_action}?",
                    "Don't forget to {safety_action}",
                    "Make sure you {safety_action}",
                    "Time to check the {safety_item}"
                ],
                "contexts": {
                    "safety_action": ["lock the door", "turn off the stove", "close the windows", "check the gas"],
                    "safety_item": ["doors", "windows", "stove", "lights"]
                }
            }
        }
    
    def _init_response_templates(self) -> Dict:
        """Initialize response templates based on cognitive levels."""
        return {
            "clear_confirmation": {
                "patterns": [
                    "Yes, I {action_past_tense}",
                    "Already done",
                    "I took it",
                    "Yes, finished",
                    "All set",
                    "I remember, yes"
                ],
                "cognitive_features": {
                    "hesitation_pauses": (0, 2),
                    "semantic_incoherence": (0.0, 0.3),
                    "confidence": (0.8, 1.0)
                }
            },
            "mild_confusion": {
                "patterns": [
                    "Um... I think I {action_past_tense}",
                    "Maybe? I'm not sure",
                    "I think so... yes",
                    "Probably, but...",
                    "I believe I did",
                    "Um, what was it again?"
                ],
                "cognitive_features": {
                    "hesitation_pauses": (2, 5),
                    "semantic_incoherence": (0.3, 0.6),
                    "confidence": (0.4, 0.7)
                }
            },
            "moderate_confusion": {
                "patterns": [
                    "I... what? Did I...?",
                    "I don't remember... maybe?",
                    "What medicine? The blue one?",
                    "I think... or was it yesterday?",
                    "Um... help me remember",
                    "Which one do you mean?"
                ],
                "cognitive_features": {
                    "hesitation_pauses": (5, 10),
                    "semantic_incoherence": (0.6, 0.8),
                    "confidence": (0.1, 0.4)
                }
            },
            "high_confusion": {
                "patterns": [
                    "I don't understand",
                    "What? I'm confused",
                    "I don't know what you mean",
                    "Help me... I'm lost",
                    "What am I supposed to do?",
                    "I can't remember anything"
                ],
                "cognitive_features": {
                    "hesitation_pauses": (8, 15),
                    "semantic_incoherence": (0.8, 1.0),
                    "confidence": (0.0, 0.2)
                }
            },
            "delay_resistance": {
                "patterns": [
                    "Later, I'm busy now",
                    "Not right now",
                    "In a minute",
                    "Can it wait?",
                    "I'll do it soon",
                    "Give me a moment"
                ],
                "cognitive_features": {
                    "hesitation_pauses": (1, 4),
                    "semantic_incoherence": (0.1, 0.4),
                    "confidence": (0.6, 0.9)
                }
            }
        }
    
    def generate_reminder_scenario(
        self, 
        category: str, 
        cognitive_level: str = "random"
    ) -> Dict:
        """Generate a complete reminder scenario."""
        if category not in self.reminder_categories:
            raise ValueError(f"Unknown category: {category}")
        
        # Select cognitive level
        if cognitive_level == "random":
            cognitive_level = random.choice(list(self.response_templates.keys()))
        
        # Generate reminder
        reminder = self._generate_reminder(category)
        
        # Generate user response
        response = self._generate_user_response(cognitive_level, reminder)
        
        # Extract features
        features = self._extract_response_features(response, cognitive_level)
        
        # Create complete scenario
        scenario = {
            "reminder_id": f"rem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
            "category": category,
            "reminder_text": reminder["text"],
            "reminder_context": reminder["context"],
            "user_response": response["text"],
            "cognitive_level": cognitive_level,
            "response_time_seconds": response["response_time"],
            "features": features,
            "labels": {
                "confusion_detected": features["confusion_detected"],
                "memory_issue": features["memory_issue"],
                "caregiver_alert_needed": features["caregiver_alert_needed"],
                "cognitive_risk_score": features["cognitive_risk_score"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return scenario
    
    def _generate_reminder(self, category: str) -> Dict:
        """Generate a reminder for the given category."""
        cat_data = self.reminder_categories[category]
        template = random.choice(cat_data["templates"])
        
        # Fill in template variables
        context = {}
        reminder_text = template
        for key, values in cat_data["contexts"].items():
            if f"{{{key}}}" in template:
                context[key] = random.choice(values)
                reminder_text = reminder_text.replace(f"{{{key}}}", context[key])
        
        return {
            "text": reminder_text,
            "context": context,
            "category": category,
            "priority": random.choice(["low", "medium", "high", "critical"])
        }
    
    def _generate_user_response(self, cognitive_level: str, reminder: Dict) -> Dict:
        """Generate user response based on cognitive level."""
        response_data = self.response_templates[cognitive_level]
        
        # Select base pattern
        base_pattern = random.choice(response_data["patterns"])
        
        # Add Pitt Corpus-based variations
        response_text = self._add_pitt_variations(base_pattern, cognitive_level)
        
        # Calculate response time based on cognitive level
        response_time = self._calculate_response_time(cognitive_level)
        
        return {
            "text": response_text,
            "response_time": response_time,
            "cognitive_level": cognitive_level
        }
    
    def _add_pitt_variations(self, base_text: str, cognitive_level: str) -> str:
        """Add variations based on Pitt Corpus patterns."""
        # Get appropriate Pitt patterns
        if cognitive_level in ["mild_confusion", "moderate_confusion", "high_confusion"]:
            patterns = self.pitt_patterns.get("dementia", [])
        else:
            patterns = self.pitt_patterns.get("control", [])
        
        if not patterns:
            return base_text
        
        # Select a pattern for reference
        pattern = random.choice(patterns)
        
        # Add hesitation based on Pitt data
        hesitation_count = int(pattern.get("hesitation_pauses", 0))
        if hesitation_count > 3:
            hesitations = ["um", "uh", "well", "you know"]
            for _ in range(min(hesitation_count // 2, 3)):
                base_text = random.choice(hesitations) + "... " + base_text
        
        # Add self-correction if indicated
        if pattern.get("self_correction", 0) > 0 and random.random() < 0.3:
            corrections = [
                "I mean...",
                "actually...", 
                "wait, no...",
                "let me think..."
            ]
            base_text = base_text + " " + random.choice(corrections) + " " + base_text.split()[0]
        
        return base_text
    
    def _calculate_response_time(self, cognitive_level: str) -> float:
        """Calculate realistic response time based on cognitive level."""
        time_ranges = {
            "clear_confirmation": (5.0, 20.0),
            "mild_confusion": (15.0, 45.0),
            "moderate_confusion": (30.0, 90.0),
            "high_confusion": (45.0, 120.0),
            "delay_resistance": (2.0, 15.0)
        }
        
        min_time, max_time = time_ranges.get(cognitive_level, (10.0, 30.0))
        return random.uniform(min_time, max_time)
    
    def _extract_response_features(self, response: Dict, cognitive_level: str) -> Dict:
        """Extract features from the response."""
        response_text = response["text"]
        
        # Use feature extractor
        try:
            features = self.feature_extractor.extract_features_normalized(
                transcript_text=response_text
            )
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            features = {}
        
        # Add synthetic features based on cognitive level
        cognitive_features = self.response_templates[cognitive_level]["cognitive_features"]
        
        # Merge with Pitt-based features if available
        for feature_name, (min_val, max_val) in cognitive_features.items():
            if feature_name not in features:
                features[feature_name] = random.uniform(min_val, max_val)
        
        # Calculate derived features
        features.update({
            "confusion_detected": cognitive_level in ["moderate_confusion", "high_confusion"],
            "memory_issue": cognitive_level in ["mild_confusion", "moderate_confusion", "high_confusion"],
            "caregiver_alert_needed": cognitive_level in ["moderate_confusion", "high_confusion"],
            "cognitive_risk_score": self._calculate_risk_score(cognitive_level, features),
            "response_coherence": 1.0 - features.get("semantic_incoherence", 0.0)
        })
        
        return features
    
    def _calculate_risk_score(self, cognitive_level: str, features: Dict) -> float:
        """Calculate overall cognitive risk score."""
        base_scores = {
            "clear_confirmation": 0.1,
            "mild_confusion": 0.4, 
            "moderate_confusion": 0.7,
            "high_confusion": 0.9,
            "delay_resistance": 0.2
        }
        
        base_score = base_scores.get(cognitive_level, 0.5)
        
        # Adjust based on features
        adjustments = 0.0
        if features.get("semantic_incoherence", 0) > 0.6:
            adjustments += 0.1
        if features.get("hesitation_pauses", 0) > 5:
            adjustments += 0.1
        if features.get("low_confidence_answers", 0) > 0.5:
            adjustments += 0.1
        
        return min(1.0, base_score + adjustments)
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        output_file: str = "data/synthetic_reminder_dataset.csv",
        distribution: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate a complete synthetic dataset for training.
        
        Args:
            num_samples: Total number of samples to generate
            output_file: Output file path
            distribution: Distribution of categories and cognitive levels
            
        Returns:
            Path to generated dataset file
        """
        if distribution is None:
            # Default distribution
            distribution = {
                "categories": {
                    "medication": 0.4,
                    "meal": 0.2, 
                    "appointment": 0.2,
                    "hygiene": 0.1,
                    "safety": 0.1
                },
                "cognitive_levels": {
                    "clear_confirmation": 0.3,
                    "mild_confusion": 0.25,
                    "moderate_confusion": 0.25,
                    "high_confusion": 0.15,
                    "delay_resistance": 0.05
                }
            }
        
        logger.info(f"Generating {num_samples} synthetic reminder scenarios...")
        
        scenarios = []
        for i in range(num_samples):
            # Select category based on distribution
            category = self._select_weighted_choice(distribution["categories"])
            
            # Select cognitive level based on distribution  
            cognitive_level = self._select_weighted_choice(distribution["cognitive_levels"])
            
            # Generate scenario
            scenario = self.generate_reminder_scenario(category, cognitive_level)
            scenarios.append(scenario)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} scenarios")
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._save_scenarios_to_csv(scenarios, output_path)
        
        logger.info(f"Saved {len(scenarios)} scenarios to {output_path}")
        return str(output_path)
    
    def _select_weighted_choice(self, weights: Dict[str, float]) -> str:
        """Select item based on weighted distribution."""
        items = list(weights.keys())
        probabilities = list(weights.values())
        return random.choices(items, weights=probabilities)[0]
    
    def _save_scenarios_to_csv(self, scenarios: List[Dict], output_path: Path):
        """Save scenarios to CSV file."""
        if not scenarios:
            logger.error("No scenarios to save")
            return
        
        # Flatten scenarios for CSV
        rows = []
        for scenario in scenarios:
            row = {
                "reminder_id": scenario["reminder_id"],
                "category": scenario["category"],
                "reminder_text": scenario["reminder_text"],
                "user_response": scenario["user_response"],
                "cognitive_level": scenario["cognitive_level"],
                "response_time_seconds": scenario["response_time_seconds"],
                "confusion_detected": scenario["labels"]["confusion_detected"],
                "memory_issue": scenario["labels"]["memory_issue"],
                "caregiver_alert_needed": scenario["labels"]["caregiver_alert_needed"],
                "cognitive_risk_score": scenario["labels"]["cognitive_risk_score"],
                "timestamp": scenario["timestamp"]
            }
            
            # Add features
            for feature_name, feature_value in scenario["features"].items():
                row[f"feature_{feature_name}"] = feature_value
            
            # Add context
            for context_key, context_value in scenario["reminder_context"].items():
                row[f"context_{context_key}"] = context_value
            
            rows.append(row)
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if rows:
                # Get all unique fieldnames from all rows
                all_fieldnames = set()
                for row in rows:
                    all_fieldnames.update(row.keys())
                fieldnames = sorted(list(all_fieldnames))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def combine_with_pitt_dataset(
        self, 
        synthetic_file: str,
        pitt_file: str, 
        output_file: str = "data/combined_reminder_training_data.csv"
    ) -> str:
        """
        Combine synthetic reminder data with processed Pitt Corpus data.
        
        This creates a comprehensive training dataset that includes:
        - Real dementia speech patterns from Pitt Corpus
        - Synthetic reminder-response scenarios
        - Combined feature space for robust training
        """
        logger.info("Combining synthetic and Pitt Corpus datasets...")
        
        combined_rows = []
        
        # Load synthetic data
        with open(synthetic_file, 'r', encoding='utf-8') as f:
            synthetic_reader = csv.DictReader(f)
            for row in synthetic_reader:
                # Mark as synthetic
                row['data_source'] = 'synthetic'
                row['original_task'] = 'reminder_response'
                combined_rows.append(row)
        
        logger.info(f"Loaded {len(combined_rows)} synthetic samples")
        
        # Load and adapt Pitt data
        pitt_count = 0
        with open(pitt_file, 'r', encoding='utf-8') as f:
            pitt_reader = csv.DictReader(f) 
            for row in pitt_reader:
                # Adapt Pitt data for reminder context
                adapted_row = {
                    'reminder_id': f"pitt_{row['participant_id']}_{row['task']}",
                    'category': 'baseline_speech',  # Pitt tasks as baseline
                    'reminder_text': f"Please describe the {row['task']} task",
                    'user_response': f"Baseline {row['task']} task response",  # Placeholder
                    'cognitive_level': 'dementia_baseline' if int(row['dementia_label']) == 1 else 'control_baseline',
                    'response_time_seconds': 60.0,  # Estimated
                    'confusion_detected': int(row['dementia_label']) == 1,
                    'memory_issue': int(row['dementia_label']) == 1,
                    'caregiver_alert_needed': int(row['dementia_label']) == 1,
                    'cognitive_risk_score': float(row['semantic_incoherence']),
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'pitt_corpus',
                    'original_task': row['task'],
                    'participant_id': row['participant_id'],
                    'dementia_label': row['dementia_label']
                }
                
                # Copy Pitt features with prefix
                for key, value in row.items():
                    if key not in ['participant_id', 'dementia_label', 'task', 'file_path']:
                        adapted_row[f'pitt_{key}'] = value
                
                combined_rows.append(adapted_row)
                pitt_count += 1
        
        logger.info(f"Loaded {pitt_count} Pitt Corpus samples")
        
        # Save combined dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if combined_rows:
                # Get all unique fieldnames
                fieldnames = set()
                for row in combined_rows:
                    fieldnames.update(row.keys())
                fieldnames = sorted(list(fieldnames))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(combined_rows)
        
        logger.info(f"Combined dataset saved to {output_path} with {len(combined_rows)} total samples")
        return str(output_path)