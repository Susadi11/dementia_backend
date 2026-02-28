import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS_PER_PERSONA = 6  # 6 users per persona = 36 total users
SESSIONS_PER_USER = 30      # 30 sessions per user
TOTAL_SESSIONS = NUM_USERS_PER_PERSONA * 6 * SESSIONS_PER_USER  # ~1080 sessions

# Define 6 Personas (aligned with your document)
PERSONAS = {
    'fast_accurate': {
        'label': 'LOW',
        'base_accuracy': 0.90,
        'base_rt_adj': 1.2,
        'accuracy_trend': 0.0,    # stable
        'rt_trend': 0.0,           # stable
        'variability': 0.05        # low noise
    },
    'slow_accurate': {
        'label': 'LOW',
        'base_accuracy': 0.88,
        'base_rt_adj': 2.0,        # slower but cognitively fine
        'accuracy_trend': 0.0,
        'rt_trend': 0.0,
        'variability': 0.07
    },
    'mild_decline': {
        'label': 'MEDIUM',
        'base_accuracy': 0.80,
        'base_rt_adj': 1.6,
        'accuracy_trend': -0.003,  # slight decline per session
        'rt_trend': 0.015,          # slight slowdown
        'variability': 0.10
    },
    'clear_decline': {
        'label': 'HIGH',
        'base_accuracy': 0.70,
        'base_rt_adj': 2.2,
        'accuracy_trend': -0.008,  # stronger decline
        'rt_trend': 0.03,
        'variability': 0.15         # more variability
    },
    'distracted': {
        'label': 'LOW',            # high variability but no true decline
        'base_accuracy': 0.82,
        'base_rt_adj': 1.5,
        'accuracy_trend': 0.0,
        'rt_trend': 0.0,
        'variability': 0.20         # high noise
    },
    'learning_effect': {
        'label': 'LOW',
        'base_accuracy': 0.65,      # starts lower
        'base_rt_adj': 2.5,
        'accuracy_trend': 0.005,    # improves initially
        'rt_trend': -0.02,          # gets faster
        'variability': 0.08
    }
}

def generate_session_data(user_id, persona_name, persona_config, session_num, start_date):
    """Generate one session's worth of data for a user"""
    
    # Calculate session date (one session every 2-3 days)
    session_date = start_date + timedelta(days=session_num * 2.5)
    
    # Apply trend over sessions
    accuracy_drift = persona_config['accuracy_trend'] * session_num
    rt_drift = persona_config['rt_trend'] * session_num
    
    # Add random noise
    noise_acc = np.random.normal(0, persona_config['variability'])
    noise_rt = np.random.normal(0, persona_config['variability'] * 0.5)
    
    # Calculate final values
    accuracy = np.clip(
        persona_config['base_accuracy'] + accuracy_drift + noise_acc,
        0.1, 1.0
    )
    
    rt_adj_session = max(
        persona_config['base_rt_adj'] + rt_drift + noise_rt,
        0.5  # minimum RT
    )
    
    # Calculate SAC and IES
    sac = accuracy / rt_adj_session
    ies = rt_adj_session / max(accuracy, 0.1)  # avoid division by zero
    
    # Simulate motor baseline (older users have higher baseline)
    motor_baseline = np.random.uniform(0.4, 1.0)
    
    # Total attempts in session
    total_attempts = np.random.randint(15, 25)
    correct = int(accuracy * total_attempts)
    errors = total_attempts - correct
    
    return {
        'user_id': user_id,
        'persona': persona_name,
        'session_num': session_num + 1,
        'session_date': session_date.strftime('%Y-%m-%d'),
        'motor_baseline': round(motor_baseline, 3),
        'total_attempts': total_attempts,
        'correct': correct,
        'errors': errors,
        'accuracy': round(accuracy, 3),
        'rt_adj_session': round(rt_adj_session, 3),
        'sac': round(sac, 3),
        'ies': round(ies, 3),
        'risk_label': persona_config['label']
    }

# Generate full dataset
data = []
user_counter = 1

for persona_name, persona_config in PERSONAS.items():
    for user_in_persona in range(NUM_USERS_PER_PERSONA):
        user_id = f"U{user_counter:03d}"
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))
        
        for session_num in range(SESSIONS_PER_USER):
            session_data = generate_session_data(
                user_id, persona_name, persona_config, session_num, start_date
            )
            data.append(session_data)
        
        user_counter += 1

# Convert to DataFrame
df = pd.DataFrame(data)

# Display summary
print("=" * 60)
print("SYNTHETIC DATASET GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nTotal Sessions: {len(df)}")
print(f"Total Users: {df['user_id'].nunique()}")
print(f"\nRisk Label Distribution:")
print(df['risk_label'].value_counts())
print(f"\nPersona Distribution:")
print(df['persona'].value_counts())
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

# Save to CSV
output_file = 'synthetic_dementia_game_data.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Dataset saved to: {output_file}")
print("=" * 60)