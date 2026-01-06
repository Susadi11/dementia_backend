import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility (change this for different datasets)
np.random.seed(42)
random.seed(42)

# Configuration
NUM_USERS_PER_PERSONA = 6
SESSIONS_PER_USER = 30

# Enhanced Personas with Realistic Noise
PERSONAS = {
    'fast_accurate': {
        'label': 'LOW',
        'base_accuracy': 0.90,
        'base_rt_adj': 1.2,
        'accuracy_trend': 0.0,
        'rt_trend': 0.0,
        'variability': 0.05
    },
    'slow_accurate': {
        'label': 'LOW',
        'base_accuracy': 0.88,
        'base_rt_adj': 2.0,
        'accuracy_trend': 0.0,
        'rt_trend': 0.0,
        'variability': 0.07
    },
    'mild_decline': {
        'label': 'MEDIUM',
        'base_accuracy': 0.80,
        'base_rt_adj': 1.6,
        'accuracy_trend': -0.003,
        'rt_trend': 0.015,
        'variability': 0.10
    },
    'clear_decline': {
        'label': 'HIGH',
        'base_accuracy': 0.70,
        'base_rt_adj': 2.2,
        'accuracy_trend': -0.008,
        'rt_trend': 0.03,
        'variability': 0.15
    },
    'distracted': {
        'label': 'LOW',
        'base_accuracy': 0.82,
        'base_rt_adj': 1.5,
        'accuracy_trend': 0.0,
        'rt_trend': 0.0,
        'variability': 0.20
    },
    'learning_effect': {
        'label': 'LOW',
        'base_accuracy': 0.65,
        'base_rt_adj': 2.5,
        'accuracy_trend': 0.005,
        'rt_trend': -0.02,
        'variability': 0.08
    }
}

def add_realistic_noise(session_num, persona_config, user_age=70):
    """
    Add multiple layers of realistic noise to make data more lifelike
    """
    noise_factors = {}
    
    # 1. SESSION-TO-SESSION NOISE (everyone has good/bad days)
    daily_fatigue = np.random.normal(0, 0.08)
    
    # 2. WITHIN-SESSION VARIABILITY (attention fluctuates)
    attention_lapses = np.random.poisson(2) * 0.02  # Random attention drops
    
    # 3. LEARNING EFFECT (first 5 sessions improve for everyone)
    if session_num <= 5:
        learning_boost = (5 - session_num) * 0.03
    else:
        learning_boost = 0
    
    # 4. AGE-RELATED MOTOR VARIANCE (older = more variable)
    motor_noise = np.random.normal(0, 0.05 * (user_age / 70))
    
    # 5. OCCASIONAL OUTLIERS (1 in 10 sessions is really bad)
    is_outlier = np.random.random() < 0.1
    outlier_factor = 0.7 if is_outlier else 1.0
    
    # 6. CIRCADIAN RHYTHM (morning vs evening performance)
    time_of_day_effect = np.random.choice([-0.02, 0, 0.02], p=[0.3, 0.4, 0.3])
    
    noise_factors['daily_fatigue'] = daily_fatigue
    noise_factors['attention_lapses'] = attention_lapses
    noise_factors['learning_boost'] = learning_boost
    noise_factors['motor_noise'] = motor_noise
    noise_factors['outlier_factor'] = outlier_factor
    noise_factors['time_of_day'] = time_of_day_effect
    
    return noise_factors

def generate_enhanced_session(user_id, persona_name, persona_config, session_num, start_date, user_age):
    """
    Generate one session with realistic noise applied
    """
    # Calculate session date
    session_date = start_date + timedelta(days=session_num * 2.5)
    
    # Get realistic noise factors
    noise = add_realistic_noise(session_num, persona_config, user_age)
    
    # Apply trend over sessions
    accuracy_drift = persona_config['accuracy_trend'] * session_num
    rt_drift = persona_config['rt_trend'] * session_num
    
    # Add base variability
    base_noise_acc = np.random.normal(0, persona_config['variability'])
    base_noise_rt = np.random.normal(0, persona_config['variability'] * 0.5)
    
    # Calculate final accuracy with ALL noise factors
    accuracy = persona_config['base_accuracy'] + accuracy_drift + base_noise_acc
    accuracy += noise['daily_fatigue']
    accuracy -= noise['attention_lapses']
    accuracy += noise['learning_boost']
    accuracy += noise['time_of_day']
    accuracy *= noise['outlier_factor']
    accuracy = np.clip(accuracy, 0.1, 1.0)
    
    # Calculate final RT with ALL noise factors
    rt_adj_session = persona_config['base_rt_adj'] + rt_drift + base_noise_rt
    rt_adj_session += noise['motor_noise']
    rt_adj_session -= noise['learning_boost'] * 0.2  # Learning makes you faster
    rt_adj_session += noise['attention_lapses'] * 0.5  # Lapses slow you down
    if noise['outlier_factor'] < 1.0:
        rt_adj_session *= 1.3  # Bad days are slower
    rt_adj_session = max(rt_adj_session, 0.5)
    
    # Calculate SAC and IES
    sac = accuracy / rt_adj_session
    ies = rt_adj_session / max(accuracy, 0.1)
    
    # Motor baseline (varies by age and individual)
    motor_baseline = np.random.uniform(0.4, 1.0) * (user_age / 70)
    
    # Total attempts (varies naturally)
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

# Generate enhanced dataset
print("🔄 Generating ENHANCED dataset with realistic noise...")
data = []
user_counter = 1

for persona_name, persona_config in PERSONAS.items():
    for user_in_persona in range(NUM_USERS_PER_PERSONA):
        user_id = f"U{user_counter:03d}"
        start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 60))
        user_age = np.random.randint(60, 85)  # Random age per user
        
        for session_num in range(SESSIONS_PER_USER):
            session_data = generate_enhanced_session(
                user_id, persona_name, persona_config, 
                session_num, start_date, user_age
            )
            data.append(session_data)
        
        user_counter += 1

# Convert to DataFrame
df_enhanced = pd.DataFrame(data)

# Display summary
print("\n" + "="*60)
print("✅ ENHANCED SYNTHETIC DATASET GENERATED")
print("="*60)
print(f"\nTotal Sessions: {len(df_enhanced)}")
print(f"Total Users: {df_enhanced['user_id'].nunique()}")
print(f"\nRisk Label Distribution:")
print(df_enhanced['risk_label'].value_counts())
print(f"\nPersona Distribution:")
print(df_enhanced['persona'].value_counts())

# Save enhanced data
df_enhanced.to_csv('synthetic_dementia_ENHANCED.csv', index=False)
print("\n💾 Saved to: synthetic_dementia_ENHANCED.csv")

# Show comparison statistics
print("\n" + "="*60)
print("📊 DATA QUALITY METRICS")
print("="*60)
print("\nAccuracy Statistics by Persona:")
print(df_enhanced.groupby('persona')['accuracy'].agg(['mean', 'std', 'min', 'max']))
print("\nReaction Time Statistics by Persona:")
print(df_enhanced.groupby('persona')['rt_adj_session'].agg(['mean', 'std', 'min', 'max']))

# Visualize noise effects
print("\n📈 Generating visualization...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Enhanced Dataset with Realistic Noise - Sample Users', fontsize=16)

for idx, persona in enumerate(['fast_accurate', 'slow_accurate', 'mild_decline', 
                                'clear_decline', 'distracted', 'learning_effect']):
    row = idx // 3
    col = idx % 3
    
    # Get one user from this persona
    sample_user = df_enhanced[df_enhanced['persona'] == persona]['user_id'].iloc[0]
    user_data = df_enhanced[df_enhanced['user_id'] == sample_user]
    
    axes[row, col].plot(user_data['session_num'], user_data['accuracy'], 
                        marker='o', label='Accuracy', alpha=0.7)
    axes[row, col].plot(user_data['session_num'], user_data['sac'], 
                        marker='s', label='SAC', alpha=0.7)
    axes[row, col].set_title(f'{persona.replace("_", " ").title()}')
    axes[row, col].set_xlabel('Session Number')
    axes[row, col].set_ylabel('Score')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_data_visualization.png', dpi=150)
print("✅ Visualization saved to: enhanced_data_visualization.png")
plt.show()

print("\n" + "="*60)
print("✨ ENHANCEMENT COMPLETE!")
print("="*60)
print("\nKey Improvements:")
print("✓ Daily fatigue variations (±8%)")
print("✓ Attention lapses (random drops)")
print("✓ Learning effects (first 5 sessions)")
print("✓ Age-based motor variance")
print("✓ Occasional outliers (10% bad sessions)")
print("✓ Circadian rhythm effects")
print("\n🎯 Your data is now much more realistic!")