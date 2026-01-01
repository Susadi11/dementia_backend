# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import joblib
# import os
# import json

# class DementiaRiskClassifier:
#     """
#     Logistic Regression classifier for dementia risk assessment
#     Classifies users into Low (0), Medium (1), or High (2) risk categories
#     """
    
#     def __init__(self):
#         self.model = LogisticRegression(
#             multi_class='multinomial',
#             solver='lbfgs',
#             max_iter=1000,
#             random_state=42
#         )
#         self.scaler = StandardScaler()
#         self.feature_names = [
#             'avg_sac', 'min_sac', 'sac_trend', 'sac_variability',
#             'avg_accuracy', 'min_accuracy', 'accuracy_trend',
#             'avg_ies', 'max_ies', 'num_sessions', 'lstm_decline_score'
#         ]
#         self.risk_labels = ['Low', 'Medium', 'High']
    
#     def prepare_features(self, sessions_data, lstm_predictions):
#         """
#         Prepare features for risk classification
        
#         Args:
#             sessions_data: List of session dictionaries
#             lstm_predictions: LSTM decline scores for sequences
            
#         Returns:
#             X: Feature matrix
#             y: Risk labels (if available in data)
#         """
#         from src.features.game_features import GameFeatureExtractor
#         from src.features.cognitive_scoring import CognitiveScorer
        
#         extractor = GameFeatureExtractor()
#         scorer = CognitiveScorer()
        
#         # Group by user
#         import pandas as pd
#         df = pd.DataFrame(sessions_data)
        
#         X = []
#         y = []
        
#         for i, (user_id, user_sessions) in enumerate(df.groupby('user_id')):
#             if len(user_sessions) < 5:  # Minimum sessions required
#                 continue
            
#             # Calculate cognitive scores for all sessions
#             cognitive_scores = []
#             for _, session in user_sessions.iterrows():
#                 session_data = {
#                     'total_matches': session['total_matches'],
#                     'total_errors': session['total_errors'],
#                     'total_attempts': session['total_matches'] + session['total_errors'],
#                     'reaction_times': [session['avg_reaction_time']] * session['total_matches'],
#                     'hints_used': session['hints_used']
#                 }
#                 scores = scorer.calculate_session_scores(session_data)
#                 cognitive_scores.append(scores)
            
#             # Calculate temporal features
#             temporal_features = scorer.calculate_temporal_features(cognitive_scores)
            
#             # Get LSTM prediction for this user (use last prediction)
#             lstm_score = lstm_predictions[min(i, len(lstm_predictions) - 1)]
            
#             # Build feature vector
#             feature_vec = [
#                 temporal_features.get('avg_sac', 0),
#                 min([s['sac_score'] for s in cognitive_scores]),
#                 temporal_features.get('sac_trend', 0),
#                 np.std([s['sac_score'] for s in cognitive_scores]),
#                 temporal_features.get('avg_accuracy', 0),
#                 min([s['accuracy_rate'] for s in cognitive_scores]),
#                 temporal_features.get('accuracy_trend', 0),
#                 np.mean([s['ies_score'] for s in cognitive_scores]),
#                 max([s['ies_score'] for s in cognitive_scores]),
#                 len(cognitive_scores),
#                 lstm_score
#             ]
            
#             X.append(feature_vec)
            
#             # Determine risk label from persona (for training)
#             persona = user_sessions.iloc[0].get('persona_type', '')
#             risk_label = self._persona_to_risk_label(persona, lstm_score)
#             y.append(risk_label)
        
#         return np.array(X), np.array(y)
    
#     def _persona_to_risk_label(self, persona_type, lstm_score):
#         """Map persona type to risk label"""
#         if persona_type in ['fast_accurate', 'slow_accurate']:
#             return 0  # Low risk
#         elif persona_type in ['distracted', 'mild_decline']:
#             return 1  # Medium risk
#         elif persona_type in ['moderate_decline', 'severe_decline']:
#             return 2  # High risk
#         else:
#             # Fallback to LSTM score
#             if lstm_score < 0.3:
#                 return 0
#             elif lstm_score < 0.6:
#                 return 1
#             else:
#                 return 2
    
#     def train(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the risk classifier
        
#         Args:
#             X_train, y_train: Training data
#             X_val, y_val: Validation data (optional)
#         """
#         # Scale features
#         X_train_scaled = self.scaler.fit_transform(X_train)
        
#         # Train model
#         self.model.fit(X_train_scaled, y_train)
        
#         # Evaluate on training set
#         train_acc = self.model.score(X_train_scaled, y_train)
#         print(f"Training accuracy: {train_acc:.4f}")
        
#         # Evaluate on validation set if provided
#         if X_val is not None and y_val is not None:
#             X_val_scaled = self.scaler.transform(X_val)
#             val_acc = self.model.score(X_val_scaled, y_val)
#             print(f"Validation accuracy: {val_acc:.4f}")
            
#             # Print classification report
#             from sklearn.metrics import classification_report
#             y_pred = self.model.predict(X_val_scaled)
#             print("\nClassification Report:")
#             print(classification_report(y_val, y_pred, target_names=self.risk_labels))
    
#     def predict(self, X):
#         """
#         Predict risk categories
        
#         Args:
#             X: Feature matrix
            
#         Returns:
#             risk_labels: Risk category names
#             probabilities: Probability for each class
#         """
#         X_scaled = self.scaler.transform(X)
#         predictions = self.model.predict(X_scaled)
#         probabilities = self.model.predict_proba(X_scaled)
        
#         risk_labels = [self.risk_labels[p] for p in predictions]
        
#         return risk_labels, probabilities
    
#     def predict_single(self, features_dict):
#         """
#         Predict risk for a single user
        
#         Args:
#             features_dict: Dictionary with feature values
            
#         Returns:
#             risk_label: Risk category name
#             probability: Confidence score
#         """
#         # Build feature vector
#         feature_vec = [
#             features_dict.get('avg_sac', 0),
#             features_dict.get('min_sac', 0),
#             features_dict.get('sac_trend', 0),
#             features_dict.get('sac_variability', 0),
#             features_dict.get('avg_accuracy', 0),
#             features_dict.get('min_accuracy', 0),
#             features_dict.get('accuracy_trend', 0),
#             features_dict.get('avg_ies', 0),
#             features_dict.get('max_ies', 0),
#             features_dict.get('num_sessions', 0),
#             features_dict.get('lstm_decline_score', 0)
#         ]
        
#         X = np.array([feature_vec])
#         risk_labels, probabilities = self.predict(X)
        
#         return risk_labels[0], probabilities[0]
    
#     def save_model(self, model_dir='models/risk_classifier'):
#         """Save model and scaler"""
#         os.makedirs(model_dir, exist_ok=True)
        
#         # Save model
#         joblib.dump(self.model, os.path.join(model_dir, 'logistic_model.pkl'))
        
#         # Save scaler
#         joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
#         # Save metadata
#         metadata = {
#             'feature_names': self.feature_names,
#             'risk_labels': self.risk_labels,
#             'model_type': 'LogisticRegression'
#         }
        
#         with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         print(f"Risk classifier saved to {model_dir}")
    
#     @classmethod
#     def load_model(cls, model_dir='models/risk_classifier'):
#         """Load saved model"""
#         instance = cls()
        
#         # Load model and scaler
#         instance.model = joblib.load(os.path.join(model_dir, 'logistic_model.pkl'))
#         instance.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
#         # Load metadata
#         with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
#             metadata = json.load(f)
        
#         instance.feature_names = metadata['feature_names']
#         instance.risk_labels = metadata['risk_labels']
        
#         return instance


# def train_risk_classifier_from_data(synthetic_data_path='data/sample/synthetic_dataset.json',
#                                     lstm_model_path='models/lstm_model'):
#     """
#     Train risk classifier using synthetic data and LSTM predictions
#     """
#     from src.models.lstm_model.lstm_trainer import LSTMDementiaPredictor
    
#     print("Loading data and models...")
    
#     # Load synthetic data
#     with open(synthetic_data_path, 'r') as f:
#         data = json.load(f)
    
#     sessions = data['sessions']
    
#     # Load trained LSTM model
#     lstm_model = LSTMDementiaPredictor.load_model(lstm_model_path)
    
#     # Get LSTM predictions
#     print("Generating LSTM predictions...")
#     X_lstm, _, _ = lstm_model.prepare_sequences(sessions)
#     lstm_predictions = lstm_model.predict(X_lstm)
    
#     # Initialize risk classifier
#     print("Preparing risk classifier features...")
#     classifier = DementiaRiskClassifier()
    
#     # Prepare features
#     X, y = classifier.prepare_features(sessions, lstm_predictions)
    
#     print(f"Prepared {len(X)} samples for risk classification")
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"Training set: {len(X_train)} samples")
#     print(f"Validation set: {len(X_val)} samples")
    
#     # Train
#     print("\nTraining risk classifier...")
#     classifier.train(X_train, y_train, X_val, y_val)
    
#     # Save model
#     print("\nSaving risk classifier...")
#     classifier.save_model('models/risk_classifier')
    
#     return classifier


# if __name__ == "__main__":
#     classifier = train_risk_classifier_from_data()