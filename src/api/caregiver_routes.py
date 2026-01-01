from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from sqlalchemy import func, desc

from src.database.models import User, GameSession, CognitiveScore, RiskAssessment
from src.database.connection import get_db_context
from src.features.cognitive_scoring import CognitiveScorer
from src.models.lstm_model.lstm_trainer import LSTMDementiaPredictor
from src.models.risk_classifier.risk_model import DementiaRiskClassifier

caregiver_bp = Blueprint('caregiver', __name__, url_prefix='/api/caregiver')


    """Get comprehensive dashboard summary for a user"""
@caregiver_bp.route('/dashboard/<user_id>/summary', methods=['GET'])
def get_dashboard_summary(user_id):
    try:
        with get_db_context() as db:
            # Get user
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get session statistics
            total_sessions = db.query(GameSession).filter(
                GameSession.user_id == user_id,
                GameSession.completed == True
            ).count()
            
            # Get recent sessions (last 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            recent_sessions = db.query(GameSession).filter(
                GameSession.user_id == user_id,
                GameSession.completed == True,
                GameSession.timestamp >= thirty_days_ago
            ).count()
            
            # Get latest cognitive score
            latest_score = db.query(CognitiveScore).join(GameSession).filter(
                GameSession.user_id == user_id
            ).order_by(desc(CognitiveScore.timestamp)).first()
            
            # Get latest risk assessment
            latest_risk = db.query(RiskAssessment).filter(
                RiskAssessment.user_id == user_id
            ).order_by(desc(RiskAssessment.timestamp)).first()
            
            # Calculate average scores (last 10 sessions)
            recent_scores = db.query(CognitiveScore).join(GameSession).filter(
                GameSession.user_id == user_id
            ).order_by(desc(CognitiveScore.timestamp)).limit(10).all()
            
            avg_accuracy = sum(s.accuracy_rate for s in recent_scores) / len(recent_scores) if recent_scores else 0
            avg_sac = sum(s.sac_score for s in recent_scores) / len(recent_scores) if recent_scores else 0
            
            summary = {
                'user': {
                    'user_id': user.user_id,
                    'name': user.name,
                    'age': user.age,
                    'date_enrolled': user.date_enrolled.isoformat(),
                    'motor_baseline_rt': user.motor_baseline_rt
                },
                'statistics': {
                    'total_sessions': total_sessions,
                    'recent_sessions_30d': recent_sessions,
                    'avg_accuracy': round(avg_accuracy, 4),
                    'avg_sac_score': round(avg_sac, 4)
                },
                'latest_score': {
                    'timestamp': latest_score.timestamp.isoformat() if latest_score else None,
                    'accuracy_rate': latest_score.accuracy_rate if latest_score else None,
                    'sac_score': latest_score.sac_score if latest_score else None,
                    'ies_score': latest_score.ies_score if latest_score else None
                } if latest_score else None,
                'risk_assessment': {
                    'timestamp': latest_risk.timestamp.isoformat() if latest_risk else None,
                    'risk_category': latest_risk.risk_category if latest_risk else 'Unknown',
                    'lstm_decline_score': latest_risk.lstm_decline_score if latest_risk else None,
                    'confidence_score': latest_risk.confidence_score if latest_risk else None,
                    'alert_triggered': latest_risk.alert_triggered if latest_risk else False
                } if latest_risk else None
            }
            
            return jsonify(summary), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

""" Get cognitive performance trends over time """
@caregiver_bp.route('/dashboard/<user_id>/trends', methods=['GET'])
def get_cognitive_trends(user_id):
    try:
        lookback_days = request.args.get('lookback_days', 90, type=int)
        limit = request.args.get('limit', 50, type=int)
        
        with get_db_context() as db:
            # Get user
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get sessions within timeframe
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            
            scores = db.query(CognitiveScore, GameSession).join(GameSession).filter(
                GameSession.user_id == user_id,
                CognitiveScore.timestamp >= cutoff_date
            ).order_by(CognitiveScore.timestamp.asc()).limit(limit).all()
            
            if not scores:
                return jsonify({
                    'user_id': user_id,
                    'trends': [],
                    'message': 'No data available for the specified period'
                }), 200
            
            # Build trend data
            trends = []
            for score, session in scores:
                trends.append({
                    'timestamp': score.timestamp.isoformat(),
                    'session_id': session.session_id,
                    'accuracy_rate': score.accuracy_rate,
                    'error_rate': score.error_rate,
                    'avg_reaction_time': score.avg_reaction_time,
                    'motor_adjusted_rt': score.motor_adjusted_rt,
                    'sac_score': score.sac_score,
                    'ies_score': score.ies_score,
                    'total_matches': session.total_matches,
                    'total_errors': session.total_errors
                })
            
            # Calculate trend statistics
            accuracies = [t['accuracy_rate'] for t in trends]
            sac_scores = [t['sac_score'] for t in trends]
            
            import numpy as np
            accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0
            sac_trend = np.polyfit(range(len(sac_scores)), sac_scores, 1)[0] if len(sac_scores) > 1 else 0
            
            return jsonify({
                'user_id': user_id,
                'timeframe': {
                    'lookback_days': lookback_days,
                    'start_date': trends[0]['timestamp'] if trends else None,
                    'end_date': trends[-1]['timestamp'] if trends else None,
                    'data_points': len(trends)
                },
                'trends': trends,
                'statistics': {
                    'accuracy_trend': round(accuracy_trend, 6),
                    'sac_trend': round(sac_trend, 6),
                    'avg_accuracy': round(sum(accuracies) / len(accuracies), 4) if accuracies else 0,
                    'avg_sac': round(sum(sac_scores) / len(sac_scores), 4) if sac_scores else 0
                }
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""Get historical risk assessments for a user"""
@caregiver_bp.route('/dashboard/<user_id>/risk-history', methods=['GET'])
def get_risk_history(user_id):
    try:
        limit = request.args.get('limit', 20, type=int)
        
        with get_db_context() as db:
            # Get user
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get risk assessments
            assessments = db.query(RiskAssessment).filter(
                RiskAssessment.user_id == user_id
            ).order_by(desc(RiskAssessment.timestamp)).limit(limit).all()
            
            history = [{
                'assessment_id': a.assessment_id,
                'timestamp': a.timestamp.isoformat(),
                'risk_category': a.risk_category,
                'lstm_decline_score': a.lstm_decline_score,
                'confidence_score': a.confidence_score,
                'alert_triggered': a.alert_triggered,
                'sessions_analyzed': a.sessions_analyzed
            } for a in assessments]
            
            return jsonify({
                'user_id': user_id,
                'assessment_count': len(history),
                'history': history
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""Get active alerts for a user"""
@caregiver_bp.route('/dashboard/<user_id>/alerts', methods=['GET'])
def get_active_alerts(user_id):
    try:
        with get_db_context() as db:
            # Get user
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get recent assessments with alerts
            recent_alerts = db.query(RiskAssessment).filter(
                RiskAssessment.user_id == user_id,
                RiskAssessment.alert_triggered == True
            ).order_by(desc(RiskAssessment.timestamp)).limit(5).all()
            
            alerts = [{
                'timestamp': a.timestamp.isoformat(),
                'risk_category': a.risk_category,
                'lstm_decline_score': a.lstm_decline_score,
                'message': f"Risk level elevated to {a.risk_category}"
            } for a in recent_alerts]
            
            return jsonify({
                'user_id': user_id,
                'active_alerts': len(alerts),
                'alerts': alerts
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


  """
    Perform comprehensive dementia risk assessment
    
    This endpoint:
    1. Loads user's session data
    2. Runs LSTM model for decline prediction
    3. Runs risk classifier
    4. Stores assessment in database
    5. Triggers alerts if necessary
    """
@caregiver_bp.route('/assess-risk/<user_id>', methods=['POST'])
def assess_dementia_risk(user_id):
    try:
        with get_db_context() as db:
            # Get user
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get user's sessions
            sessions = db.query(GameSession, CognitiveScore).join(CognitiveScore).filter(
                GameSession.user_id == user_id,
                GameSession.completed == True
            ).order_by(GameSession.timestamp.asc()).all()
            
            if len(sessions) < 5:
                return jsonify({
                    'error': 'Insufficient data for risk assessment',
                    'message': 'At least 5 completed sessions required',
                    'current_sessions': len(sessions)
                }), 400
            
            # Prepare session data
            session_data = []
            for game_session, cog_score in sessions:
                session_data.append({
                    'user_id': user_id,
                    'session_number': len(session_data),
                    'total_matches': game_session.total_matches,
                    'total_errors': game_session.total_errors,
                    'hints_used': game_session.hints_used,
                    'avg_reaction_time': cog_score.avg_reaction_time,
                    'accuracy_rate': cog_score.accuracy_rate,
                    'sac_score': cog_score.sac_score,
                    'ies_score': cog_score.ies_score
                })
            
            # Load LSTM model
            try:
                lstm_model = LSTMDementiaPredictor.load_model('models/lstm_model')
            except Exception as e:
                return jsonify({
                    'error': 'LSTM model not available',
                    'message': 'Please train the model first'
                }), 503
            
            # Get LSTM prediction
            X_lstm, _, _ = lstm_model.prepare_sequences(session_data)
            if len(X_lstm) == 0:
                return jsonify({
                    'error': 'Insufficient data for LSTM',
                    'message': 'Need at least 10 sessions for temporal analysis'
                }), 400
            
            lstm_score = float(lstm_model.predict(X_lstm)[-1])  # Latest prediction
            
            # Load risk classifier
            try:
                risk_classifier = DementiaRiskClassifier.load_model('models/risk_classifier')
            except Exception as e:
                return jsonify({
                    'error': 'Risk classifier not available',
                    'message': 'Please train the classifier first'
                }), 503
            
            # Prepare features for risk classifier
            cognitive_scores = [{'sac_score': s['sac_score'], 
                               'accuracy_rate': s['accuracy_rate'],
                               'ies_score': s['ies_score']} for s in session_data]
            
            scorer = CognitiveScorer()
            temporal_features = scorer.calculate_temporal_features(cognitive_scores)
            temporal_features['lstm_decline_score'] = lstm_score
            
            # Get risk prediction
            risk_label, probabilities = risk_classifier.predict_single(temporal_features)
            confidence = float(max(probabilities))
            
            # Determine if alert should be triggered
            alert_triggered = risk_label in ['Medium', 'High']
            
            # Save assessment
            assessment = RiskAssessment(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                lstm_decline_score=lstm_score,
                risk_category=risk_label,
                confidence_score=confidence,
                alert_triggered=alert_triggered,
                sessions_analyzed=len(session_data)
            )
            
            db.add(assessment)
            db.commit()
            
            # Return assessment
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'assessment': {
                    'assessment_id': assessment.assessment_id,
                    'timestamp': assessment.timestamp.isoformat(),
                    'risk_category': risk_label,
                    'lstm_decline_score': round(lstm_score, 4),
                    'confidence_score': round(confidence, 4),
                    'alert_triggered': alert_triggered,
                    'sessions_analyzed': len(session_data),
                    'probabilities': {
                        'Low': round(float(probabilities[0]), 4),
                        'Medium': round(float(probabilities[1]), 4),
                        'High': round(float(probabilities[2]), 4)
                    }
                },
                'recommendation': self._get_recommendation(risk_label, lstm_score)
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

   """Generate recommendation based on risk assessment"""
def _get_recommendation(risk_category, lstm_score):
    recommendations = {
        'Low': {
            'message': 'Cognitive function appears stable. Continue regular monitoring.',
            'actions': [
                'Continue regular gameplay sessions',
                'Maintain cognitive stimulation activities',
                'Schedule routine check-up in 3 months'
            ]
        },
        'Medium': {
            'message': 'Mild cognitive changes detected. Increased monitoring recommended.',
            'actions': [
                'Increase gameplay frequency to 3-4 times per week',
                'Consider additional cognitive assessments (MMSE/MoCA)',
                'Monitor for changes in daily activities',
                'Schedule follow-up in 1 month'
            ]
        },
        'High': {
            'message': 'Significant cognitive decline indicators. Professional evaluation strongly recommended.',
            'actions': [
                'Schedule urgent appointment with healthcare provider',
                'Consider comprehensive neurological evaluation',
                'Discuss with family members about care planning',
                'Increase monitoring frequency'
            ]
        }
    }
    
    return recommendations.get(risk_category, recommendations['Low'])