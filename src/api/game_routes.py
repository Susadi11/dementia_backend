from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
from sqlalchemy.orm import Session

from src.database.models import GameSession, GameAction, CognitiveScore, User
from src.database.connection import get_db_context
from src.features.cognitive_scoring import CognitiveScorer
from src.features.game_features import GameFeatureExtractor

game_bp = Blueprint('game', __name__, url_prefix='/api/game')

@game_bp.route('/session/start', methods=['POST'])
def start_session():
    """
    Initialize a new game session
    
    Request body:
    {
        "user_id": "USR001",
        "difficulty_level": 1,
        "total_cards": 12
    }
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        difficulty_level = data.get('difficulty_level', 1)
        total_cards = data.get('total_cards', 12)
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Generate session ID
        session_id = f"SES_{uuid.uuid4().hex[:12]}"
        
        with get_db_context() as db:
            # Verify user exists
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Create new session
            new_session = GameSession(
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                difficulty_level=difficulty_level,
                total_cards=total_cards,
                completed=False
            )
            
            db.add(new_session)
            db.commit()
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'user_id': user_id,
                'difficulty_level': difficulty_level,
                'total_cards': total_cards,
                'timestamp': new_session.timestamp.isoformat()
            }), 201
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@game_bp.route('/session/<session_id>/action', methods=['POST'])
def log_action(session_id):
    """
    Log a game action
    
    Request body:
    {
        "action_type": "match",
        "card_id": 5,
        "card_pair": "5,12",
        "reaction_time_ms": 2340,
        "is_correct": true
    }
    """
    try:
        data = request.json
        action_type = data.get('action_type')
        
        if not action_type:
            return jsonify({'error': 'action_type is required'}), 400
        
        with get_db_context() as db:
            # Verify session exists
            session = db.query(GameSession).filter(
                GameSession.session_id == session_id
            ).first()
            
            if not session:
                return jsonify({'error': 'Session not found'}), 404
            
            # Create action record
            action = GameAction(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                action_type=action_type,
                card_id=data.get('card_id'),
                card_pair=data.get('card_pair'),
                reaction_time_ms=data.get('reaction_time_ms', 0),
                is_correct=data.get('is_correct')
            )
            
            db.add(action)
            
            # Update session counters
            if action_type == 'match' and data.get('is_correct'):
                session.total_matches += 1
            elif action_type == 'error' or (action_type == 'match' and not data.get('is_correct')):
                session.total_errors += 1
            elif action_type == 'hint':
                session.hints_used += 1
            
            db.commit()
            
            return jsonify({
                'status': 'success',
                'action_id': action.action_id,
                'session_id': session_id
            }), 201
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@game_bp.route('/session/<session_id>/complete', methods=['POST'])
def complete_session(session_id):
    """
    Complete a game session and calculate cognitive scores
    
    Request body:
    {
        "session_duration": 125.5,
        "interruptions": 1
    }
    """
    try:
        data = request.json or {}
        session_duration = data.get('session_duration', 0)
        interruptions = data.get('interruptions', 0)
        
        with get_db_context() as db:
            # Get session with actions
            session = db.query(GameSession).filter(
                GameSession.session_id == session_id
            ).first()
            
            if not session:
                return jsonify({'error': 'Session not found'}), 404
            
            # Get user for motor baseline
            user = db.query(User).filter(User.user_id == session.user_id).first()
            motor_baseline = user.motor_baseline_rt if user else 0
            
            # Get all actions
            actions = db.query(GameAction).filter(
                GameAction.session_id == session_id
            ).all()
            
            # Extract features
            extractor = GameFeatureExtractor()
            action_data = [{
                'action_type': a.action_type,
                'timestamp': a.timestamp,
                'reaction_time_ms': a.reaction_time_ms,
                'is_correct': a.is_correct
            } for a in actions]
            
            features = extractor.extract_from_actions(action_data)
            
            # Calculate cognitive scores
            scorer = CognitiveScorer(motor_baseline=motor_baseline)
            
            session_data = {
                'total_matches': session.total_matches,
                'total_errors': session.total_errors,
                'total_attempts': session.total_matches + session.total_errors,
                'reaction_times': [a.reaction_time_ms for a in actions 
                                 if a.reaction_time_ms > 0],
                'hints_used': session.hints_used
            }
            
            scores = scorer.calculate_session_scores(session_data)
            
            # Update session
            session.session_duration = session_duration
            session.interruptions = interruptions
            session.completed = True
            
            # Save cognitive scores
            cognitive_score = CognitiveScore(
                session_id=session_id,
                user_id=session.user_id,
                timestamp=datetime.utcnow(),
                accuracy_rate=scores['accuracy_rate'],
                error_rate=scores['error_rate'],
                avg_reaction_time=scores['avg_reaction_time'],
                motor_adjusted_rt=scores['motor_adjusted_rt'],
                sac_score=scores['sac_score'],
                ies_score=scores['ies_score']
            )
            
            db.add(cognitive_score)
            db.commit()
            
            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'completed': True,
                'scores': scores,
                'features': features
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@game_bp.route('/session/<session_id>/status', methods=['GET'])
def get_session_status(session_id):
    """Get current session status"""
    try:
        with get_db_context() as db:
            session = db.query(GameSession).filter(
                GameSession.session_id == session_id
            ).first()
            
            if not session:
                return jsonify({'error': 'Session not found'}), 404
            
            action_count = db.query(GameAction).filter(
                GameAction.session_id == session_id
            ).count()
            
            return jsonify({
                'session_id': session.session_id,
                'user_id': session.user_id,
                'difficulty_level': session.difficulty_level,
                'total_cards': session.total_cards,
                'total_matches': session.total_matches,
                'total_errors': session.total_errors,
                'hints_used': session.hints_used,
                'action_count': action_count,
                'completed': session.completed,
                'timestamp': session.timestamp.isoformat()
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@game_bp.route('/sessions/user/<user_id>', methods=['GET'])
def get_user_sessions(user_id):
    """Get all sessions for a user"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        with get_db_context() as db:
            sessions = db.query(GameSession).filter(
                GameSession.user_id == user_id
            ).order_by(GameSession.timestamp.desc()).limit(limit).all()
            
            session_list = [{
                'session_id': s.session_id,
                'timestamp': s.timestamp.isoformat(),
                'difficulty_level': s.difficulty_level,
                'total_matches': s.total_matches,
                'total_errors': s.total_errors,
                'completed': s.completed
            } for s in sessions]
            
            return jsonify({
                'user_id': user_id,
                'session_count': len(session_list),
                'sessions': session_list
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500