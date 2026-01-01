from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid

from src.database.models import User, Caregiver
from src.database.connection import get_db_context

user_bp = Blueprint('user', __name__, url_prefix='/api/users')

"""register a new user"""
@user_bp.route('/register', methods=['POST'])
def register_user():
    try:
        data = request.json
        name = data.get('name')
        age = data.get('age')
        
        if not name or not age:
            return jsonify({'error': 'name and age are required'}), 400
        
        if age < 0 or age > 120:
            return jsonify({'error': 'Invalid age'}), 400
        
        # Generate user ID
        user_id = f"USR_{uuid.uuid4().hex[:8]}"
        
        with get_db_context() as db:
            # Check if caregiver exists (if provided)
            caregiver_id = data.get('caregiver_id')
            if caregiver_id:
                caregiver = db.query(Caregiver).filter(
                    Caregiver.caregiver_id == caregiver_id
                ).first()
                if not caregiver:
                    return jsonify({'error': 'Caregiver not found'}), 404
            
            # Create new user
            new_user = User(
                user_id=user_id,
                name=name,
                age=age,
                caregiver_id=caregiver_id,
                date_enrolled=datetime.utcnow()
            )
            
            db.add(new_user)
            db.commit()
            
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'name': name,
                'age': age,
                'caregiver_id': caregiver_id,
                'date_enrolled': new_user.date_enrolled.isoformat()
            }), 201
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""Get user profile"""
@user_bp.route('/<user_id>/profile', methods=['GET'])
def get_user_profile(user_id):
    try:
        with get_db_context() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get session count
            from src.database.models import GameSession
            session_count = db.query(GameSession).filter(
                GameSession.user_id == user_id
            ).count()
            
            return jsonify({
                'user_id': user.user_id,
                'name': user.name,
                'age': user.age,
                'date_enrolled': user.date_enrolled.isoformat(),
                'caregiver_id': user.caregiver_id,
                'motor_baseline_rt': user.motor_baseline_rt,
                'session_count': session_count
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    """Update user profile"""
@user_bp.route('/<user_id>/profile', methods=['PUT'])
def update_user_profile(user_id):
    try:
        data = request.json
        
        with get_db_context() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Update fields if provided
            if 'name' in data:
                user.name = data['name']
            if 'age' in data:
                user.age = data['age']
            if 'caregiver_id' in data:
                # Verify caregiver exists
                if data['caregiver_id']:
                    caregiver = db.query(Caregiver).filter(
                        Caregiver.caregiver_id == data['caregiver_id']
                    ).first()
                    if not caregiver:
                        return jsonify({'error': 'Caregiver not found'}), 404
                user.caregiver_id = data['caregiver_id']
            
            db.commit()
            
            return jsonify({
                'status': 'success',
                'user_id': user.user_id,
                'name': user.name,
                'age': user.age,
                'caregiver_id': user.caregiver_id
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    """ Set motor baseline for user from calibration task"""
@user_bp.route('/calibrate', methods=['POST'])
def calibrate_motor_baseline():
    try:
        data = request.json
        user_id = data.get('user_id')
        reaction_times = data.get('reaction_times', [])
        
        if not user_id or not reaction_times:
            return jsonify({'error': 'user_id and reaction_times required'}), 400
        
        # Calculate baseline as median of calibration trials
        import numpy as np
        baseline_rt = float(np.median(reaction_times))
        
        with get_db_context() as db:
            user = db.query(User).filter(User.user_id == user_id).first()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            user.motor_baseline_rt = baseline_rt
            db.commit()
            
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'motor_baseline_rt': baseline_rt,
                'calibration_trials': len(reaction_times)
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

 """Register a new caregiver"""
@user_bp.route('/caregiver/register', methods=['POST'])
def register_caregiver():
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        
        if not name or not email:
            return jsonify({'error': 'name and email are required'}), 400
        
        # Generate caregiver ID
        caregiver_id = f"CG_{uuid.uuid4().hex[:8]}"
        
        with get_db_context() as db:
            # Check if email already exists
            existing = db.query(Caregiver).filter(
                Caregiver.email == email
            ).first()
            
            if existing:
                return jsonify({'error': 'Email already registered'}), 409
            
            # Create new caregiver
            new_caregiver = Caregiver(
                caregiver_id=caregiver_id,
                name=name,
                email=email,
                phone=data.get('phone'),
                date_registered=datetime.utcnow()
            )
            
            db.add(new_caregiver)
            db.commit()
            
            return jsonify({
                'status': 'success',
                'caregiver_id': caregiver_id,
                'name': name,
                'email': email,
                'phone': data.get('phone')
            }), 201
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


"""Get all patients for a caregiver"""
@user_bp.route('/caregiver/<caregiver_id>/patients', methods=['GET'])
def get_caregiver_patients(caregiver_id):
    try:
        with get_db_context() as db:
            caregiver = db.query(Caregiver).filter(
                Caregiver.caregiver_id == caregiver_id
            ).first()
            
            if not caregiver:
                return jsonify({'error': 'Caregiver not found'}), 404
            
            patients = db.query(User).filter(
                User.caregiver_id == caregiver_id
            ).all()
            
            patient_list = [{
                'user_id': p.user_id,
                'name': p.name,
                'age': p.age,
                'date_enrolled': p.date_enrolled.isoformat()
            } for p in patients]
            
            return jsonify({
                'caregiver_id': caregiver_id,
                'caregiver_name': caregiver.name,
                'patient_count': len(patient_list),
                'patients': patient_list
            }), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500