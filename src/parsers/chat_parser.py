"""
Chat Message Parser

Parses text and audio input from mobile chat interface.

Author: Research Team
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    message_id: str
    user_id: str
    text: str
    timestamp: datetime
    audio_features: Optional[Dict] = None
    message_type: str = 'text'  # 'text' or 'voice'


@dataclass
class ChatSession:
    """Represents a complete chat session."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages: List[ChatMessage] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class ChatParser:
    """Parse chat data from mobile app."""
    
    @staticmethod
    def parse_message(message_data: Dict) -> ChatMessage:
        """
        Parse a single chat message.
        
        Args:
            message_data: Dictionary with message data
            
        Returns:
            ChatMessage object
        """
        return ChatMessage(
            message_id=message_data.get('id', ''),
            user_id=message_data.get('user_id', ''),
            text=message_data.get('text', ''),
            timestamp=message_data.get('timestamp', datetime.now()),
            audio_features=message_data.get('audio_features'),
            message_type=message_data.get('type', 'text')
        )
    
    @staticmethod
    def parse_session(session_data: Dict) -> ChatSession:
        """
        Parse a complete chat session.
        
        Args:
            session_data: Dictionary with session data
            
        Returns:
            ChatSession object
        """
        session = ChatSession(
            session_id=session_data.get('id', ''),
            user_id=session_data.get('user_id', ''),
            start_time=session_data.get('start_time', datetime.now()),
            end_time=session_data.get('end_time')
        )
        
        # Parse all messages in session
        messages_data = session_data.get('messages', [])
        for msg_data in messages_data:
            message = ChatParser.parse_message(msg_data)
            session.messages.append(message)
        
        return session
    
    @staticmethod
    def extract_text_from_session(session: ChatSession) -> str:
        """
        Extract combined text from all messages.
        
        Args:
            session: ChatSession object
            
        Returns:
            Combined text from all messages
        """
        texts = [msg.text for msg in session.messages if msg.text]
        return ' '.join(texts)


__all__ = ["ChatMessage", "ChatSession", "ChatParser"]
