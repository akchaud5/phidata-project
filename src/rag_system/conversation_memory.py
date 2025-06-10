import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    id: str
    session_id: str
    user_message: str
    assistant_response: str
    context_used: List[Dict[str, Any]]
    citations: List[str]
    timestamp: str
    response_quality: float
    search_query: Optional[str] = None
    search_results_count: Optional[int] = None

@dataclass
class ConversationSession:
    id: str
    user_id: Optional[str]
    title: str
    created_at: str
    updated_at: str
    turns: List[ConversationTurn]
    context_summary: str
    total_turns: int
    is_active: bool

class ConversationMemory:
    def __init__(self, storage_path: str = "./data/conversations.json", 
                 max_sessions: int = 100, session_ttl_days: int = 30):
        self.storage_path = storage_path
        self.max_sessions = max_sessions
        self.session_ttl_days = session_ttl_days
        self.sessions: Dict[str, ConversationSession] = {}
        self.load_conversations()
    
    def load_conversations(self):
        """Load conversations from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for session_data in data:
                    # Convert turns back to objects
                    turns = [ConversationTurn(**turn_data) for turn_data in session_data['turns']]
                    session_data['turns'] = turns
                    session = ConversationSession(**session_data)
                    self.sessions[session.id] = session
            
            # Clean up expired sessions
            self._cleanup_expired_sessions()
            
            logger.info(f"Loaded {len(self.sessions)} conversation sessions")
        except FileNotFoundError:
            logger.info("No existing conversations file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
    
    def save_conversations(self):
        """Save conversations to storage"""
        try:
            # Convert to serializable format
            sessions_data = []
            for session in self.sessions.values():
                session_dict = asdict(session)
                sessions_data.append(session_dict)
            
            with open(self.storage_path, 'w') as f:
                json.dump(sessions_data, f, indent=2)
            
            logger.info(f"Saved {len(self.sessions)} conversation sessions")
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def create_session(self, user_id: Optional[str] = None, title: str = None) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        
        if not title:
            title = f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ConversationSession(
            id=session_id,
            user_id=user_id,
            title=title,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            turns=[],
            context_summary="",
            total_turns=0,
            is_active=True
        )
        
        self.sessions[session_id] = session
        self._enforce_session_limit()
        self.save_conversations()
        
        logger.info(f"Created new conversation session: {session_id}")
        return session_id
    
    def add_turn(self, session_id: str, user_message: str, assistant_response: str,
                 context_used: List[Dict[str, Any]] = None, citations: List[str] = None,
                 response_quality: float = 0.0, search_query: str = None,
                 search_results_count: int = None) -> str:
        """Add a turn to a conversation session"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new one")
            session_id = self.create_session()
        
        session = self.sessions[session_id]
        turn_id = str(uuid.uuid4())
        
        turn = ConversationTurn(
            id=turn_id,
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            context_used=context_used or [],
            citations=citations or [],
            timestamp=datetime.now().isoformat(),
            response_quality=response_quality,
            search_query=search_query,
            search_results_count=search_results_count
        )
        
        session.turns.append(turn)
        session.total_turns += 1
        session.updated_at = datetime.now().isoformat()
        
        # Update context summary
        self._update_context_summary(session)
        
        self.save_conversations()
        
        logger.info(f"Added turn to session {session_id}")
        return turn_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session"""
        return self.sessions.get(session_id)
    
    def get_session_history(self, session_id: str, last_n_turns: int = None) -> List[ConversationTurn]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if not session:
            return []
        
        if last_n_turns:
            return session.turns[-last_n_turns:]
        return session.turns
    
    def get_context_for_session(self, session_id: str, max_context_length: int = 2000) -> str:
        """Get relevant context from conversation history"""
        session = self.get_session(session_id)
        if not session:
            return ""
        
        # Start with session summary
        context_parts = []
        if session.context_summary:
            context_parts.append(f"Previous conversation summary: {session.context_summary}")
        
        # Add recent turns
        recent_turns = session.turns[-3:]  # Last 3 turns for immediate context
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response[:200]}...")
        
        # Combine and truncate if needed
        full_context = "\n".join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        return full_context
    
    def search_conversations(self, query: str, user_id: Optional[str] = None) -> List[ConversationTurn]:
        """Search through conversation history"""
        results = []
        query_lower = query.lower()
        
        for session in self.sessions.values():
            # Filter by user if specified
            if user_id and session.user_id != user_id:
                continue
            
            for turn in session.turns:
                # Search in user messages and assistant responses
                if (query_lower in turn.user_message.lower() or 
                    query_lower in turn.assistant_response.lower()):
                    results.append(turn)
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda t: t.timestamp, reverse=True)
        return results
    
    def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[ConversationSession]:
        """Get all sessions for a user"""
        sessions = [
            session for session in self.sessions.values()
            if session.user_id == user_id and (not active_only or session.is_active)
        ]
        
        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].title = title
        self.sessions[session_id].updated_at = datetime.now().isoformat()
        self.save_conversations()
        return True
    
    def deactivate_session(self, session_id: str) -> bool:
        """Deactivate a session"""
        if session_id not in self.sessions:
            return False
        
        self.sessions[session_id].is_active = False
        self.sessions[session_id].updated_at = datetime.now().isoformat()
        self.save_conversations()
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id not in self.sessions:
            return False
        
        del self.sessions[session_id]
        self.save_conversations()
        logger.info(f"Deleted session {session_id}")
        return True
    
    def get_conversation_analytics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics about conversations"""
        relevant_sessions = [
            session for session in self.sessions.values()
            if not user_id or session.user_id == user_id
        ]
        
        if not relevant_sessions:
            return {}
        
        total_sessions = len(relevant_sessions)
        total_turns = sum(session.total_turns for session in relevant_sessions)
        active_sessions = sum(1 for session in relevant_sessions if session.is_active)
        
        # Calculate average response quality
        all_turns = [turn for session in relevant_sessions for turn in session.turns]
        avg_quality = sum(turn.response_quality for turn in all_turns) / len(all_turns) if all_turns else 0
        
        # Most active session
        most_active_session = max(relevant_sessions, key=lambda s: s.total_turns) if relevant_sessions else None
        
        # Recent activity
        now = datetime.now()
        recent_sessions = [
            session for session in relevant_sessions
            if datetime.fromisoformat(session.updated_at) > now - timedelta(days=7)
        ]
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_turns': total_turns,
            'average_turns_per_session': total_turns / total_sessions if total_sessions > 0 else 0,
            'average_response_quality': avg_quality,
            'most_active_session': {
                'id': most_active_session.id,
                'title': most_active_session.title,
                'turns': most_active_session.total_turns
            } if most_active_session else None,
            'recent_activity': len(recent_sessions),
            'sessions_last_7_days': len(recent_sessions)
        }
    
    def _update_context_summary(self, session: ConversationSession):
        """Update the context summary for a session"""
        if len(session.turns) < 3:
            return
        
        # Create a summary of the conversation so far
        topics = set()
        key_points = []
        
        for turn in session.turns:
            # Extract potential topics (simple keyword extraction)
            words = turn.user_message.lower().split()
            for word in words:
                if len(word) > 5 and word.isalpha():  # Potential topic words
                    topics.add(word)
        
        # Create summary
        topics_list = list(topics)[:5]  # Top 5 topics
        session.context_summary = f"Topics discussed: {', '.join(topics_list)}. Total turns: {session.total_turns}"
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        cutoff_date = datetime.now() - timedelta(days=self.session_ttl_days)
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if datetime.fromisoformat(session.updated_at) < cutoff_date:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _enforce_session_limit(self):
        """Ensure we don't exceed the maximum number of sessions"""
        if len(self.sessions) <= self.max_sessions:
            return
        
        # Sort sessions by updated_at and remove oldest
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1].updated_at
        )
        
        sessions_to_remove = len(self.sessions) - self.max_sessions
        for i in range(sessions_to_remove):
            session_id = sorted_sessions[i][0]
            del self.sessions[session_id]
        
        logger.info(f"Removed {sessions_to_remove} old sessions to enforce limit")
    
    def export_session(self, session_id: str, format: str = 'json') -> str:
        """Export a session in specified format"""
        session = self.get_session(session_id)
        if not session:
            return ""
        
        if format.lower() == 'json':
            return json.dumps(asdict(session), indent=2)
        
        elif format.lower() == 'markdown':
            md_content = f"# {session.title}\n\n"
            md_content += f"**Created:** {session.created_at}\n"
            md_content += f"**Total Turns:** {session.total_turns}\n\n"
            
            for i, turn in enumerate(session.turns, 1):
                md_content += f"## Turn {i}\n\n"
                md_content += f"**User:** {turn.user_message}\n\n"
                md_content += f"**Assistant:** {turn.assistant_response}\n\n"
                if turn.citations:
                    md_content += f"**Citations:** {', '.join(turn.citations)}\n\n"
                md_content += "---\n\n"
            
            return md_content
        
        else:
            return str(asdict(session))
    
    def clear_all_conversations(self):
        """Clear all conversation data"""
        self.sessions.clear()
        self.save_conversations()
        logger.info("Cleared all conversation data")