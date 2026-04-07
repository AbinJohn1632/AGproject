import uuid, json, os, time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Chat"
    alias: str = ""
    buffer: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class SessionManager:
    def __init__(self, storage_file: str = "sessions.json"):
        self.storage_file = storage_file
        self.sessions: Dict[str, Session] = {}
        self.active_session_id: Optional[str] = None
        self.load()

    def create_session(self, name: str = None) -> Session:
        if not name:
            name = f"Chat {time.strftime('%Y-%m-%d %H:%M')}"
        session = Session(name=name)
        self.sessions[session.id] = session
        self.active_session_id = session.id
        self.save()
        return session

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session_id == session_id:
                self.active_session_id = None
            self.save()

    def rename_session(self, session_id: str, new_name: str):
        if session_id in self.sessions:
            self.sessions[session_id].name = new_name
            self.save()

    def set_session_alias(self, session_id: str, alias: str):
        if session_id in self.sessions:
            self.sessions[session_id].alias = alias
            self.save()

    def update_session_buffer(self, session_id: str, buffer: List[str]):
        if session_id in self.sessions:
            self.sessions[session_id].buffer = buffer
            self.sessions[session_id].timestamp = time.time()
            self.save()

    def save(self):
        data = {
            "active_session_id": self.active_session_id,
            "sessions": [s.to_dict() for s in self.sessions.values()]
        }
        with open(self.storage_file, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    self.active_session_id = data.get("active_session_id")
                    for s_data in data.get("sessions", []):
                        s = Session.from_dict(s_data)
                        self.sessions[s.id] = s
            except Exception as e:
                print(f"Error loading sessions: {e}")
                self.sessions = {}
                self.active_session_id = None
        
        # Ensure at least one session exists
        if not self.sessions:
            self.create_session("Initial Session")

    def get_active_session(self) -> Optional[Session]:
        if self.active_session_id and self.active_session_id in self.sessions:
            return self.sessions[self.active_session_id]
        return None

    def set_active_session(self, session_id: str):
        if session_id in self.sessions:
            self.active_session_id = session_id
            self.save()
