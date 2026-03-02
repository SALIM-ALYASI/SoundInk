import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from core.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    
    # User's wallet (characters they can synthesize)
    token_balance = Column(Integer, default=50000) 
    
    # Track when they joined
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to history
    generations = relationship("GenerationHistory", back_populates="owner")

class GenerationHistory(Base):
    __tablename__ = "generation_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"))
    
    # The session_id correlates to the folder in storage/sessions/
    session_id = Column(String, unique=True, index=True)
    voice_id = Column(String)
    
    # How much it cost
    characters_used = Column(Integer, default=0)
    
    # Has this been deleted during cleanup?
    media_deleted = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="generations")
