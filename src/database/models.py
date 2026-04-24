"""
Модели базы данных PostgreSQL.

Хранит: пользователей, сессии, логи, метрики.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Text, Float, Integer, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(255))
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    sessions = relationship("Session", back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    agent_version = Column(String(50))
    queries_count = Column(Integer, default=0)

    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), ForeignKey("sessions.id"), nullable=False)
    query = Column(Text, nullable=False)
    response = Column(Text)
    confidence = Column(Float)
    documents_used = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    was_clarification = Column(Boolean, default=False)

    session = relationship("Session", back_populates="interactions")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
