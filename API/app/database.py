from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Float, Text
from sqlalchemy.sql import func
import datetime
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from app.models import HistoryCreate

from fastapi import FastAPI

# =============================================================================
# DATABASE
# =============================================================================

DATABASE_URL = 'sqlite+aiosqlite:///./history.db'

engine = create_async_engine(
    DATABASE_URL, 
    echo=True
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass


class History(Base):

    __tablename__ = 'request_history'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
        )
    
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processing_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    mood: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


@asynccontextmanager
async def lifespan(app: FastAPI):

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await engine.dispose()

async def get_db() -> AsyncGenerator[AsyncSession, None]:

    async with async_session_maker() as session:

        try:
            yield session
        finally:
            await session.close()

async def history_to_db(history_data: HistoryCreate, db: AsyncSession) -> History:
    history = History(
        user_id=history_data.user_id,
        user_agent=history_data.user_agent,
        file_name=history_data.file_name,
        file_size=history_data.file_size,
        processing_time=history_data.processing_time,
        mood=history_data.mood,
        error_message=history_data.error_message
    )

    db.add(history)
    await db.commit()
    await db.refresh(history)
    return history