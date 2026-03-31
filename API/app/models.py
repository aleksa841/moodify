from typing import Optional, AsyncGenerator
from pydantic import BaseModel
import datetime


class HistoryCreate(BaseModel):
    user_id: Optional[int] = None
    user_agent: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None
    mood: Optional[str] = None
    error_message: Optional[str] = None

class HistoryGet(BaseModel):
    id: int
    created_at: datetime.datetime
    user_id: Optional[int] = None
    user_agent: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None
    mood: Optional[str] = None
    error_message: Optional[str] = None

class Headears(BaseModel):
    user_id: int
    user_agent: str

class ModelResultGet(BaseModel):
    headers: Headears
    mood: str
    features: dict

class StatsPart(BaseModel):
    count: int
    mean: float
    median: float
    q95: float
    q99: float

class StatsGet(BaseModel):
    stats_time_by_requests_ok: StatsPart
    stats_time_by_requests_error: StatsPart
    stats_file_size_in_bytes: StatsPart