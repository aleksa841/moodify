from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
import io
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# импорты pydantic моделей
from app.models import HistoryCreate, HistoryGet, ModelResultGet, StatsGet

# импорты БД setup, lifespan
from app.database import History, lifespan, get_db, history_to_db

# импорты для извлечения фичей и предсказания настроения
from app.extract_audio_features import features_extraction
from app.ml_model import get_prediction

# импорт для получения статистики по БД
from app.request_stats import get_stats


app = FastAPI(
    title='FastAPI Moodify',
    description='Классификация треков по настроению',
    lifespan=lifespan
)

@app.post(
        '/forward', 
        response_model=ModelResultGet,
        summary='Получить настроение трека'
        )

async def forward(
    audio: UploadFile = File(),
    user_id: int = Header(default=0),
    user_agent: str = Header(default=''),
    db: AsyncSession = Depends(get_db), 
):
    
    start_time = time.perf_counter()

    if audio is None or not audio.content_type.startswith('audio/'):

        processing_time = (time.perf_counter() - start_time)

        # формируем данные с ошибкой и отправляем в бд
        history_data = HistoryCreate(
            user_id=user_id,
            user_agent=user_agent,
            file_name=audio.filename if audio is not None else None,
            file_size=None,
            processing_time=processing_time,
            mood=None,
            error_message='no_audiofile',
        )

        await history_to_db(history_data, db)

        raise HTTPException(
                status_code=400,
                detail='Bad Request'
        )
    
    audio_bytes = await audio.read()
    file_size = len(audio_bytes)

    audio_ = io.BytesIO(audio_bytes)
    # извлекаем аудио-фичи
    df_features = features_extraction(audio_)

    # получаем предсказание настроения
    label = get_prediction(df_features)

    if label is None:
        processing_time = time.perf_counter() - start_time

        # формируем данные с ошибкой и отправляем в бд
        history_data = HistoryCreate(
            user_id=user_id,
            user_agent=user_agent,
            file_name=audio.filename,
            file_size=file_size,
            processing_time=processing_time,
            mood=None,
            error_message='model_failed',
        )

        await history_to_db(history_data, db)

        raise HTTPException(
            status_code=403,
            detail='Модель не смогла обработать данные'
        )
    
    processing_time = time.perf_counter() - start_time

    history_data = HistoryCreate(
        user_id=user_id,
        user_agent=user_agent,
        file_name=audio.filename,
        file_size=file_size,
        processing_time=processing_time,
        mood=str(label),
        error_message=None,
    )

    await history_to_db(history_data, db)

    # несколько признаков для примера
    features_example = df_features.iloc[0, 0:30].round(4).to_dict() 
    response_data = {
        'headers': {
            'user_id': user_id, 
            'user_agent': user_agent},
        'mood': label,
        'features': features_example
    }
    return response_data


@app.get(
    '/history', 
    response_model=list[HistoryGet],
    summary='Получить историю запросов'
)
async def history(
    skip: int = 0, 
    limit: int = 100,
    db: AsyncSession = Depends(get_db), 
):
    
    result = await db.execute(
        select(History).offset(skip).limit(limit)
    )

    return list(result.scalars().all())


@app.get(
    '/stats', 
    response_model=StatsGet,
    summary='Получить статистику запросов'
)
async def stats(
    db: AsyncSession = Depends(get_db)
):
    # получаем данные из БД, отдельно по запросам с ошибками и без
    result_ok = await db.execute(
        select(History.processing_time)\
            .where(History.processing_time.is_not(None))\
            .where(History.error_message.is_(None))
    )

    result_error = await db.execute(
        select(History.processing_time)\
            .where(History.processing_time.is_not(None))\
            .where(History.error_message.is_not(None))
    )

    result_size = await db.execute(
        select(History.file_size)\
            .where(History.file_size.is_not(None))
    )
    
    result_ok_list = list(result_ok.scalars().all())
    result_error_list = list(result_error.scalars().all())
    result_size_list = list(result_size.scalars().all())
    
    return {
        'stats_time_by_requests_ok': get_stats(result_ok_list),
        'stats_time_by_requests_error': get_stats(result_error_list),
        'stats_file_size_in_bytes': get_stats(result_size_list)
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'app.main:app',
        host='127.0.0.1',
        port=8000,
        reload=True,
        log_level='info'
    )