# server/main.py   Доступен по http://127.0.0.1:8000/
import time
import os
import shutil
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression

# Импортируем классы моделей
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Импортируем наши Pydantic модели
from .config_models import FitRequest, PredictRequest, ModelNameConfig
from .config import settings

SUPPORTED_MODELS = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
    'CatBoostClassifier': CatBoostClassifier,
}


training_semaphore: asyncio.Semaphore = None

LOADED_MODELS=dict()

def _train_model_sync(
    model_name: str,
    model_path: str,
    model_type: str,
    hyperparameters: Dict,
    X: List[List[float]],
    y: List[int]
):
    """
    Это синхронная функция обучения модели, предназначения для запуска в ProcessPoolExecutor.
    """
    try:
        if os.path.exists(model_path):
            return f"Модель по пути '{model_path}' уже существует."
        
        if model_type not in SUPPORTED_MODELS:
            return f"Тип модели '{model_type}' не поддерживается."
        
        model_class = SUPPORTED_MODELS[model_type]   
        model = model_class(**hyperparameters)
        model.fit(X, y)
        joblib.dump(model, model_path)
        
        return f"Модель {model_name} успешно загружена и находится в {model_path}"
        
    except Exception as e:
        return f"Ошибка при обучении модели: {e}"




        
@asynccontextmanager
async def lifespan(app: FastAPI):
    global training_semaphore
    training_semaphore = asyncio.Semaphore(settings.TRAINING_CORES)
    app.state.process_pool = ProcessPoolExecutor(max_workers=1)
    print(f"Сервер запущен. Пул процессов на {settings.TRAINING_CORES} воркера(ов) создан.")
    
    yield
    
    app.state.process_pool.shutdown(wait=True) #Ждем пока все процессы завершат свою работу
    print("Пул процессов остановлен.")

app = FastAPI(title="ML Model Server", lifespan=lifespan)



#-------------------------------------------------------------------------#-------------------------------------------------------------------------

# Эндпоинты API

@app.get('/')
def read_root():
    return {"message": "Добро пожаловать на сервер для ML моделей!"}

@app.post('/fit')
async def fit_model(request: FitRequest):

    if training_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="Все вычислительные ресурсы для обучения заняты. Пожалуйста, повторите запрос позже."
        )
    
    os.makedirs(settings.MODELS_PATH, exist_ok=True)
    model_name = request.config.model_name
    model_path = os.path.join(settings.MODELS_PATH, f'{model_name}.joblib')
    
    model_type = request.config.model_type
    hyperparameters = request.config.model_dump(exclude={'model_name', 'model_type'})
    X = request.X
    y = request.y
    async with training_semaphore:
        print(f"Захвачен слот для обучения модели '{request.config.model_name}'.")
        loop = asyncio.get_running_loop()
        error_message = await loop.run_in_executor(
            app.state.process_pool,
            _train_model_sync,
            model_name,
            model_path,
            model_type,
            hyperparameters,
            X,
            y
        )
        print(f"Слот для обучения модели '{request.config.model_name}' освобожден.")


    if error_message:
        raise HTTPException(status_code=400, detail=error_message)

    return {
        "message": "Задача обучения модели запущена в фоновом режиме.",
        "model_name": model_name,
        "path": model_path
    }


@app.post('/predict')
def predict(request: PredictRequest):
    '''Функция, которая возвращает предсказания модели model_name'''
    model_name = request.config.model_name
    
    if model_name not in LOADED_MODELS:
        raise HTTPException(
            status_code=404, 
            detail=f"Модель '{model_name}' не загружена для инференса. Сначала вызовите эндпоинт /load."
        )
        
    model = LOADED_MODELS[model_name]

    try:
        predictions_array = model.predict(request.X)
        
        predictions = predictions_array.tolist()

        return {
            "model_name": model_name,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка при выполнении предсказания моделью '{model_name}': {e}"
        )

@app.post('/load')
def load_model(config: ModelNameConfig):
    '''Функция для загрузки обученной модели в память для инференса'''
    model_name = config.model_name
    model_path = os.path.join(settings.MODELS_PATH, f'{model_name}.joblib')
    
    if model_name in LOADED_MODELS:
        return {"message": f"Модель {model_name} уже загружена"}
        
    if len(LOADED_MODELS) >= settings.MAX_INFERENCE_MODELS:
        raise HTTPException(
            status_code=409, # Conflict
            detail=f"Достигнут лимит одновременно загруженных моделей ({settings.MAX_INFERENCE_MODELS}). "
                   f"Текущие загруженные модели: {list(LOADED_MODELS.keys())}. "
                   "Выгрузите одну из них перед загрузкой новой."
        )

        
    try:
        model = joblib.load(model_path)
        LOADED_MODELS[model_name] = model
        return {"message": f"Модель '{model_name}' успешно загружена в память."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить модель '{model_name}': {e}")

@app.post('/unload')
def unload_model(config: ModelNameConfig):
    '''Функция для выгрузки обученной модели из память'''
    model_name = config.model_name

    if model_name not in LOADED_MODELS:
        return {"message": f"Модель '{model_name}' уже выгружена из памяти."} 

    del LOADED_MODELS[model_name]
    return {"message": f"Модель '{model_name}' успешно выгружена из памяти."}

@app.post('/remove')
def remove_model(config: ModelNameConfig):
    """Удалить файл обученной модели с диска."""
    model_name = config.model_name

    if model_name in LOADED_MODELS:
        raise HTTPException(status_code=409, detail=f"Модель '{model_name}' сейчас загружена для инференса. Выгрузите ее перед удалением.")

    model_path = os.path.join(settings.MODELS_PATH, f'{model_name}.joblib')

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Файл модели '{model_name}' не найден на диске.")

    try:
        os.remove(model_path)
        return {"message": f"Файл модели '{model_name}' успешно удален с диска."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении файла модели: {e}")


@app.post('/remove_all')
def remove_all_models():
    if LOADED_MODELS:
        raise HTTPException(status_code=409, detail=f"Невозможно удалить все модели, так как некоторые из них загружены в память. Сначала выгрузите все модели.")

    folder = settings.MODELS_PATH
    if not os.path.isdir(folder):
        return {"message": "Директория с моделями не существует. Нечего удалять."}
    
    try:
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        
        return {"message": f"Все модели из директории '{folder}' были успешно удалены."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла ошибка при удалении директории с моделями: {e}")






    