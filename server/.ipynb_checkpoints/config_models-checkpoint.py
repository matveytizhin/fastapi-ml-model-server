from pydantic import BaseModel, Field
from typing import List, Literal, Union

class ModelNameConfig(BaseModel):
    '''
    Базовая конфигурация модели
    '''
    model_name: str = Field(..., max_length=50, description="Уникальное имя модели")


class LogisticRegressionConfig(ModelNameConfig):
    """Конфигурация для обучения sklearn.linear_model.LogisticRegression."""
    
    model_type: Literal['LogisticRegression']
    
    penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = 'l2'
    C: float = Field(1.0, gt=0)
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = 'lbfgs'


class RandomForestConfig(ModelNameConfig):
    """Конфигурация для обучения sklearn.ensemble.RandomForestClassifier."""

    model_type: Literal['RandomForestClassifier']
    
    n_estimators: int = Field(100, gt=0)
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'


class XGBoostConfig(ModelNameConfig):
    """Конфигурация для обучения xgboost.XGBClassifier."""

    model_type: Literal['XGBClassifier']
    
    n_estimators: int = Field(100, gt=0)
    learning_rate: float = Field(0.1, gt=0)
    max_depth: int = Field(3, ge=0)


class LightGBMConfig(ModelNameConfig):
    """Конфигурация для обучения lightgbm.LGBMClassifier."""

    model_type: Literal['LGBMClassifier']
    
    n_estimators: int = Field(100, gt=0)
    learning_rate: float = Field(0.1, gt=0)
    num_leaves: int = Field(31, gt=1)

class CatBoostConfig(ModelNameConfig):
    """Конфигурация для обучения catboost.CatBoostClassifier."""
    model_type: Literal['CatBoostClassifier']
    
    iterations: int = Field(1000, gt=0)
    learning_rate: float = Field(0.03, gt=0)
    depth: int = Field(6, ge=1, le=16)
    verbose: bool = False

AnyModelConfig = Union[
    LogisticRegressionConfig,
    RandomForestConfig,
    XGBoostConfig,
    LightGBMConfig,
    CatBoostConfig,
]

class FitRequest(BaseModel):
    '''Pydantic-модель для тела POST-запроса на эндпоинт /fit''' 
    X: List[List[float]] = Field(..., description="Матрица признаков для обучения")
    y: List[int] = Field(..., description="Вектор целевой переменной")

    config: AnyModelConfig = Field(..., discriminator='model_type')
    

class PredictRequest(BaseModel):
    '''Pydantic-модель для тела POST-запроса на эндпоинт /predict'''
    X: List[List[float]] = Field(...)
    config: ModelNameConfig = Field(...)
    












    