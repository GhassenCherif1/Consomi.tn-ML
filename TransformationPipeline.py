
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from LogScaling import LogScaling
class TransformationPipeline:

    def __init__(self) -> None:
        pass
    
    def preprocess(self,d):
        cat_cols = d.select_dtypes('object').columns
        num_cols = d.select_dtypes(include=['float64',"int64",'float32']).columns

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        ) 

        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )

        preprocessor = ColumnTransformer([
            ("log_transform", LogScaling(), num_cols),
            ("num_pipeline", num_pipeline, num_cols),
            ("cat_pipelines",cat_pipeline,cat_cols)
            ], remainder= 'passthrough')
        
        return preprocessor