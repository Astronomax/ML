import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


continuous_columns = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add', 'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Total_Bsmt_SF', 'First_Flr_SF', 'Second_Flr_SF', 'Low_Qual_Fin_SF', 'Gr_Liv_Area', 'Bsmt_Full_Bath', 'Bsmt_Half_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Garage_Cars', 'Garage_Area', 'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch', 'Three_season_porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val', 'Mo_Sold', 'Year_Sold', 'Longitude', 'Latitude']
categorical_columns = ['MS_SubClass', 'MS_Zoning', 'Street', 'Alley', 'Lot_Shape', 'Land_Contour', 'Utilities', 'Lot_Config', 'Land_Slope', 'Neighborhood', 'Condition_1', 'Condition_2', 'Bldg_Type', 'House_Style', 'Overall_Qual', 'Overall_Cond', 'Roof_Style', 'Roof_Matl', 'Exterior_1st', 'Exterior_2nd', 'Mas_Vnr_Type', 'Exter_Qual', 'Exter_Cond', 'Foundation', 'Bsmt_Qual', 'Bsmt_Cond', 'Bsmt_Exposure', 'BsmtFin_Type_1', 'BsmtFin_Type_2', 'Heating', 'Heating_QC', 'Central_Air', 'Electrical', 'Kitchen_Qual', 'Functional', 'Fireplace_Qu', 'Garage_Type', 'Garage_Finish', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive', 'Pool_QC', 'Fence', 'Misc_Feature', 'Sale_Type', 'Sale_Condition']


class BaseDataPreprocessor(TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]]=None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()
        pass

    def fit(self, data, *args):
        self.scaler.fit(data if self.needed_columns is None else data[self.needed_columns])
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        return self.scaler.transform(data if self.needed_columns is None else data[self.needed_columns])


class OneHotPreprocessor(BaseDataPreprocessor):
    def __init__(self, **kwargs):
        super(OneHotPreprocessor, self).__init__(continuous_columns, **kwargs)
        self.one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    def fit(self, data, *args):
        super().fit(data)
        self.one_hot_encoder.fit(data[categorical_columns])
        return self

    def transform(self, data):
        return np.concatenate((super().transform(data), self.one_hot_encoder.transform(data[categorical_columns])), axis=1)



def make_ultimate_pipeline():
    return Pipeline([('onehot', OneHotPreprocessor()),
                     ('ridge', Ridge())])