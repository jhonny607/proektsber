import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.num_features = [col for col in df.columns if df[col].dtype != 'object' and len(df[col].unique()) <= 25 ]
        self.kat_features = [col for col in df.columns  if col not in self.num_features]

    def fill_void(self):
        num_col = self.num_features
        kat_col = self.kat_features
        for col in num_col:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        for col in kat_col:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def delete_corr(self, threshold=0.8):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_features = []
        for col in upper_triangle.columns:
            if any(upper_triangle[col] > threshold):
                corr_features.append(col)
        if corr_features:
            self.df = self.df.drop(columns=corr_features)
    def delete_errors(self):
        if 'full_sq' in self.df.columns:
            self.df = self.df[( self.df['full_sq'] >= 10) & (self.df['full_sq'] <= 500)]
        if 'full_sq' in self.df.columns and 'life_sq' in self.df.columns:
            self.df = self.df[(self.df['full_sq'] >= self.df['life_sq'])]
        if 'floor' in self.df.columns and 'max_floor' in self.df.columns:
            self.df = self.df[self.df['floor'] <= self.df['max_floor']]
        if 'kitch_sq' in self.df.columns and 'life_sq' in self.df.columns:
            self.df = self.df[self.df['kitch_sq'] < self.df['life_sq']]
        for col, znach in [['mkad_km',60],['ttk_km',30],['sadovoe_km',30]]:
            if col in self.df.columns:
                self.df = self.df[(self.df[col] < znach) & (self.df[col] >0)]
    def create_new_features(self):
        self.df['sq_rat'] = self.df['life_sq'] / (self.df['full_sq']+1)
        self.df['kitch_rat'] = self.df['kitch_sq'] / (self.df['life_sq']+1)
        self.df['floor_rat'] = self.df['floor'] / (self.df['max_floor']+1)
        self.df['first_floor'] = (self.df['floor'] <= 1).astype(int)
        self.df['room_sq'] = self.df['life_sq'] /( self.df['num_room']+1)
        self.df['density'] = self.df['num_room'] / (self.df['full_sq']+1)
        self.df['ageofbuilding'] = 2025 - self.df['build_year']
        self.df['new_build'] = (self.df['ageofbuilding'] <=10).astype(int)
        self.df['old_build'] = (self.df['ageofbuilding'] >=11).astype(int)
        self.df['lux'] = (self.df['full_sq'] * self.df['num_room']) / (self.df['ageofbuilding'] + 1)
        self.df['healthcare_access'] = self.df['hospital_beds_raion'] / (self.df['raion_popul'] + 1)

    def target_encoding(self, target_col='price_doc', categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            categorical_cols = [col for col in categorical_cols
                                if col not in ['timestamp', target_col]]
        for col in categorical_cols:
            if col in self.df.columns:
                map_enc = self.df.groupby(col)[target_col].mean().to_dict()
                self.df[f'{col}_target_encoding'] = self.df[col].map(map_enc)
                global_mean = self.df[target_col].mean()
                self.df[f'{col}_target_encoding'] = self.df[f'{col}_target_encoding'].fillna(global_mean)
    def label_encode(self):
        for col in self.kat_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
    def remove_kvazi(self):
        self.df = self.df.replace([np.inf,-np.inf],np.nan)
        self.df = self.df.dropna()

    def run_pipeline(self):
        self.fill_void()
        self.delete_errors()
        self.create_new_features()
        self.delete_corr()
        self.target_encoding()
        self.label_encode()
        self.remove_kvazi()
        return self.df



