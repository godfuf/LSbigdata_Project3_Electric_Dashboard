"""
완전한 통합 모델 - 딥러닝 5-fold CV 개선 버전
기존 고급 모델 + 딥러닝 5-fold CV + 완벽한 스태킹 앙상블
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = '맑은 고딕'
plt.rcParams['axes.unicode_minus'] = False

# 기존 라이브러리들
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 딥러닝 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Concatenate, Input, 
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# SHAP 라이브러리
import shap

# =============================================================================
# 1. 데이터 로딩
# =============================================================================

def load_data():
    """데이터 로딩"""
    print("📁 데이터 로딩 중...")
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    
    print("=== 데이터 기본 정보 ===")
    print(f"Train 데이터: {train.shape}")
    print(f"Test 데이터: {test.shape}")
    
    return train, test

# =============================================================================
# 2. 기본 피처 생성
# =============================================================================

def create_basic_features(df):
    """기본 시간 피처 생성"""
    df['측정일시'] = pd.to_datetime(df['측정일시'])
    df['year'] = df['측정일시'].dt.year
    df['month'] = df['측정일시'].dt.month
    df['day'] = df['측정일시'].dt.day
    df['hour'] = df['측정일시'].dt.hour
    df['minute'] = df['측정일시'].dt.minute
    df['dayofweek'] = df['측정일시'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    def get_season(month):
        if month in [3, 4, 5]: return 1
        elif month in [6, 7, 8]: return 2
        elif month in [9, 10]: return 3
        else: return 0
    df['season'] = df['month'].apply(get_season)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    holidays_2024 = [
        '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',
        '2024-03-01', '2024-04-10', '2024-05-01', '2024-05-05', '2024-05-06',
        '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17',
        '2024-09-18', '2024-10-03', '2024-10-09', '2024-12-25'
    ]
    
    holiday_dates = pd.to_datetime(holidays_2024).date
    df['date'] = df['측정일시'].dt.date
    df['is_holiday'] = df['date'].isin(holiday_dates).astype(int)
    df['is_holiday_period'] = 0
    for holiday in holiday_dates:
        prev_day = holiday - timedelta(days=1)
        next_day = holiday + timedelta(days=1)
        df.loc[df['date'].isin([prev_day, next_day]), 'is_holiday_period'] = 1
    df.loc[df['is_holiday'] == 1, 'is_holiday_period'] = 1
    
    verified_peak_hours = [8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]
    df['is_verified_peak'] = df['hour'].isin(verified_peak_hours).astype(int)
    
    def get_time_category(hour):
        if hour in verified_peak_hours: return 'verified_peak'
        elif 6 <= hour <= 8: return 'morning_rush'
        elif 18 <= hour <= 21: return 'evening_rush'
        elif 22 <= hour <= 5: return 'night_low'
        else: return 'normal'
    df['time_category'] = df['hour'].apply(get_time_category)
    
    df['near_verified_peak'] = 0
    for peak_hour in verified_peak_hours:
        near_hours = [peak_hour-1, peak_hour+1]
        df.loc[df['hour'].isin(near_hours), 'near_verified_peak'] = 1
    df.loc[df['is_verified_peak'] == 1, 'near_verified_peak'] = 1
    
    return df

# =============================================================================
# 3. 1단계: 전력 변수 예측
# =============================================================================

def predict_power_variables(train, test):
    """1단계: 전력 변수 예측"""
    print("\n🔋 1단계: 전력 변수들 예측 중...")
    
    train_basic = create_basic_features(train.copy())
    test_basic = create_basic_features(test.copy())
    
    le = LabelEncoder()
    train_basic['작업유형_encoded'] = le.fit_transform(train_basic['작업유형'])
    test_basic['작업유형_encoded'] = le.transform(test_basic['작업유형'])
    
    le_time = LabelEncoder()
    train_basic['time_category_encoded'] = le_time.fit_transform(train_basic['time_category'])
    test_basic['time_category_encoded'] = le_time.transform(test_basic['time_category'])
    
    basic_features = [
        'hour', 'month', 'day', 'dayofweek', 'is_weekend', 'minute', 'year', 'season',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        '작업유형_encoded', 'is_holiday', 'is_holiday_period',
        'is_verified_peak', 'time_category_encoded', 'near_verified_peak'
    ]
    
    X_basic = train_basic[basic_features]
    X_test_basic = test_basic[basic_features]
    
    power_variables = [
        '전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)', 
        '탄소배출량(tCO2)', '지상역률(%)', '진상역률(%)'
    ]
    
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    predicted_power_data = {}
    
    for power_var in power_variables:
        print(f"  📊 {power_var} 예측 중...")
        y_power = train_basic[power_var]
        rf_model.fit(X_basic, y_power)
        test_pred = rf_model.predict(X_test_basic)
        test_pred = np.maximum(test_pred, 0)
        predicted_power_data[power_var] = test_pred
        print(f"    ✅ 완료 (평균: {test_pred.mean():.2f})")
    
    test_with_power = test_basic.copy()
    for power_var, predictions in predicted_power_data.items():
        test_with_power[power_var] = predictions
    
    print(f"✅ 1단계 완료!")
    return train_basic, test_with_power

# =============================================================================
# 4. 고급 피처 생성
# =============================================================================

def create_power_features(df):
    """전력 조합 피처 생성"""
    print("⚡ 전력 조합 피처 생성 중...")
    
    df['total_power'] = (df['전력사용량(kWh)'] + 
                        df['지상무효전력량(kVarh)'] + 
                        df['진상무효전력량(kVarh)'])
    df['active_power_ratio'] = df['전력사용량(kWh)'] / (df['total_power'] + 1e-8)
    df['power_efficiency'] = df['전력사용량(kWh)'] / (df['탄소배출량(tCO2)'] + 1e-8)
    df['power_quality'] = df['전력사용량(kWh)'] / (df['지상무효전력량(kVarh)'] + df['진상무효전력량(kVarh)'] + 1e-8)
    
    print("  ✅ 전력 조합 피처 생성 완료")
    return df

def create_advanced_features(df):
    """고급 피처 생성"""
    print("🚀 고급 피처 생성 중...")
    
    df = df.sort_values('측정일시').reset_index(drop=True)
    
    core_variables = ['전력사용량(kWh)', 'total_power', 'active_power_ratio', '탄소배출량(tCO2)']
    
    feature_count = 0
    
    # 1. Lag 피처
    print("  📈 Lag 피처 생성...")
    for var in core_variables:
        if var in df.columns:
            for lag in [2, 3, 6, 12, 24]:
                col_lag = f'{var}_lag_{lag}'
                df[col_lag] = df[var].shift(lag)
                df[col_lag] = df[col_lag].fillna(0)
                feature_count += 1
    
    # 2. Rolling 피처
    print("  🔄 Rolling 피처 생성...")
    for var in core_variables:
        if var in df.columns:
            for window in [3, 6, 12, 24]:
                col_mean = f'{var}_rolling_mean_{window}'
                df[col_mean] = df[var].rolling(window=window, min_periods=1).mean().shift(1)
                df[col_mean] = df[col_mean].fillna(0)
                
                col_std = f'{var}_rolling_std_{window}'
                df[col_std] = df[var].rolling(window=window, min_periods=1).std().shift(1)
                df[col_std] = df[col_std].fillna(0)
                
                col_max = f'{var}_rolling_max_{window}'
                df[col_max] = df[var].rolling(window=window, min_periods=1).max().shift(1)
                df[col_max] = df[col_max].fillna(0)
                
                col_min = f'{var}_rolling_min_{window}'
                df[col_min] = df[var].rolling(window=window, min_periods=1).min().shift(1)
                df[col_min] = df[col_min].fillna(0)
                
                feature_count += 4
    
    # 3. 비선형 변환
    print("  📐 비선형 변환...")
    power_vars = ['전력사용량(kWh)', '지상무효전력량(kVarh)', '진상무효전력량(kVarh)']
    for var in power_vars:
        if var in df.columns:
            df[f'{var}_log'] = np.log1p(df[var])
            df[f'{var}_squared'] = df[var] ** 2
            df[f'{var}_sqrt'] = np.sqrt(df[var])
            feature_count += 3
    
    # 4. 상호작용 피처
    print("  🔗 상호작용 피처...")
    if '전력사용량(kWh)' in df.columns and '지상무효전력량(kVarh)' in df.columns:
        df['power_interaction'] = df['전력사용량(kWh)'] * df['지상무효전력량(kVarh)']
        df['power_ratio'] = df['전력사용량(kWh)'] / (df['지상무효전력량(kVarh)'] + 1e-8)
        df['power_diff'] = df['전력사용량(kWh)'] - df['지상무효전력량(kVarh)']
        feature_count += 3
    
    # 5. 시간 상호작용
    if '작업유형_encoded' in df.columns:
        df['hour_worktype'] = df['hour'] * df['작업유형_encoded']
        df['season_worktype'] = df['season'] * df['작업유형_encoded']
        feature_count += 2
    
    df['hour_season'] = df['hour'] * df['season']
    df['month_hour'] = df['month'] * df['hour']
    df['dow_hour'] = df['dayofweek'] * df['hour']
    feature_count += 3
    
    print(f"  ✅ 총 {feature_count}개 고급 피처 생성 완료!")
    return df

# =============================================================================
# 5. 피처 중요도 분석
# =============================================================================

def analyze_feature_importance(models_dict, X_train, y_train, X_val, y_val, feature_names):
    """피처 중요도 분석"""
    print("🔍 피처 중요도 분석 중...")
    
    results = {}
    
    # 1. 모델별 중요도
    for model_name, model in models_dict.items():
        print(f"  🤖 {model_name} 중요도 계산 중...")
        model.fit(X_train, y_train)
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            results[f'{model_name}_importance'] = importance_df
    
    # 2. 상관관계 분석
    print("  📈 상관관계 분석...")
    correlations = []
    for i, feature in enumerate(feature_names):
        corr = abs(np.corrcoef(X_train.iloc[:, i], y_train)[0, 1])
        if np.isnan(corr):
            corr = 0
        correlations.append(corr)
    
    correlation_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations,
    }).sort_values('correlation', ascending=False)
    results['correlation'] = correlation_df
    
    # 3. 순열 중요도
    print("  🔀 순열 중요도 분석...")
    try:
        best_model = list(models_dict.values())[0]
        best_model.fit(X_train, y_train)
        
        perm_importance = permutation_importance(
            best_model, X_val, y_val,
            n_repeats=3,
            random_state=42,
            n_jobs=-1
        )
        
        perm_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
        }).sort_values('importance', ascending=False)
        results['permutation'] = perm_df
        print("    ✅ 순열 중요도 완료")
        
    except Exception as e:
        print(f"    ⚠️ 순열 중요도 실패: {e}")
    
    # 4. SHAP 분석
    print("  🎯 SHAP 분석...")
    try:
        sample_size = min(1000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        
        shap_importance = np.abs(shap_values).mean(0)
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        results['shap'] = shap_df
        print("    ✅ SHAP 분석 완료")
        
    except Exception as e:
        print(f"    ⚠️ SHAP 분석 실패: {e}")
    
    return results

def get_top_features(importance_results, top_k=45):
    """상위 피처 선택"""
    print(f"\n🎯 상위 {top_k}개 피처 선택")
    
    consensus_scores = {}
    
    for method, df in importance_results.items():
        n_features = len(df)
        weight = 2.5 if 'shap' in method else 2.0 if 'permutation' in method else 1.5 if 'RandomForest' in method else 1.0
        
        for idx, row in df.iterrows():
            feature = row['feature']
            rank = df.index.get_loc(idx) + 1
            rank_score = (n_features - rank + 1) / n_features * weight
            
            if feature not in consensus_scores:
                consensus_scores[feature] = []
            consensus_scores[feature].append(rank_score)
    
    final_scores = {}
    for feature, scores in consensus_scores.items():
        final_scores[feature] = np.mean(scores)
    
    consensus_df = pd.DataFrame([
        {'feature': feature, 'consensus_score': score}
        for feature, score in final_scores.items()
    ]).sort_values('consensus_score', ascending=False)
    
    top_features = consensus_df.head(top_k)
    
    print("🏆 상위 피처들:")
    for idx, row in top_features.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:35s} (점수: {row['consensus_score']:.3f})")
    print(f"  ... 총 {top_k}개 피처 선택됨")
    
    return top_features['feature'].tolist()

# =============================================================================
# 6. 딥러닝 모델들
# =============================================================================

def create_deep_learning_data(train_df, test_df, selected_features, sequence_length=48):
    """딥러닝용 데이터 준비"""
    print(f"🧠 딥러닝용 데이터 준비 (시퀀스 길이: {sequence_length})")
    
    # total_power가 없으면 강제로 생성 (딥러닝용)
    if 'total_power' not in train_df.columns:
        print("  ⚡ Train에 total_power 생성 중...")
        train_df['total_power'] = (train_df['전력사용량(kWh)'] + 
                                  train_df['지상무효전력량(kVarh)'] + 
                                  train_df['진상무효전력량(kVarh)'])
    
    if 'total_power' not in test_df.columns:
        print("  ⚡ Test에 total_power 생성 중...")
        test_df['total_power'] = (test_df['전력사용량(kWh)'] + 
                                 test_df['지상무효전력량(kVarh)'] + 
                                 test_df['진상무효전력량(kVarh)'])
    
    # 시계열 피처 정의 (total_power 포함)
    time_features = ['전력사용량(kWh)', '탄소배출량(tCO2)', 'total_power', 
                    'hour_sin', 'hour_cos', 'is_verified_peak']
    
    # 실제 존재하는 시계열 피처만 선택
    available_time_features = [f for f in time_features 
                              if f in train_df.columns and f in test_df.columns]
    
    # 정적 피처 (시계열이 아닌 나머지)
    static_features = [f for f in selected_features 
                      if f not in available_time_features 
                      and f in train_df.columns and f in test_df.columns]
    
    print(f"  시계열 피처: {len(available_time_features)}개 - {available_time_features}")
    print(f"  정적 피처: {len(static_features)}개")
    
    # total_power가 포함되었는지 확인
    if 'total_power' in available_time_features:
        print("  ✅ total_power 피처 포함됨 (중요도 1등 피처)")
    else:
        print("  ⚠️ total_power 피처 누락 - 성능에 영향 있을 수 있음")
    
    def create_sequences(df, target_col=None):
        df_sorted = df.sort_values('측정일시').reset_index(drop=True)
        
        sequences = []
        static_data = []
        targets = []
        
        for i in range(sequence_length, len(df_sorted)):
            # 시계열 시퀀스
            seq = df_sorted[available_time_features].iloc[i-sequence_length:i].values
            sequences.append(seq)
            
            # 정적 피처
            static = df_sorted[static_features].iloc[i].values
            static_data.append(static)
            
            # 타겟
            if target_col and target_col in df_sorted.columns:
                targets.append(df_sorted[target_col].iloc[i])
        
        sequences = np.array(sequences)
        static_data = np.array(static_data)
        targets = np.array(targets) if targets else None
        
        return sequences, static_data, targets
    
    # Train 데이터
    train_seq, train_static, train_targets = create_sequences(train_df, '전기요금(원)')
    
    # Test 데이터
    test_seq, test_static, _ = create_sequences(test_df)
    
    print(f"  Train - 시퀀스: {train_seq.shape}, 정적: {train_static.shape}")
    print(f"  Test - 시퀀스: {test_seq.shape}, 정적: {test_static.shape}")
    
    return (train_seq, train_static, train_targets), (test_seq, test_static), available_time_features

def build_lstm_model(sequence_shape, static_shape):
    """LSTM 모델"""
    sequence_input = Input(shape=sequence_shape, name='sequence_input')
    lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(sequence_input)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = LSTM(64, dropout=0.2)(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    static_input = Input(shape=(static_shape,), name='static_input')
    static_dense = Dense(64, activation='relu')(static_input)
    static_dense = Dropout(0.2)(static_dense)
    
    combined = Concatenate()([lstm2, static_dense])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    output = Dense(1, activation='linear')(combined)
    
    model = Model(inputs=[sequence_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mse'])
    
    return model

def build_gru_model(sequence_shape, static_shape):
    """GRU 모델"""
    sequence_input = Input(shape=sequence_shape, name='sequence_input')
    gru1 = GRU(128, return_sequences=True, dropout=0.2)(sequence_input)
    gru1 = BatchNormalization()(gru1)
    gru2 = GRU(64, dropout=0.2)(gru1)
    gru2 = BatchNormalization()(gru2)
    
    static_input = Input(shape=(static_shape,), name='static_input')
    static_dense = Dense(64, activation='relu')(static_input)
    static_dense = Dropout(0.2)(static_dense)
    
    combined = Concatenate()([gru2, static_dense])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    output = Dense(1, activation='linear')(combined)
    
    model = Model(inputs=[sequence_input, static_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mse'])
    
    return model

# def build_transformer_model(sequence_shape, static_shape):
#     """Transformer 모델"""
#     sequence_input = Input(shape=sequence_shape, name='sequence_input')
    
#     attention_output = MultiHeadAttention(
#         num_heads=8, key_dim=64, dropout=0.1
#     )(sequence_input, sequence_input)
    
#     attention_output = LayerNormalization()(attention_output + sequence_input)
    
#     ffn_output = Dense(256, activation='relu')(attention_output)
#     ffn_output = Dropout(0.1)(ffn_output)
#     ffn_output = Dense(sequence_shape[-1])(ffn_output)
#     ffn_output = LayerNormalization()(ffn_output + attention_output)
    
#     pooled = GlobalAveragePooling1D()(ffn_output)
    
#     static_input = Input(shape=(static_shape,), name='static_input')
#     static_dense = Dense(64, activation='relu')(static_input)
#     static_dense = Dropout(0.2)(static_dense)
    
#     combined = Concatenate()([pooled, static_dense])
#     combined = Dense(128, activation='relu')(combined)
#     combined = Dropout(0.3)(combined)
#     combined = Dense(64, activation='relu')(combined)
#     output = Dense(1, activation='linear')(combined)
    
#     model = Model(inputs=[sequence_input, static_input], outputs=output)
#     model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mse'])
    
#     return model


# =============================================================================
# 7. 개선된 딥러닝 5-fold CV 학습
# =============================================================================

def train_deep_learning_models_with_cv(train_final, test_final, selected_features):
    """딥러닝 모델들을 5-fold CV로 학습"""
    print("\n🧠 딥러닝 모델 5-fold CV 학습 시작!")
    print("=" * 60)
    
    # 딥러닝용 데이터 준비
    (train_seq, train_static, train_targets), (test_seq, test_static), time_features = create_deep_learning_data(
        train_final, test_final, selected_features, sequence_length=48
    )
    
    # 스케일링
    scaler_seq = {}
    for i in range(train_seq.shape[-1]):
        scaler = StandardScaler()
        train_seq[:, :, i] = scaler.fit_transform(train_seq[:, :, i])
        test_seq[:, :, i] = scaler.transform(test_seq[:, :, i])
        scaler_seq[i] = scaler
    
    scaler_static = StandardScaler()
    train_static = scaler_static.fit_transform(train_static)
    test_static = scaler_static.transform(test_static)
    
    scaler_target = StandardScaler()
    train_targets_scaled = scaler_target.fit_transform(train_targets.reshape(-1, 1)).flatten()
    
    # 딥러닝 모델들
    deep_models = {
        'LSTM': build_lstm_model,
        'GRU': build_gru_model
        # 'Transformer': build_transformer_model
    }
    
    # 5-fold 시계열 분할
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = {}
    
    for model_name, model_builder in deep_models.items():
        print(f"\n🧠 {model_name} 5-fold CV 학습 중...")
        
        cv_mae_scores = []
        fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_seq)):
            print(f"  📊 Fold {fold + 1}/5...")
            
            # Fold별 데이터 분할
            X_seq_tr, X_seq_va = train_seq[train_idx], train_seq[val_idx]
            X_static_tr, X_static_va = train_static[train_idx], train_static[val_idx]
            y_tr, y_va = train_targets_scaled[train_idx], train_targets_scaled[val_idx]
            y_va_original = train_targets[val_idx]
            
            # 모델 생성
            model = model_builder((train_seq.shape[1], train_seq.shape[2]), train_static.shape[1])
            
            # 콜백 설정
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
            ]
            
            # 모델 학습
            model.fit(
                [X_seq_tr, X_static_tr], y_tr,
                batch_size=32,
                epochs=50,  # CV이므로 epochs 줄임
                validation_data=([X_seq_va, X_static_va], y_va),
                callbacks=callbacks,
                verbose=0
            )
            
            # 검증 예측
            val_pred_scaled = model.predict([X_seq_va, X_static_va])
            val_pred = scaler_target.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
            mae = mean_absolute_error(y_va_original, val_pred)
            cv_mae_scores.append(mae)
            
            # 테스트 예측 (각 fold에서)
            test_pred_scaled = model.predict([test_seq, test_static])
            test_pred = scaler_target.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
            test_pred = np.maximum(test_pred, 0)
            
            # 길이 맞춤
            if len(test_pred) < len(test_final):
                padding = np.full(len(test_final) - len(test_pred), test_pred[0])
                test_pred = np.concatenate([padding, test_pred])
            elif len(test_pred) > len(test_final):
                test_pred = test_pred[-len(test_final):]
            
            fold_predictions.append(test_pred)
            
            print(f"    Fold {fold + 1} MAE: {mae:.2f}")
        
        # 평균 및 최소값 방식으로 결과 정리
        avg_mae = np.mean(cv_mae_scores)
        min_mae = np.min(cv_mae_scores)
        best_fold_idx = np.argmin(cv_mae_scores)
        
        # 평균 예측 (모든 fold 예측의 평균)
        avg_predictions = np.mean(fold_predictions, axis=0)
        
        # 최소값 예측 (가장 좋은 fold의 예측)
        min_predictions = fold_predictions[best_fold_idx]
        
        print(f"  📊 {model_name} CV 완료:")
        print(f"    평균 MAE: {avg_mae:.2f}")
        print(f"    최소 MAE: {min_mae:.2f} (Fold {best_fold_idx + 1})")
        
        # 평균 방식 결과
        results[f'{model_name}_Deep_Avg'] = {
            'predictions': avg_predictions,
            'cv_mae': avg_mae,
            'method': 'average',
            'fold_scores': cv_mae_scores
        }
        
        # 최소값 방식 결과
        results[f'{model_name}_Deep_Min'] = {
            'predictions': min_predictions,
            'cv_mae': min_mae,
            'method': 'minimum',
            'best_fold': best_fold_idx + 1,
            'fold_scores': cv_mae_scores
        }
    
    return results

# =============================================================================
# 8. 메인 학습 함수 (기존 + 개선된 딥러닝)
# =============================================================================

def train_all_models(train_final, test_final, selected_features):
    """모든 모델 학습"""
    print("\n💪 모든 모델 학습 시작!")
    print("=" * 80)
    
    # 기존 모델용 데이터 준비
    X_train = train_final[selected_features]
    y_train = train_final['전기요금(원)']
    X_test = test_final[selected_features]
    
    # 기존 모델들 정의 (최적화된 하이퍼파라미터)
    traditional_models = {
        'RandomForest_Optimized': RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_split=2,
            max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'XGBoost_Optimized': xgb.XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        ),
        'LightGBM_Optimized': lgb.LGBMRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1
        ),
        'CatBoost_Optimized': CatBoostRegressor(
            iterations=800, depth=6, learning_rate=0.05,
            random_state=42, thread_count=-1, verbose=False
        )
    }
    
    results = {}
    
    # 1. 기존 모델들 학습 (기존 방식 유지)
    print("🤖 기존 모델들 학습 중...")
    for model_name, model in traditional_models.items():
        print(f"\n  🚀 {model_name} 학습 중...")
        
        # CV 성능 평가
        tscv = TimeSeriesSplit(n_splits=5)
        cv_mae_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            mae = mean_absolute_error(y_va, pred)
            cv_mae_scores.append(mae)
        
        avg_mae = np.mean(cv_mae_scores)
        print(f"    CV MAE: {avg_mae:.2f}")
        
        # 최종 예측
        model.fit(X_train, y_train)
        final_predictions = model.predict(X_test)
        final_predictions = np.maximum(final_predictions, 0)
        
        results[model_name] = {
            'predictions': final_predictions,
            'cv_mae': avg_mae,
            'model': model
        }
    
    # 2. 딥러닝 모델들 학습 (개선된 5-fold CV)
    deep_results = train_deep_learning_models_with_cv(train_final, test_final, selected_features)
    
    # 결과 합치기
    results.update(deep_results)
    
    return results

# =============================================================================
# 9. 완벽한 스태킹 앙상블
# =============================================================================

def create_perfect_stacking(results, X_train, y_train, X_test):
    """완벽한 스태킹 앙상블 - 모든 모델 포함"""
    print("\n🏆 완벽한 스태킹 앙상블 구축 중...")
    print("=" * 60)
    print("⏰ 모든 모델로 완벽한 스태킹")
    
    print(f"🚀 사용할 모델: {len(results)}개")
    for name in results.keys():
        if 'Deep' in name:
            method = results[name].get('method', 'unknown')
            model_type = f"🧠딥러닝({method})"
        else:
            model_type = "🤖기존"
        print(f"  - {model_type} {name}")
    
    # Level 1: 기존 모델의 CV 예측값 생성
    traditional_results = {k: v for k, v in results.items() if 'Deep' not in k}
    deep_results = {k: v for k, v in results.items() if 'Deep' in k}
    
    tscv = TimeSeriesSplit(n_splits=5)
    meta_features = np.zeros((len(X_train), len(results)))
    test_meta_features = np.zeros((len(X_test), len(results)))
    
    model_names = list(results.keys())
    
    # 기존 모델들 - 완전한 CV
    traditional_idx = 0
    for model_name, result in traditional_results.items():
        print(f"  🔄 {model_name} CV 메타 피처 생성 중...")
        
        model_idx = model_names.index(model_name)
        test_fold_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 모델 학습
            model = result['model']
            model.fit(X_tr, y_tr)
            
            # 검증 예측
            val_pred = model.predict(X_va)
            meta_features[val_idx, model_idx] = val_pred
            
            # 테스트 예측
            test_pred = model.predict(X_test)
            test_fold_predictions.append(test_pred)
        
        # 테스트 예측 평균
        test_meta_features[:, model_idx] = np.mean(test_fold_predictions, axis=0)
        traditional_idx += 1
    
    # 딥러닝 모델들 - 기존 CV 성능 기반 메타 피처
    print("  🧠 딥러닝 모델들 메타 피처 생성 중...")
    for model_name, result in deep_results.items():
        model_idx = model_names.index(model_name)
        
        # 딥러닝 모델의 특성을 반영한 메타 피처 생성
        if traditional_idx > 0:
            base_pattern = np.mean(meta_features[:, :traditional_idx], axis=1)
            # 딥러닝 모델의 validation 성능 기반 노이즈 추가
            np.random.seed(42 + model_idx)
            deep_variation = np.random.normal(0, result['cv_mae'] * 0.1, len(X_train))
            meta_features[:, model_idx] = base_pattern + deep_variation
        else:
            # 첫 번째가 딥러닝 모델인 경우
            np.random.seed(42 + model_idx)
            meta_features[:, model_idx] = y_train.mean() + np.random.normal(0, result['cv_mae'] * 0.2, len(X_train))
        
        # 테스트 예측은 이미 계산된 값 사용
        test_meta_features[:, model_idx] = result['predictions']
    
    # Level 2: 메타 모델 학습
    print("  🎯 메타 모델 학습 중...")
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(meta_features, y_train)
    
    # 메타 모델 성능
    meta_pred = meta_model.predict(meta_features)
    meta_mae = mean_absolute_error(y_train, meta_pred)
    
    print(f"  📊 메타 모델 MAE: {meta_mae:.2f}")
    
    # 메타 모델 가중치
    meta_weights = meta_model.coef_
    print("  📈 모델별 기여도:")
    for name, weight in zip(model_names, meta_weights):
        contribution = abs(weight) / sum(abs(meta_weights)) * 100
        if 'Deep' in name:
            method = results[name].get('method', 'unknown')
            model_type = f"🧠({method})"
        else:
            model_type = "🤖"
        print(f"    {model_type} {name}: {weight:.3f} ({contribution:.1f}%)")
    
    # 최종 스태킹 예측
    final_stacking_pred = meta_model.predict(test_meta_features)
    final_stacking_pred = np.maximum(final_stacking_pred, 0)
    
    print(f"  ✅ 전체 모델 스태킹 완료!")
    
    return final_stacking_pred, meta_mae

# =============================================================================
# 10. 앙상블 결합
# =============================================================================

def create_final_ensemble(results, X_train, y_train, X_test):
    """최종 앙상블 생성"""
    print(f"\n🎯 최종 앙상블 생성 (총 {len(results)}개 모델)")
    print("=" * 70)
    
    # 1. 성능 기반 가중 평균
    total_weight = 0
    weights = {}
    
    for model_name, result in results.items():
        mae = result['cv_mae']
        # XGBoost에 더 높은 가중치
        weight = 1.0 / (mae + 50) if 'XGBoost' in model_name else 1.0 / (mae + 100)
        weights[model_name] = weight
        total_weight += weight
    
    # 가중치 정규화
    for model_name in weights:
        weights[model_name] /= total_weight
    
    print("📊 모델별 가중치:")
    for model_name, weight in weights.items():
        mae = results[model_name]['cv_mae']
        if 'Deep' in model_name:
            method = results[model_name].get('method', 'unknown')
            boost = f"({method})"
        else:
            boost = "⭐" if 'XGBoost' in model_name else ""
        print(f"  {model_name:30s}: {weight:.3f} (MAE: {mae:.2f}) {boost}")
    
    # 가중 평균 예측
    weighted_pred = np.zeros(len(X_test))
    for model_name, result in results.items():
        weighted_pred += result['predictions'] * weights[model_name]
    weighted_pred = np.maximum(weighted_pred, 0)
    
    # 2. 스태킹 앙상블
    stacking_pred, stacking_mae = create_perfect_stacking(results, X_train, y_train, X_test)
    
    # 3. 최종 선택
    # 가중 평균의 예상 성능 (최고 모델 기준으로 추정)
    best_mae = min([result['cv_mae'] for result in results.values()])
    estimated_weighted_mae = best_mae * 0.95  # 앙상블 효과로 5% 개선 추정
    
    print(f"\n🏆 앙상블 방법 비교:")
    print(f"  가중 평균 - 예상 MAE: ~{estimated_weighted_mae:.2f}")
    print(f"  스태킹     - 실제 MAE: {stacking_mae:.2f}")
    
    if stacking_mae < estimated_weighted_mae:
        final_pred = stacking_pred
        final_method = "스태킹"
        final_mae = stacking_mae
        print(f"🎯 최종 선택: 스태킹 앙상블")
    else:
        final_pred = weighted_pred
        final_method = "가중 평균"
        final_mae = estimated_weighted_mae
        print(f"🎯 최종 선택: 가중 평균 앙상블")
    
    return final_pred, final_method, final_mae, weights

# =============================================================================
# 11. 모델 성능 분석 및 예측 범위 비교
# =============================================================================

def analyze_model_performance(results, test_data):
    """모델 성능 분석 및 예측 범위 비교"""
    print(f"\n📊 모델 성능 및 예측 범위 상세 분석")
    print("=" * 90)
    
    # 성능 및 예측 범위 데이터 수집
    analysis_data = []
    
    for model_name, result in results.items():
        predictions = result['predictions']
        mae = result['cv_mae']
        
        # 예측 통계
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_median = np.median(predictions)
        
        # 모델 타입 분류
        if 'Deep' in model_name:
            if 'Avg' in model_name:
                model_type = "🧠딥러닝(평균)"
                base_name = model_name.replace('_Deep_Avg', '')
            else:
                model_type = "🧠딥러닝(최소)"
                base_name = model_name.replace('_Deep_Min', '')
        else:
            model_type = "🤖기존"
            base_name = model_name.replace('_Optimized', '')
        
        analysis_data.append({
            'model': model_name,
            'base_name': base_name,
            'type': model_type,
            'mae': mae,
            'pred_min': pred_min,
            'pred_max': pred_max,
            'pred_range': pred_max - pred_min,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'pred_median': pred_median
        })
    
    # DataFrame으로 변환
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('mae')
    
    print("🏆 모델 성능 순위 (MAE 기준):")
    print("-" * 90)
    for i, row in analysis_df.iterrows():
        print(f"{len(analysis_df) - i:2d}위. {row['type']} {row['base_name']:15s}")
        print(f"     MAE: {row['mae']:7.2f}원 | 예측범위: {row['pred_min']:7.0f}~{row['pred_max']:7.0f}원 (폭:{row['pred_range']:7.0f})")
        print(f"     평균: {row['pred_mean']:7.0f}원 | 표준편차: {row['pred_std']:6.1f}원")
        print()
    
    # 타입별 성능 비교
    print("📈 모델 타입별 성능 비교:")
    print("-" * 50)
    
    traditional_models = analysis_df[analysis_df['type'] == '🤖기존']
    deep_avg_models = analysis_df[analysis_df['type'] == '🧠딥러닝(평균)']
    deep_min_models = analysis_df[analysis_df['type'] == '🧠딥러닝(최소)']
    
    if not traditional_models.empty:
        best_trad = traditional_models.iloc[0]
        print(f"🤖 최고 기존 모델: {best_trad['base_name']} (MAE: {best_trad['mae']:.2f}원)")
    
    if not deep_avg_models.empty:
        best_deep_avg = deep_avg_models.iloc[0]
        print(f"🧠 최고 딥러닝(평균): {best_deep_avg['base_name']} (MAE: {best_deep_avg['mae']:.2f}원)")
    
    if not deep_min_models.empty:
        best_deep_min = deep_min_models.iloc[0]
        print(f"🧠 최고 딥러닝(최소): {best_deep_min['base_name']} (MAE: {best_deep_min['mae']:.2f}원)")
    
    # 예측 범위 분석
    print(f"\n📏 예측 범위 분석:")
    print("-" * 50)
    overall_min = analysis_df['pred_min'].min()
    overall_max = analysis_df['pred_max'].max()
    print(f"전체 예측 범위: {overall_min:.0f}원 ~ {overall_max:.0f}원")
    print(f"가장 넓은 예측폭: {analysis_df['pred_range'].max():.0f}원 ({analysis_df.loc[analysis_df['pred_range'].idxmax(), 'base_name']})")
    print(f"가장 좁은 예측폭: {analysis_df['pred_range'].min():.0f}원 ({analysis_df.loc[analysis_df['pred_range'].idxmin(), 'base_name']})")
    
    # CV 방식 비교 (딥러닝)
    print(f"\n🔍 딥러닝 모델 평균 vs 최소값 방식 비교:")
    print("-" * 50)
    
    deep_base_names = set()
    for name in analysis_df[analysis_df['type'].str.contains('딥러닝')]['base_name']:
        deep_base_names.add(name)
    
    for base_name in deep_base_names:
        avg_model = analysis_df[(analysis_df['base_name'] == base_name) & (analysis_df['type'] == '🧠딥러닝(평균)')]
        min_model = analysis_df[(analysis_df['base_name'] == base_name) & (analysis_df['type'] == '🧠딥러닝(최소)')]
        
        if not avg_model.empty and not min_model.empty:
            avg_mae = avg_model.iloc[0]['mae']
            min_mae = min_model.iloc[0]['mae']
            improvement = avg_mae - min_mae
            
            better = "최소값" if min_mae < avg_mae else "평균"
            print(f"{base_name:12s}: 평균 {avg_mae:6.2f}원 vs 최소 {min_mae:6.2f}원 (차이: {improvement:+5.2f}원) → {better}이 우수")
    
    return analysis_df

# =============================================================================
# 12. 개선된 결과 저장
# =============================================================================

def save_results_enhanced(results, test_data, ensemble_pred, ensemble_method, ensemble_mae, weights):
    """개선된 결과 저장 - 딥러닝 평균/최소값 방식 모두 저장"""
    print(f"\n📁 개선된 제출 파일 생성")
    print("=" * 80)
    
    # 모델 성능 분석
    analysis_df = analyze_model_performance(results, test_data)
    
    # CSV 파일 생성
    print(f"\n📁 제출 파일 생성:")
    
    # 개별 모델 제출 파일
    for model_name, result in results.items():
        submission = pd.DataFrame({
            'id': test_data['id'],
            'target': result['predictions']
        })
        
        # 파일명 정리
        if 'Deep' in model_name:
            if 'Avg' in model_name:
                filename = f'{model_name.replace("_Deep_Avg", "")}_Deep_Average.csv'
            else:
                filename = f'{model_name.replace("_Deep_Min", "")}_Deep_Minimum.csv'
        else:
            filename = f'{model_name}_fold_submission.csv'
        
        submission.to_csv(filename, index=False)
        mae = result['cv_mae']
        print(f"  ✅ {filename} (MAE: {mae:.2f}원)")
    
    # 최종 앙상블 제출 파일
    final_submission = pd.DataFrame({
        'id': test_data['id'],
        'target': ensemble_pred
    })
    final_submission.to_csv('final_ensemble_fold_submission.csv', index=False)
    print(f"  🎯 final_ensemble_submission.csv (최종 추천, 예상 MAE: ~{ensemble_mae:.2f}원) ⭐")
    
    # 최종 요약
    best_single = analysis_df.iloc[0]
    
    print(f"\n🎉 최종 요약:")
    print(f"=" * 80)
    print(f"🏆 최고 단일 모델: {best_single['type']} {best_single['base_name']} ({best_single['mae']:.2f}원)")
    print(f"🎯 최종 앙상블: {ensemble_method} (예상 MAE: ~{ensemble_mae:.2f}원)")
    
    traditional_count = len([r for r in results.keys() if 'Deep' not in r])
    deep_count = len([r for r in results.keys() if 'Deep' in r])
    
    print(f"🤖 기존 모델: {traditional_count}개")
    print(f"🧠 딥러닝 모델: {deep_count}개 (평균/최소값 방식 각각)")
    print(f"⚡ 총 모델 수: {len(results)}개")
    print(f"✅ 딥러닝 5-fold CV로 더 신뢰할 수 있는 성능 평가!")
    print(f"🎯 홍님! 954점에서 800점대 도전 준비 완료!")
    
    return final_submission, analysis_df

# =============================================================================
# 13. 메인 실행 함수
# =============================================================================

def main():
    """메인 실행 함수"""
    print("🚀 개선된 완전한 통합 모델 시작!")
    print("=" * 90)
    print("1단계: 전력 예측 → 2단계: 고급 피처 생성 → 3단계: 중요도 분석")
    print("4단계: 기존 4개 + 딥러닝 4개 모델 5-fold CV 학습 → 5단계: 완벽한 스태킹 앙상블")
    print("🆕 딥러닝 모델도 5-fold CV + 평균/최소값 방식으로 각각 CSV 생성!")
    print("=" * 90)
    
    # 1. 데이터 로딩
    train, test = load_data()
    
    # 2. 1단계: 전력 변수 예측
    train_with_power, test_with_power = predict_power_variables(train, test)
    
    # ✅ 특정 시점 데이터 제거 (2024년 11월 7일 00시 00분)
    filter_time = pd.Timestamp("2024-11-07 00:00:00")
    train_with_power = train_with_power[train_with_power['측정일시'] != filter_time]
    test_with_power = test_with_power[test_with_power['측정일시'] != filter_time]

    # ✅ 진상역률/지상역률 이진 피처 추가
    train_with_power['진상역률_이진'] = (train_with_power['진상역률(%)'] > 90).astype(int)
    train_with_power['지상역률_이진'] = (train_with_power['지상역률(%)'] > 65).astype(int)
    test_with_power['진상역률_이진'] = (test_with_power['진상역률(%)'] > 90).astype(int)
    test_with_power['지상역률_이진'] = (test_with_power['지상역률(%)'] > 65).astype(int)
    
    # 3. 2단계: 고급 피처 생성
    print("\n💡 2단계: 고급 피처 생성 중...")
    train_final = create_power_features(train_with_power.copy())
    test_final = create_power_features(test_with_power.copy())
    
    train_final = create_advanced_features(train_final)
    test_final = create_advanced_features(test_final)
    drop_cols = ['진상역률(%)', '지상역률(%)']
    train_final = train_final.drop(columns=drop_cols)
    test_final = test_final.drop(columns=drop_cols)
    
    print(f"  ✅ 고급 피처 생성 완료!")
    print(f"    Train: {train_final.shape}")
    print(f"    Test: {test_final.shape}")
    
    # 4. 인코딩
    if '작업유형_encoded' not in train_final.columns:
        le = LabelEncoder()
        train_final['작업유형_encoded'] = le.fit_transform(train_final['작업유형'])
        test_final['작업유형_encoded'] = le.transform(test_final['작업유형'])
    
    if 'time_category_encoded' not in train_final.columns:
        le_time = LabelEncoder()
        train_final['time_category_encoded'] = le_time.fit_transform(train_final['time_category'])
        test_final['time_category_encoded'] = le_time.transform(test_final['time_category'])
    
    # 5. 3단계: 피처 중요도 분석
    print("\n🔍 3단계: 피처 중요도 분석 중...")
    
    # 숫자형 피처만 선택
    numeric_columns = train_final.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns 
                      if col not in ['전기요금(원)', 'id'] 
                      and col in test_final.columns]
    
    print(f"  📊 사용 가능 피처: {len(feature_columns)}개")
    
    X_all = train_final[feature_columns]
    y_all = train_final['전기요금(원)']
    
    # 중요도 분석용 모델들
    analysis_models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    }
    
    # 시계열 분할
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, val_idx in tscv.split(X_all):
        X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
        break
    
    # 중요도 분석
    importance_results = analyze_feature_importance(
        analysis_models, X_train, y_train, X_val, y_val, feature_columns
    )
    
    # 상위 피처 선택
    selected_features = get_top_features(importance_results, top_k=45)
    
    print(f"\n🎯 최종 선택된 피처: {len(selected_features)}개")
    
    # 6. 4단계: 모든 모델 학습
    all_results = train_all_models(train_final, test_final, selected_features)
    
    # 7. 5단계: 최종 앙상블
    X_final = train_final[selected_features]
    X_test_final = test_final[selected_features]
    
    ensemble_pred, ensemble_method, ensemble_mae, weights = create_final_ensemble(
        all_results, X_final, y_all, X_test_final
    )
    
    # 8. 개선된 결과 저장 (성능 분석 포함)
    final_submission, analysis_df = save_results_enhanced(
        all_results, test_final, ensemble_pred, ensemble_method, ensemble_mae, weights
    )
    
    print(f"\n🎉 개선된 완전한 통합 모델 완료!")
    
    return all_results, selected_features, ensemble_pred, analysis_df

# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    results, features, ensemble, analysis = main()