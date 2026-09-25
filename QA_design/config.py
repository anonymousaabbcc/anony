import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('URBANBIND_ROOT', Path(__file__).resolve().parents[1]))
DATASET_ROOT = Path(os.environ.get('URBANBIND_DATA_ROOT', PROJECT_ROOT / 'dataset'))
SOURCE_CITY_ORDER = ['NYCTAXI', 'BIKECHI', 'NYC-BIKE']
TARGET_CITY_ORDER = ['BIKEDC']
ALL_CITY_ORDER = SOURCE_CITY_ORDER + TARGET_CITY_ORDER
CITY_ORDER = SOURCE_CITY_ORDER

def _data_path(key, filename):
    return Path(os.environ.get(key, DATASET_ROOT / filename))

RAW_CITY_CONFIG = {
    'NYCTAXI': {'format': 'grid', 'path': _data_path('URBANBIND_NYCTAXI_PATH', 'nyctaxi.grid'), 'expected_shape': (2160, 2, 10, 20), 'grid_shape': (10, 20), 'interval_minutes': 60, 'role': 'source'},
    'BIKECHI': {'format': 'grid', 'path': _data_path('URBANBIND_BIKECHI_PATH', 'bikechi.grid'), 'expected_shape': (2208, 2, 15, 18), 'grid_shape': (15, 18), 'interval_minutes': 60, 'role': 'source'},
    'NYC-BIKE': {'format': 'h5', 'path': _data_path('URBANBIND_NYCBIKE_PATH', 'nycbike.h5'), 'expected_shape': (4392, 2, 16, 8), 'grid_shape': (16, 8), 'interval_minutes': 60, 'role': 'source'},
    'BIKEDC': {'format': 'grid', 'path': _data_path('URBANBIND_BIKEDC_PATH', 'bikedc.grid'), 'expected_shape': (2208, 2, 16, 8), 'grid_shape': (16, 8), 'interval_minutes': 60, 'role': 'heldout_target'}
}
CITY_CONFIG = RAW_CITY_CONFIG
CITY_SLUG = {'NYCTAXI': 'nyctaxi', 'BIKECHI': 'bikechi', 'NYC-BIKE': 'nycbike', 'BIKEDC': 'bikedc'}
SPLIT_ROOT = Path(os.environ.get('URBANBIND_SPLIT_ROOT', DATASET_ROOT / 'processed_splits'))
QA_SOURCE_OUTPUT_ROOT = Path(os.environ.get('URBANBIND_QA_SOURCE_ROOT', PROJECT_ROOT / 'outputs' / 'qa' / 'source'))
QA_TARGET_OUTPUT_ROOT = Path(os.environ.get('URBANBIND_QA_TARGET_ROOT', PROJECT_ROOT / 'outputs' / 'qa' / 'target'))
QA_OUTPUT_ROOT = QA_SOURCE_OUTPUT_ROOT
SPLITS = ('train', 'valid', 'test')
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2
HISTORY_LEN = 8
PRED_LEN = 4
WINDOW_LEN = HISTORY_LEN + PRED_LEN
HORIZONS = [1, 2, 3, 4]
CHANNEL_TO_INDEX = {'inflow': 0, 'outflow': 1}
INDEX_TO_CHANNEL = {0: 'inflow', 1: 'outflow'}
TEXT_DECIMALS = 1
EPS = 1e-06
TREND_SLOPE_THRESHOLD = 0.01
TREND_RANGE_THRESHOLD = 0.1
TREND_SIGN_CHANGES = 3
RATIO_DECIMALS = 3
FINEGRAIN_ZERO_CAP = 0.25
FINEGRAIN_Q_LOW = 0.33
FINEGRAIN_Q_HIGH = 0.67
FINEGRAIN_ACTIVITY_WEIGHT = 0.5
FINEGRAIN_DYNAMICS_WEIGHT = 0.5
FINEGRAIN_SAMPLING_POLICY = 'historical_only_zero_preserving_stratified'
SYSTEM_PROMPT = 'You analyze hourly urban flow maps. Each city image contains eight consecutive historical time steps, labeled t=1 through t=8. Prediction horizons h=1 through h=4 refer to future steps after the displayed history and are not included in the input images. For whole-city views, whole_city denotes the mean flow over the complete native grid. For coarse and local views, the REGION LAYOUT panel defines the spatial location of each region ID, and the same partition is used at all eight historical steps. For fine-grained views, F(r,c) denotes the native grid cell at one-based row r and column c; row increases from top to bottom and column increases from left to right. Colors inside an urban map represent original grid-cell flow values. When numeric annotations are shown, they are computed only from the displayed historical maps. If a question asks for a dominant spatial unit and multiple units have exactly the same maximum value, use the first unit in row-major order: coarse_1 < coarse_2 < ...; local_1 < local_2 < ...; F(1,1) < F(1,2) < ... . For hotspot-movement questions, do not use that single-unit tie break: the hotspot position at each time is the centroid of all tied maximum units. Use the question to identify the requested city or cities, channel or channels, historical time step or steps, and spatial unit or units. Answer using the requested canonical format.'
