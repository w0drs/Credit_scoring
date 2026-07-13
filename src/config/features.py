"""Конфигурация полей для ML модели"""

MODEL_FEATURES = [
    'education_cd',
    'gender_cd',
    'age',
    'car_own_flg',
    'car_type_flg',
    'appl_rej_cnt',
    'good_work_flg',
    'Score_bki',
    'out_request_cnt',
    'region_rating',
    'home_address_cd',
    'work_address_cd',
    'income',
    'SNA',
    'first_time_cd',
    'Air_flg'
]

FEATURE_TYPES = {
    'education_cd': 'category',
    'gender_cd': 'category',
    'age': 'numeric',
    'car_own_flg': 'category',
    'car_type_flg': 'category',
    'appl_rej_cnt': 'numeric',
    'good_work_flg': 'numeric',
    'Score_bki': 'numeric',
    'out_request_cnt': 'numeric',
    'region_rating': 'numeric',
    'home_address_cd': 'numeric',
    'work_address_cd': 'numeric',
    'income': 'numeric',
    'SNA': 'numeric',
    'first_time_cd': 'numeric',
    'Air_flg': 'category'
}