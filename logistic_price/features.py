"""Select P4's ordered inputs from the shared feature-table v1."""
from src.common import features as common_features

GROUPS = {
    'calendar': ['slot_sin', 'slot_cos', 'season_sin', 'season_cos', 'weekend',
                 *[f'weekday_{i}' for i in range(7)]],
    'history': ['nsw_demand_lag1', 'nsw_price_lag1', 'nsw_demand_lag7',
                'nsw_price_lag7', 'prev_price_mean'],
    'cross_state': ['qld_demand_lag1'],
    'rolling28': ['nsw_demand_mean28', 'nsw_demand_std28', 'nsw_price_mean28', 'nsw_price_std28'],
}
FEATURES = [col for group in GROUPS.values() for col in group]
MARKET = ['nsw_demand', 'nsw_price', 'qld_demand', 'vic_demand', 'vic_price']
WEATHER = ['syd_temp', 'syd_rh', 'syd_cloud', 'syd_solar']
RAW = [*MARKET, *WEATHER]


def build_features(data):
    """Use the shared batch calculation, retaining the original 22-column order."""
    return common_features.build_features(data).loc[:, FEATURES]


def build_features_for_day(history, weather_next):
    """Use the full past prefix and 48 weather rows required by shared v1."""
    return common_features.build_features_for_day(history, weather_next).loc[:, FEATURES]
