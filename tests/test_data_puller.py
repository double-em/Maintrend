import pytest
import json
import pandas as pd
import numpy as np
from util.data_puller import down_time_transformer, product_produced_transformer

class TestDowntimeTransformer:

    def test_nan_values(self, downtime_data):
        df = down_time_transformer(downtime_data)

        assert df.isna().sum().sum() == 0

    def test_length(self, downtime_data):
        df = down_time_transformer(downtime_data)

        assert len(df.values) == 3

    def test_times_down(self, downtime_data):
        df = down_time_transformer(downtime_data)

        assert df.iloc[0]['times_down'] == 7

    def test_columns_exists(self, downtime_data):
        df = down_time_transformer(downtime_data)
        df.reset_index(inplace=True)

        assert 'timestamp' in df
        assert 'downtime' in df
        assert 'times_down' in df
        assert 'comment' in df

class TestProductProducedTransformer:

    def test_nan_values(self, production_data):
        df = product_produced_transformer(production_data)

        assert df.isna().sum().sum() == 0

    def test_length(self, production_data):
        df = product_produced_transformer(production_data)

        assert len(df.values) == 7

    def test_columns_exists(self, production_data):
        df = product_produced_transformer(production_data)
        df.reset_index(inplace=True)

        assert 'timestamp' in df
        assert 'produced' in df