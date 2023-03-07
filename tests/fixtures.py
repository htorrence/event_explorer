import pandas as pd
import pytest
from datetime import datetime


@pytest.fixture()
def df():
    return pd.read_csv('./tests/test_data.csv')

