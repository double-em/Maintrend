import os
import pytest

#def pytest_generate_tests(metafunc):
#        os.environ['API_BASE_URL'] = str(os.environ.get('API_CHANNEL'))

@pytest.fixture
def use_api_url(monkeypatch):
    print(os.environ.get('API_BASE_URL'))
    monkeypatch.setenv('API_BASE_URL', os.environ.get(['API_BASE_URL']))

@pytest.fixture
def use_api_channel(monkeypatch):
    monkeypatch.setenv("API_CHANNEL", os.environ.get('API_CHANNEL'))

@pytest.fixture
def use_api_f(monkeypatch):
    monkeypatch.setenv("API_F", os.environ.get('API_F'))

@pytest.fixture
def use_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", os.environ.get('API_KEY'))