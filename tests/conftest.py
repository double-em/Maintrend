import os
import shutil
import pytest


#def pytest_generate_tests(metafunc):
#        os.environ['API_BASE_URL'] = str(os.environ.get('API_CHANNEL'))

# @pytest.fixture
# def use_api_url(monkeypatch):
#     print(os.environ.get('API_BASE_URL'))
#     monkeypatch.setenv('API_BASE_URL', os.environ.get(['API_BASE_URL']))

# @pytest.fixture
# def use_api_channel(monkeypatch):
#     monkeypatch.setenv("API_CHANNEL", os.environ.get('API_CHANNEL'))

# @pytest.fixture
# def use_api_f(monkeypatch):
#     monkeypatch.setenv("API_F", os.environ.get('API_F'))

# @pytest.fixture
# def use_api_key(monkeypatch):
#     monkeypatch.setenv("API_KEY", os.environ.get('API_KEY'))

@pytest.fixture
def datadir(tmp_path, request):

    filename = request.module.__file__
    test_dir, _ = os.path.split(filename)

    files_list = os.listdir(test_dir)
    
    for file_name in files_list:
        full_file_name = os.path.join(test_dir, file_name)

        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, tmp_path)
    
    return tmp_path

@pytest.fixture
def downtime_data(datadir):
    import json
    
    with open(datadir / 'downtime_test_data.json') as json_file:
        return json.load(json_file)

@pytest.fixture
def production_data(datadir):
    import json

    with open(datadir / 'production_test_data.json') as json_file:
        return json.load(json_file)