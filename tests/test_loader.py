import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data.loader import load_json, load_dataset
import os
import json

def test_load_json(tmp_path):
    test_file = tmp_path / "test.json"
    test_data = {"courses": [{"name": "Test"}]}
    test_file.write_text(json.dumps(test_data))
    loaded = load_json(str(test_file))
    assert loaded == test_data

def test_load_dataset_dict_format(tmp_path):
    test_file = tmp_path / "courses.json"
    test_file.write_text(json.dumps({"courses": [{"name": "Sec Course"}]}))
    dataset = load_dataset(str(test_file))
    assert isinstance(dataset, list)
    assert dataset[0]["name"] == "Sec Course"
