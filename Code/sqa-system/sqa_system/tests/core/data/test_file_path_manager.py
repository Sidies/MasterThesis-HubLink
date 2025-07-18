import os
import pytest
import json
from unittest.mock import MagicMock, Mock, patch
from sqa_system.core.data.file_path_manager import FilePathManager

@pytest.fixture
def file_path_manager():
    return FilePathManager()

def test_init(file_path_manager):
    """Test the initialization of the FilePathManager."""
    # check if default file paths are given
    file_path = FilePathManager.FILE_PATHS_JSON
    if not os.path.exists(file_path):
        return
    
    # else load the contents and check if the file path manager
    # also provides these file paths
    with open(file_path, 'r', encoding='utf-8') as f:
        file_paths = json.load(f)
        
    # assert that the length of the file paths is the same
    assert len(file_paths) == len(file_path_manager.get_all_file_names())
    
def test_get_path(file_path_manager):
    """Test the get_path method of the FilePathManager."""
    # check if default file paths are given
    file_path = FilePathManager.FILE_PATHS_JSON
    if not os.path.exists(file_path):
        return
    
    # else load the contents and check if the file path manager
    # also provides these file paths
    with open(file_path, 'r', encoding='utf-8') as f:
        file_paths = json.load(f)
    
    for file_name in file_paths:
        assert file_path_manager.get_path(file_name)