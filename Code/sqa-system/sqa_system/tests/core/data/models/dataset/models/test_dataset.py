from datetime import datetime

import pytest

from pydantic import BaseModel
from sqa_system.core.data.models.dataset.base.dataset import Dataset


class TestEntry(BaseModel):
    __test__ = False
    name: str
    value: int
    
@pytest.fixture
def sample_dataset():
    initial_data = {
        "entry1": TestEntry(name="Entry 1", value=1),
        "entry2": TestEntry(name="Entry 2", value=2),
    }
    return Dataset[TestEntry](name="Test Dataset", data=initial_data)
    
def test_dataset_initialization():
    """Test the initialization of the dataset."""
    initial_data = {
        "entry1": TestEntry(name="Entry 1", value=1),
        "entry2": TestEntry(name="Entry 2", value=2),
    }
    dataset = Dataset[TestEntry](name="Test Dataset", data=initial_data)
    assert dataset.name == "Test Dataset"
    assert isinstance(dataset.id, str)
    assert isinstance(dataset.created_at, datetime)
    assert isinstance(dataset.updated_at, datetime)
    assert len(dataset.get_all_entries()) == 2
    assert dataset.get_entry("entry1").name == "Entry 1"
    assert dataset.get_entry("entry2").value == 2
    
def test_add_entry(sample_dataset):
    """Test adding a single entry to the dataset."""
    entry = TestEntry(name="Test Entry", value=42)
    sample_dataset.add_entry("test_id_2", entry)
    assert len(sample_dataset.get_all_entries()) == 3
    assert sample_dataset.get_entry("test_id_2") == entry

def test_add_entries(sample_dataset):
    """Test adding multiple entries to the dataset."""
    new_entries = {
        "entry3": TestEntry(name="Entry 3", value=3),
        "entry4": TestEntry(name="Entry 4", value=4),
    }
    sample_dataset.add_entries(new_entries)
    assert len(sample_dataset.get_all_entries()) == 4
    assert sample_dataset.get_entry("entry3").name == "Entry 3"
    assert sample_dataset.get_entry("entry4").value == 4
    
def test_add_entries_with_existing_keys(sample_dataset):
    """Test adding multiple entries with existing keys to the dataset."""
    new_entries = {
        "entry1": TestEntry(name="New Entry 1", value=10),
        "entry3": TestEntry(name="Entry 3", value=3),
    }
    sample_dataset.add_entries(new_entries)
    assert len(sample_dataset.get_all_entries()) == 3
    assert sample_dataset.get_entry("entry1").name == "Entry 1"
    assert sample_dataset.get_entry("entry3").name == "Entry 3"
    
def test_update_entry(sample_dataset):
    """Test updating an existing entry in the dataset."""
    updated_entry = TestEntry(name="Updated Entry", value=100)
    sample_dataset.update_entry("entry1", updated_entry)
    assert sample_dataset.get_entry("entry1") == updated_entry
    
def test_update_nonexistent_entry(sample_dataset):
    """Test updating a nonexistent entry in the dataset."""
    entry = TestEntry(name="Test Entry", value=42)
    with pytest.raises(KeyError):
        sample_dataset.update_entry("nonexistent_id", entry)

def test_delete_entry(sample_dataset):
    """Test deleting an entry from the dataset."""
    sample_dataset.delete_entry("entry1")
    assert len(sample_dataset.get_all_entries()) == 1
    with pytest.raises(KeyError):
        sample_dataset.get_entry("entry1")

def test_delete_nonexistent_entry(sample_dataset):
    """Test deleting a nonexistent entry from the dataset."""
    with pytest.raises(KeyError):
        sample_dataset.delete_entry("nonexistent_id")
        
def test_get_all_entries(sample_dataset):
    """Test getting all entries from the dataset."""
    all_entries = sample_dataset.get_all_entries()
    assert len(all_entries) == 2
    assert {entry.name for entry in all_entries} == {"Entry 1", "Entry 2"}
    
    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])
