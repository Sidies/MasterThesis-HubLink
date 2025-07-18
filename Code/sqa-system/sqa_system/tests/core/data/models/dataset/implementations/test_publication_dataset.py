from datetime import datetime

import pytest

from sqa_system.core.data.models.dataset.base.dataset import Dataset
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.models.dataset.implementations.publication_dataset import PublicationDataset

from typing import List

@pytest.fixture
def sample_publications() -> List[Publication]:
    return [
        Publication(doi="abcd",
                    authors=["Author 1", "Author 2"],
                    title="Test Publication",
                    year=2024,
                    venue="Test Venue",
                    abstract="Test Abstract",
                    research_field="Test Research Field",
                    full_text="Test Full Text"),
        Publication(doi="efgh",
                    authors=["Author 3", "Author 4"],
                    title="Test Publication 2",
                    year=2024,
                    venue="Test Venue 2",
                    abstract="Test Abstract 2",
                    research_field="Test Research Field 2",
                    full_text="Test Full Text 2"),
    ]

@pytest.fixture
def sample_publication_dataset():
    initial_data = {
        "publication1": Publication(doi="ijkl",
                                   authors=["Author 1", "Author 2"],
                                   title="Test Publication",
                                   year=2024,
                                   venue="Test Venue",
                                   abstract="Test Abstract",
                                   research_field="Test Research Field",
                                   full_text="Test Full Text"),
        "publication2": Publication(doi="mnop",
                                   authors=["Author 3", "Author 4"],
                                   title="Test Publication 2",
                                   year=2024,
                                   venue="Test Venue 2",
                                   abstract="Test Abstract 2",
                                   research_field="Test Research Field 2",
                                   full_text="Test Full Text 2"),
    }
    return PublicationDataset(name="Test Publication Dataset", data=initial_data)

def test_initialization(sample_publication_dataset):
    """Test the initialization of the PublicationDataset."""
    assert sample_publication_dataset.name == "Test Publication Dataset"
    assert len(sample_publication_dataset.get_all_entries()) == 2

def test_add_entry(sample_publication_dataset):
    """Test adding a single entry to the PublicationDataset."""
    entry = Publication(doi="qrst",
                        authors=["Author 5", "Author 6"],
                        title="Test Publication 3",
                        year=2024,
                        venue="Test Venue 3",
                        abstract="Test Abstract 3",
                        research_field="Test Research Field 3",
                        full_text="Test Full Text 3")
    sample_publication_dataset.add_entry("test_id_1", entry)
    assert len(sample_publication_dataset.get_all_entries()) == 3
    assert sample_publication_dataset.get_entry("test_id_1") == entry
        
def test_add_entries(sample_publication_dataset):
    """Test adding multiple entries to the PublicationDataset."""
    entries = {
        "publication3": Publication(doi="qrst",
                    authors=["Author 5", "Author 6"],
                    title="Test Publication 3",
                    year=2024,
                    venue="Test Venue 3",
                    abstract="Test Abstract 3",
                    research_field="Test Research Field 3",
                    full_text="Test Full Text 3"),
        "publication4": Publication(doi="uvwx",
                    authors=["Author 7", "Author 8"],
                    title="Test Publication 4",
                    year=2024,
                    venue="Test Venue 4",
                    abstract="Test Abstract 4",
                    research_field="Test Research Field 4",
                    full_text="Test Full Text 4"),
    }
    sample_publication_dataset.add_entries(entries)
    assert len(sample_publication_dataset.get_all_entries()) == 4
    assert sample_publication_dataset.get_entry("publication3") == entries["publication3"]
    assert sample_publication_dataset.get_entry("publication4") == entries["publication4"]
    
def test_get_all_entries(sample_publication_dataset):
    """Test getting all entries from the PublicationDataset."""
    assert len(sample_publication_dataset.get_all_entries()) == 2
    assert {entry.doi for entry in sample_publication_dataset.get_all_entries()} == {"ijkl", "mnop"}

def test_update_entry(sample_publication_dataset):
    """Test updating an entry in the PublicationDataset."""
    entry = sample_publication_dataset.get_entry("publication1")
    entry.title = "Updated Title"
    sample_publication_dataset.update_entry("publication1", entry)
    assert sample_publication_dataset.get_entry("publication1").title == "Updated Title"
    
def test_delete_entry(sample_publication_dataset):
    """Test deleting an entry from the PublicationDataset."""
    sample_publication_dataset.delete_entry("publication1")
    assert len(sample_publication_dataset.get_all_entries()) == 1
