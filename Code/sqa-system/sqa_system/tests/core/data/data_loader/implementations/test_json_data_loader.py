import pytest
from sqa_system.core.data.data_loader.implementations.json_publication_loader import JsonPublicationLoader
from sqa_system.core.data.file_path_manager import FilePathManager

def test_json_data_loader():
    """Test the JsonPublicationLoader with our json file."""
    path_manager = FilePathManager()
    json_file_path = path_manager.get_path("merged_ecsa_icsa.json")

    field_mapping = {
            "doi": "doi",
            "authors": "authors",
            "title": "title",
            "year": "year",
            "venue": "venue_name",
            "abstract": "abstract",
            "research_field": "track",
            "full_text": "fulltext"
    }
    json_loader = JsonPublicationLoader(field_mapping=field_mapping)

    dataset = json_loader.load("test_dataset", str(json_file_path))

    assert dataset is not None
    assert len(dataset.get_all_entries()) == 153
    
    # check if a specific entry exists    
    entry = dataset.get_entry("10.1007/978-3-030-86044-8_21")
    assert entry is not None
    assert entry.title == "How Software Architects Focus Their Attention"
    assert entry.year == 2021
    
def test_extract_annotations():
    """Test the extraction of annotations."""
    field_mapping = {
        "doi": "doi",
        "authors": "authors",
        "title": "title",
        "year": "year",
        "venue": "venue_name",
        "abstract": "abstract",
        "research_field": "track",
        "full_text": "fulltext"
    }
    json_loader = JsonPublicationLoader(field_mapping=field_mapping)
    extraction_text = "Meta Data{Paper class{Validation Research}, Research Level{Primary Research}, Kind{full}}, Validity{Input Data{used}, Threats To Validity{External Validity, Internal Validity, Construct Validity, Confirmability}, Referenced Threats To Validity Guideline}, First Research Object{First Evaluation{Evaluation Method{Questionnaire}, Properties{Quality in Use{Context coverage}}}, Research Object{Architecture Decision Making}}"
    
    extraction, _ = json_loader._extract_annotations("annotations", extraction_text)

    assert extraction == {
        "Meta Data": {
            "Paper class": "Validation Research",
            "Research Level": "Primary Research",
            "Kind": "full"
        },
        "Validity": {
            "Input Data": "used",
            "Threats To Validity": [
                "External Validity",
                "Internal Validity",
                "Construct Validity",
                "Confirmability"
            ],
            "Referenced Threats To Validity Guideline": True
        },
        "First Research Object": {
            "First Evaluation": {
                "Evaluation Method": "Questionnaire",
                "Properties": {
                    "Quality in Use": "Context coverage"
                }
            },
            "Research Object": "Architecture Decision Making"
        }
    }
    
    extraction_text_2 = "First Research Object{Research Object{Architecture Analysis Method}, First Evaluation{Properties{Product Quality{Functional Suitability}}, Evaluation Method{Technical Experiment}}}, Second Research Object{Research Object{Architecture Optimization Method}}, Validity{Tool Support{available}, Replication Package, Input Data{available}, Threats To Validity{External Validity, Internal Validity}}, Meta Data{Research Level{Primary Research}, Kind{full}, Paper class{Proposal of Solution, Validation Research}}"
    
    extraction, _ = json_loader._extract_annotations("annotations", extraction_text_2)
    
    assert extraction == {
        "First Research Object": {
            "Research Object": "Architecture Analysis Method",
            "First Evaluation": {
                "Properties": {
                    "Product Quality": "Functional Suitability"
                },
                "Evaluation Method": "Technical Experiment"
            }
        },
        "Second Research Object": {
            "Research Object": "Architecture Optimization Method"
        },
        "Validity": {
            "Tool Support": "available",
            "Replication Package": "true",
            "Input Data": "available",
            "Threats To Validity": [
                "External Validity",
                "Internal Validity"
            ]
        },
        "Meta Data": {
            "Research Level": "Primary Research",
            "Kind": "full",
            "Paper class": [
                "Proposal of Solution",
                "Validation Research"
            ]
        }
    }

    
if __name__ == "__main__":
    import sys
    pytest.main([sys.argv[0], "-v"])