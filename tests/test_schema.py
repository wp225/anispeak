from anispeakarchive.corpus.schema import _abstract_from_inverted_index
from anispeakarchive.corpus.schema import to_web_doc

def test_abstract_reconstruction():
    inv = {"hello": [0], "world": [1]}
    txt = _abstract_from_inverted_index(inv)
    assert txt == "hello world"


def test_web_doc_text_includes_metadata_fields() -> None:
    raw = {
        "id": "https://openalex.org/W123",
        "title": "Bird Song Analysis",
        "publication_year": 2024,
        "publication_date": "2024-03-01",
        "summary": "A study of bird vocalization corpora.",
        "abstract_inverted_index": {"Bird": [0], "calls": [1]},
        "authorships": [
            {
                "author": {"display_name": "Jane Doe"},
                "institutions": [{"display_name": "Cornell University"}],
                "raw_affiliation_strings": ["Cornell Lab of Ornithology"],
            }
        ],
        "best_oa_location": {
            "landing_page_url": "https://example.org/paper",
            "pdf_url": "https://example.org/paper.pdf",
            "source": {"display_name": "Journal of Bioacoustics"},
        },
        "species": [{"name": "zebra finch", "category": "bird"}],
        "stages": [{"stage": "classification"}],
        "features": [{"feature": "pitch"}],
        "modalities": [{"modality": "audio"}],
        "domains": [{"domain": "bioacoustics"}],
        "contexts": [{"context": "field"}],
        "research_tradition": "computational",
        "paper_type": "dataset",
        "social_scale": "individual",
        "has_dataset": True,
        "has_code": True,
    }
    doc = to_web_doc(raw, include_abstract=True, include_affiliations=True)
    text = doc["text"]
    assert "Title: Bird Song Analysis" in text
    assert "Summary: A study of bird vocalization corpora." in text
    assert "Abstract: Bird calls" in text
    assert "Authors: Jane Doe" in text
    assert "Institutions: Cornell University" in text
    assert "Venue: Journal of Bioacoustics" in text
    assert "Year: 2024" in text
    assert "Publication date: 2024-03-01" in text
    assert "Resources: has dataset, has code" in text
    assert "VENUE:Journal of Bioacoustics" in text
    assert "YEAR:2024" in text
    assert "AUTHOR:Jane Doe" in text
    assert "INSTITUTION:Cornell University" in text
