import pytest
from doclens.fields import extract_fields, value_appears_in_text


def test_extract_explicit_values():
    s = "Invoice no: INV-204\nTotal payable: INR 860\nApplication deadline: 25 October 2026\n"
    got = extract_fields(s)["fields"]
    assert got["invoice_number"] == "INV-204"
    assert got["total_payable"] == "INR 860"
    assert got["application_deadline"] == "25 October 2026"
    assert got["meeting_date"] is None


def test_hindi_fields():
    s = "आवेदन जमा करने की अंतिम तिथि: 25 अक्टूबर 2026\nसमीक्षा बैठक: 4 नवंबर 2026"
    got = extract_fields(s)["fields"]
    assert got["application_deadline"] == "25 अक्टूबर 2026"
    assert got["meeting_date"] == "4 नवंबर 2026"


def test_match_is_not_proof():
    assert value_appears_in_text("  INR 860 ", "total payable: INR   860")
    assert not value_appears_in_text("INR 960", "total payable: INR 860")
    with pytest.raises(TypeError): extract_fields(None)
