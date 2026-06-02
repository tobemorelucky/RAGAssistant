import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from query_parser import parse_query


def test_jpm_not_ba_false_positive():
    result = parse_query("If JPM went bankrupted by the end by 2021 Q1, what would happen?")
    assert result["company"] == "jpmorgan"
    assert result["matched_company_alias"] == "JPM"


def test_boeing_effective_tax_rate():
    result = parse_query("How does Boeing's effective tax rate in FY2022 compare to FY2021?")
    assert result["company"] == "boeing"
    assert 2022 in result["years"]
    assert 2021 in result["years"]
    assert "effective tax rate" in result["metrics"]


def test_adobe_operating_margin():
    result = parse_query("Does Adobe have an improving operating margin profile as of FY2022?")
    assert result["company"] == "adobe"
    assert 2022 in result["years"]
    assert "operating margin" in result["metrics"]


def test_amd_quick_ratio_fy22():
    result = parse_query("Does AMD have a reasonably healthy liquidity profile based on its quick ratio for FY22?")
    assert result["company"] == "amd"
    assert 2022 in result["years"]
    assert "quick ratio" in result["metrics"]
