"""
Clinical Input Validation & Sanitization Module
================================================
Guards machine learning models against invalid data types, negative numbers, 
physiologically impossible measurements, and biologically inconsistent ratios.
"""

from typing import Dict, List, Tuple, Union, Optional

# Standard adult clinical reference ranges & human survivable bounds
PHYSIOLOGICAL_BOUNDS = {
    "HGB": {
        "name": "Hemoglobin",
        "unit": "g/dL",
        "min": 2.0,
        "max": 25.0,
        "ref_min": 12.0,
        "ref_max": 17.5,
        "desc": "Oxygen-carrying protein in red blood cells."
    },
    "RBC": {
        "name": "Red Blood Cell Count",
        "unit": "M/µL",
        "min": 1.0,
        "max": 9.0,
        "ref_min": 3.8,
        "ref_max": 5.9,
        "desc": "Number of red blood cells per microliter of blood."
    },
    "MCV": {
        "name": "Mean Corpuscular Volume",
        "unit": "fL",
        "min": 40.0,
        "max": 150.0,
        "ref_min": 80.0,
        "ref_max": 100.0,
        "desc": "Average volume (size) of individual red blood cells."
    },
    "MCH": {
        "name": "Mean Corpuscular Hemoglobin",
        "unit": "pg",
        "min": 10.0,
        "max": 50.0,
        "ref_min": 27.0,
        "ref_max": 33.0,
        "desc": "Average mass of hemoglobin per red blood cell."
    },
    "MCHC": {
        "name": "Mean Corpuscular Hemoglobin Concentration",
        "unit": "g/dL",
        "min": 20.0,
        "max": 45.0,
        "ref_min": 31.0,
        "ref_max": 36.0,
        "desc": "Average concentration of hemoglobin inside red blood cells."
    }
}


def validate_blood_parameters(
    hgb: Union[float, int, str],
    rbc: Union[float, int, str],
    mcv: Union[float, int, str],
    mch: Union[float, int, str],
    mchc: Union[float, int, str]
) -> Tuple[bool, Union[Dict[str, float], str], List[str]]:
    """
    Validates complete blood count parameters across three defensive tiers:
      Tier 1: Type & Null validation (detects strings, empty, None)
      Tier 2: Physiological range checks (detects negatives, zero, absurd values)
      Tier 3: Biological plausibility & Wintrobe consistency checks

    Returns:
        (is_valid, cleaned_data_or_error_message, list_of_clinical_warnings)
    """
    raw_inputs = {
        "HGB": hgb,
        "RBC": rbc,
        "MCV": mcv,
        "MCH": mch,
        "MCHC": mchc
    }
    
    cleaned_values: Dict[str, float] = {}
    warnings: List[str] = []

    # -------------------------------------------------------------
    # TIER 1: Numeric Type & Null Interception
    # -------------------------------------------------------------
    for param, val in raw_inputs.items():
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return False, f"Missing input: '{param}' ({PHYSIOLOGICAL_BOUNDS[param]['name']}) cannot be empty.", []
        
        try:
            float_val = float(val)
        except (ValueError, TypeError):
            return (
                False,
                f"Type error: '{param}' received '{val}'. All blood test measurements must be valid numeric values.",
                []
            )
        
        # Check for NaN / Infinity
        if float_val != float_val or float_val in (float('inf'), float('-inf')):
            return False, f"Invalid value: '{param}' cannot be NaN or Infinite.", []

        cleaned_values[param] = round(float_val, 2)

    # -------------------------------------------------------------
    # TIER 2: Sign and Physiological Range Validation
    # -------------------------------------------------------------
    for param, val in cleaned_values.items():
        bounds = PHYSIOLOGICAL_BOUNDS[param]
        
        if val <= 0:
            return (
                False,
                f"Physiological violation: {bounds['name']} ({param}) cannot be zero or negative ({val} {bounds['unit']}).",
                []
            )

        if val < bounds["min"] or val > bounds["max"]:
            return (
                False,
                f"Out-of-range error: {bounds['name']} ({param}) value of {val} {bounds['unit']} falls outside human survivable laboratory limits ({bounds['min']} - {bounds['max']} {bounds['unit']}).",
                []
            )

    # -------------------------------------------------------------
    # TIER 3: Biological Cross-Parameter Plausibility (Wintrobe Ratio)
    # -------------------------------------------------------------
    # Expected MCHC = (MCH / MCV) * 100
    calc_mchc = (cleaned_values["MCH"] / cleaned_values["MCV"]) * 100.0
    diff = abs(cleaned_values["MCHC"] - calc_mchc)
    
    if diff > 10.0:
        warnings.append(
            f"Biological Consistency Advisory: Reported MCHC ({cleaned_values['MCHC']} g/dL) deviates from "
            f"Wintrobe calculated ratio (MCH/MCV * 100 = {calc_mchc:.1f} g/dL). "
            f"Please verify patient laboratory test values."
        )

    # Expected MCH = (HGB / RBC) * 10
    calc_mch = (cleaned_values["HGB"] / cleaned_values["RBC"]) * 10.0
    mch_diff = abs(cleaned_values["MCH"] - calc_mch)
    if mch_diff > 8.0:
        warnings.append(
            f"Biological Consistency Advisory: Reported MCH ({cleaned_values['MCH']} pg) deviates from "
            f"expected hemoglobin content per cell ((HGB/RBC)*10 = {calc_mch:.1f} pg)."
        )

    return True, cleaned_values, warnings


def get_parameter_status(param_name: str, value: float) -> Dict[str, str]:
    """
    Evaluates whether a single parameter falls in the Low, Normal, or High clinical range.
    Useful for rendering informative badges in reports and the Web UI.
    """
    param = param_name.upper()
    if param not in PHYSIOLOGICAL_BOUNDS:
        return {"status": "UNKNOWN", "badge_class": "secondary", "message": "Unknown parameter"}

    meta = PHYSIOLOGICAL_BOUNDS[param]
    ref_min = meta["ref_min"]
    ref_max = meta["ref_max"]

    if value < ref_min:
        return {
            "status": "LOW",
            "badge_class": "warning",
            "message": f"Below normal adult reference range ({ref_min}-{ref_max} {meta['unit']})"
        }
    elif value > ref_max:
        return {
            "status": "HIGH",
            "badge_class": "danger",
            "message": f"Above normal adult reference range ({ref_min}-{ref_max} {meta['unit']})"
        }
    else:
        return {
            "status": "NORMAL",
            "badge_class": "success",
            "message": f"Within healthy reference range ({ref_min}-{ref_max} {meta['unit']})"
        }
