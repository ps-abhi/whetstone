from validate import validate_response, Verdict

GOOD = '{"prescription": {"prescriptionId": "RX123456", "patient": {"patientId": "PT123456", "firstName": "Jane", "lastName": "Doe", "dateOfBirth": "1980-01-01", "gender": "female", "medicalConditions": ["Asthma", "Hypertension"]}, "medication": {"medicationId": "MD123456", "name": "Albuterol", "dosage": 200, "units": "mcg", "instructions": "Inhale 2 puffs every 4-6 hours as needed for shortness of breath.", "refills": 3}, "pharmacy": {"pharmacyId": "PH123456", "name": "CVS Pharmacy", "address": "123 Main Street, Anytown, CA 12345", "phone": "(123) 456-7890"}, "datePrescribed": "2023-03-08", "dateExpires": "2023-09-07", "status": "active"}}'

def test_valid():
    assert validate_response(GOOD) == Verdict.VALID

def test_truncated():
    assert validate_response(GOOD, "length") == Verdict.TRUNCATED

def test_parse_error():
    assert validate_response('{"prescription": {') == Verdict.PARSE_ERROR

def test_schema_error():
    assert validate_response('{"prescription": {"status": "foo"}}') == Verdict.SCHEMA_ERROR
