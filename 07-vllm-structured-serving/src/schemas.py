from typing import Literal, Optional
from pydantic import BaseModel
from datasets import load_dataset

class Patient(BaseModel):
    patientId : str
    firstName : str
    lastName : str 
    dateOfBirth : str
    gender  : str 
    medicalConditions : Optional[list[str]] = None  

class Medications(BaseModel):
    medicationId : str
    name : str
    dosage : float
    units : str 
    instructions : str 
    refills : int

class Pharmacy(BaseModel):
    pharmacyId : str 
    name : str
    address : str
    phone : Optional[str] = None 

class Prescription(BaseModel):
    prescriptionId : str
    patient : Patient
    medication : Medications
    pharmacy : Pharmacy
    datePrescribed : str
    dateExpires : Optional[str] = None
    status : Optional[Literal["active", "inactive", "expired"]] = None 

class PrescriptionRecord(BaseModel):
    prescription : Prescription
