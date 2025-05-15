from django import forms

class PrediccionIndividualForm(forms.Form):
    nombre = forms.CharField(label='Nombre de la embarazada', max_length=100)
    
    Age = forms.FloatField(label='Edad de la madre', min_value=10, max_value=60)  # C1
    BMI = forms.FloatField(label='IMC', min_value=10)  # C2
    Gestational_age_of_delivery = forms.FloatField(label='Edad gestacional al parto (semanas)', min_value=20, max_value=45)  # C3
    Gravidity = forms.IntegerField(label='Gravidez', min_value=0)  # C4
    Parity = forms.IntegerField(label='Paridad', min_value=0)  # C5
    Initial_onset_symptoms = forms.ChoiceField(label='Síntoma inicial',
        choices=[(0, 'Edema'), (1, 'Hipertensión'), (2, 'FGR')])  # C6
    Gestational_age_of_IOS_onset = forms.FloatField(label='Edad gestacional al inicio de los síntomas', min_value=10, max_value=45)  # C7
    Interval_IOS_to_delivery = forms.FloatField(label='Intervalo entre síntomas iniciales y parto (días)', min_value=0)  # C8
    Gestational_age_hypertension_onset = forms.FloatField(label='Edad gestacional al inicio de hipertensión', min_value=10, max_value=45)  # C9
    Interval_hypertension_to_delivery = forms.FloatField(label='Intervalo entre hipertensión y parto (días)', min_value=0)  # C10
    Gestational_age_edema_onset = forms.FloatField(label='Edad gestacional al inicio de edema', min_value=10, max_value=45)  # C11
    Interval_edema_to_delivery = forms.FloatField(label='Intervalo entre edema y parto (días)', min_value=0)  # C12
    Gestational_age_proteinuria_onset = forms.FloatField(label='Edad gestacional al inicio de proteinuria', min_value=10, max_value=45)  # C13
    Interval_proteinuria_to_delivery = forms.FloatField(label='Intervalo entre proteinuria y parto (días)', min_value=0)  # C14
    Expectant_treatment = forms.FloatField(label='Tratamiento expectante (1=sí, 0=no)', min_value=0, max_value=1)  # C15
    Antihypertensive_before_hosp = forms.FloatField(label='Antihipertensivos antes de hospitalización (1=sí, 0=no)', min_value=0, max_value=1)  # C16
    Past_history = forms.ChoiceField(label='Antecedentes',
        choices=[(0, 'Ninguno'), (1, 'Hipertensión'), (2, 'SOP')])  # C17
    Maximum_systolic_blood_pressure = forms.FloatField(label='Presión sistólica máxima', min_value=80)  # C18
    Maximum_diastolic_blood_pressure = forms.FloatField(label='Presión diastólica máxima', min_value=40)  # C19
    Reasons_for_delivery = forms.ChoiceField(label='Razón para el parto',
        choices=[(0, 'HELLP'), (1, 'Distres fetal'), (2, 'Disfunción orgánica'),
                 (3, 'Hipertensión incontrolada'), (4, 'Edema'), (5, 'FGR')])  # C20
    Mode_of_delivery = forms.ChoiceField(label='Modo de parto',
        choices=[(0, 'Cesárea'), (1, 'Parto vaginal')])  # C21
    Maximum_BNP_value = forms.FloatField(label='Valor máximo de BNP')  # C22
    Maximum_creatinine_value = forms.FloatField(label='Creatinina máxima')  # C23
    Maximum_uric_acid = forms.FloatField(label='Ácido úrico máximo')  # C24
    Maximum_proteinuria = forms.FloatField(label='Proteinuria máxima')  # C25
    Maximum_total_protein = forms.FloatField(label='Proteína total máxima')  # C26
    Maximum_albumin = forms.FloatField(label='Albúmina máxima')  # C27
    Maximum_ALT = forms.FloatField(label='ALT máxima')  # C28
    Maximum_AST = forms.FloatField(label='AST máxima')  # C29
    Maximum_platelet = forms.FloatField(label='Plaquetas máximas')  # C30
