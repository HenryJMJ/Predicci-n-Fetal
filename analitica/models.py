from django.db import models
from django.utils import timezone

class HistorialPrediccion(models.Model):
    nombre = models.CharField(max_length=100, null=True, blank=True)

    fecha = models.DateTimeField(default=timezone.now)

    # Campos C1 a C30 con valores por defecto
    C1 = models.IntegerField(verbose_name="Age", default=0)
    C2 = models.FloatField(verbose_name="BMI", default=0.0)
    C3 = models.IntegerField(verbose_name="Gestational age of delivery", default=0)
    C4 = models.IntegerField(verbose_name="Gravidity", default=0)
    C5 = models.IntegerField(verbose_name="Parity", default=0)
    C6 = models.IntegerField(verbose_name="Initial onset symptoms", default=0)
    C7 = models.IntegerField(verbose_name="Gestational age of IOS onset", default=0)
    C8 = models.IntegerField(verbose_name="Interval IOS to delivery", default=0)
    C9 = models.IntegerField(verbose_name="Gestational age hypertension onset", default=0)
    C10 = models.IntegerField(verbose_name="Interval hypertension to delivery", default=0)
    C11 = models.IntegerField(verbose_name="Gestational age edema onset", default=0)
    C12 = models.IntegerField(verbose_name="Interval edema to delivery", default=0)
    C13 = models.IntegerField(verbose_name="Gestational age proteinuria onset", default=0)
    C14 = models.IntegerField(verbose_name="Interval proteinuria to delivery", default=0)
    C15 = models.IntegerField(verbose_name="Expectant treatment", default=0)
    C16 = models.IntegerField(verbose_name="Antihypertensive before hosp", default=0)
    C17 = models.IntegerField(verbose_name="Past history", default=0)
    C18 = models.FloatField(verbose_name="Max systolic BP", default=0.0)
    C19 = models.FloatField(verbose_name="Max diastolic BP", default=0.0)
    C20 = models.IntegerField(verbose_name="Reasons for delivery", default=0)
    C21 = models.IntegerField(verbose_name="Mode of delivery", default=0)
    C22 = models.FloatField(verbose_name="Max BNP", default=0.0)
    C23 = models.FloatField(verbose_name="Max creatinine", default=0.0)
    C24 = models.FloatField(verbose_name="Max uric acid", default=0.0)
    C25 = models.FloatField(verbose_name="Max proteinuria", default=0.0)
    C26 = models.FloatField(verbose_name="Max total protein", default=0.0)
    C27 = models.FloatField(verbose_name="Max albumin", default=0.0)
    C28 = models.FloatField(verbose_name="Max ALT", default=0.0)
    C29 = models.FloatField(verbose_name="Max AST", default=0.0)
    C30 = models.FloatField(verbose_name="Max platelet", default=0.0)

    # Resultados de los modelos
    resultado_log = models.CharField(max_length=10)
    resultado_svm = models.CharField(max_length=10)
    resultado_rna = models.CharField(max_length=10)
    resultado_mcd = models.CharField(max_length=50)

    def __str__(self):
        return f"Predicción del {self.fecha.strftime('%Y-%m-%d %H:%M')}"
