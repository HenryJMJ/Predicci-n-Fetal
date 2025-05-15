from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('entrenamiento/', views.entrenamiento, name='entrenamiento'),
    path('prediccion/individual/', views.prediccion_individual, name='prediccion_individual'),
    path('prediccion/lote/', views.prediccion_lote, name='prediccion_lote'),
    path('historial/', views.historial_predicciones, name='historial_predicciones'),
    path('historial/<int:id>/', views.detalles_prediccion, name='detalles_prediccion')

]
