#!/bin/bash
# Script para ativar o ambiente virtual do projeto Diabetes GOP
echo "Ativando ambiente virtual do projeto Diabetes GOP..."
source venv/bin/activate
echo "Ambiente virtual ativado!"
echo "Pacotes principais instalados:"
echo "- pandas >= 1.5.0"
echo "- numpy >= 1.24.0"
echo "- matplotlib >= 3.5.0"
echo "- seaborn >= 0.11.0"
echo "- scikit-learn >= 1.2.0"
echo "- jupyter >= 1.0.0"
echo "- imbalanced-learn >= 0.10.0"
echo ""
echo "Para usar o Jupyter Lab: jupyter lab"
echo "Para usar o Jupyter Notebook: jupyter notebook"
echo "Para desativar o ambiente: deactivate"
