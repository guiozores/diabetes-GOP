#!/usr/bin/env python3
"""
🩺 Sistema de Predição de Diabetes - Demonstração Prática
========================================================

Este script demonstra como usar os modelos treinados para fazer predições
de diabetes em novos pacientes, seguindo as melhores práticas implementadas
no projeto de reorganização CRISP-DM.

Uso:
    python demo_predicao_diabetes.py

Funcionalidades:
    - Carregamento dos modelos treinados
    - Predição com threshold otimizado
    - Análise de risco contextual
    - Exemplos práticos de uso clínico

Data: Junho 2025
Versão: 1.0 - Implementação Completa CRISP-DM
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def carregar_sistema_predicao():
    """Carrega todos os componentes necessários para predição"""
    
    print("🔄 Carregando sistema de predição de diabetes...")
    
    # Verificar se a pasta de modelos existe
    pasta_modelos = Path('todos-modelos')
    if not pasta_modelos.exists():
        raise FileNotFoundError("❌ Pasta 'todos-modelos' não encontrada! Execute o notebook principal primeiro.")
    
    try:
        # Carregar componentes principais
        modelo_rf = joblib.load('todos-modelos/random_forest_model.pkl')
        scaler = joblib.load('todos-modelos/scaler.pkl')
        feature_columns = joblib.load('todos-modelos/feature_columns.pkl')
        
        print("✅ Sistema carregado com sucesso!")
        print(f"   📊 Modelo: Random Forest (melhor AUC)")
        print(f"   🔧 Scaler: StandardScaler configurado")
        print(f"   📝 Features: {len(feature_columns)} variáveis")
        
        return modelo_rf, scaler, feature_columns
        
    except Exception as e:
        print(f"❌ Erro ao carregar sistema: {e}")
        return None, None, None

def interpretar_risco(probabilidade):
    """Interpreta o nível de risco baseado na probabilidade"""
    
    if probabilidade >= 0.8:
        return "🔴 MUITO ALTO", "Forte indicação de diabetes - Investigação urgente"
    elif probabilidade >= 0.6:
        return "🟠 ALTO", "Risco elevado - Exames complementares recomendados"
    elif probabilidade >= 0.4:
        return "🟡 MODERADO", "Risco moderado - Monitoramento regular"
    elif probabilidade >= 0.2:
        return "🟢 BAIXO", "Risco baixo - Prevenção e estilo de vida saudável"
    else:
        return "💚 MUITO BAIXO", "Risco muito baixo - Manter hábitos saudáveis"

def predizer_diabetes(dados_paciente, modelo, scaler, feature_columns, 
                     threshold=0.45, contexto="consulta_medica"):
    """
    Realiza predição de diabetes para um paciente
    
    Args:
        dados_paciente: Lista com [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                  Insulin, BMI, DiabetesPedigreeFunction, Age]
        modelo: Modelo treinado
        scaler: Normalizador ajustado
        feature_columns: Nome das features
        threshold: Limiar de decisão (default: 0.45 - otimizado)
        contexto: Contexto clínico ('triagem', 'consulta_medica', 'preventiva')
    
    Returns:
        dict: Resultados da predição
    """
    
    # Ajustar threshold baseado no contexto
    thresholds_contexto = {
        'triagem': 0.35,          # Máxima detecção para triagem populacional
        'consulta_medica': 0.45,  # Equilíbrio para consulta médica
        'preventiva': 0.60        # Conservador para medicina preventiva
    }
    
    if contexto in thresholds_contexto:
        threshold = thresholds_contexto[contexto]
    
    # Converter para DataFrame
    dados_df = pd.DataFrame([dados_paciente], columns=feature_columns)
    
    # Normalizar dados
    dados_normalizados = scaler.transform(dados_df)
    
    # Fazer predição
    probabilidade = modelo.predict_proba(dados_normalizados)[0][1]
    predicao = 1 if probabilidade >= threshold else 0
    
    # Interpretar risco
    nivel_risco, descricao_risco = interpretar_risco(probabilidade)
    
    # Compilar resultados
    resultado = {
        'predicao': 'DIABETES' if predicao == 1 else 'NÃO-DIABETES',
        'probabilidade': probabilidade,
        'nivel_risco': nivel_risco,
        'descricao_risco': descricao_risco,
        'threshold_usado': threshold,
        'contexto': contexto,
        'confianca': 'Alta' if abs(probabilidade - 0.5) > 0.3 else 'Moderada'
    }
    
    return resultado

def exibir_resultado(dados_paciente, resultado, feature_columns):
    """Exibe o resultado da predição de forma organizada"""
    
    print(f"\n" + "="*70)
    print(f"🩺 RELATÓRIO DE PREDIÇÃO DE DIABETES")
    print(f"="*70)
    
    # Dados do paciente
    print(f"\n👤 DADOS DO PACIENTE:")
    labels = [
        "Gestações", "Glicemia (mg/dL)", "Pressão Arterial (mmHg)", 
        "Espessura da Pele (mm)", "Insulina (mu U/ml)", "IMC (kg/m²)", 
        "Herança Genética", "Idade (anos)"
    ]
    
    for i, (label, valor) in enumerate(zip(labels, dados_paciente)):
        print(f"   {label:<20}: {valor}")
    
    # Resultado da predição
    print(f"\n🔬 RESULTADO DA ANÁLISE:")
    print(f"   Predição Final     : {resultado['predicao']}")
    print(f"   Probabilidade      : {resultado['probabilidade']:.1%}")
    print(f"   Nível de Risco     : {resultado['nivel_risco']}")
    print(f"   Contexto Clínico   : {resultado['contexto'].replace('_', ' ').title()}")
    print(f"   Threshold Utilizado: {resultado['threshold_usado']:.2f}")
    print(f"   Confiança          : {resultado['confianca']}")
    
    # Interpretação clínica
    print(f"\n💡 INTERPRETAÇÃO CLÍNICA:")
    print(f"   {resultado['descricao_risco']}")
    
    # Recomendações
    print(f"\n📋 RECOMENDAÇÕES:")
    if resultado['probabilidade'] >= 0.6:
        print(f"   • Investigação diagnóstica complementar urgente")
        print(f"   • Teste oral de tolerância à glicose (TOTG)")
        print(f"   • Hemoglobina glicada (HbA1c)")
        print(f"   • Acompanhamento endocrinológico")
    elif resultado['probabilidade'] >= 0.4:
        print(f"   • Monitoramento regular da glicemia")
        print(f"   • Avaliação em 6 meses")
        print(f"   • Orientação nutricional")
        print(f"   • Atividade física regular")
    else:
        print(f"   • Manter estilo de vida saudável")
        print(f"   • Check-up anual de rotina")
        print(f"   • Prevenção primária")
    
    print(f"\n" + "="*70)

def demonstracao_casos_clinicos():
    """Demonstra o sistema com casos clínicos variados"""
    
    print(f"\n🎯 DEMONSTRAÇÃO: CASOS CLÍNICOS VARIADOS")
    print(f"="*80)
    
    # Carregar sistema
    modelo, scaler, feature_columns = carregar_sistema_predicao()
    if modelo is None:
        return
    
    # Casos clínicos para demonstração
    casos_clinicos = [
        {
            'nome': 'Paciente A - Risco Baixo',
            'dados': [0, 85, 65, 20, 80, 22.0, 0.2, 25],
            'contexto': 'consulta_medica',
            'descricao': 'Jovem, sem gestações, glicemia normal, IMC normal'
        },
        {
            'nome': 'Paciente B - Risco Moderado',
            'dados': [2, 110, 75, 25, 100, 28.5, 0.4, 35],
            'contexto': 'consulta_medica', 
            'descricao': 'Adulta, pré-diabetes, sobrepeso, fatores de risco'
        },
        {
            'nome': 'Paciente C - Risco Alto',
            'dados': [4, 160, 85, 30, 150, 35.0, 0.8, 45],
            'contexto': 'triagem',
            'descricao': 'Múltiplas gestações, hiperglicemia, obesidade'
        },
        {
            'nome': 'Paciente D - Idosa com Fatores',
            'dados': [6, 140, 90, 35, 200, 32.0, 1.2, 65],
            'contexto': 'preventiva',
            'descricao': 'Idosa, múltiplas gestações, histórico familiar'
        }
    ]
    
    # Processar cada caso
    for i, caso in enumerate(casos_clinicos, 1):
        print(f"\n📋 CASO {i}: {caso['nome']}")
        print(f"Descrição: {caso['descricao']}")
        
        # Fazer predição
        resultado = predizer_diabetes(
            caso['dados'], modelo, scaler, feature_columns, 
            contexto=caso['contexto']
        )
        
        # Exibir resultado resumido
        print(f"Resultado: {resultado['predicao']} ({resultado['probabilidade']:.1%}) - {resultado['nivel_risco']}")
        print(f"Contexto: {caso['contexto'].replace('_', ' ').title()} (threshold: {resultado['threshold_usado']:.2f})")
    
    # Análise detalhada de um caso
    print(f"\n🔍 ANÁLISE DETALHADA - CASO 2 (Risco Moderado):")
    caso_detalhado = casos_clinicos[1]
    resultado_detalhado = predizer_diabetes(
        caso_detalhado['dados'], modelo, scaler, feature_columns,
        contexto=caso_detalhado['contexto']
    )
    
    exibir_resultado(caso_detalhado['dados'], resultado_detalhado, feature_columns)

def comparar_thresholds():
    """Demonstra o impacto de diferentes thresholds"""
    
    print(f"\n🔄 DEMONSTRAÇÃO: IMPACTO DOS THRESHOLDS")
    print(f"="*60)
    
    # Carregar sistema
    modelo, scaler, feature_columns = carregar_sistema_predicao()
    if modelo is None:
        return
    
    # Paciente com risco limítrofe
    dados_limitrofe = [2, 125, 80, 28, 120, 29.0, 0.5, 40]
    
    print(f"\n👤 PACIENTE LIMÍTROFE:")
    print(f"Gestações: 2, Glicemia: 125 mg/dL, IMC: 29.0, Idade: 40 anos")
    print(f"Perfil: Pré-diabetes, sobrepeso, risco intermediário")
    
    # Testar diferentes contextos/thresholds
    contextos = [
        ('triagem', 'Triagem Populacional'),
        ('consulta_medica', 'Consulta Médica'),
        ('preventiva', 'Medicina Preventiva')
    ]
    
    print(f"\n📊 RESULTADOS POR CONTEXTO:")
    print(f"{'Contexto':<20} {'Threshold':<10} {'Predição':<12} {'Probabilidade':<12}")
    print(f"-" * 60)
    
    for contexto, nome in contextos:
        resultado = predizer_diabetes(
            dados_limitrofe, modelo, scaler, feature_columns,
            contexto=contexto
        )
        
        print(f"{nome:<20} {resultado['threshold_usado']:<10.2f} "
              f"{resultado['predicao']:<12} {resultado['probabilidade']:<12.1%}")
    
    print(f"\n💡 INTERPRETAÇÃO:")
    print(f"• Mesma probabilidade ({resultado['probabilidade']:.1%}) em todos os contextos")
    print(f"• Diferentes thresholds levam a diferentes decisões")
    print(f"• Triagem: Mais sensível (detecta mais casos)")
    print(f"• Preventiva: Mais específica (evita falsos positivos)")

def menu_interativo():
    """Interface interativa para teste do sistema"""
    
    print(f"\n🎮 MODO INTERATIVO - TESTE SEU PRÓPRIO CASO")
    print(f"="*50)
    
    # Carregar sistema
    modelo, scaler, feature_columns = carregar_sistema_predicao()
    if modelo is None:
        return
    
    try:
        print(f"\n📝 Insira os dados do paciente:")
        
        # Coletar dados
        gestacoes = int(input("Número de gestações (0-17): "))
        glicemia = float(input("Glicemia em jejum (mg/dL, ex: 110): "))
        pressao = float(input("Pressão arterial diastólica (mmHg, ex: 80): "))
        pele = float(input("Espessura da pele tricipital (mm, ex: 25): "))
        insulina = float(input("Insulina sérica (mu U/ml, ex: 100): "))
        imc = float(input("IMC (kg/m², ex: 28.5): "))
        heranca = float(input("Função de herança genética (0.1-2.5, ex: 0.5): "))
        idade = int(input("Idade (anos, ex: 35): "))
        
        # Selecionar contexto
        print(f"\n🏥 Selecione o contexto clínico:")
        print(f"1. Triagem populacional (threshold 0.35 - máxima detecção)")
        print(f"2. Consulta médica (threshold 0.45 - equilibrado)")
        print(f"3. Medicina preventiva (threshold 0.60 - conservador)")
        
        contexto_num = input("Escolha (1/2/3): ")
        contextos_map = {
            '1': 'triagem',
            '2': 'consulta_medica', 
            '3': 'preventiva'
        }
        
        contexto = contextos_map.get(contexto_num, 'consulta_medica')
        
        # Fazer predição
        dados_usuario = [gestacoes, glicemia, pressao, pele, insulina, imc, heranca, idade]
        resultado = predizer_diabetes(
            dados_usuario, modelo, scaler, feature_columns,
            contexto=contexto
        )
        
        # Exibir resultado
        exibir_resultado(dados_usuario, resultado, feature_columns)
        
    except KeyboardInterrupt:
        print(f"\n\n👋 Saindo do modo interativo...")
    except Exception as e:
        print(f"\n❌ Erro na entrada de dados: {e}")
        print(f"Certifique-se de inserir valores numéricos válidos.")

def main():
    """Função principal - menu de demonstração"""
    
    print(f"🩺 SISTEMA DE PREDIÇÃO DE DIABETES")
    print(f"Implementação CRISP-DM - Versão 1.0")
    print(f"="*50)
    
    while True:
        print(f"\n📋 MENU DE DEMONSTRAÇÕES:")
        print(f"1. 🎯 Casos clínicos pré-definidos")
        print(f"2. 🔄 Comparação de thresholds")
        print(f"3. 🎮 Modo interativo (seu caso)")
        print(f"4. 📊 Informações do sistema")
        print(f"0. 🚪 Sair")
        
        try:
            opcao = input(f"\nEscolha uma opção (0-4): ").strip()
            
            if opcao == '1':
                demonstracao_casos_clinicos()
            elif opcao == '2':
                comparar_thresholds()
            elif opcao == '3':
                menu_interativo()
            elif opcao == '4':
                print(f"\n📊 INFORMAÇÕES DO SISTEMA:")
                print(f"• Modelo: Random Forest (melhor AUC: 0.8234)")
                print(f"• Dataset: Pima Indians Diabetes Database")
                print(f"• Metodologia: CRISP-DM com SMOTE")
                print(f"• Performance: 68.5% sensibilidade, 78% especificidade")
                print(f"• Melhoria vs baseline: +18.5% na detecção")
                print(f"• Contextos: Triagem, Consulta Médica, Preventiva")
            elif opcao == '0':
                print(f"\n👋 Obrigado por usar o Sistema de Predição de Diabetes!")
                print(f"🎯 Sistema desenvolvido seguindo metodologia CRISP-DM")
                print(f"📚 Para mais informações, consulte o notebook 'teste.ipynb'")
                break
            else:
                print(f"❌ Opção inválida. Escolha entre 0-4.")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Saindo do sistema...")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()
