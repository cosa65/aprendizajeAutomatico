#!/usr/bin/env python
# coding: utf-8

# In[198]:


import numpy as np
import pandas as pd
import operator

from collections import Counter


# In[199]:


def construir_arbol(instancias, etiquetas, altura, criterio):
    #print("Instancias")
    #print(instancias)
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterio)
    # Criterio de corte: ¿Hay ganancia?
    if ganancia == 0 or (altura != None and altura == 0):
        #  Si no hay ganancia en separar, o llegue a la altura requerida, corto la recursion. 
        return Hoja(etiquetas)
    else: 
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        altura_rec = None if(altura == None) else altura - 1
        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen, altura_rec, criterio)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, altura_rec, criterio)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)


# In[200]:


# Definición de la estructura del árbol. 

class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas))


class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecho
     
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho
        
        
# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor
    
    def cumple(self, instancia):
        return instancia[self.atributo] <= self.valor
    
    def __repr__(self):
        return "¿Es el valor para {} igual a {}?".format(self.atributo, self.valor)


# In[201]:


#Como hago para refactorizar esto?
def gini(etiquetas):
    diccionario = dict(Counter(etiquetas))
    suma = 0
    for etiqueta in diccionario.keys():
        suma += (diccionario[etiqueta]/len(etiquetas))**2
    impureza = 1 - suma
    return impureza

def ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    etiquetas = np.concatenate((etiquetas_rama_izquierda, etiquetas_rama_derecha))
    n_izq = len(etiquetas_rama_izquierda)
    n_der = len(etiquetas_rama_derecha)
    n = len(etiquetas)
    
    gini_total = gini(etiquetas)
    gini_izq = gini(etiquetas_rama_izquierda)
    gini_der = gini(etiquetas_rama_derecha)
    
    ganancia_gini = gini_total - ((n_izq/n)*gini_izq + (n_der/n)*gini_der)
    
    return ganancia_gini

def entropia(etiquetas):
    diccionario = dict(Counter(etiquetas))
    entropia = 0
    for etiqueta in diccionario.keys():
        proporcion = diccionario[etiqueta]/len(etiquetas)
        entropia += -proporcion*log(proporcion,2)
    return entropia

def ganancia_entropia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    etiquetas = np.concatenate((etiquetas_rama_izquierda, etiquetas_rama_derecha))
    n_izq = len(etiquetas_rama_izquierda)
    n_der = len(etiquetas_rama_derecha)
    n = len(etiquetas)
    
    entropia_total = entropia(etiquetas)
    entropia_izq = entropia(etiquetas_rama_izquierda)
    entropia_der = entropia(etiquetas_rama_derecha)
    
    ganancia_entropia = entropia_total - ((n_izq/n)*entropia_izq + (n_der/n)*entropia_der)
    
    return ganancia_gini
    


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    # COMPLETAR (recomendamos utilizar máscaras para este punto)
    instancias_cumplen = pd.DataFrame(columns=instancias.columns)
    instancias_no_cumplen = pd.DataFrame(columns=instancias.columns)
     
    columna_etiqueta = pd.DataFrame(data=etiquetas, index=instancias.index, columns=['etiqueta'])
    instancias_con_etiqueta = pd.concat([instancias, columna_etiqueta], axis=1)
    print(type(instancias_con_etiqueta))
    instancias_cumplen_etiqueta = instancias_con_etiqueta.where(pregunta.cumple(instancias_con_etiqueta)).dropna()
    instancias_no_cumplen_etiqueta = instancias_con_etiqueta.mask(pregunta.cumple(instancias_con_etiqueta)).dropna()
    etiquetas_cumplen = instancias_cumplen_etiqueta.loc[:,'etiqueta'].tolist()
    etiquetas_no_cumplen = instancias_no_cumplen_etiqueta.loc[:,'etiqueta'].tolist()
    
    del instancias_cumplen_etiqueta['etiqueta']
    del instancias_no_cumplen_etiqueta['etiqueta']
    
    instancias_cumplen = instancias_cumplen_etiqueta
    instancias_no_cumplen = instancias_no_cumplen_etiqueta
    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen


# In[202]:


def encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterio):
    max_ganancia = 0
    mejor_pregunta = None
    for columna in instancias.columns:
        for valor in set(instancias[columna]):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            #Ver como refactorizar
            if(criterio == 'gini'):
                ganancia = ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            else:
                ganancia = ganancia_entropia(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            #print("La ganancia para la pregunta {}, es {}".format(pregunta, ganancia))
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta
    #print("La mejor pregunta es {}, con una ganancia de {}".format(mejor_pregunta, max_ganancia))        
    return max_ganancia, mejor_pregunta


def imprimir_arbol(arbol, spacing=""):
    if isinstance(arbol, Hoja):
        print (spacing + "Hoja:", arbol.cuentas)
        return

    print (spacing + str(arbol.pregunta))

    print (spacing + '--> True:')
    imprimir_arbol(arbol.sub_arbol_izquierdo, spacing + "  ")

    print (spacing + '--> False:')
    imprimir_arbol(arbol.sub_arbol_derecho, spacing + "  ")


# In[203]:


X = pd.DataFrame([["Sol","Calor","Alta","Debil"],
                ["Sol","Calor","Alta","Fuerte"],
                ["Nublado","Calor","Alta","Debil"],
                ["Lluvia","Templado","Alta","Debil"],
                ["Lluvia","Frio","Normal","Debil"],
                ["Lluvia","Frio","Normal","Fuerte"],
                ["Nublado","Frio","Normal","Fuerte"],
                ["Sol","Templado","Alta","Debil"],
                ["Sol","Frio","Normal","Debil"],
                ["Lluvia","Templado","Normal","Debil"],
                ["Sol","Templado","Normal","Fuerte"],
                ["Nublado","Templado","Alta","Fuerte"],
                ["Nublado","Calor","Normal","Debil"],
                ["Lluvia","Templado","Alta","Fuerte"]],
                columns = ['Cielo', 'Temperatura', 'Humedad', 'Viento'])

y = ['No', 'No', 'Si', 'Si', 'Si', 'No', 'Si', 'No', 'Si', 'Si', 'Si', 'Si', 'Si', 'No']


# In[204]:


#arbol = construir_arbol(X, y, 3, operator.eq)
#imprimir_arbol(arbol)


# ## Resultado esperado
# 
# ```
# ¿Es el valor para Cielo igual a Nublado?
# --> True:
#   ¿Es el valor para Temperatura igual a Frio?
#   --> True:
#     Hoja: {'Si': 1}
#   --> False:
#     ¿Es el valor para Temperatura igual a Templado?
#     --> True:
#       Hoja: {'Si': 1}
#     --> False:
#       Hoja: {'Si': 2}
# --> False:
#   ¿Es el valor para Humedad igual a Normal?
#   --> True:
#     ¿Es el valor para Viento igual a Fuerte?
#     --> True:
#       Hoja: {'No': 1, 'Si': 1}
#     --> False:
#       ¿Es el valor para Cielo igual a Sol?
#       --> True:
#         Hoja: {'Si': 1}
#       --> False:
#         Hoja: {'Si': 2}
#   --> False:
#     ¿Es el valor para Cielo igual a Sol?
#     --> True:
#       ¿Es el valor para Temperatura igual a Templado?
#       --> True:
#         Hoja: {'No': 1}
#       --> False:
#         Hoja: {'No': 2}
#     --> False:
#       Hoja: {'Si': 1, 'No': 1}
# ```

# ## Parte 2 (opcional)
# Protocolo sklearn para clasificadores. Completar el protocolo requerido por sklearn. Deben completar la función predict utilizando el árbol para predecir valores de nuevas instancias. 
# 

# In[205]:


def predecir(arbol, x_t):
    if isinstance(arbol, Hoja):
        return max(arbol.cuentas, key=arbol.cuentas.get)
    
    if(arbol.pregunta.cumple(x_t)):
        return predecir(arbol.sub_arbol_izquierdo, x_t)
    else:
        return predecir(arbol.sub_arbol_derecho, x_t)
        
class MiClasificadorArbol(): 
    def __init__(self, columnas=None):
        self.arbol = None
        self.columnas = columnas
    
    def fit(self, X_train, y_train, max_depth=None, criterion='gini'):
        self.arbol = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train, max_depth, criterion)
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df) 
            print(x_t, "predicción ->", prediction)
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy
        

# Ejemplo de uso
clf = MiClasificadorArbol()

# Tomar en cuenta que sklearn espera numpy arrays:
clf.fit(np.array(X), y, 3)
clf.score(np.array(X), y)

clf2 = MiClasificadorArbol(['Cielo', 'Temperatura', 'Humedad', 'Viento'])
clf2.fit(np.array(X), y)
clf2.score(np.array(X),y)





