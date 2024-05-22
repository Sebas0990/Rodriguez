import numpy as np
from collections import defaultdict

class Recomendador:
    def __init__(self):
        self.REGLAS = defaultdict(list)
        self.conjunto_items_frecuentes = None
        self.base_datos = []
        self.precios = []

    def eclat(self, transacciones, conteo_minimo_apoyo):
        """
        Implementa el algoritmo Eclat para encontrar conjuntos de ítems frecuentes.
        """
        print("eclat")
        item_conjunto_ids = defaultdict(set)
        for tid, transaccion in enumerate(transacciones):
            for item in transaccion:
                item_conjunto_ids[item].add(tid)

        item_conjunto_ids = {item: tids for item, tids in item_conjunto_ids.items() if len(tids) >= conteo_minimo_apoyo}

        def eclat_recursivo(prefijo, items_conjunto_ids, conjuntos_frecuentes):
            sorted_items = sorted(items_conjunto_ids.items(), key=lambda x: len(x[1]), reverse=True)
            for i, (item, conjunto_ids_i) in enumerate(sorted_items):
                nuevo_conjunto = prefijo + (item,)
                conjuntos_frecuentes.append((nuevo_conjunto, len(conjunto_ids_i)))
                sufijo_conjunto_ids = {}
                for item_j, conjunto_ids_j in sorted_items[i + 1:]:
                    nuevo_conjunto_ids = conjunto_ids_i & conjunto_ids_j
                    if len(nuevo_conjunto_ids) >= conteo_minimo_apoyo:
                        sufijo_conjunto_ids[item_j] = nuevo_conjunto_ids
                eclat_recursivo(nuevo_conjunto, sufijo_conjunto_ids, conjuntos_frecuentes)

        conjuntos_frecuentes = []
        eclat_recursivo(tuple(), item_conjunto_ids, conjuntos_frecuentes)
        self.conjunto_items_frecuentes = conjuntos_frecuentes

    def calcular_soportes(self, D, X, Y=None):
        """
        Calcula los soportes de los conjuntos de ítems X y X, Y en la base de datos D.
        """
        print("calcular_soportes")
        conteo_X, conteo_XY, conteo_Y = 0, 0, 0 if Y else None
        for transaccion in D:
            tiene_X = set(X).issubset(transaccion)
            tiene_Y = set(Y).issubset(transaccion) if Y else False
            if tiene_X:
                conteo_X += 1
                if Y and tiene_Y:
                    conteo_XY += 1
            if Y and tiene_Y:
                conteo_Y += 1
        sup_X = conteo_X / len(D)
        sup_XY = conteo_XY / len(D)
        sup_Y = conteo_Y / len(D) if Y is not None else None
        return sup_X, sup_XY, sup_Y
    
    def crear_reglas_asociacion(self, F, confianza_minima, transacciones):
        """
        Crea reglas de asociación basadas en conjuntos de ítems frecuentes.
        """
        print("crear_reglas_asociacion")
        B = defaultdict(list)
        soportes_conjunto_items = {frozenset(conjunto): soporte for conjunto, soporte in F}
        for conjunto, soporte in F:
            if len(conjunto) > 1:
                for i in range(len(conjunto)):
                    antecedente = frozenset([conjunto[i]])
                    consecuente = frozenset(conjunto[:i] + conjunto[i+1:])
                    soporte_antecedente = soportes_conjunto_items.get(antecedente, 0)
                    if soporte_antecedente > 0:
                        confianza = soporte / soporte_antecedente
                        if confianza >= confianza_minima:
                            metricas = {
                                'confianza': confianza  
                            }
                            B[antecedente].append((consecuente, metricas))
        return B

    def entrenar(self, precios, base_datos, conteo_minimo_apoyo=10, confianza_minima=0.1):
        """
        Entrena el recomendador con la base de datos dada.
        """
        print("entrenamiento")
        self.base_datos = base_datos
        self.precios = precios
        self.eclat(base_datos, conteo_minimo_apoyo)
        self.REGLAS = self.crear_reglas_asociacion(self.conjunto_items_frecuentes, confianza_minima=confianza_minima, transacciones=base_datos)
        return self
    
    def obtener_recomendaciones(self, carrito, max_recomendaciones=5):
        """
        Obtiene recomendaciones basadas en el carrito de compras dado.
        """
        print("recomendaciones")
        print(carrito)
        precios_normalizados = self.precios

        recomendaciones = {}
        for antecedente, reglas in self.REGLAS.items():
            if antecedente.issubset(carrito):
                for consecuente, metricas in reglas:
                    for item in consecuente:
                        if item not in carrito:
                            factor_precio = precios_normalizados[item] if item < len(precios_normalizados) else 0
                            factor_metrica = metricas['confianza']
                            puntaje = factor_metrica * (1 + factor_precio)
                            recomendaciones[item] = recomendaciones.get(item, []) + [puntaje]

        promedio_recomendaciones = {item: sum(puntajes) / len(puntajes) for item, puntajes in recomendaciones.items()}
        recomendaciones_ordenadas = sorted(promedio_recomendaciones.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in recomendaciones_ordenadas[:max_recomendaciones]]
