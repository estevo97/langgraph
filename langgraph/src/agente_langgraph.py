# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os 
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.llm import crear_llm
import asyncio
from src.cliente_mcp import ejecutar_herramienta_mcp
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


@tool
def calculator(expression: str) -> str:
    """Útil para realizar cálculos matemáticos exactos. Ingresa la expresión matemática."""
    try:
        result = eval(expression, {"__builtins__": None})
        return str(result)
    except Exception as e:
        return f"Error en el cálculo: {e}"


# EL ESTADO Y EL MODELO

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str 

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
llm = crear_llm(
    proveedor="gemini"
)


# LOS NODOS

def nodo_enrutador(state: AgentState):
    """Analiza la petición y decide a qué agente enviarla."""
    mensaje_usuario = state["messages"][-1].content
    
    prompt = (
        "Eres un sistema de enrutamiento. Tu única tarea es clasificar "
        "el mensaje del usuario.\n"
        "Si el usuario pide cálculos, números o resolver problemas lógicos, "
        "responde: matematico\n"
        "Si el usuario pide escribir, rimas, literatura o algo creativo, "
        "responde: poeta\n"
        "Si el usuario pide consultar o listar documentos de una categoría, "
        "responde: listar_documentos\n"
        "Si el usuario pide clasificar u organizar los documentos pendientes "
        "de recepción, responde: clasificar_documentos\n"
        "Si el usuario hace preguntas científicas, responde: cientifico\n"
        "Responde ÚNICAMENTE con una de estas cinco opciones: "
        "matematico, poeta, listar_documentos, "
        "clasificar_documentos o cientifico.\n"
        "En minúsculas y sin puntuación.\n"
        f"Mensaje: {mensaje_usuario}"
    )
    
    respuesta = llm.invoke(prompt)
    
    contenido = respuesta.content
    
    # Verificamos si Gemini nos ha devuelto una lista de bloques
    if isinstance(contenido, list):
        # Extraemos el texto del primer bloque
        if len(contenido) > 0 and isinstance(contenido[0], dict) and "text" in contenido[0]:
            texto = contenido[0]["text"]
        else:
            texto = str(contenido[0])
    else:
        # Si es un texto normal
        texto = str(contenido)
        
    destino = texto.strip().lower()

    
    # por seguridad, por lo que pueda devolver el llm
    if "matematico" in destino:
        destino = "matematico"
    elif "listar_documentos" in destino:
        destino = "listar_documentos"
    elif "clasificar_documentos" in destino:
        destino = "clasificar_documentos"
    elif "poeta" in destino:
        destino = "poeta"
    elif "cientifico" in destino:
        destino = "cientifico"
    else:
        destino = "poeta"
        
    print(f" [Enrutador] Desviando consulta hacia: {destino.upper()}")
    
    return {"next_agent": destino}

def nodo_matematico(state: AgentState):
    """Es experto en números, tiene calculadora integrada)"""
    print("[Agente Matemático] calculando...")
    
    #Creamos un sub-agente exclusivo para este nodo, dándole la herramienta
    math_agent = create_react_agent(llm, tools=[calculator])
    
    #Definimos su personalidad (inyectada de forma segura para tu versión de LangGraph)
    prompt_sistema = SystemMessage(
        content="Eres un matemático brillante pero muy arrogante y gruñón. "
                "USA OBLIGATORIAMENTE la herramienta 'calculator' para resolver la operación. "
                "Quéjate de que te hagan preguntas tan aburridas y fáciles."
    )
    
    #Preparamos los mensajes: Personalidad + Historial del usuario
    mensajes_para_agente = [prompt_sistema] + state["messages"]
    
    #Ejecutamos el sub-agente
    respuesta_sub_agente = math_agent.invoke({"messages": mensajes_para_agente})
    
    #Extraemos solo el mensaje final que nos dio el sub-agente para devolverlo al grafo principal
    respuesta_final = respuesta_sub_agente["messages"][-1]
    
    return {"messages": [respuesta_final]}

def nodo_poeta(state: AgentState):
    """Un experto en letras (Llamada simple, sin herramientas)."""
    print("[Agente Poeta] Recurriendo a las musas...")
    
    prompt_sistema = SystemMessage(
        content="Eres un poeta clásico y dramático del siglo XIX. "
                "Responde usando una poesía que rime. Usa palabras grandilocuentes."
    )
    
    # Como el poeta no necesita herramientas, hacemos un invoke() directo al LLM
    respuesta = llm.invoke([prompt_sistema] + state["messages"])
    
    return {"messages": [respuesta]}


def nodo_cientifico(state: AgentState):
    """Un experto científico que responde con rigor y evidencia."""
    print("[Agente Científico] Analizando los hechos...")

    prompt_sistema = SystemMessage(
        content=(
            "Eres un científico experto en física, química, biología "
            "y otras disciplinas científicas. "
            "Responde con rigor, precisión y basándote en evidencias "
            "y conocimientos científicos contrastados. "
            "Explica los conceptos complejos de forma clara y didáctica. "
            "Distingue entre hechos demostrados, hipótesis y teorías. "
            "Si no conoces la respuesta con suficiente certeza, "
            "reconócelo en lugar de inventar información. "
            "Utiliza un tono profesional, objetivo y analítico.\n"
        )
    )

    respuesta = llm.invoke([prompt_sistema] + state["messages"])

    return {"messages": [respuesta]}


async def nodo_listar_documentos(state: AgentState):
    """Lista los documentos de una categoría."""

    mensaje_usuario = state["messages"][-1].content.lower()

    categorias = {
        "deportes",
        "tecnologia",
        "ciencia",
        "economia",
        "politica",
        "educacion",
    }

    # Identificar la categoría indicada por el usuario
    tematica = next(
        (categoria for categoria in categorias if categoria in mensaje_usuario),
        None
    )

    if tematica is None:
        respuesta = (
            "Indica una categoría válida: deportes, tecnologia, "
            "ciencia, economia, politica o educacion."
        )
    else:
        resultado = await ejecutar_herramienta_mcp(
            "listar_archivos",
            {"tematica": tematica}
        )

        if not resultado:
            respuesta = f"No hay documentos en la categoría {tematica}."
        else:
            archivos = "\n".join(f"- {archivo}" for archivo in resultado)
            respuesta = f"Documentos de {tematica}:\n{archivos}"

    return {"messages": [AIMessage(content=respuesta)]}

async def nodo_clasificar_documentos(state: AgentState):
    """Lee los documentos de recepción, los clasifica y los mueve."""

    print("[Clasificador] Buscando documentos pendientes...")

    # 1. Obtener los archivos de recepción
    resultado = await ejecutar_herramienta_mcp(
        "listar_recepcion",
        {}
    )

    archivos = [
        bloque["text"]
        for bloque in resultado
        if bloque.get("type") == "text"
    ]

    if not archivos:
        return {
            "messages": [
                AIMessage(content="No hay documentos pendientes de clasificar.")
            ]
        }

    resultados = []

    # 2. Procesar cada documento
    for nombre_archivo in archivos:

        print(f"[Clasificador] Analizando: {nombre_archivo}")

        # Leer el documento
        contenido_resultado = await ejecutar_herramienta_mcp(
            "leer_archivo",
            {"nombre_archivo": nombre_archivo}
        )

        contenido = contenido_resultado[0]["text"]

        # 3. Preguntar a Gemini por la categoría
        prompt = f"""
        Clasifica el siguiente documento en UNA de estas categorías:

        deportes
        tecnologia
        ciencia
        economia
        politica
        educacion

        Devuelve únicamente el nombre de la categoría,
        en minúsculas y sin explicaciones.

        Nombre del archivo: {nombre_archivo}

        Contenido:
        {contenido}
        """

        respuesta = llm.invoke(prompt)
        contenido_respuesta = respuesta.content

        if isinstance(contenido_respuesta, list):
            tematica = contenido_respuesta[0]["text"].strip().lower()
        else:
            tematica = str(contenido_respuesta).strip().lower()

        # 4. Validar la categoría
        categorias_validas = {
            "deportes",
            "tecnologia",
            "ciencia",
            "economia",
            "politica",
            "educacion",
        }

        if tematica not in categorias_validas:
            resultados.append(
                f"No se pudo clasificar {nombre_archivo}: categoría no válida."
            )
            continue

        # 5. Mover el archivo a su categoría
        movimiento = await ejecutar_herramienta_mcp(
            "mover_archivo",
            {
                "nombre_archivo": nombre_archivo,
                "tematica": tematica
            }
        )

        # Extraer el texto de la respuesta MCP
        if isinstance(movimiento, list):
            texto_movimiento = "\n".join(
                bloque["text"]
                for bloque in movimiento
                if bloque.get("type") == "text"
            )
        else:
            texto_movimiento = str(movimiento)

        resultados.append(texto_movimiento)

    respuesta_final = "\n".join(resultados)

    return {
        "messages": [
            AIMessage(content=respuesta_final)
        ]
    }

#  CONSTRUCCIÓN DEL GRAFO

grafo = StateGraph(AgentState)

grafo.add_node("enrutador", nodo_enrutador)
grafo.add_node("matematico", nodo_matematico)
grafo.add_node("poeta", nodo_poeta)
grafo.add_node("cientifico", nodo_cientifico)
grafo.add_node("listar_documentos", nodo_listar_documentos)
grafo.add_node("clasificar_documentos", nodo_clasificar_documentos)
grafo.add_edge(START, "enrutador")

def leer_decision(state: AgentState):
    return state["next_agent"]

grafo.add_conditional_edges(
    "enrutador",
    leer_decision,
    {
        "matematico": "matematico",
        "cientifico": "cientifico",
        "listar_documentos": "listar_documentos",
        "clasificar_documentos": "clasificar_documentos",
        "poeta": "poeta"
    }
)

grafo.add_edge("matematico", END)
grafo.add_edge("poeta", END)
grafo.add_edge("cientifico", END)
grafo.add_edge("listar_documentos", END)
grafo.add_edge("clasificar_documentos", END)

app = grafo.compile()


# PRUEBILLAS

# PRUEBAS

async def hacer_pregunta(texto):
    print("\n")
    print(f"Usuario: {texto}")

    inputs = {"messages": [HumanMessage(content=texto)]}

    respuesta_final = await app.ainvoke(inputs)

    print(f"Respuesta Final:\n{respuesta_final['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(
        hacer_pregunta("Clasifica los documentos pendientes de recepción")
    )

