"""
agent.py — Loop agéntico (ReAct).
El LLM razona, decide qué herramienta usar, observa el resultado, y repite.

CONFIGURACIÓN: Cambia LLM_PROVIDER según tu API.
Soporta: "openai" (compatible con Kimi/DeepSeek/Groq), "anthropic", "google"
"""

import json
from tools import TOOLS_SCHEMA, TOOLS_MAP

# ─── CONFIGURACIÓN ────────────────────────────────────────────
# Elige tu provider y modelo. Si usas Kimi, DeepSeek, Groq, etc.
# la mayoría son compatibles con el formato OpenAI.

LLM_PROVIDER = "openai"  # "openai" | "anthropic" | "google"

# Para APIs compatibles con OpenAI (Kimi, DeepSeek, Groq, etc.)
# cambia base_url y api_key:
OPENAI_CONFIG = {
    "api_key": "TU_API_KEY",
    "base_url": "https://api.moonshot.cn/v1",  # ← Kimi
    # "base_url": "https://api.deepseek.com",  # ← DeepSeek
    # "base_url": "https://api.groq.com/openai/v1",  # ← Groq
    # "base_url": "https://api.openai.com/v1",  # ← OpenAI
    "model": "moonshot-v1-8k",  # ← cambia al modelo de tu API
}

MAX_ITERATIONS = 6  # máx pasos del agente antes de forzar respuesta

SYSTEM_PROMPT = """Eres un agente musical inteligente. Tu trabajo es recomendar canciones
basándote en el historial de escucha del usuario y el estilo que pide.

TIENES ACCESO A ESTAS HERRAMIENTAS que trabajan sobre un dataset local de ~5000 canciones:
- search_songs: busca canciones por similitud de letras (TF-IDF + coseno)
- get_user_profile: analiza el historial de escucha del usuario
- filter_by: filtra resultados por género/artista

PROCESO:
1. Primero analiza el perfil del usuario con get_user_profile
2. Luego busca canciones con search_songs usando keywords relevantes
3. Opcionalmente filtra con filter_by si el usuario pide un género específico
4. Finalmente da tu recomendación razonada

Siempre explica POR QUÉ recomiendas cada canción, conectándola con el gusto del usuario.
Responde en español."""


# ─── Cliente LLM ──────────────────────────────────────────────
def create_client():
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        return OpenAI(
            api_key=OPENAI_CONFIG["api_key"],
            base_url=OPENAI_CONFIG["base_url"],
        )
    elif LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        return Anthropic(api_key=OPENAI_CONFIG["api_key"])
    else:
        raise ValueError(f"Provider no soportado: {LLM_PROVIDER}")


client = create_client()


def chat_completion(messages, tools=None):
    """Llama al LLM y devuelve la respuesta."""
    if LLM_PROVIDER == "openai":
        kwargs = {
            "model": OPENAI_CONFIG["model"],
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return client.chat.completions.create(**kwargs)

    elif LLM_PROVIDER == "anthropic":
        # Adaptar formato Anthropic
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        msgs = [m for m in messages if m["role"] != "system"]
        return client.messages.create(
            model=OPENAI_CONFIG["model"],
            max_tokens=1024,
            system=system,
            messages=msgs,
            tools=[
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in (tools or [])
            ],
        )


# ─── Extraer tool calls de la respuesta ──────────────────────
def extract_tool_calls(response):
    """Extrae las llamadas a herramientas de la respuesta del modelo."""
    if LLM_PROVIDER == "openai":
        msg = response.choices[0].message
        if msg.tool_calls:
            return [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
        return []

    elif LLM_PROVIDER == "anthropic":
        return [
            {
                "id": block.id,
                "name": block.name,
                "args": block.input,
            }
            for block in response.content
            if block.type == "tool_use"
        ]


def extract_text(response):
    """Extrae el texto de la respuesta."""
    if LLM_PROVIDER == "openai":
        return response.choices[0].message.content or ""
    elif LLM_PROVIDER == "anthropic":
        return "".join(
            block.text for block in response.content if block.type == "text"
        )


def get_stop_reason(response):
    """Verifica si el modelo quiere seguir usando herramientas."""
    if LLM_PROVIDER == "openai":
        return response.choices[0].finish_reason  # "tool_calls" | "stop"
    elif LLM_PROVIDER == "anthropic":
        return response.stop_reason  # "tool_use" | "end_turn"


# ─── Loop agéntico ────────────────────────────────────────────
def run_agent(user_input: str, on_step=None):
    """
    Ejecuta el agente. Devuelve la respuesta final.

    on_step: callback opcional que se llama en cada paso del agente.
             Recibe (step_type, step_data) donde step_type es:
             "thinking", "tool_call", "tool_result", "final"
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for iteration in range(MAX_ITERATIONS):
        # Llamar al LLM
        if on_step:
            on_step("thinking", {"iteration": iteration + 1})

        response = chat_completion(messages, tools=TOOLS_SCHEMA)
        stop_reason = get_stop_reason(response)

        # ¿Quiere usar herramientas?
        tool_calls = extract_tool_calls(response)

        if not tool_calls:
            # Respuesta final — no quiere más herramientas
            final_text = extract_text(response)
            if on_step:
                on_step("final", {"text": final_text})
            return final_text

        # Ejecutar cada herramienta
        if LLM_PROVIDER == "openai":
            # Añadir el mensaje del asistente con los tool_calls
            messages.append(response.choices[0].message)

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            if on_step:
                on_step("tool_call", {"name": tool_name, "args": tool_args})

            # Ejecutar la herramienta
            if tool_name in TOOLS_MAP:
                result = TOOLS_MAP[tool_name](**tool_args)
            else:
                result = json.dumps({"error": f"Herramienta '{tool_name}' no existe"})

            if on_step:
                on_step("tool_result", {"name": tool_name, "result": result[:500]})

            # Añadir resultado al historial
            if LLM_PROVIDER == "openai":
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            elif LLM_PROVIDER == "anthropic":
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": result,
                        }
                    ],
                })

    # Si llegamos al máximo de iteraciones, respuesta de emergencia
    return "He analizado varias opciones pero no he podido llegar a una conclusión clara. ¿Puedes darme más detalles?"


# ─── Ejecución directa (para testing) ────────────────────────
if __name__ == "__main__":
    def print_step(step_type, data):
        if step_type == "thinking":
            print(f"\n🧠 Pensando... (paso {data['iteration']})")
        elif step_type == "tool_call":
            print(f"🔧 → {data['name']}({data['args']})")
        elif step_type == "tool_result":
            print(f"📊 ← resultado: {data['result'][:200]}...")
        elif step_type == "final":
            print(f"\n{'='*50}")
            print(data["text"])

    resultado = run_agent(
        "He escuchado mucho Radiohead y Bright Eyes últimamente. "
        "Quiero algo melancólico con guitarras acústicas.",
        on_step=print_step,
    )
