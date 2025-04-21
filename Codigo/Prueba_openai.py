import os
import openai
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

# Obtener la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    client = openai.OpenAI()  # Crear cliente en la nueva versión
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "Dime si la API funciona correctamente"}]
    )
    print("✅ Conexión exitosa con OpenAI")
    print("Respuesta de la API:", response.choices[0].message.content)
except openai.AuthenticationError:
    print("❌ Error de autenticación: La clave API es incorrecta o no está activa.")
except openai.RateLimitError:
    print("❌ Error de cuota: Has superado tu límite de uso de la API.")
except Exception as e:
    print(f"❌ Otro error: {e}")