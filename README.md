# Chatbot Universitario - TFG

Este proyecto implementa un sistema de chatbot para responder preguntas a partir de documentos académicos. Forma parte de mi Proyecto de Fin de Grado.

## Requisitos y configuración

Para que el sistema funcione correctamente, es necesario realizar los siguientes pasos previos:

1. **Instalar las dependencias**
Se puede incluir un entorno virtual con el documento requirement.txt, pero en caso de que no se haga, el proyecto incluye un archivo `requirements.txt` con todas las librerías necesarias. Para instalarlo, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
pip install -r requirements.txt
```

2. **Añadir una clave de API de OpenAI**

El sistema utiliza la API de OpenAI (modelo GPT) para generar las respuestas. Por tanto, es imprescindible disponer de una clave de API válida.  
No se proporciona una clave API ya que cada una es personal, es lo mismo que una contraseña de Correo.
Una vez tengas la clave, en el archivo `.env` de la carpeta raíz, añade la clave API:

```env
OPENAI_API_KEY=tu_clave_aqui
```

3. **Ubicación del proyecto**

El sistema está diseñado para ejecutarse desde una carpeta ubicada en el **escritorio del dispositivo**.  
Algunas rutas internas están previstas para funcionar correctamente solo si el proyecto se encuentra en esa ubicación.  
Se recomienda no modificar la ruta base salvo que se actualicen también las rutas internas en el código.
