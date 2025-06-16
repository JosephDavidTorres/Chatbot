
document.addEventListener("DOMContentLoaded", function () {
    const toggleBtn = document.getElementById("toggle-chatbot");
    const chatInterface = document.getElementById("chat-interface");
    const sendBtn = document.getElementById("send-btn");
    const inputField = document.getElementById("chat-input");
    const chatLog = document.getElementById("chat-log");

    toggleBtn.addEventListener("click", () => {
        if (chatInterface.style.display === "none") {
            chatInterface.style.display = "block";
            inputField.focus();
        } else {
            chatInterface.style.display = "none";
        }
    });

    async function enviarPregunta() {
        const pregunta = inputField.value.trim();
        if (!pregunta) return;

        const userMsg = document.createElement("div");
        userMsg.textContent = "Tú: " + pregunta;
        chatLog.appendChild(userMsg);

        inputField.value = "";

        try {
            const response = await fetch("http://localhost:8000/preguntar", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ pregunta }),
            });

            const data = await response.json();
            const botMsg = document.createElement("div");
            botMsg.textContent = "Chatbot: " + data.respuesta;
            chatLog.appendChild(botMsg);
            chatLog.scrollTop = chatLog.scrollHeight;
        } catch (error) {
            const errorMsg = document.createElement("div");
            errorMsg.textContent = "Chatbot: Error de conexión con el servidor.";
            chatLog.appendChild(errorMsg);
        }
    }

    sendBtn.addEventListener("click", enviarPregunta);
    inputField.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            enviarPregunta();
        }
    });
});
