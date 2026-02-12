from typing import List, Dict

class MemoryManager:
    """
    Gestiona la memoria de la conversación.
    Funciona como una lista inteligente que recuerda los últimos mensajes.
    """
    
    def __init__(self, limit: int = 10):
        # limit: Cuantos mensajes recordamos (para no gastar demasiados tokens)
        self.history: List[Dict[str, str]] = []
        self.limit = limit
        
    def add_user_message(self, text: str):
        """Guardamos lo que dice el usuario"""
        self.history.append({"role": "user", "content": text})
        self._cleanup()
        
    def add_assistant_message(self, text: str):
        """Guardamos lo que responde el bot"""
        self.history.append({"role": "assistant", "content": text})
        self._cleanup()
        
    def get_messages(self) -> List[Dict[str, str]]:
        """Devuelve la historia completa para enviarla al LLM"""
        return self.history
    
    def _cleanup(self):
        """Si la memoria se llena, borramos lo más antiguo (olvido selectivo)"""
        if len(self.history) > self.limit:
            # Quitamos los mensajes más viejos del principio
            self.history = self.history[-self.limit:]
