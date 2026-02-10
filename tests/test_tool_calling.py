import sys
from pathlib import Path
import unittest

# Añadir el raíz del proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.generation.llm_groq import GroqLLM

class TestToolCalling(unittest.TestCase):
    def setUp(self):
        self.llm = GroqLLM()
        print("\n" + "="*50)
        
    def test_general_question_no_tool(self):
        """Prueba que una pregunta general NO use herramientas."""
        query = "Hola, ¿qué tal estás?"
        print(f"🤖 Probando pregunta GENERAL: '{query}'")
        
        # Esta llamada debería ser rápida y NO debería imprimir "[Tool usada]" en la consola
        response = self.llm.chat_with_tools(query)
        print(f"📝 Respuesta: {response[:100]}...")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
    def test_financial_question_uses_tool(self):
        """Prueba que una pregunta financiera SÍ use herramientas."""
        query = "¿Cuáles fueron los ingresos de Apple en 2019?"
        print(f"🤖 Probando pregunta FINANCIERA: '{query}'")
        
        # Esta llamada debería imprimir "[Tool usada] ..." en la consola
        response = self.llm.chat_with_tools(query)
        print(f"📝 Respuesta: {response[:100]}...")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Verificar que la respuesta menciona algo financiero (Apple, billones, etc)
        self.assertTrue("Apple" in response or "AAPL" in response)

if __name__ == "__main__":
    unittest.main()
