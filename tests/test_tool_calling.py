import sys
from pathlib import Path
import unittest

# Anadir el raiz del proyecto al path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.generation.llm_groq import GroqLLM

class TestToolCalling(unittest.TestCase):
    def setUp(self):
        self.llm = GroqLLM()
        print("\n" + "="*50)
        
    def test_general_question_no_tool(self):
        """Prueba que una pregunta general NO use herramientas."""
        query = "Hola, que tal estas?"
        print(f"Probando pregunta GENERAL: '{query}'")
        
        response = self.llm.chat_with_tools(query)
        print(f"Respuesta: {response[:100]}...")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
    def test_financial_question_uses_tool(self):
        """Prueba que una pregunta financiera SI use herramientas."""
        query = "Cuales fueron los ingresos de Apple en 2019?"
        print(f"Probando pregunta FINANCIERA: '{query}'")
        
        response = self.llm.chat_with_tools(query)
        print(f"Respuesta: {response[:100]}...")
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertTrue("Apple" in response or "AAPL" in response)

if __name__ == "__main__":
    unittest.main()
