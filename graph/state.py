from typing import List, TypedDict

#Grafiğin güncel durumu. Bunları State isimli bi dosyaya bu şekilde tipleri ile kaydedersek,
#Diğer kısımlarda, özellikle fonksiyon oluşturduğumuz nodes klasörü içindeki sınıflarda bu veritiplerini kullanabiliriz.
class GraphState(TypedDict):
    """
      Represents the state of our graph.

      Attributes:
          question: question
          generation: LLM generation
          web_search: whether to add search
          documents: list of documents
      """

    question: str
    generation: str
    web_search: bool
    documents: List[str]