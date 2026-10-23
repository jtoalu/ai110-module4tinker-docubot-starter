"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

from operator import index
from pydoc import text
import os, re
import glob

class DocuBot:
    SECTION_SIZE = 500

    STOP_WORDS = {
        "a", "an", "and", "are", "can", "do", "does", "find", "for",
        "i", "in", "is", "me", "of", "on", "please", "the", "to",
        "you",
    }

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        # self.index = self.build_index(self.documents)

        self.sections = self.split_documents(self.documents)
        self.index = self.build_index(self.sections)

    @classmethod
    def split_text(cls, text):
        """Split text into paragraphs, breaking long paragraphs into bounded sections."""
        sections = []
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            remaining = paragraph.strip()
            while len(remaining) > cls.SECTION_SIZE:
                split_at = remaining.rfind(" ", 0, cls.SECTION_SIZE + 1)
                if split_at <= 0:
                    split_at = cls.SECTION_SIZE
                sections.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
            if remaining:
                sections.append(remaining)
        return sections

    @classmethod
    def split_documents(cls, documents):
        """Return (filename, section) pairs for all non-empty document sections."""
        return [
            (filename, section)
            for filename, text in documents
            for section in cls.split_text(text)
        ]

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """

    def build_index(self, sections):
        """Map each lowercase word to the document sections containing it."""
        index = {}
        for filename, section in sections:
            tokens = dict.fromkeys(re.findall(r"\w+", section.lower()))
            for token in tokens:
                index.setdefault(token, []).append((filename, section))
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        """Return the number of distinct query words present in the text."""
        query_words = {
            word
            for word in re.findall(r"\w+", query.lower())
            if word not in self.STOP_WORDS
    }
        text_words = set(re.findall(r"\w+", text.lower()))
        return sum(word in text_words for word in query_words)

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        """Return the top matching (filename, section) pairs by relevance."""
        query_words = {
            word
            for word in re.findall(r"\w+", query.lower())
            if word not in self.STOP_WORDS
        }
        if not query_words:
            return []

        indexed_sections = {
            section
            for word in query_words
            for section in self.index.get(word, [])
        }

        scored_results = []
        for filename, section in self.sections:
            if (filename, section) not in indexed_sections:
                continue
            score = self.score_document(query, section)
            if score > 0:
                scored_results.append((score, filename, section))

        scored_results.sort(key=lambda result: result[0], reverse=True)
        minimum_evidence_score = min(2, len(query_words))
        ''' this code is stricter than intended. the app does not return "Where is the auth token generated?"
        if not scored_results or scored_results[0][0] < len(query_words):
            return []
        '''
        if not scored_results or scored_results[0][0] < minimum_evidence_score:
            return []
        return [
            (filename, section)
            for _, filename, section in scored_results[:top_k]
        ]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
