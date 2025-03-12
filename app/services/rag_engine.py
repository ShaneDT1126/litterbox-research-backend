from typing import List, Dict, Any, Tuple, Optional
import os
import re
import time
import uuid

from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import get_logger
from app.services.conversation import ConversationMemory
from app.services.vector_store import ChromaVectorStore

logger = get_logger(__name__)


class RAGEngine:
    """Service for processing queries using RAG approach"""

    def __init__(self, vector_store_path: Optional[str] = None):
        """Initialize RAG engine"""
        # Initialize OpenAI API
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Initialize conversation memory
        self.conversation_memory = ConversationMemory()

        # Initialize ChromaDB vector store
        self.vector_store = ChromaVectorStore(
            persist_directory=vector_store_path
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=settings.LLM_TEMPERATURE,
            model=settings.LLM_MODEL
        )

        # Create multi-query retriever
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vector_store.get_retriever(
                search_kwargs={"k": settings.RETRIEVER_K}
            ),
            llm=self.llm
        )

        # Confusion indicators for detection
        self.confusion_indicators = [
            "confused", "don't understand", "unclear", "lost", "difficult",
            "i'm not sure", "don't get it", "confusing", "not helpful",
            "too complicated", "i don't know", "what do you mean",
            "not clear", "can you explain", "i'm lost"
        ]

        # Topic keywords for detection (potential to add if necessary)
        self.topic_keywords = {
            "CPU Architecture": ["cpu", "processor", "alu", "control unit", "registers", "instruction cycle",
                                 "fetch", "decode", "execute", "von neumann"],
            "Cache Memory": ["cache", "memory hierarchy", "hit", "miss", "replacement", "write policy",
                             "direct mapped", "set associative", "fully associative"],
            "Memory Systems": ["ram", "rom", "memory", "dram", "sram", "virtual memory", "paging",
                               "segmentation", "tlb", "page table", "memory management"],
            "Instruction Set Architecture": ["isa", "instruction set", "opcodes", "addressing modes", "cisc", "risc",
                                             "arm", "x86", "mips", "instruction format"],
            "Pipelining": ["pipeline", "stage", "hazard", "forwarding", "stall", "branch prediction",
                           "data hazard", "control hazard", "structural hazard"],
            "I/O Systems": ["input", "output", "i/o", "bus", "interrupt", "dma", "device", "controller",
                            "peripheral", "usb", "pci"],
            "Performance": ["performance", "speedup", "benchmark", "amdahl", "cpi", "mips", "flops",
                            "throughput", "latency", "clock rate"]
        }

        # Core template
        self.core_template = """
        You are LitterBox, an AI companion specialized in Computer Architecture that uses advanced scaffolding techniques to guide learning. Your purpose is to guide users through inquiries by breaking down complex topics into manageable steps, fostering critical thinking, and encouraging self-discovery.

        Core Communication Principles:
        1. Never provide direct, comprehensive answers
        2. Respond with strategic, thought-provoking questions
        3. Guide users to discover knowledge independently
        4. Adapt support level dynamically based on user's responses

        Current topic: {topic}
        Current scaffolding level: {level}

        {level_specific_instructions}

        Remember:
        - Guide, don't tell
        - Promote independent thinking
        - Build systematic understanding
        - Connect concepts meaningfully
        - Celebrate the learning process
        """

        # Level-specific instructions
        self.level_specific_instructions = {
            1: """
            Level 1 (Foundation Discovery):
            - Purpose: Assess baseline understanding
            - Interaction Style: Exploratory, open-ended questioning
            - Key Techniques:
              * Elicit prior knowledge
              * Use simple, non-technical language
              * Encourage initial hypothesis formation

            Example Approach:
            "I'm intrigued by your interest in this topic. What comes to your mind when you hear about [concept]? Can you share any initial thoughts or experiences that might relate?"

            Scaffolding Techniques:
            1. Break down complex concepts into simple parts
            2. Use analogies to explain difficult concepts
            3. Ask basic comprehension questions
            4. Provide encouragement and positive reinforcement
            """,

            2: """
            Level 2 (Conceptual Development):
            - Purpose: Deepen understanding through guided exploration
            - Interaction Style: Strategic questioning, partial guidance
            - Key Techniques:
              * Connect new information to existing knowledge
              * Introduce progressive complexity
              * Encourage hypothesis testing

            Example Approach:
            "Interesting observation! How might that connect to what we previously discussed about [related concept]? What patterns or connections do you see?"

            Scaffolding Techniques:
            1. Provide hints rather than full explanations
            2. Ask questions that require application of concepts
            3. Encourage the student to make connections between concepts
            4. Challenge misconceptions with guiding questions
            """,

            3: """
            Level 3 (Independent Reasoning):
            - Purpose: Promote advanced problem-solving and synthesis
            - Interaction Style: Minimal guidance, complex scenario exploration
            - Key Techniques:
              * Challenge user to apply knowledge
              * Encourage system-level thinking
              * Promote self-evaluation and reflection

            Example Approach:
            "Now that you've explored this concept, how would you design a solution that addresses multiple perspectives? What trade-offs would you consider?"

            Scaffolding Techniques:
            1. Ask challenging questions that require synthesis of multiple concepts
            2. Provide minimal guidance, focusing on verification
            3. Encourage the student to evaluate trade-offs and design decisions
            4. Prompt for deeper analysis of implications and consequences
            """
        }

        # Confusion template for handling confused students
        self.confusion_template = """
        You are LitterBox, an AI companion specialized in Computer Architecture.

        The student is confused about {topic}. They said: "{query}"

        Current scaffolding level: {level} (1 = highest support, 3 = lowest support)

        Your task is to:
        1. Acknowledge their confusion
        2. Provide a SIMPLE explanation of {topic} using analogies and everyday examples
        3. Break down the concept into smaller, more manageable parts
        4. Use appropriate scaffolding techniques for level {level}
        5. End with a simple question to check understanding

        Remember to be encouraging and supportive. If this is a new topic for them, start with a clear introduction
        of what {topic} is before providing any deeper explanations.
        """

        logger.info("RAG Engine initialized successfully")

    def is_introduction(self, query: str) -> bool:
        """Detect if the query is just an introduction or greeting"""
        introduction_patterns = [
            r"(?i)hello.*name is",
            r"(?i)hi.*name is",
            r"(?i)i am [a-z]+",
            r"(?i)my name is",
            r"(?i)^hello$",
            r"(?i)^hi$",
            r"(?i)^hey$",
            r"(?i)nice to meet you",
            r"(?i)^greetings$",
            r"(?i)^good (morning|afternoon|evening)$"
        ]

        # Clean the query by removing punctuation and extra spaces
        clean_query = query.strip().lower()

        # Check if it's a simple greeting
        simple_greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if clean_query.rstrip("!.,?") in simple_greetings:
            return True

        return any(re.search(pattern, query) for pattern in introduction_patterns)

    def is_confused(self, query: str) -> bool:
        """Detect if the student is expressing confusion"""
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in self.confusion_indicators)

    def detect_topic(self, query: str, retrieved_docs: List[Document] = None) -> str:
        """Detect the topic of the query based on keywords and retrieved documents"""

        # Check if this is just an acknowledgment
        acknowledgment_patterns = [
            r"(?i)^(i understand|i get it|that makes sense|i see|got it|thank you|thanks).*$",
            r"(?i)^(now i (understand|get|see)).*$",
            r"(?i)^(that('s| is) (clear|helpful)).*$"
        ]

        is_acknowledgment = any(re.match(pattern, query) for pattern in acknowledgment_patterns)

        if is_acknowledgment:
            logger.info("Detected acknowledgment - will maintain current topic")
            return "MAINTAIN_CURRENT_TOPIC"  # Special flag to maintain current topi

        # Check if this is a general greeting or introduction
        if self.is_introduction(query):
            return "Introduction"

        # Clean the query
        query_lower = query.lower().strip()

        # If the query is very short (less than 3 words) and not a specific question, it might not be a specific topic
        if len(query_lower.split()) < 3 and "?" not in query_lower:
            # Check if it's an expression of confusion
            if self.is_confused(query_lower):
                return "Confusion"  # Special marker for confusion
            return "Topic Selection"

        # Count keyword matches for each topic
        topic_scores = {topic: 0 for topic in self.topic_keywords}

        # Check query for keywords
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    topic_scores[topic] += 1

        # Check retrieved documents for topic metadata
        if retrieved_docs:
            # Count topics in retrieved documents
            doc_topics = {}
            for doc in retrieved_docs:
                if "topic" in doc.metadata:
                    doc_topic = doc.metadata["topic"]
                    if doc_topic in doc_topics:
                        doc_topics[doc_topic] += 1
                    else:
                        doc_topics[doc_topic] = 1

            # Add document topics to scores with higher weight
            for topic, count in doc_topics.items():
                if topic in topic_scores:
                    topic_scores[topic] += count * 2  # Give more weight to document metadata

        # Get topic with highest score
        max_score = 0
        detected_topic = "General Computer Architecture"

        for topic, score in topic_scores.items():
            if score > max_score:
                max_score = score
                detected_topic = topic

        # If the max score is still 0, it means we couldn't detect a specific topic
        if max_score == 0:
            # Check if it's an expression of confusion
            if self.is_confused(query_lower):
                return "Confusion"  # Special marker for confusion
            return "Topic Selection"

        return detected_topic

    def determine_scaffolding_level(self, conversation_history: List[Dict], current_level: int = 2) -> int:
        """Determine appropriate scaffolding level based on conversation history"""
        if not conversation_history:
            return current_level

        # Only consider recent exchanges (last 5)
        recent_exchanges = conversation_history[-5:]

        # Count indicators of understanding
        understanding_indicators = sum(
            1 for exchange in recent_exchanges
            if any(phrase in exchange.get("student_message", "").lower()
                   for phrase in
                   ["i understand", "that makes sense", "got it", "i see", "clear", "understood",
                    "right", "okay", "perfect", "great", "i get it"
                    ])
        )

        # Count indicators of confusion
        confusion_indicators = sum(
            1 for exchange in recent_exchanges
            if any(phrase in exchange.get("student_message", "").lower()
                   for phrase in self.confusion_indicators)
        )

        # Calculate adjustment
        adjustment = understanding_indicators - confusion_indicators

        # Apply adjustment to current level
        new_level = current_level + adjustment

        # Ensure level is between 1 and 3
        return max(1, min(new_level, 3))

    def get_most_recent_topic(self, history: List[Dict]) -> str:
        """Get the most recent non-general topic from conversation history"""
        if not history:
            return "General Computer Architecture"

        # Look through history in reverse to find the most recent topic
        for exchange in reversed(history):
            topic = exchange.get("topic")
            if topic and topic not in ["Introduction", "Topic Selection", "Confusion"]:
                return topic

        return "General Computer Architecture"

    def process_query(self,
                      query: str,
                      conversation_id: str = "new",
                      student_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the single template approach
        Returns: A response dictionary with all relevant information
        """
        # Get conversation history or create a new conversation
        conversation_id, history = self._check_conversation_id(conversation_id, student_id)

        # Log conversation details
        self._log_conversation_details(conversation_id, history)

        # Determine current scaffolding level from conversation history
        current_scaffolding_level = self._get_current_scaffolding_level(conversation_id, history)

        # Gather all contextual information for the query
        context = self._gather_query_context(query, history, current_scaffolding_level)

        # Handle special cases
        if context["is_introduction"]:
            return self._handle_introduction(query, conversation_id, context["scaffolding_level"], student_id)

        # Retrieve relevant documents
        retrieved_docs = self._retrieve_relevant_documents(query, context["detected_topic"])

        # Format conversation history and extract document content
        formatted_history = self._format_conversation_history(history)
        relevant_content = self._extract_document_content(retrieved_docs)

        # Generate response using the single template
        response_text = self._generate_response_with_single_template(
            query=query,
            history=formatted_history,
            content=relevant_content,
            context=context
        )

        # Format sources for citation
        sources = self._format_sources(retrieved_docs, context["detected_topic"])

        # Update scaffolding level in memory if it changed
        if context["scaffolding_level"] != current_scaffolding_level:
            self.conversation_memory.update_scaffolding_level(conversation_id, context["scaffolding_level"])

        # Store the exchange in conversation memory
        self._store_exchange(
            conversation_id,
            query,
            response_text,
            context["detected_topic"],
            context["scaffolding_level"],
            student_id
        )

        # Create and return response object
        return self._create_response_object(
            response_text,
            sources,
            context["scaffolding_level"],
            context["detected_topic"],
            conversation_id
        )

    def _check_conversation_id(self, conversation_id: str, student_id: Optional[str]) -> Tuple[str, List[Dict]]:
        """Check if conversation exists and retrieve history"""
        return self.conversation_memory.get_conversation(conversation_id, student_id)

    def _log_conversation_details(self, conversation_id: str, history: List[Dict]):
        """Log details about the current conversation"""
        logger.info(f"Conversation ID: {conversation_id}")
        logger.info(f"History length: {len(history)}")
        for i, exchange in enumerate(history):
            logger.info(f"Exchange {i}:")
            logger.info(f"  Student: {exchange.get('student_message', 'N/A')}")
            logger.info(f"  Assistant: {exchange.get('assistant_message', 'N/A')[:50]}...")
            logger.info(f"  Topic: {exchange.get('topic', 'N/A')}")

    def _get_current_scaffolding_level(self, conversation_id: str, history: List[Dict]) -> int:
        """Get the current scaffolding level based on history"""
        if history:
            return self.conversation_memory.get_current_scaffolding_level(conversation_id)
        return settings.DEFAULT_SCAFFOLDING_LEVEL

    def _handle_introduction(self, query: str, conversation_id: str,
                             scaffolding_level: int, student_id: Optional[str]) -> Dict[str, Any]:
        """Handle introduction queries using the single template"""
        # Extract name if possible
        name_match = re.search(r"(?i)name is (\w+)", query)
        name = name_match.group(1) if name_match else ""

        # Create a minimal context for introduction
        context = {
            "detected_topic": "Introduction",
            "scaffolding_level": scaffolding_level,
            "student_name": name,
            "is_introduction": True,
            "is_confused": False,
            "is_topic_transition": False,
            "previous_topic": None
        }

        # Generate introduction response using the single template
        response_text = self._generate_response_with_single_template(
            query=query,
            history="",  # No history for introduction
            content="",  # No content needed for introduction
            context=context
        )

        # Store the exchange in conversation memory
        self._store_exchange(
            conversation_id,
            query,
            response_text,
            "Introduction",
            scaffolding_level,
            student_id
        )

        # Prepare response object
        return self._create_response_object(
            response_text,
            [],  # No sources for introduction
            scaffolding_level,
            "Introduction",
            conversation_id
        )

    def _is_student_confused(self, query: str) -> bool:
        """Check if the student is expressing confusion"""
        confusion_phrases = [
            "confused", "don't understand", "unclear", "lost", "difficult",
            "i'm not sure", "im not sure", "don't get it", "confusing", "not helpful",
            "too complicated", "i don't know", "what do you mean", "teach me",
            "not clear", "can you explain", "i'm lost", "dont know", "do not know"
        ]

        query_lower = query.lower()
        for phrase in confusion_phrases:
            if phrase in query_lower:
                logger.info(f"Detected confusion phrase: '{phrase}'")
                return True
        return False

    def _get_most_recent_topic(self, history: List[Dict]) -> Optional[str]:
        """Find the most recent substantive topic from conversation history"""
        for exchange in reversed(history):
            topic = exchange.get("topic")
            if topic and topic not in ["Introduction", "Topic Selection"]:
                logger.info(f"Found substantive topic: {topic}")
                return topic
        logger.info("No substantive topic found in history")
        return None

    def _handle_topic_selection(self, query: str, conversation_id: str,
                                scaffolding_level: int, student_id: Optional[str]) -> Dict[str, Any]:
        """Handle topic selection queries"""
        response_text = """I'd be happy to help you with Computer Organization Architecture! 

        To provide the most relevant information, could you specify which topic you'd like to explore? Here are some key areas we could discuss:

        1. CPU Architecture - components, instruction cycle, registers
        2. Cache Memory - hierarchy, mapping techniques, replacement policies
        3. Memory Systems - RAM, ROM, virtual memory
        4. Instruction Set Architecture - CISC vs RISC, addressing modes
        5. Pipelining - stages, hazards, optimization
        6. I/O Systems - buses, interrupts, DMA
        7. Performance - metrics, benchmarking, optimization

        Which of these topics would you like to learn more about?"""

        # Store the exchange in conversation memory
        self._store_exchange(
            conversation_id,
            query,
            response_text,
            "Topic Selection",
            scaffolding_level,
            student_id
        )

        # Prepare response object
        return self._create_response_object(
            response_text,
            [],  # No sources for topic selection
            scaffolding_level,
            "Topic Selection",
            conversation_id
        )

    def _retrieve_relevant_documents(self, query: str, detected_topic: str) -> List[Document]:
        """Retrieve and filter relevant documents for the query"""
        # Only retrieve documents if we have a specific topic
        if detected_topic not in ["Introduction", "Topic Selection", "General Computer Architecture"]:
            # Retrieve relevant documents using multiquery
            retrieved_docs = self.retriever.invoke(query)

            # Filter out irrelevant documents if we have a specific topic
            if detected_topic != "General Computer Architecture":
                filtered_docs = []
                for doc in retrieved_docs:
                    # Check if document content is relevant to the topic
                    doc_content = doc.page_content.lower()
                    topic_keywords = self.topic_keywords.get(detected_topic, [])

                    # Count keyword matches in document
                    keyword_matches = sum(1 for keyword in topic_keywords if keyword in doc_content)

                    # Only include document if it has sufficient keyword matches
                    if keyword_matches >= 1:
                        filtered_docs.append(doc)

                # Use filtered documents if we have enough, otherwise use original
                if len(filtered_docs) >= 2:
                    return filtered_docs
                return retrieved_docs
            return retrieved_docs
        return []  # For general topics, don't retrieve documents

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history with enhanced topic tracking"""
        if not history:
            return ""

        # Include more context from previous exchanges
        recent_history = history[-5:] if len(history) >= 5 else history
        history_parts = []

        # Add a header to emphasize the importance of the conversation history
        history_parts.append("--- PREVIOUS CONVERSATION ---")

        # Track topics discussed
        topics_discussed = []
        key_terms = []

        for i, exchange in enumerate(recent_history):
            student_msg = exchange.get('student_message', '')
            assistant_msg = exchange.get('assistant_message', '')
            topic = exchange.get('topic', 'Unknown')

            # Track topics
            if topic not in ["Introduction", "Topic Selection"] and topic not in topics_discussed:
                topics_discussed.append(topic)

            # Extract potential key terms from assistant messages (simplified approach)
            potential_terms = re.findall(r'\*\*(.*?)\*\*', assistant_msg)
            key_terms.extend(potential_terms)

            # Add more context about each exchange
            history_parts.append(f"Exchange {i + 1} (Topic: {topic}):")
            history_parts.append("Student: " + student_msg)
            history_parts.append("Assistant: " + assistant_msg)
            history_parts.append("---")

        # Add topic summary
        if topics_discussed:
            history_parts.append(f"Main topics discussed: {', '.join(topics_discussed)}")

        # Add key terms summary
        if key_terms:
            history_parts.append(f"Key terms mentioned: {', '.join(set(key_terms))}")

        history_parts.append("--- CURRENT EXCHANGE ---")

        return "\n".join(history_parts)

    def _extract_document_content(self, retrieved_docs: List[Document]) -> str:
        """Extract relevant content from retrieved documents"""
        content_parts = []
        if retrieved_docs:
            for i, doc in enumerate(retrieved_docs[:3]):
                content_parts.append(f"DOCUMENT {i + 1}:\n{doc.page_content}")
                logger.info(f"Retrieved document {i + 1} with topic: {doc.metadata.get('topic', 'unknown')}")
        return "\n\n".join(content_parts)

    def _extract_student_name(self, history: List[Dict]) -> Optional[str]:
        """Extract student name from conversation history if available"""
        if not history:
            return None

        # Check introduction message for name
        for exchange in history:
            student_msg = exchange.get('student_message', '').lower()
            if "name is" in student_msg or "i am" in student_msg or "i'm" in student_msg:
                # Try to extract name using regex
                name_match = re.search(r"(?:my name is|i am|i'm) (\w+)", student_msg, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).capitalize()
                    logger.info(f"Extracted student name: {name}")
                    return name

        return None

    def _format_sources(self, retrieved_docs: List[Document], topic: str) -> List[Dict]:
        """Format sources for citation"""
        sources = []
        for doc in retrieved_docs[:3]:
            source = {
                "content": doc.page_content[:200] + "...",  # Truncate for brevity
                "topic": doc.metadata.get("topic", topic),
                "confidence": 0.8,  # Placeholder confidence score
                "metadata": doc.metadata
            }
            sources.append(source)
        return sources

    def _check_conversation_id(self, conversation_id: str, student_id: Optional[str]) -> Tuple[str, List[Dict]]:
        """Check if conversation exists and retrieve history"""
        return self.conversation_memory.get_conversation(conversation_id, student_id)

    def _store_exchange(self, conversation_id: str, query: str, response: str,
                        topic: str, level: int, student_id: Optional[str]) -> None:
        """Store an exchange in the conversation memory"""
        self.conversation_memory.add_exchange(
            conversation_id=conversation_id,
            student_message=query,
            assistant_message=response,
            topic=topic,
            scaffolding_level=level,
            student_id=student_id
        )

    def _create_response_object(self, response_text: str, sources: List[Dict],
                                level: int, topic: str, conversation_id: str) -> Dict[str, Any]:
        """Create a standardized response object"""
        return {
            "response": response_text,
            "sources": sources,
            "scaffolding_level": level,
            "topic": topic,
            "conversation_id": conversation_id
        }

    def _is_first_time_topic(self, history: List[Dict], topic: str) -> bool:
        """Check if this is the first time discussing a topic"""
        if not history:
            return True

        for exchange in history:
            if exchange.get("topic") == topic:
                return False
        return True

    def _gather_query_context(self, query: str, history: List[Dict], current_scaffolding_level: int) -> Dict:
        """Gather all contextual information needed for response generation"""

        # Detect the topic
        detected_topic = self.detect_topic(query)

        # Initialize context dictionary
        context = {
            "is_introduction": self.is_introduction(query) and not history,
            "is_confused": self._is_student_confused(query),
            "detected_topic": detected_topic,
            "scaffolding_level": current_scaffolding_level,
            "is_term_refresh_request": False,
            "is_acknowledgment": detected_topic == "MAINTAIN_CURRENT_TOPIC"
        }

        # If this is an acknowledgment or we need to maintain the current topic
        if context["is_acknowledgment"] and history:
            previous_topic = self._get_most_recent_topic(history)
            if previous_topic and previous_topic not in ["Introduction"]:
                context["detected_topic"] = previous_topic
                logger.info(f"Maintaining previous topic: {previous_topic}")

        # Extract student name if available
        context["student_name"] = self._extract_student_name(history)

        # Determine if this is a topic transition
        previous_topic = self._get_most_recent_topic(history) if history else None
        context["previous_topic"] = previous_topic
        context["is_topic_transition"] = (previous_topic and
                                          previous_topic != context["detected_topic"] and
                                          context["detected_topic"] not in ["Introduction", "Topic Selection",
                                                                            "General Computer Architecture"])

        # Adjust scaffolding level based on context
        if context["is_confused"] and context["scaffolding_level"] > 1:
            # Lower scaffolding level if student is confused
            logger.info(
                f"Lowering scaffolding level due to confusion: {context['scaffolding_level']} -> {context['scaffolding_level'] - 1}")
            context["scaffolding_level"] = context["scaffolding_level"] - 1
        elif not context["is_confused"] and len(history) >= 3:
            # Potentially adjust scaffolding level based on demonstrated understanding
            adjusted_level = self.determine_scaffolding_level(history, current_scaffolding_level)
            if adjusted_level != current_scaffolding_level:
                logger.info(
                    f"Adjusting scaffolding level based on understanding: {current_scaffolding_level} -> {adjusted_level}")
                context["scaffolding_level"] = adjusted_level

        # Log the context for debugging
        logger.info(f"Query context: {context}")

        return context

    def _generate_response_with_single_template(self, query: str, history: str, content: str, context: Dict) -> str:
        """Generate a response using a refined template that balances guidance with conciseness"""

        # Revised system template
        system_template = """
        You are LitterBox, an AI companion specialized in Computer Architecture that guides learning through discovery. Your core purpose is to help users develop understanding by thinking through concepts themselves rather than receiving direct answers.

        FUNDAMENTAL RULE:
        - NEVER provide direct, complete answers to questions
        - Instead, guide the student to discover answers through appropriate scaffolding

        SCAFFOLDING APPROACH:

        Level 1 (High Support):
        - Provide partial explanations that point in the right direction
        - Use concrete analogies and examples that illuminate concepts
        - Break complex ideas into smaller, manageable components
        - Ask 1-2 focused guiding questions (not more)
        - Balance: 70% guided explanation, 30% thoughtful questions

        Level 2 (Moderate Support):
        - Offer minimal explanations that highlight key principles
        - Ask 1-2 more challenging questions that prompt connections
        - Suggest approaches for thinking about the problem
        - Balance: 50% guided explanation, 50% thoughtful questions

        Level 3 (Minimal Support):
        - Provide very limited direct guidance
        - Ask 1-2 challenging questions that require synthesis
        - Prompt for analysis of trade-offs and implications
        - Balance: 30% guided explanation, 70% thoughtful questions

        COMMUNICATION STYLE:
        - Be concise and focused - keep responses under 150 words when possible
        - Focus on one aspect of the topic at a time rather than covering everything
        - Use natural, conversational language that varies in structure
        - Avoid repetitive patterns, phrases, and response structures
        - Sound like a thoughtful peer rather than an instructor

        RESPONSE STRUCTURE:
        - Start with a brief framing of the concept (1-2 sentences)
        - Provide a partial explanation or analogy (2-3 sentences)
        - Ask 1-2 focused questions (not more)
        - Keep paragraphs short (2-3 sentences maximum)
        - Use white space effectively

        PROHIBITED PATTERNS:
        - NEVER provide complete, direct answers to questions
        - Don't ask more than 2 questions in a single response
        - Don't overwhelm with too many concepts at once
        - Don't write lengthy, dense paragraphs
        - Don't start multiple responses with the same phrases
        - Avoid excessive enthusiasm or praise

        GUIDING TECHNIQUES:
        - Focus on one key aspect rather than the entire concept
        - Use strategic hints that point toward discovery
        - Analogies that illuminate complex concepts
        - Partial explanations that invite completion
        - Progressive guidance across multiple exchanges
        
        HANDLING CLOSURE:
        - When a student indicates understanding or thanks you, respect that as closure
        - Do not continue teaching or asking questions when a student signals they're done
        - Provide a brief, positive acknowledgment and offer to help with other topics
        - Recognize phrases like "I understand now", "thank you", "got it" as signals to conclude
        - Keep closure responses brief (2-3 sentences) and don't introduce new material
        """

        # Create a user message that provides all necessary context
        user_message = """
        CURRENT QUERY: {query}

        TOPIC: {topic}

        SCAFFOLDING LEVEL: {level}

        CONTEXTUAL INFORMATION:
        - Student name: {name}
        - Is introduction: {is_introduction}
        - Student appears confused: {is_confused}
        - Topic transition: {is_transition}
        - Previous topic: {previous_topic}
        - Term refresh request: {is_term_refresh}

        CONVERSATION HISTORY:
        {history}

        RELEVANT CONTENT:
        {content}

        CRITICAL GUIDELINES:
        1. NEVER provide a direct, complete answer to the question
        2. Guide the student to discover the answer through appropriate scaffolding
        3. Be concise and focused - aim for 100-150 words when possible
        4. Ask only 1-2 questions (not more) in your response
        5. Focus on one aspect of the topic rather than trying to cover everything
        6. Use short paragraphs (2-3 sentences) and effective white space
        7. Sound conversational but efficient

        SPECIAL INSTRUCTIONS:
        {special_instructions}
        """

        # Determine special instructions based on context
        special_instructions = ""

        if context.get("is_term_refresh_request", False):
            special_instructions = """
            The student is asking about terms:
            - Focus on 2-3 key terms maximum, not all of them
            - Use the terms naturally in context
            - Ask if one of these terms is what they're trying to recall
            """
        elif context.get("is_acknowledgment", False):
            special_instructions = """
            The student is acknowledging understanding and possibly concluding this topic.

            IMPORTANT: When a student says they understand or thank you, DO NOT continue teaching the same topic or ask follow-up questions about it.

            Instead:
            1. Briefly acknowledge their understanding with a short, positive response (1 sentence)
            2. Express that you're available for further questions on this or other topics
            3. Mention they can ask about other computer architecture topics if they wish
            4. Keep this response brief (2-3 sentences total) and conclusive
            5. DO NOT introduce new concepts or ask questions about the current topic

            Example: "Happy to help! If you have any other questions about cache memory or want to explore another computer architecture topic, just let me know."
            """

        elif context.get("is_confused", False):
            special_instructions = """
            The student is expressing confusion:
            - Offer a simpler analogy (1-2 sentences)
            - Break down just one aspect of the concept
            - Ask one very basic question to check understanding
            """
        elif context.get("is_topic_transition", False):
            special_instructions = """
            The student is transitioning to a new topic:
            - Briefly introduce just one aspect of the new topic
            - Ask one question about their familiarity with this aspect
            """

        # Add topic-specific instructions for complex topics
        if context["detected_topic"] == "Cache Memory" or context["detected_topic"] == "Memory Systems":
            special_instructions += """
            This is a complex topic with formulas and calculations:
            - Focus on one component of the calculation at a time
            - Use a simple example if appropriate
            - Don't introduce all variables at once
            - Guide them to build the formula step by step across multiple exchanges
            """

        # Check for thank you messages
        thank_you_patterns = [
            r"(?i)thank(s| you)",
            r"(?i)got it",
            r"(?i)(i )?understand( it)? now",
            r"(?i)that('s| is) (helpful|clear|good)"
        ]

        is_thank_you = any(re.search(pattern, query) for pattern in thank_you_patterns)

        if is_thank_you and not context.get("is_acknowledgment", False):
            context["is_acknowledgment"] = True
            special_instructions = """
                The student is thanking you or indicating they understand.

                IMPORTANT: When a student thanks you or says they understand, DO NOT continue teaching the same topic or ask follow-up questions about it.

                Instead:
                1. Briefly acknowledge with a short, positive response (1 sentence)
                2. Express that you're available for further questions on this or other topics
                3. Mention they can ask about other computer architecture topics if they wish
                4. Keep this response brief (2-3 sentences total) and conclusive
                5. DO NOT introduce new concepts or ask questions about the current topic

                Example: "You're welcome! If you have any other questions about cache memory or want to explore another computer architecture topic, just let me know."
                """

        # Create the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_message)
        ])

        # Format the messages with all context variables
        messages = chat_prompt.format_messages(
            query=query,
            topic=context["detected_topic"],
            level=context["scaffolding_level"],
            name=context["student_name"] if context["student_name"] else "the student",
            is_introduction=str(context["is_introduction"]),
            is_confused=str(context["is_confused"]),
            is_transition=str(context["is_topic_transition"]),
            previous_topic=context["previous_topic"] if context["previous_topic"] else "none",
            is_term_refresh=str(context.get("is_term_refresh_request", False)),
            special_instructions=special_instructions,
            history=history,
            content=content
        )

        # Generate the response
        logger.info(f"Generating concise response for topic: {context['detected_topic']}")
        response = self.llm.invoke(messages)
        return response.content