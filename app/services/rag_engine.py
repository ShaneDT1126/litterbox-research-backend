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

        # Multiple choice question patterns for detection
        self.mcq_patterns = [
            r"[A-D]\.\s.*\n[A-D]\.\s.*",  # Multiple lines with A. B. C. D.
            r"[A-D]\)\s.*\n[A-D]\)\s.*",  # Multiple lines with A) B) C) D)
            r"is\s+[A-D]\s+the\s+(correct\s+|right\s+)?answer",  # "is A the correct answer"
            r"answer\s+this\s+for\s+me",  # "answer this for me"
            r"which\s+(option|answer)\s+is\s+correct",  # "which option is correct"
            r"the\s+answer\s+is\s+[A-D]",  # "the answer is A"
            r"^[A-D]$",  # Just a letter A, B, C, or D
            r"(option|answer)\s+[A-D]",  # "option A" or "answer B"
            r"is\s+the\s+answer\s+[A-D]",  # "is the answer A"
            r"is\s+it\s+[A-D]",  # "is it A"
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

    def is_multiple_choice_question(self, query: str) -> bool:
        """Detect if the query is a multiple-choice question"""
        query_lower = query.lower()

        # Check for common MCQ patterns
        for pattern in self.mcq_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.info(f"Detected multiple-choice question pattern: {pattern}")
                return True

        # Check for options in the query
        option_patterns = [
            r"[A-D]\.\s",  # A. B. C. D.
            r"[A-D]\)\s",  # A) B) C) D)
            r"\([A-D]\)\s",  # (A) (B) (C) (D)
        ]

        for pattern in option_patterns:
            matches = re.findall(pattern, query)
            if len(matches) >= 2:  # At least two options found
                logger.info(f"Detected multiple options in question: {matches}")
                return True

        return False

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

    def is_assignment_question(self, query: str) -> bool:
        """Detect if the query appears to be from an assignment or homework"""

        # Assignment question patterns
        assignment_patterns = [
            r"what is the (minimum|maximum|number of|value of|result of)",
            r"how many [a-z]+ (are|is) (needed|required|used)",
            r"calculate the",
            r"compute the",
            r"find the",
            r"determine the",
            r"solve for",
            r"what (will|would) be the",
            r"the answer is \d+",
            r"is the answer \d+",
            r"is \d+ (the correct answer|right)",
            r"program.*return",
            r"register.*spill",
            r"pipeline.*cycle",
            r"cache.*miss",
            r"memory.*refresh"
        ]

        # Check for assignment patterns
        for pattern in assignment_patterns:
            if re.search(pattern, query.lower(), re.IGNORECASE):
                logger.info(f"Detected assignment question pattern: {pattern}")
                return True

        # Also check if it's a multiple-choice question
        return self.is_multiple_choice_question(query)

    def is_step_by_step_request(self, query: str) -> bool:
        """Detect if the student is asking for a step-by-step solution"""
        step_patterns = [
            r"(?i)step[ -]by[ -]step",
            r"(?i)do it for me",
            r"(?i)show me how",
            r"(?i)walk me through",
            r"(?i)guide me through",
            r"(?i)explain the (steps|process)",
            r"(?i)how (do|would|can) (i|you) (solve|calculate|compute|find)",
            r"(?i)what (is|are) the steps",
            r"(?i)how to (solve|calculate|compute|find)",
            r"(?i)can you (solve|calculate|compute|find) (it|this)"
        ]

        return any(re.search(pattern, query) for pattern in step_patterns)

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

        # Check if this is an assignment question
        is_assignment = self.is_assignment_question(query)

        # Check if this is a step-by-step request
        is_step_request = self.is_step_by_step_request(query)

        # Apply special handling for assignment questions and step-by-step requests
        if is_assignment:
            logger.info("Detected assignment question - applying special handling")
            query = query + " [NOTE: This appears to be an assignment question. Remember to use scaffolded guidance instead of providing the direct answer. Help the student think through the problem without revealing the final answer.]"

        if is_step_request:
            logger.info("Detected step-by-step request - applying special handling")
            query = query + " [NOTE: The student is asking for a step-by-step process. Provide guidance on the approach and methodology, but DO NOT complete the final step or reveal the answer. Stop your explanation before reaching the conclusion and ask the student what they think the answer would be.]"

        # Gather all contextual information for the query
        context = self._gather_query_context(query, history, current_scaffolding_level)

        # Add flags to context
        context["is_assignment"] = is_assignment
        context["is_step_request"] = is_step_request

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

        # Post-process the response to catch any direct answers that might slip through
        response_text = self._filter_direct_answers(response_text, context["is_assignment"], context["is_step_request"])

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

    def _filter_direct_answers(self, response: str, is_assignment: bool, is_step_request: bool) -> str:
        """Filter out direct answers to assignment questions and step-by-step requests"""
        if not (is_assignment or is_step_request):
            return response

        # Patterns that might indicate a direct answer
        direct_answer_patterns = [
            # Multiple choice answers
            r"[Tt]he (correct |right |)answer is ([A-D])",
            r"[Oo]ption ([A-D]) is (correct|right)",
            r"([A-D]) is the (correct|right) (answer|option)",
            r"[Tt]he (correct|right) (answer|option) is ([A-D])",

            # Numerical answers
            r"[Tt]he (correct |right |)(answer|result|value|number|minimum|maximum) is (\d+)",
            r"[Tt]herefore,? the (answer|result|value|number|minimum|maximum) is (\d+)",
            r"[Tt]hus,? the (answer|result|value|number|minimum|maximum) is (\d+)",
            r"[Ss]o,? the (answer|result|value|number|minimum|maximum) is (\d+)",
            r"[Tt]he (total|final) (number of |)(registers|cycles|time|value) (needed|required|used|is) (\d+)",
            r"[Ww]e need (\d+) (registers|cycles)",
            r"[Tt]he (program|code|calculation) (requires|needs|uses) (\d+)",
            r"[Tt]he answer to (this|the) (question|problem) is (\d+)",
            r"[Tt]he (correct|right) (answer|solution) is (\d+)",
            r"[Cc]onclusion\s*\n*[^.]*(\d+)[^.]*\.",
            r"[Tt]herefore, (\d+) (registers|cycles) (are|is) (needed|required|used)",
            r"[Ss]o, the answer is ([A-D]|option [A-D]|\d+)",
            r"[Tt]he total (refresh |)time is (\d+)",
            r"[Tt]he total number of clock cycles (needed |required |)is (\d+)",
        ]

        # Check if the response contains a direct answer
        for pattern in direct_answer_patterns:
            match = re.search(pattern, response)
            if match:
                logger.warning(f"Detected direct answer in response: {match.group(0)}")

                # Replace the direct answer with a scaffolded approach
                disclaimer = "\nI should guide you through this problem rather than providing a direct answer. Let's work through it step by step:\n\n"

                # Remove the direct answer statement
                modified_response = re.sub(pattern, "[Let's think about this...]", response)

                # Add guidance on how to approach the problem
                guidance = "\nTo solve this problem, consider the key concepts involved. What approach would you take to analyze this step by step?"

                return disclaimer + modified_response + guidance

        # Check for conclusion sections that might contain direct answers
        conclusion_section = re.search(r"(Conclusion|In conclusion|To summarize|Therefore)[:\s\n]+(.*?)(\n\n|$)",
                                       response, re.IGNORECASE | re.DOTALL)
        if conclusion_section:
            conclusion_text = conclusion_section.group(2)
            # Check if the conclusion contains a number (likely an answer)
            if re.search(r"\d+", conclusion_text) or re.search(r"[Tt]he answer is", conclusion_text) or re.search(
                    r"[Ss]o, the", conclusion_text):
                logger.warning(f"Detected potential direct answer in conclusion: {conclusion_text}")

                # Replace the conclusion with a more open-ended one
                if is_step_request:
                    new_conclusion = "Now that we've worked through most of the steps, what do you think the final answer would be? Try to complete the last step yourself based on what we've discussed."
                else:
                    new_conclusion = "By working through this problem step by step and analyzing the key concepts, you can determine the answer. What do you think the answer is based on the analysis we've done?"

                modified_response = response.replace(conclusion_section.group(0), f"Conclusion:\n{new_conclusion}")
                return modified_response

        # Additional check specifically for step-by-step responses
        if is_step_request:
            # Check for final answer statements at the end of the response
            last_paragraph = response.split('\n\n')[-1]
            if re.search(r"(the answer is|so,? the|therefore,? the|we get|we have|result is|total is) (\d+|[A-D])",
                         last_paragraph, re.IGNORECASE):
                # Replace the last paragraph with an open-ended question
                paragraphs = response.split('\n\n')
                paragraphs[
                    -1] = "Now that we've worked through most of the steps, what do you think the final answer would be? Try to complete the last step yourself based on what we've discussed."
                return '\n\n'.join(paragraphs)

        return response

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

        # Check if this is a verification request
        is_verification, has_proposed_answer = self._is_verification_request(query)

        # Check if this is a multiple-choice question
        is_mcq = self.is_multiple_choice_question(query)

        # Initialize context dictionary
        context = {
            "is_introduction": self.is_introduction(query) and not history,
            "is_confused": self._is_student_confused(query),
            "detected_topic": detected_topic,
            "scaffolding_level": current_scaffolding_level,
            "is_term_refresh_request": False,
            "is_acknowledgment": detected_topic == "MAINTAIN_CURRENT_TOPIC",
            "is_verification_request": is_verification,
            "has_proposed_answer": has_proposed_answer,
            "is_mcq": is_mcq
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

        # Base System template
        system_template = """
        You are LitterBox, an AI companion specialized in Computer Architecture that guides learning through discovery. Your core purpose is to help users develop understanding by thinking through concepts themselves rather than receiving direct answers.

        FUNDAMENTAL RULES:
        - For verification requests where the student's answer is CORRECT, provide direct confirmation and explanation
        - For verification requests where the student's answer is INCORRECT, DONT provide answer instead use scaffolding to guide them toward the correct answer
        - For multiple-choice questions, NEVER directly state which option (A, B, C, D) is correct
        - For calculation problems, NEVER provide the final numerical answer directly
        - For register allocation problems, NEVER state the final number of registers needed
        - For pipeline execution problems, NEVER state the final number of cycles
        - For any assignment-like question, NEVER provide the final answer
        - When asked for step-by-step guidance, STOP before the final step and ask the student to complete it
        - For all questions, guide the student to discover answers through appropriate scaffolding
        - If you are asked about anything unrelated or irrelevant, DON'T ENTERTAIN it and let them know you only answer to Computer Organization Architecture related questions. (1 sentence maximum)

        ASSIGNMENT QUESTION HANDLING:
        - When a student asks about an assignment or homework question:
        1. NEVER directly provide the final answer (e.g., "The answer is 6 registers")
        2. NEVER conclude with a definitive statement about the answer
        3. Instead, guide the student through the reasoning process step by step
        4. Ask questions that help the student reach the conclusion themselves
        5. If you walk through calculations, stop before the final answer and ask the student what they think
        6. End with an open-ended question rather than a definitive conclusion
        7. Focus on the methodology and approach rather than the final result

        STEP-BY-STEP REQUEST HANDLING:
        - When a student asks for a step-by-step process:
        1. Break down the problem-solving approach into clear steps
        2. Explain the methodology and reasoning for each step
        3. Work through the initial steps of the problem
        4. STOP before the final calculation or conclusion
        5. Ask the student what they think the next step or final answer would be
        6. Do not provide a "Conclusion" section that states the answer
        7. End with a question that encourages the student to complete the process
        
        MULTIPLE-CHOICE QUESTION HANDLING:
        - When a student asks about a multiple-choice question:
          1. NEVER directly identify which option (A, B, C, D) is correct
          2. NEVER use phrases like "the answer is X" or "option X is correct"
          3. Instead, discuss the concepts related to the question
          4. Guide the student through the reasoning process for evaluating each option
          5. Ask questions that help the student eliminate incorrect options
          6. Focus on the underlying principles rather than the specific answer

        VERIFICATION APPROACH:
        - When a student asks you to verify their answer or understanding:
          1. First determine if their answer is correct or incorrect
          2. If CORRECT: Directly confirm they are right and provide supporting explanation
          3. If INCORRECT: Use appropriate scaffolding based on their level to guide them toward the correct answer
          4. For calculations, show the correct calculation steps when confirming a correct answer
          5. For conceptual understanding, reinforce correct understanding with additional context

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

        # A user message that provides all necessary context
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
        - Verification request: {is_verification_request}
        - Has proposed answer: {has_proposed_answer}
        - Is multiple-choice question: {is_mcq}

        CONVERSATION HISTORY:
        {history}

        RELEVANT CONTENT:
        {content}

        CRITICAL GUIDELINES:
        1. For verification requests where the student's answer is CORRECT, provide direct confirmation and explanation
        2. For verification requests where the student's answer is INCORRECT, use scaffolding to guide them toward the correct answer
        3. For multiple-choice questions, NEVER directly state which option (A, B, C, D) is correct
        4. For calculation problems, NEVER provide the final numerical answer directly
        5. Be concise and focused - aim for 100-150 words when possible
        6. Ask only 1-2 questions (not more) in your response
        7. Focus on one aspect of the topic rather than trying to cover everything
        8. Use short paragraphs (2-3 sentences) and effective white space
        9. Sound conversational but efficient

        SPECIAL INSTRUCTIONS:
        {special_instructions}
        """

        # Determine special instructions based on context
        special_instructions = ""

        if context.get("is_step_request", False):
            special_instructions = """
            The student is asking for a step-by-step process:

            CRITICAL: When providing steps, you MUST NOT complete the final step or reveal the answer.

            Instead:
            1. Break down the problem-solving approach into clear steps
            2. Explain the methodology and reasoning for each step
            3. Work through the initial steps of the problem
            4. STOP before the final calculation or conclusion
            5. Ask the student what they think the next step or final answer would be
            6. Do not provide a "Conclusion" section that states the answer
            7. End with a question that encourages the student to complete the process

            Example approach: "Let me guide you through the process. First, we need to... Now, based on these steps, what do you think the answer would be?"
            """

        elif context.get("is_assignment", False):
            special_instructions = """
            This is an assignment-like question:

            CRITICAL: You MUST NOT provide the final answer.

            Instead:
            1. Guide the student through the reasoning process step by step
            2. Break down the problem into smaller parts
            3. Ask questions that help the student think through each step
            4. If you walk through calculations, stop before the final answer and ask what they think
            5. End with an open-ended question rather than a definitive conclusion
            6. NEVER state "the answer is X" or "therefore, we need X registers/cycles"
            7. NEVER include a conclusion that states the final answer

            Example approach: "Let's analyze this problem step by step. First, let's identify what we're looking for..."
            """

        elif context.get("is_mcq", False):
            special_instructions = """
                    This is a multiple-choice question:

                    CRITICAL: You MUST NOT reveal which option is correct.

                    Instead:
                    1. Discuss the concepts related to the question
                    2. Guide the student through the reasoning process for evaluating each option
                    3. Ask questions that help the student think through the problem
                    4. Focus on the underlying principles rather than the specific answer
                    5. If the student proposes an answer, do not confirm or deny it directly
                    6. Avoid phrases like "the answer is X" or "option X is correct"

                    Example approach: "This question is about [concept]. Let's think about what [concept] means and how it applies here. What do you think option A means in this context?"
                    """

        elif context.get("is_term_refresh_request", False):
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
        elif context.get("is_verification_request", False) and context.get("has_proposed_answer", False):
            special_instructions = """
            The student is asking you to verify their answer or understanding:

            CRITICAL APPROACH:

            1. First, determine if their answer/understanding is CORRECT or INCORRECT based on the relevant content

            2. If their answer is CORRECT:
               - Directly confirm they are right (e.g., "Yes, your answer of X is correct.")
               - Provide a clear explanation of why it's correct
               - For calculations, show the complete calculation steps to reinforce understanding
               - For concepts, provide additional context to deepen their understanding
               - No need to ask further questions about this specific point

            3. If their answer is INCORRECT:
               - DO NOT directly state they are wrong or provide the correct answer
               - Use scaffolding appropriate to their level to guide them toward the correct answer
               - For calculations, ask about a specific step or assumption that might be incorrect
               - For concepts, ask a targeted question that helps them reconsider their understanding

            Remember: Only provide direct confirmation when they are correct. Otherwise, use scaffolding to guide them.
            """
        elif context.get("is_introduction", False):
            special_instructions = """
                This is a greeting/introduction:
                - Respond in a friendly, conversational way
                - DO NOT treat this as a teaching opportunity about introductions
                - Simply welcome the student and briefly introduce yourself as LitterBox
                - Ask how you can help with Computer Architecture topics
                - Keep it brief (2-3 sentences)
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

        # Topic-specific instructions for complex topics
        if context["detected_topic"] == "Cache Memory" or context["detected_topic"] == "Memory Systems":
            special_instructions += """
            This is a complex topic with formulas and calculations:
            - Focus on one component of the calculation at a time
            - Use a simple example if appropriate
            - Don't introduce all variables at once
            - Guide them to build the formula step by step across multiple exchanges
            """

        # Checks for thank you messages
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

        # Creates the chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_message)
        ])

        # Formats the messages with all context variables
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
            is_verification_request=str(context.get("is_verification_request", False)),
            has_proposed_answer=str(context.get("has_proposed_answer", False)),
            is_mcq=str(context.get("is_mcq", False)),
            special_instructions=special_instructions,
            history=history,
            content=content
        )

        # Generate the response
        logger.info(f"Generating concise response for topic: {context['detected_topic']}")
        response = self.llm.invoke(messages)
        return response.content

    def _is_verification_request(self, query: str) -> Tuple[bool, bool]:
        """
        Detect if the query is asking for verification of an answer or understanding.
        Returns a tuple: (is_verification, has_proposed_answer)
        """
        verification_patterns = [
            r"(?i)is (it|this|that) correct",
            r"(?i)am i (right|correct)",
            r"(?i)is my (answer|understanding|explanation|interpretation|calculation|result|solution) (right|correct)",
            r"(?i)did i (understand|calculate|solve|get|explain) (it|this) (right|correctly)",
            r"(?i)check my (work|calculation|answer|understanding)",
            r"(?i)verify my (answer|calculation|result|understanding)",
            r"(?i)i (think|believe|calculated|got|found|understand) [^.?!]+ (is|am) i (right|correct)",
            r"(?i)my (calculations?|answers?|results?|understanding|explanation) (is|are) [^.?!]+",
            r"(?i)does that (sound|seem) (right|correct)"
        ]

        # Check if the query contains a verification request
        is_verification = any(re.search(pattern, query) for pattern in verification_patterns)

        # Check if the query contains a proposed answer
        # This could be a numerical value, a specific term, or a statement of understanding
        has_proposed_answer = bool(
            # Check for numerical values
            re.search(r'\d+(\.\d+)?', query) or
            # Check for statements like "I think X is Y" or "X means Y"
            re.search(r"(?i)(i think|i believe|means|is defined as|refers to|is when|is a|is the)", query) or
            # Check for specific technical terms in quotes or emphasized
            re.search(r'"[^"]+"|\*[^\*]+\*', query)
        )

        return is_verification, has_proposed_answer