"""
Chatbot Service using Fine-tuned LLaMA 3.2 3B with LoRA
Trained on DailyDialog dataset for dementia care conversations
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
import re
from datetime import datetime

# Import NLP Engine for intelligent context detection
from src.features.conversational_ai.nlp.nlp_engine import NLPEngine

# Import ModelLoader for loading models from registry
from src.utils.model_loader import get_model_loader

import logging
import transformers.pytorch_utils
import transformers.generation.utils

logger = logging.getLogger(__name__)

# --- MONKEY PATCH FOR MPS COMPATIBILITY ---
# Fixes "IndexError: tuple index out of range" in isin_mps_friendly on Apple Silicon
# when special tokens are scalar tensors (0-d).
def isin_mps_friendly_patched(elements, test_elements):
    if test_elements.ndim == 0:
        test_elements = test_elements.unsqueeze(0)
    if elements.ndim == 0:
        elements = elements.unsqueeze(0)
    
    # Original logic from transformers library
    return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()

# Apply patch to likely locations
transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly_patched
if hasattr(transformers.generation.utils, "isin_mps_friendly"):
    transformers.generation.utils.isin_mps_friendly = isin_mps_friendly_patched
logger.info("🔧 Applied MPS compatibility patch for transformers library")
# ------------------------------------------


class DementiaChatbot:
    """
    Chatbot for dementia care using fine-tuned LLaMA 3.2 3B model.
    Model is trained on DailyDialog dataset for natural conversations.
    """

    def __init__(
        self,
        base_model_name: str = None,
        lora_adapter_path: str = None,
        device: str = None
    ):
        """
        Initialize the chatbot with fine-tuned model.

        Args:
            base_model_name: HuggingFace model ID for base LLaMA model (defaults to env var LLAMA_BASE_MODEL or registry)
            lora_adapter_path: HuggingFace model ID or local path to LoRA adapter (defaults to env var LLAMA_LORA_ADAPTER or registry)
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        # Try to load from model registry first, then env vars, then hardcoded defaults
        loader = get_model_loader()
        llama_model_info = loader.get_model_info("llama_3_2_3b_dementia_care")

        # Get base model and adapter from registry or fallbacks
        if llama_model_info and llama_model_info.get("model_source") == "huggingface":
            registry_base_model = llama_model_info.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
            registry_adapter = llama_model_info.get("lora_adapter", "susadi/hale-empathy-3b")
        else:
            registry_base_model = "meta-llama/Llama-3.2-3B-Instruct"
            registry_adapter = "susadi/hale-empathy-3b"

        # Priority: passed args > env vars > registry > hardcoded defaults
        self.base_model_name = base_model_name or os.getenv("LLAMA_BASE_MODEL", registry_base_model)
        self.lora_adapter_path = lora_adapter_path or os.getenv("LLAMA_LORA_ADAPTER", registry_adapter)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        self.model = None
        self.tokenizer = None
        self.conversation_history: Dict[str, list] = {}  # session_id -> messages

        # Initialize NLP Engine for context-aware prompting
        try:
            logger.info("Initializing NLP Engine for intelligent prompting...")
            self.nlp_engine = NLPEngine(
                enable_semantic=False,  # Disable heavy semantic analysis for speed
                enable_emotion=True,
                enable_linguistic=True,
                device="cpu"  # Use CPU for NLP to avoid GPU conflicts
            )
            logger.info("[SUCCESS] NLP Engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NLP Engine: {e}. Using simple prompting.")
            self.nlp_engine = None

        self._load_model()

    def _load_model(self):
        """Load base model and apply LoRA adapter."""
        try:
            # Get HuggingFace token from environment (for private models)
            hf_token = os.getenv("LLAMA_HF_TOKEN")
            if hf_token:
                logger.info("HuggingFace token found - can access private models")

            logger.info(f"Loading base model: {self.base_model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                token=hf_token  # Use token for private models
            )

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                token=hf_token  # Use token for private models
            )

            # Load and apply LoRA adapter (from HuggingFace or local path)
            logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_adapter_path,  # Works with both HuggingFace ID and local path
                    token=hf_token  # Use token for private models
                )
                logger.info("LoRA adapter loaded successfully")
            except Exception as adapter_error:
                logger.warning(f"Could not load LoRA adapter: {adapter_error}")
                logger.warning("Using base model without LoRA")
                self.model = base_model

            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)

            # Set to eval mode
            self.model.eval()

            logger.info("[SUCCESS] Chatbot model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.5,
        top_p: float = 0.9,
        use_history: bool = True
    ) -> Dict[str, Any]:
        """
        Generate chatbot response for user message.

        Args:
            user_message: The user's input text
            user_id: User identifier
            session_id: Optional session ID for conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            use_history: Whether to use conversation history

        Returns:
            Dict with response, session_id, and metadata
        """
        try:
            # Create session ID if not provided
            if session_id is None:
                session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Initialize session history if needed
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Adaptive max_tokens based on message length and type
            msg_lower = user_message.lower().strip()
            msg_word_count = len(msg_lower.split())
            if msg_word_count <= 3:  # Short messages like "hi", "hello", "I feel sad"
                max_tokens = min(max_tokens, 60)
            elif '?' in user_message or any(msg_lower.startswith(q) for q in ['what', 'how', 'can', 'tell', 'explain', 'describe']):
                max_tokens = min(max_tokens, 100)  # Questions get moderate length
            else:
                max_tokens = min(max_tokens, 80)  # Default moderate

            # Build conversation context
            if use_history and self.conversation_history[session_id]:
                # Include recent conversation history (last 5 exchanges)
                recent_history = self.conversation_history[session_id][-10:]  # Last 5 Q&A pairs
                context_messages = recent_history + [user_message]
                prompt = self._build_prompt(context_messages, session_id)
            else:
                prompt = self._build_prompt([user_message], session_id)

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            # Prepare token IDs - ensure they are integers, not tuples/lists
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            
            # Handle case where token IDs might be lists/tuples
            if isinstance(pad_token_id, (list, tuple)):
                pad_token_id = pad_token_id[0] if len(pad_token_id) > 0 else None
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0] if len(eos_token_id) > 0 else None
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )

            # Decode response
            # Handle different return types from generate()
            if isinstance(outputs, tuple):
                generated_ids = outputs[0][0]
            elif hasattr(outputs, 'sequences'):
                generated_ids = outputs.sequences[0]
            else:
                generated_ids = outputs[0]
            
            full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Extract just the assistant's response
            response_text = self._extract_response(full_response, prompt)

            # If extraction failed, try with skip_special_tokens
            if not response_text or len(response_text.strip()) == 0:
                full_response_clean = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if full_response_clean.startswith(prompt):
                    response_text = full_response_clean[len(prompt):].strip()
                else:
                    response_text = full_response_clean.strip()

            # Post-process: clean training data leakage and enforce brevity
            response_text = self._clean_response(response_text)

            # Store in conversation history
            self.conversation_history[session_id].append(user_message)
            self.conversation_history[session_id].append(response_text)

            # Detect if response might need safety monitoring
            safety_warnings = self._check_safety(response_text, user_message)

            return {
                "response": response_text,
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": self.base_model_name,
                    "adapter": self.lora_adapter_path,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "conversation_length": len(self.conversation_history[session_id])
                },
                "safety_warnings": safety_warnings if safety_warnings else None
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _analyze_user_context(self, user_message: str, session_id: str) -> Dict[str, Any]:
        """
        Analyze user message to detect context signals for adaptive prompting.

        Returns dict with:
            - is_greeting: bool
            - is_first_message: bool
            - is_question: bool
            - wants_list: bool
            - asking_datetime: bool
            - confusion: bool
            - emotion: str
            - current_datetime: str
        """
        context = {
            "is_greeting": False,
            "is_first_message": False,
            "is_question": False,
            "wants_list": False,
            "asking_datetime": False,
            "confusion": False,
            "emotion": "neutral",
            "current_datetime": ""
        }

        msg_lower = user_message.lower().strip()

        # Check if this is the first message in the session
        if session_id not in self.conversation_history or len(self.conversation_history[session_id]) == 0:
            context["is_first_message"] = True

        # 1. Detect greetings
        greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
            r'^(hi|hello|hey)[\s!,.]',
        ]
        if any(re.search(pattern, msg_lower) for pattern in greeting_patterns):
            context["is_greeting"] = True

        # 2. Detect questions
        if '?' in user_message or any(msg_lower.startswith(q) for q in
            ['what', 'where', 'when', 'who', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']):
            context["is_question"] = True

        # 3. Detect list requests
        list_keywords = ['list', 'steps', 'how to', 'ways to', 'options', 'things', 'items', 'bullet']
        if any(keyword in msg_lower for keyword in list_keywords):
            context["wants_list"] = True

        # 4. Detect date/time requests
        datetime_keywords = [
            'what time', 'what day', 'what date', 'what is the time', 'what is the date',
            'what\'s the time', 'what\'s the date', 'time is it', 'date is it',
            'today\'s date', 'current time', 'current date', 'what year', 'tell me the time',
            'tell me the date'
        ]
        if any(keyword in msg_lower for keyword in datetime_keywords):
            context["asking_datetime"] = True
            # Get current date and time
            now = datetime.now()
            context["current_datetime"] = now.strftime("%A, %B %d, %Y at %I:%M %p")

        # 5. Detect confusion or memory issues
        confusion_keywords = [
            "forget", "can't remember", "don't remember", "confused", "lost",
            "where am i", "what day", "don't know", "not sure", "unclear"
        ]
        if any(keyword in msg_lower for keyword in confusion_keywords):
            context["confusion"] = True

        # 6. Use NLP Engine for emotion detection (if available)
        if self.nlp_engine:
            try:
                analysis = self.nlp_engine.analyze(user_message)
                if analysis.emotion_analysis:
                    context["emotion"] = analysis.emotion_analysis.dominant_emotion
                    # Check for emotional distress
                    if analysis.emotion_analysis.dominant_emotion in ['fear', 'sadness', 'anger']:
                        context["emotion"] = analysis.emotion_analysis.dominant_emotion
            except Exception as e:
                logger.debug(f"NLP emotion analysis failed: {e}")

        return context

    def _build_prompt(self, messages: list, session_id: str = None) -> str:
        """
        Build intelligent, context-aware prompt using NLP analysis.
        Uses LLaMA's instruction format with adaptive system prompts.
        """
        # Get the latest user message
        user_message = messages[-1] if messages else ""

        # Analyze context
        context = self._analyze_user_context(user_message, session_id or "default")

        # Build adaptive system prompt based on context
        system_prompt = self._build_adaptive_system_prompt(context, user_message)

        if len(messages) == 1:
            # Single message - simple format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{messages[0]}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Multi-turn conversation
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

            # Add conversation history
            for i, msg in enumerate(messages):
                if i % 2 == 0:  # User message
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>"
                else:  # Assistant message
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>"

            # Add final user message if needed
            if len(messages) % 2 == 1:
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def _build_adaptive_system_prompt(self, context: Dict[str, Any], user_message: str) -> str:
        """
        Build an adaptive system prompt based on detected context signals.
        """
        # Build introduction for first greeting
        introduction = ""
        if context["is_greeting"] and context["is_first_message"]:
            introduction = "\n- This is the first message. Introduce yourself as 'Hale', a helpful assistant here to support them."

        # Add current datetime if asking
        datetime_info = ""
        if context["asking_datetime"] and context["current_datetime"]:
            datetime_info = f"\n\nCURRENT DATE AND TIME: {context['current_datetime']}\n- The user is asking about the date or time. Provide this information clearly."

        base_prompt = """You are Hale, a friendly AI care companion for elderly individuals.

CRITICAL RULES:{introduction}
- Keep ALL responses SHORT: 2-3 sentences maximum.
- For greetings, reply in ONE sentence only.
- You are an AI assistant, NOT a human. Never claim to be a real person, therapist, or counselor.
- NEVER mention personal details like your education, location, license, or experience.
- NEVER share phone numbers, helpline numbers, or website URLs.
- NEVER claim to be from a specific city, state, or country. You can answer questions about places if the user asks.
- If the user is emotional, respond with warmth and empathy in 2-3 short sentences.
- If the user asks a question, give a brief helpful answer.
- Use simple, clear language suitable for elderly users.
- Be warm but concise.{datetime_info}

Context signals (for guidance only, not to be mentioned):
- First message: {is_first_message}
- Greeting: {is_greeting}
- Question: {is_question}
- Wants list: {wants_list}
- Asking date/time: {asking_datetime}
- Confusion: {confusion}
- Emotional tone: {emotion}"""

        # Fill in the template with context
        adaptive_prompt = base_prompt.format(
            introduction=introduction,
            datetime_info=datetime_info,
            user_message=user_message,
            is_first_message="Yes" if context["is_first_message"] else "No",
            is_greeting="Yes" if context["is_greeting"] else "No",
            is_question="Yes" if context["is_question"] else "No",
            wants_list="Yes" if context["wants_list"] else "No",
            asking_datetime="Yes" if context["asking_datetime"] else "No",
            confusion="Yes" if context["confusion"] else "No",
            emotion=context["emotion"]
        )

        return adaptive_prompt

    def _extract_response(self, full_response: str, prompt: str) -> str:
        """Extract just the assistant's response from full generation."""
        # Look for the assistant's response between tags
        assistant_start = "<|start_header_id|>assistant<|end_header_id|>"

        # Find the last occurrence of assistant header (our generated response)
        if assistant_start in full_response:
            # Split by assistant headers
            parts = full_response.split(assistant_start)
            if len(parts) > 1:
                # Get the last part (our generated response)
                response = parts[-1]

                # Clean up special tokens
                response = response.replace("<|eot_id|>", "")
                response = response.replace("<|end_of_text|>", "")
                response = response.replace("<|begin_of_text|>", "")
                response = response.strip()

                return response

        # Fallback: try simple removal of prompt
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
            response = response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            return response

        # Last resort: return cleaned full response
        return full_response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

    def _clean_response(self, response: str) -> str:
        """
        Post-process response to remove training data leakage and enforce quality.
        Removes phone numbers, personal identity claims, URLs, and location-specific content.
        """
        if not response:
            return response

        # 1. Remove phone numbers (any format)
        response = re.sub(r'\b\d{1,2}[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', response)
        response = re.sub(r'\b1-\d{3}-\d{3}-\d{4}\b', '', response)

        # 2. Remove URLs and email addresses
        response = re.sub(r'https?://\S+', '', response)
        response = re.sub(r'www\.\S+', '', response)
        response = re.sub(r'\S+@\S+\.\S+', '', response)

        # 3. Remove therapist/counselor identity claims (training data leakage)
        identity_patterns = [
            r"I[' ]?a?m a (licensed|certified|professional|registered)[\w\s]*?(counselor|therapist|psychologist|social worker|clinician)[\w\s,]*?\.",
            r"I have (a |an )?(Master'?s?|Bachelor'?s?|PhD|doctorate)[\w\s]*?(degree|in)[\w\s,]*?\.",
            r"I[' ]?a?m licensed in[\w\s,]*?\.",
            r"I have over \d+[\w\s]*?experience[\w\s,]*?\.",
            r"I[' ]?a?m a (Texas|Georgia|Florida|California|New York)[\w\s]*?\.",
            r"I[' ]?a?m a member of[\w\s,]*?\.",
            r"I have worked with[\w\s,]*?(children|adolescents|adults|couples|families)[\w\s,]*?\.",
            r"I am skilled in[\w\s,]*?\.",
        ]
        for pattern in identity_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # 4. Remove references to specific organizations/helplines
        helpline_patterns = [
            r"(call|contact|reach out to)[\w\s]*?(NAMI|National Alliance|Suicide Prevention|Crisis|Helpline|Hotline)[\w\s,().\d-]*?\.",
            r"(NAMI|National Alliance on Mental)[\w\s,().\d-]*?\.",
        ]
        for pattern in helpline_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        # 5. Clean up extra whitespace from removals
        response = re.sub(r'\s{2,}', ' ', response)
        response = re.sub(r'\s+\.', '.', response)
        response = re.sub(r'\.\s*\.', '.', response)
        response = response.strip()

        # 6. Truncate at the last complete sentence (avoid cut-off text)
        if response and response[-1] not in '.!?':
            # Find the last sentence-ending punctuation
            last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_period > 20:  # Only truncate if we have enough text
                response = response[:last_period + 1]

        # 7. If response is empty after cleaning, provide a fallback
        if not response or len(response.strip()) < 5:
            response = "I hear you. How can I help you today?"

        return response.strip()

    def _check_safety(self, response: str, user_message: str) -> Optional[list]:
        """
        Basic safety check for concerning patterns.
        Returns list of warnings or None.
        """
        warnings = []

        # Check for memory-related concerns in user message
        memory_keywords = ["forget", "can't remember", "lost", "confused", "where am i"]
        if any(keyword in user_message.lower() for keyword in memory_keywords):
            warnings.append("memory_concern_detected")

        # Check for distress indicators
        distress_keywords = ["help", "scared", "alone", "emergency"]
        if any(keyword in user_message.lower() for keyword in distress_keywords):
            warnings.append("potential_distress")

        return warnings if warnings else None

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False

    def get_session_history(self, session_id: str) -> Optional[list]:
        """Get conversation history for a session."""
        return self.conversation_history.get(session_id)


# Global instance (singleton pattern)
_chatbot_instance: Optional[DementiaChatbot] = None


def get_chatbot() -> DementiaChatbot:
    """
    Get or create global chatbot instance.
    Singleton pattern to avoid loading model multiple times.
    """
    global _chatbot_instance

    if _chatbot_instance is None:
        logger.info("Initializing chatbot for the first time...")
        _chatbot_instance = DementiaChatbot()

    return _chatbot_instance
