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

from src.features.conversational_ai.nlp.nlp_engine import NLPEngine
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
    return elements.tile(test_elements.shape[0], 1).eq(test_elements.unsqueeze(1)).sum(dim=0).bool().squeeze()

transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly_patched
if hasattr(transformers.generation.utils, "isin_mps_friendly"):
    transformers.generation.utils.isin_mps_friendly = isin_mps_friendly_patched
logger.info("🔧 Applied MPS compatibility patch for transformers library")
# ------------------------------------------


class DementiaChatbot:

    def __init__(
        self,
        base_model_name: str = None,
        lora_adapter_path: str = None,
        device: str = None
    ):
        loader = get_model_loader()
        llama_model_info = loader.get_model_info("llama_3_2_3b_dementia_care")

        if llama_model_info and llama_model_info.get("model_source") == "huggingface":
            registry_base_model = llama_model_info.get("base_model", "meta-llama/Llama-3.2-3B-Instruct")
            registry_adapter = llama_model_info.get("lora_adapter", "susadi/hale-empathy-3b")
        else:
            registry_base_model = "meta-llama/Llama-3.2-3B-Instruct"
            registry_adapter = "susadi/hale-empathy-3b"

        # Priority: passed args > env vars > registry > hardcoded defaults
        self.base_model_name = base_model_name or os.getenv("LLAMA_BASE_MODEL", registry_base_model)
        self.lora_adapter_path = lora_adapter_path or os.getenv("LLAMA_LORA_ADAPTER", registry_adapter)

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
        self.conversation_history: Dict[str, list] = {}

        try:
            logger.info("Initializing NLP Engine for intelligent prompting...")
            self.nlp_engine = NLPEngine(
                enable_semantic=False,
                enable_emotion=True,
                enable_linguistic=True,
                device="cpu"
            )
            logger.info("[SUCCESS] NLP Engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NLP Engine: {e}. Using simple prompting.")
            self.nlp_engine = None

        self._load_model()

    def _load_model(self):
        try:
            hf_token = os.getenv("LLAMA_HF_TOKEN")
            if hf_token:
                logger.info("HuggingFace token found - can access private models")

            logger.info(f"Loading base model: {self.base_model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                token=hf_token
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                token=hf_token
            )

            logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            try:
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_adapter_path,
                    token=hf_token
                )
                logger.info("LoRA adapter loaded successfully")
            except Exception as adapter_error:
                logger.warning(f"Could not load LoRA adapter: {adapter_error}")
                logger.warning("Using base model without LoRA")
                self.model = base_model

            # Only needed when not using device_map="auto" (cuda case handles placement itself)
            if self.device != "cuda":
                self.model = self.model.to(self.device)

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
        try:
            if session_id is None:
                session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []

            # Cap tokens based on message type to keep responses concise
            msg_lower = user_message.lower().strip()
            msg_word_count = len(msg_lower.split())
            if msg_word_count <= 3:
                max_tokens = min(max_tokens, 60)
            elif '?' in user_message or any(msg_lower.startswith(q) for q in ['what', 'how', 'can', 'tell', 'explain', 'describe']):
                max_tokens = min(max_tokens, 100)
            else:
                max_tokens = min(max_tokens, 80)

            if use_history and self.conversation_history[session_id]:
                recent_history = self.conversation_history[session_id][-10:]
                context_messages = recent_history + [user_message]
                prompt = self._build_prompt(context_messages, session_id)
            else:
                prompt = self._build_prompt([user_message], session_id)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)

            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id

            # generate() can return list/tuple token IDs depending on config
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

            if isinstance(outputs, tuple):
                generated_ids = outputs[0][0]
            elif hasattr(outputs, 'sequences'):
                generated_ids = outputs.sequences[0]
            else:
                generated_ids = outputs[0]

            full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            response_text = self._extract_response(full_response, prompt)

            if not response_text or len(response_text.strip()) == 0:
                full_response_clean = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if full_response_clean.startswith(prompt):
                    response_text = full_response_clean[len(prompt):].strip()
                else:
                    response_text = full_response_clean.strip()

            response_text = self._clean_response(response_text)

            self.conversation_history[session_id].append(user_message)
            self.conversation_history[session_id].append(response_text)

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

        if session_id not in self.conversation_history or len(self.conversation_history[session_id]) == 0:
            context["is_first_message"] = True

        greeting_patterns = [
            r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
            r'^(hi|hello|hey)[\s!,.]',
        ]
        if any(re.search(pattern, msg_lower) for pattern in greeting_patterns):
            context["is_greeting"] = True

        if '?' in user_message or any(msg_lower.startswith(q) for q in
            ['what', 'where', 'when', 'who', 'why', 'how', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']):
            context["is_question"] = True

        list_keywords = ['list', 'steps', 'how to', 'ways to', 'options', 'things', 'items', 'bullet']
        if any(keyword in msg_lower for keyword in list_keywords):
            context["wants_list"] = True

        datetime_keywords = [
            'what time', 'what day', 'what date', 'what is the time', 'what is the date',
            'what\'s the time', 'what\'s the date', 'time is it', 'date is it',
            'today\'s date', 'current time', 'current date', 'what year', 'tell me the time',
            'tell me the date'
        ]
        if any(keyword in msg_lower for keyword in datetime_keywords):
            context["asking_datetime"] = True
            now = datetime.now()
            context["current_datetime"] = now.strftime("%A, %B %d, %Y at %I:%M %p")

        confusion_keywords = [
            "forget", "can't remember", "don't remember", "confused", "lost",
            "where am i", "what day", "don't know", "not sure", "unclear"
        ]
        if any(keyword in msg_lower for keyword in confusion_keywords):
            context["confusion"] = True

        if self.nlp_engine:
            try:
                analysis = self.nlp_engine.analyze(user_message)
                if analysis.emotion_analysis:
                    context["emotion"] = analysis.emotion_analysis.dominant_emotion
                    if analysis.emotion_analysis.dominant_emotion in ['fear', 'sadness', 'anger']:
                        context["emotion"] = analysis.emotion_analysis.dominant_emotion
            except Exception as e:
                logger.debug(f"NLP emotion analysis failed: {e}")

        return context

    def _build_prompt(self, messages: list, session_id: str = None) -> str:
        user_message = messages[-1] if messages else ""
        context = self._analyze_user_context(user_message, session_id or "default")
        system_prompt = self._build_adaptive_system_prompt(context, user_message)

        if len(messages) == 1:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{messages[0]}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

            for i, msg in enumerate(messages):
                if i % 2 == 0:
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg}<|eot_id|>"
                else:
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg}<|eot_id|>"

            if len(messages) % 2 == 1:
                prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def _build_adaptive_system_prompt(self, context: Dict[str, Any], user_message: str) -> str:
        introduction = ""
        if context["is_greeting"] and context["is_first_message"]:
            introduction = "\n- This is the first message. Introduce yourself as 'Hale', a helpful assistant here to support them."

        datetime_info = ""
        if context["asking_datetime"] and context["current_datetime"]:
            datetime_info = f"\n\nCURRENT DATE AND TIME: {context['current_datetime']}\n- The user is asking about the date or time. Provide this information clearly."

        confusion_guidance = ""
        if context["confusion"]:
            confusion_guidance = (
                "\n- The user seems confused or disoriented. Acknowledge their feelings warmly,"
                " stay grounded in the present moment, and gently suggest they speak with a"
                " family member or caregiver who knows them well."
            )

        base_prompt = """You are Hale, a friendly AI care companion for elderly individuals.

CRITICAL RULES:{introduction}
- Keep ALL responses SHORT: 2-3 sentences maximum.
- For greetings, reply in ONE sentence only.
- NEVER start your response with "I am Hale" or any self-introduction. Do NOT repeat your name or identity in every message. Only introduce yourself if this is the very first greeting message (is_first_message=Yes AND is_greeting=Yes), and even then keep it to one short sentence.
- You are an AI assistant, NOT a human. Never claim to be a real person, therapist, or counselor.
- NEVER mention personal details like your education, location, license, or experience.
- NEVER share phone numbers, helpline numbers, or website URLs.
- NEVER claim to be from a specific city, state, or country. You can answer questions about places if the user asks.
- If the user is emotional, respond with warmth and empathy in 2-3 short sentences.
- If the user asks a question, give a brief helpful answer.
- Use simple, clear language suitable for elderly users.
- Be warm but concise.

MEMORY RULES (always apply):
- You have NO knowledge of this person's personal history, medical records, or any previous conversations outside of this current session.
- Only reference information the user has told you during THIS conversation.
- NEVER say things like "I remember you said", "last time we spoke", "as we discussed before", "I know you have", or any phrase implying prior knowledge you don't have.
- NEVER claim to know the user's name, conditions, family, or past events unless they told you in this session.
- If asked about something you don't know, say honestly: "I don't have that information — could you tell me more?"{confusion_guidance}{datetime_info}

Context signals (for guidance only, not to be mentioned):
- First message: {is_first_message}
- Greeting: {is_greeting}
- Question: {is_question}
- Wants list: {wants_list}
- Asking date/time: {asking_datetime}
- Confusion: {confusion}
- Emotional tone: {emotion}"""

        return base_prompt.format(
            introduction=introduction,
            datetime_info=datetime_info,
            confusion_guidance=confusion_guidance,
            user_message=user_message,
            is_first_message="Yes" if context["is_first_message"] else "No",
            is_greeting="Yes" if context["is_greeting"] else "No",
            is_question="Yes" if context["is_question"] else "No",
            wants_list="Yes" if context["wants_list"] else "No",
            asking_datetime="Yes" if context["asking_datetime"] else "No",
            confusion="Yes" if context["confusion"] else "No",
            emotion=context["emotion"]
        )

    def _extract_response(self, full_response: str, prompt: str) -> str:
        assistant_start = "<|start_header_id|>assistant<|end_header_id|>"

        if assistant_start in full_response:
            parts = full_response.split(assistant_start)
            if len(parts) > 1:
                response = parts[-1]
                response = response.replace("<|eot_id|>", "")
                response = response.replace("<|end_of_text|>", "")
                response = response.replace("<|begin_of_text|>", "")
                return response.strip()

        # Fallback: strip prompt prefix
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
            return response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        return full_response.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

    def _clean_response(self, response: str) -> str:
        if not response:
            return response

        # Strip self-introduction the fine-tuned model prepends on every response
        response = re.sub(
            r'^I\s+am\s+Hale[,.]?\s+a\s+friendly\s+AI\s+(care\s+)?companion\s+for\s+elderly\s+individuals[,.]?\s*',
            '', response, flags=re.IGNORECASE
        )
        response = response.strip()

        # Phone numbers
        response = re.sub(r'\b\d{1,2}[-.]?\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', response)
        response = re.sub(r'\b1-\d{3}-\d{3}-\d{4}\b', '', response)

        # URLs and emails
        response = re.sub(r'https?://\S+', '', response)
        response = re.sub(r'www\.\S+', '', response)
        response = re.sub(r'\S+@\S+\.\S+', '', response)

        # Therapist/counselor identity claims (training data leakage)
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

        # Helpline/organization references
        helpline_patterns = [
            r"(call|contact|reach out to)[\w\s]*?(NAMI|National Alliance|Suicide Prevention|Crisis|Helpline|Hotline)[\w\s,().\d-]*?\.",
            r"(NAMI|National Alliance on Mental)[\w\s,().\d-]*?\.",
        ]
        for pattern in helpline_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        response = re.sub(r'\s{2,}', ' ', response)
        response = re.sub(r'\s+\.', '.', response)
        response = re.sub(r'\.\s*\.', '.', response)
        response = response.strip()

        if response and response[-1] not in '.!?':
            last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_period > 20:
                response = response[:last_period + 1]

        # False memory claims — phrases implying the bot remembers the user's past
        false_memory_patterns = [
            r"let'?s\s+try\s+to\s+remember\b[^.!?]*[.!?]?",
            r"let\s+me\s+help\s+you\s+remember\b[^.!?]*[.!?]?",
            r"i'?ll?\s+help\s+you\s+recall\b[^.!?]*[.!?]?",
            r"do\s+you\s+remember\s+when\b[^.!?]*[.!?]?",
            r"as\s+we\s+(discussed|talked\s+about)\s+before\b[^.!?]*[.!?]?",
            r"last\s+time\s+you\s+told\s+me\b[^.!?]*[.!?]?",
            r"i\s+remember\s+you\s+(said|told|mentioned)\b[^.!?]*[.!?]?",
            r"we\s+(talked|spoke|chatted)\s+about\s+this\s+before\b[^.!?]*[.!?]?",
            r"i\s+know\s+you\s+(have|had|are|were|suffer|suffer from)\b[^.!?]*[.!?]?",
            r"based\s+on\s+(our\s+previous|your\s+history|what\s+you.ve\s+shared\s+before)\b[^.!?]*[.!?]?",
            r"i\s+recall\s+(you|that\s+you)\b[^.!?]*[.!?]?",
            r"from\s+(our\s+last|our\s+previous|your\s+last)\s+(session|conversation|chat)\b[^.!?]*[.!?]?",
            r"as\s+i\s+(mentioned|said)\s+before\b[^.!?]*[.!?]?",
            r"you\s+(previously|already)\s+told\s+me\b[^.!?]*[.!?]?",
        ]
        for pattern in false_memory_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)

        response = re.sub(r'\s{2,}', ' ', response).strip()

        if not response or len(response.strip()) < 5:
            response = "I hear you. How can I help you today?"

        return response.strip()

    def _check_safety(self, response: str, user_message: str) -> Optional[list]:
        warnings = []

        memory_keywords = ["forget", "can't remember", "lost", "confused", "where am i"]
        if any(keyword in user_message.lower() for keyword in memory_keywords):
            warnings.append("memory_concern_detected")

        distress_keywords = ["help", "scared", "alone", "emergency"]
        if any(keyword in user_message.lower() for keyword in distress_keywords):
            warnings.append("potential_distress")

        return warnings if warnings else None

    def clear_session(self, session_id: str) -> bool:
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False

    def get_session_history(self, session_id: str) -> Optional[list]:
        return self.conversation_history.get(session_id)


_chatbot_instance: Optional[DementiaChatbot] = None


def get_chatbot() -> DementiaChatbot:
    global _chatbot_instance

    if _chatbot_instance is None:
        logger.info("Initializing chatbot for the first time...")
        _chatbot_instance = DementiaChatbot()

    return _chatbot_instance
