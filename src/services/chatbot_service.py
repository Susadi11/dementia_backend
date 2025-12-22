"""
Chatbot Service using Fine-tuned LLaMA 3.2 1B with LoRA
Trained on DailyDialog dataset for dementia care conversations
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class DementiaChatbot:
    """
    Chatbot for dementia care using fine-tuned LLaMA 3.2 1B model.
    Model is trained on DailyDialog dataset for natural conversations.
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        lora_adapter_path: str = "models/llama_32_1B_dailydialog_final",
        device: str = None
    ):
        """
        Initialize the chatbot with fine-tuned model.

        Args:
            base_model_name: HuggingFace model ID for base LLaMA model
            lora_adapter_path: Path to your trained LoRA adapter
            device: Device to run on ('cuda', 'mps', or 'cpu')
        """
        self.base_model_name = base_model_name
        self.lora_adapter_path = Path(lora_adapter_path)

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

        self._load_model()

    def _load_model(self):
        """Load base model and apply LoRA adapter."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # Load and apply LoRA adapter
            if self.lora_adapter_path.exists():
                logger.info(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    str(self.lora_adapter_path)
                )
                logger.info("LoRA adapter loaded successfully")
            else:
                logger.warning(f"LoRA adapter not found at {self.lora_adapter_path}, using base model")
                self.model = base_model

            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)

            # Set to eval mode
            self.model.eval()

            logger.info("✅ Chatbot model loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
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

            # Build conversation context
            if use_history and self.conversation_history[session_id]:
                # Include recent conversation history (last 5 exchanges)
                recent_history = self.conversation_history[session_id][-10:]  # Last 5 Q&A pairs
                context_messages = recent_history + [user_message]
                prompt = self._build_prompt(context_messages)
            else:
                prompt = self._build_prompt([user_message])

            # Generate response
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response (keep special tokens to properly extract)
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract just the assistant's response
            response_text = self._extract_response(full_response, prompt)

            # If extraction failed, try with skip_special_tokens
            if not response_text or len(response_text.strip()) == 0:
                full_response_clean = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Try simple extraction: everything after the prompt
                if full_response_clean.startswith(prompt):
                    response_text = full_response_clean[len(prompt):].strip()
                else:
                    # Fallback: just use the decoded output
                    response_text = full_response_clean.strip()

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
                    "adapter": str(self.lora_adapter_path.name),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "conversation_length": len(self.conversation_history[session_id])
                },
                "safety_warnings": safety_warnings if safety_warnings else None
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _build_prompt(self, messages: list) -> str:
        """
        Build chat prompt from messages.
        Uses LLaMA's instruction format.
        """
        if len(messages) == 1:
            # Single message - simple format
            system_prompt = (
                "You are a helpful, empathetic assistant for elderly individuals, "
                "especially those with memory concerns. Be patient, clear, and supportive."
            )
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{messages[0]}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Multi-turn conversation
            system_prompt = (
                "You are a helpful, empathetic assistant for elderly individuals, "
                "especially those with memory concerns. Be patient, clear, and supportive."
            )
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
