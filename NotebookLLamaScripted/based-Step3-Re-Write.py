#!/usr/bin/env python
# coding: utf-8

import torch
import warnings
import pickle
from typing import Optional, Dict, Union, List, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
import openai
from enum import Enum
import utils
import ast

warnings.filterwarnings('ignore')

class ModelType(Enum):
    OPENAI = "openai"
    LLAMA = "llama"

@dataclass
class ModelConfig:
    model_type: ModelType
    model_name: str
    hf_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None

class TranscriptRewriter:
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_config.model_type == ModelType.LLAMA:
            if not model_config.hf_token:
                raise ValueError("HuggingFace token required for Llama model")
            self._setup_llama()
        else:
            self._setup_openai()
            
        self.system_prompt = """
        You are an international oscar winning screenwriter

        You have been working with multiple award winning podcasters.

        Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

        Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

        Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

        Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

        Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

        Make sure the tangents speaker 2 provides are quite wild or interesting. 

        Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

        REMEMBER THIS WITH YOUR HEART
        The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

        For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

        It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

        Please re-write to make it as characteristic as possible

        START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

        STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

        IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

        Example of response:
        [
            ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
            ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
            ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
            ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
        ]
        """

    def _setup_llama(self):
        """Setup Llama model with HuggingFace"""
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_config.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=self.model_config.hf_token
        )

    def _setup_openai(self):
        """Setup OpenAI client"""
        if not self.model_config.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.model_config.openai_api_key
        if self.model_config.openai_org_id:
            openai.organization = self.model_config.openai_org_id

    def load_transcript(self, filename: str) -> Optional[str]:
        """Load transcript from pickle file"""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            print(f"Error loading transcript: {str(e)}")
            return None

    def parse_response(self, response: str) -> List[Tuple[str, str]]:
        """Parse the model response into a list of tuples"""
        try:
            # Clean the response string and evaluate it as a Python literal
            cleaned_response = response.strip()
            if not cleaned_response.startswith('['):
                # Try to find the first occurrence of '[' and strip everything before it
                start_idx = cleaned_response.find('[')
                if start_idx != -1:
                    cleaned_response = cleaned_response[start_idx:]
            
            return ast.literal_eval(cleaned_response)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return []

    def generate_rewrite_llama(self, input_text: str) -> List[Tuple[str, str]]:
        """Generate rewritten transcript using Llama model"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )
        
        return self.parse_response(outputs[0]["generated_text"][-1]['content'])

    def generate_rewrite_openai(self, input_text: str) -> List[Tuple[str, str]]:
        """Generate rewritten transcript using OpenAI API"""
        response = openai.chat.completions.create(
            model=self.model_config.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=1,
            max_tokens=4096  # Adjust based on model limits
        )
        return self.parse_response(response.choices[0].message.content)

    def process_transcript(self, input_path: str, output_path: str) -> bool:
        """Process input transcript to generate rewritten version"""
        try:
            # Load input transcript
            print("\nLoading input transcript...")
            input_text = self.load_transcript(input_path)
            if not input_text:
                return False

            # Generate rewrite
            print("\nGenerating rewritten transcript...")
            if self.model_config.model_type == ModelType.LLAMA:
                rewritten = self.generate_rewrite_llama(input_text)
            else:
                rewritten = self.generate_rewrite_openai(input_text)

            if not rewritten:
                print("Error: Failed to generate rewritten transcript")
                return False

            # Save rewritten transcript
            print("\nSaving rewritten transcript...")
            with open(output_path, 'wb') as file:
                pickle.dump(rewritten, file)
            print(f"Rewritten transcript saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing transcript: {str(e)}")
            return False

def main():
    # Load credentials
    creds = utils.load_all_creds()

    # Configure model
    model_config = ModelConfig(
        model_type=ModelType.OPENAI,  # or ModelType.LLAMA
        model_name="gpt-4o-mini",  # or "meta-llama/Llama-2-8b-chat-hf"
        openai_api_key=creds["OPENAI_API_KEY"],
        openai_org_id=creds["OPENAI_ORG_ID"],
        hf_token=creds["HF_TOKEN"]
    )
    
    rewriter = TranscriptRewriter(model_config)
    
    # Process transcript
    input_pkl = "sampleV1Transcripts/Causes-of-the-War-of-1812_.pkl"
    output_pkl = "sampleV2Transcripts/Causes-of-the-War-of-1812_.pkl"
    
    success = rewriter.process_transcript(input_pkl, output_pkl)
    if success:
        print("Transcript rewriting completed successfully!")
    else:
        print("Transcript rewriting failed!")

if __name__ == "__main__":
    main()
