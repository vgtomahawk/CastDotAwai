#!/usr/bin/env python
# coding: utf-8

import torch
import warnings
import pickle
from typing import Optional, Dict, Union, List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import Accelerator
import openai
from enum import Enum
import utils

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

class TranscriptWriter:
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
        You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

        We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

        You have won multiple podcast awards for your writing.
        
        Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

        Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

        Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

        Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

        Make sure the tangents speaker 2 provides are quite wild or interesting. 

        Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 

        It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

        ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
        DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
        DO NOT GIVE CHAPTER TITLES
        IT SHOULD STRICTLY BE THE DIALOGUES
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

    def read_input_file(self, filename: str) -> Optional[str]:
        """Read input file with multiple encoding attempts"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read file using {encoding} encoding.")
                return content
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.")
                return None
            except IOError:
                print(f"Error: Could not read file '{filename}'.")
                return None
        
        print(f"Error: Could not decode file with any common encoding.")
        return None

    def generate_transcript_llama(self, input_text: str) -> str:
        """Generate transcript using Llama model"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=8126,
            temperature=1,
        )
        
        return outputs[0]["generated_text"][-1]['content']

    def generate_transcript_openai(self, input_text: str) -> str:
        """Generate transcript using OpenAI API"""
        response = openai.chat.completions.create(
            model=self.model_config.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=1,
            max_tokens=4096  # Adjust based on model limits
        )
        return response.choices[0].message.content

    def process_file(self, input_path: str, output_path: str) -> bool:
        """Process input file to generate podcast transcript"""
        try:
            # Read input file
            print("\nReading input file...")
            input_text = self.read_input_file(input_path)
            if not input_text:
                return False

            # Generate transcript
            print("\nGenerating transcript...")
            if self.model_config.model_type == ModelType.LLAMA:
                transcript = self.generate_transcript_llama(input_text)
            else:
                transcript = self.generate_transcript_openai(input_text)

            # Save transcript
            print("\nSaving transcript...")
            with open(output_path, 'wb') as file:
                pickle.dump(transcript, file)
            print(f"Transcript saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return False

def main():
    # Load credentials
    creds = utils.load_all_creds()

    # Configure model
    model_config = ModelConfig(
        model_type=ModelType.OPENAI,  # or ModelType.LLAMA
        model_name="gpt-4o-mini",  # or "meta-llama/Llama-2-70b-chat-hf"
        openai_api_key=creds["OPENAI_API_KEY"],
        openai_org_id=creds["OPENAI_ORG_ID"],
        hf_token=creds["HF_TOKEN"]
    )
    
    writer = TranscriptWriter(model_config)
    
    # Process file
    input_txt = "sampleOutputs/Causes-of-the-War-of-1812_.txt"
    output_pkl = "sampleV1Transcripts/Causes-of-the-War-of-1812_.pkl"
    
    success = writer.process_file(input_txt, output_pkl)
    if success:
        print("Transcript generation completed successfully!")
    else:
        print("Transcript generation failed!")

if __name__ == "__main__":
    main()
