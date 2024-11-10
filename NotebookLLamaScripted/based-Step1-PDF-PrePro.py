#!/usr/bin/env python
# coding: utf-8

import os
import torch
import warnings
import PyPDF2
from typing import Optional, Dict, Union, List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
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

class PDFPreProcessor:
    def __init__(
        self,
        model_config: ModelConfig,
        chunk_size: int = 1000,
        max_chars: int = 100000
    ):
        self.model_config = model_config
        self.chunk_size = chunk_size
        self.max_chars = max_chars
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_config.model_type == ModelType.LLAMA:
            if not model_config.hf_token:
                raise ValueError("HuggingFace token required for Llama model")
            self._setup_llama()
        else:
            self._setup_openai()
            
        self.system_prompt = """
        You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

        The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

        Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive.

        Please be smart with what you remove and be creative ok?

        Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

        Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

        PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION ETC LIKES

        ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
        Here is the text:
        """

    def _setup_llama(self):
        """Setup Llama model with HuggingFace token"""
        accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            device_map=self.device,
            token=self.model_config.hf_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            use_safetensors=True,
            token=self.model_config.hf_token
        )
        self.model, self.tokenizer = accelerator.prepare(self.model, self.tokenizer)

    def _setup_openai(self):
        """Setup OpenAI client"""
        if not self.model_config.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.model_config.openai_api_key
        if self.model_config.openai_org_id:
            openai.organization = self.model_config.openai_org_id

    def validate_pdf(self, file_path: str) -> bool:
        """Validate if file exists and is PDF"""
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return False
        if not file_path.lower().endswith('.pdf'):
            print("Error: File is not a PDF")
            return False
        return True

    def get_pdf_metadata(self, file_path: str) -> Optional[Dict]:
        """Extract PDF metadata"""
        if not self.validate_pdf(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata
                }
                return metadata
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            return None

    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """Extract text content from PDF"""
        if not self.validate_pdf(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                
                extracted_text = []
                total_chars = 0
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if total_chars + len(text) > self.max_chars:
                        remaining_chars = self.max_chars - total_chars
                        extracted_text.append(text[:remaining_chars])
                        print(f"Reached {self.max_chars} character limit at page {page_num + 1}")
                        break
                    
                    extracted_text.append(text)
                    total_chars += len(text)
                    print(f"Processed page {page_num + 1}/{num_pages}")
                
                final_text = '\n'.join(extracted_text)
                print(f"\nExtraction complete! Total characters: {len(final_text)}")
                return final_text
                
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return None

    def create_word_bounded_chunks(self, text: str) -> List[str]:
        """Split text into word-bounded chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def process_chunk_llama(self, text_chunk: str) -> str:
        """Process text chunk using Llama model"""
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text_chunk},
        ]
        
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=512
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def process_chunk_openai(self, text_chunk: str) -> str:
        """Process text chunk using OpenAI API"""
        response = openai.chat.completions.create(
            model=self.model_config.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text_chunk}
            ],
            temperature=1e-6,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()

    def process_text(self, text: str) -> str:
        """Process entire text by chunks"""
        chunks = self.create_word_bounded_chunks(text)
        processed_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            
            if self.model_config.model_type == ModelType.LLAMA:
                processed_chunk = self.process_chunk_llama(chunk)
            else:
                processed_chunk = self.process_chunk_openai(chunk)
                
            processed_chunks.append(processed_chunk)
            
        return "\n".join(processed_chunks)

    def process_pdf(self, input_path: str, output_path: str) -> bool:
        """Process PDF file end-to-end"""
        try:
            # Extract metadata
            metadata = self.get_pdf_metadata(input_path)
            if metadata:
                print("\nPDF Metadata:")
                print(f"Number of pages: {metadata['num_pages']}")
                print("Document info:")
                for key, value in metadata['metadata'].items():
                    print(f"{key}: {value}")

            # Extract text
            print("\nExtracting text...")
            extracted_text = self.extract_text_from_pdf(input_path)
            if not extracted_text:
                return False

            # Process text
            print("\nProcessing text...")
            processed_text = self.process_text(extracted_text)

            # Save processed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            print(f"\nProcessed text saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False

def main():
    
    creds = utils.load_all_creds()

    # Example usage
    model_config = ModelConfig(
        model_type=ModelType.OPENAI,  # or ModelType.LLAMA
        model_name="gpt-4o-mini",  # or "meta-llama/Llama-2-7b-chat-hf"
        openai_api_key=creds["OPENAI_API_KEY"],  # Required for OpenAI
        openai_org_id=creds["OPENAI_ORG_ID"],  # Optional for OpenAI
        hf_token=creds["HF_TOKEN"]  # Required for Llama
    )
    
    processor = PDFPreProcessor(model_config)
    
    # Process a PDF file
    input_pdf_path = "sampleInputs/Causes-of-the-War-of-1812_.pdf"
    output_txt_path = "sampleOutputs/Causes-of-the-War-of-1812_.txt"
    
    success = processor.process_pdf(input_pdf_path, output_txt_path)
    if success:
        print("PDF processing completed successfully!")
    else:
        print("PDF processing failed!")

if __name__ == "__main__":
    main()
