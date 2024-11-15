#!/usr/bin/env python
# coding: utf-8

import pickle
import ast
from typing import Optional, List, Tuple, BinaryIO
from dataclasses import dataclass
from elevenlabs import play, save
from elevenlabs.client import ElevenLabs
from tqdm import tqdm
import io
import wave
import numpy as np
from pydub import AudioSegment
import utils
import openai

@dataclass
class TTSConfig:
    api_key: str
    model_name: str = "eleven_multilingual_v2"
    speaker1_voice: str = "Laura"
    speaker2_voice: str = "Brian"
    openai_api_key: Optional[str] = None
    openai_org_id: Optional[str] = None

class PodcastGenerator:
    def __init__(self, config: TTSConfig):
        """Initialize the podcast generator with ElevenLabs configuration"""
        self.config = config
        self.client = ElevenLabs(api_key=config.api_key)
        self.final_audio = AudioSegment.empty()
        self._setup_openai()

    def _setup_openai(self):
        """Setup OpenAI client"""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key required")
        
        openai.api_key = self.config.openai_api_key
        if self.config.openai_org_id:
            openai.organization = self.config.openai_org_id

    def generate_audio(self, text: str, voice: str) -> bytes:
        """Generate audio using ElevenLabs API"""
        try:
            audio = self.client.generate(
                text=text,
                voice=voice,
                model=self.config.model_name
            )
            return audio
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            return None

    def audio_bytes_to_segment(self, audio_bytes: bytes) -> Optional[AudioSegment]:
        """Convert audio bytes to AudioSegment"""
        try:
            # Convert bytes to AudioSegment using io.BytesIO
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="mp3"  # ElevenLabs returns MP3
            )
            return audio_segment
        except Exception as e:
            print(f"Error converting audio bytes to segment: {str(e)}")
            return None

    def load_transcript(self, filepath: str) -> List[Tuple[str, str]]:
        """Load and parse the transcript pickle file"""
        try:
            with open(filepath, 'rb') as file:
                transcript_text = pickle.load(file)
            return transcript_text
        except Exception as e:
            print(f"Error loading transcript: {str(e)}")
            return []

    def process_transcript(self, transcript: List[Tuple[str, str]]) -> bool:
        """Process the entire transcript and generate audio"""
        try:
            for speaker, text in tqdm(transcript, desc="Generating podcast segments"):
                # Select voice based on speaker
                voice = self.config.speaker1_voice if speaker == "Speaker 1" else self.config.speaker2_voice
                
                # Generate audio for segment
                audio_bytes = self.generate_audio(text, voice)
                if audio_bytes is None:
                    print(f"Failed to generate audio for segment: {text[:50]}...")
                    continue
                
                # Convert to AudioSegment
                audio_segment = self.audio_bytes_to_segment(audio_bytes)
                if audio_segment is None:
                    continue
                
                # Add to final audio with a small pause between segments
                if len(self.final_audio) > 0:
                    # Add 0.5 second silence between segments
                    self.final_audio += AudioSegment.silent(duration=500)
                self.final_audio += audio_segment
                
            return True
        except Exception as e:
            print(f"Error processing transcript: {str(e)}")
            return False

    def save_podcast(self, output_path: str) -> bool:
        """Save the final podcast as an MP3 file"""
        try:
            self.final_audio.export(
                output_path,
                format="mp3",
                bitrate="192k",
                parameters=["-q:a", "0"]
            )
            return True
        except Exception as e:
            print(f"Error saving podcast: {str(e)}")
            return False

    def test_voices(self, test_text: str = "Hello! This is a test of the text to speech system."):
        """Test both voices with a sample text"""
        print("\nTesting Speaker 1 (Laura)...")
        audio1 = self.generate_audio(test_text, self.config.speaker1_voice)
        if audio1:
            play(audio1)
        
        print("\nTesting Speaker 2 (Brian)...")
        audio2 = self.generate_audio(test_text, self.config.speaker2_voice)
        if audio2:
            play(audio2)


    def play_podcast(self, file_path: str, notebook_mode: bool = False) -> bool:
        """Play the generated podcast using ElevenLabs player with fallbacks.
    
        Args:
            file_path: Path to the generated MP3 file
                notebook_mode: Whether to play in notebook (using IPython.display) or regular mode
        """
        try:
            # Try ElevenLabs play first
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                try:
                    from elevenlabs import play
                    play(audio_data)
                    return True
                except Exception as e:
                    print(f"ElevenLabs playback failed: {str(e)}, trying fallback methods...")

            # Fallback to notebook display if in notebook mode
            if notebook_mode:
                try:
                    from IPython.display import Audio, display
                    display(Audio(file_path))
                    return True
                except ImportError:
                    print("IPython not available, falling back to regular audio playback")
        
            # Final fallback to pydub
            audio = AudioSegment.from_mp3(file_path)
            from pydub.playback import play as pydub_play
            pydub_play(audio)
            return True
        
        except Exception as e:
            print(f"Error playing podcast: {str(e)}")
            return False

    def play_podcast_online(self, transcript: List[Tuple[str, str]]) -> bool:
        """Play the podcast segments directly as they're generated, without saving"""
        try:
            for speaker, text in tqdm(transcript, desc="Playing podcast segments"):
                # Select voice based on speaker
                voice = self.config.speaker1_voice if speaker == "Speaker 1" else self.config.speaker2_voice
            
                # Generate and play audio for segment
                audio = self.generate_audio(text, voice)
                if audio is None:
                    print(f"Failed to generate audio for segment: {text[:50]}...")
                    continue
                
                # Play the segment
                play(audio)
            
            return True
        except Exception as e:
            print(f"Error playing podcast: {str(e)}")
            return False


    def update_transcript_with_evidence(self, previous_turns: List[Tuple[str, str]], remaining_turns: List[Tuple[str, str]], evidence_file: str) -> List[Tuple[str, str]]:
        """Update remaining transcript turns based on new evidence"""
        try:
            # Read the evidence
            with open(evidence_file, 'r') as f:
                new_evidence = f.read()

            # Create prompt for GPT-4
            context = "\n".join([f"{speaker}: {text}" for speaker, text in previous_turns])
            remaining = "\n".join([f"{speaker}: {text}" for speaker, text in remaining_turns])
        
            prompt = f"""Given a podcast discussion so far:
                    {context}
                    And new evidence provided by a listener:
                    {new_evidence}
                    Please update the remaining part of the discussion:
                    {remaining}
                    The first speaker should acknowledge the listener's contribution with "Oh, our cool listener barged in with..." and then naturally incorporate the new perspective. Keep the same speaking style and engagement level, but adjust the content to reflect this new information.
                    Return strictly in this format: [("Speaker 1", "text"), ("Speaker 2", "text"), ...]"""

            # Get updated transcript from OpenAI
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=1e-6
            )
        
            # Parse response and return updated turns
            updated_turns = ast.literal_eval(response.choices[0].message.content)
            return updated_turns
    
        except Exception as e:
            print(f"Error updating transcript: {str(e)}")
            return remaining_turns  # Return original if update fails

    def play_podcast_online_with_bargein(self, transcript: List[Tuple[str, str]], bargein_at_turn: int, evidence_file: str) -> bool:
        """Play podcast with barge-in capability at specified turn"""
        try:
            # Split transcript into before and after bargein point
            previous_turns = transcript[:bargein_at_turn]
            remaining_turns = transcript[bargein_at_turn:]
        
            # Play up to the barge-in point
            for speaker, text in tqdm(previous_turns, desc="Playing podcast segments (pre-bargein)"):
                voice = self.config.speaker1_voice if speaker == "Speaker 1" else self.config.speaker2_voice
                audio = self.generate_audio(text, voice)
                if audio:
                    play(audio)
        
            print("\n🎤 Listener barge-in with new evidence! Updating transcript...\n")
        
            # Update remaining transcript with new evidence
            updated_turns = self.update_transcript_with_evidence(
                previous_turns, remaining_turns, evidence_file
            )
        
            # Play updated remainder
            for speaker, text in tqdm(updated_turns, desc="Playing podcast segments (post-bargein)"):
                voice = self.config.speaker1_voice if speaker == "Speaker 1" else self.config.speaker2_voice
                audio = self.generate_audio(text, voice)
                if audio:
                    play(audio)
                
            return True
        
        except Exception as e:
            print(f"Error in barge-in podcast: {str(e)}")
            return False



def main():
    # Load credentials
    creds = utils.load_all_creds()
    
    # Configure TTS
    config = TTSConfig(
        api_key=creds["ELEVEN_API_KEY"],
        model_name="eleven_multilingual_v2",
        speaker1_voice="Laura",
        speaker2_voice="Brian",
        openai_api_key=creds["OPENAI_API_KEY"],
        openai_org_id=creds["OPENAI_ORG_ID"],
    )
    
    # Initialize generator
    generator = PodcastGenerator(config)
    
    # Optional: Test voices
    print("Testing voices...")
    generator.test_voices()
    
    # Process podcast
    input_pkl = "samplev2Transcripts/Causes-of-the-War-of-1812_.pkl"
    output_mp3 = "samplePodcasts/Causes-of-the-War-of-1812_.mp3"
    
    print("\nLoading transcript...")
    transcript = generator.load_transcript(input_pkl)
    if not transcript:
        print("Failed to load transcript!")
        return
    
 
    
    print("Playing podcast with barge-in...")
    generator.play_podcast_online_with_bargein(transcript, bargein_at_turn=6, evidence_file="sampleBargeInEvidences/1812British.txt")

    print("Playing the podcast directly in online fashion first")
    generator.play_podcast_online(transcript)

    save_for_offline_use_and_check_play = False
    if save_for_offline_use_and_check_play:
        print("\nGenerating podcast for offline use")
        if generator.process_transcript(transcript):
            print("\nSaving podcast for offline use")
            if generator.save_podcast(output_mp3):
                print(f"\nPodcast for offline use successfully generated and saved to: {output_mp3}")
            else:
                print("\nFailed to save podcast for offline use!")
        else:
            print("\nFailed to generate podcast for offline use!")

        print("Playing Out Podcast For Offline Use Loading it After saving")
        generator.play_podcast("samplePodcasts/Causes-of-the-War-of-1812_.mp3") 

if __name__ == "__main__":
    main()
