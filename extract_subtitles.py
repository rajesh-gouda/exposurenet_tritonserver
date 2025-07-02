#!/usr/bin/env python3
"""
Subtitle extraction script for batch processing videos.
Extracts subtitles from all MP4 files in the videos directory and saves to JSON.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import openai

# Local imports
from config import Config
from cache import CacheManager
import logging

logger = logging.getLogger(__name__)

# Check OpenAI version for compatibility
try:
    from openai import OpenAI

    OPENAI_V1 = True
except ImportError:
    OPENAI_V1 = False


class SubtitleExtractor:
    """Handles batch subtitle extraction from videos."""

    def __init__(self):
        """Initialize the OpenAI client."""
        Config.setup_openai()

        # Initialize OpenAI client based on version
        if OPENAI_V1:
            self.client = OpenAI()
        else:
            self.client = None  # Use legacy API calls

    def _call_openai_chat(
        self, messages: List[Dict], max_tokens: int = 2000, temperature: float = 0.3
    ) -> str:
        """
        Unified OpenAI API call that works with both v0.x and v1.x versions.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation

        Returns:
            str: Response content
        """
        try:
            if OPENAI_V1:
                # Use new API (v1.x)
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            else:
                # Use legacy API (v0.x)
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")

    def transcribe_video(
        self, video_path: str
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """
        Transcribe video and return subtitle segments.

        Args:
            video_path (str): Path to the video file

        Returns:
            Optional[List[Dict]]: List of transcription segments or None if transcription fails
        """
        user_id = "batch_extraction"
        audio_path = CacheManager.get_file_path(user_id, video_path, "audio.mp3")
        transcription_path = CacheManager.get_file_path(
            user_id, video_path, "transcript.json"
        )

        # Check for cached transcription
        cached_transcript = CacheManager.load_cache(transcription_path)
        if cached_transcript:
            print(f"Using cached transcription for {Path(video_path).name}")
            return cached_transcript

        # Extract audio if needed
        if not audio_path.exists():
            print(f"Extracting audio from {Path(video_path).name}...")
            os.system(
                f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -b:a 64k -f mp3 "{audio_path}"'
            )

            if not os.path.exists(audio_path):
                print(f"Failed to extract audio from video: {video_path}")
                return None

        # Transcribe audio using OpenAI Whisper API
        print(f"Transcribing audio for {Path(video_path).name}...")
        try:
            with open(audio_path, "rb") as audio_file:
                if OPENAI_V1:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )
                else:
                    response = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                    )

            if not response:
                return None

            # Extract segments from the response
            if OPENAI_V1:
                segments = response.segments if hasattr(response, "segments") else []
                transcription = [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                    }
                    for segment in segments
                ]
            else:
                segments = response.get("segments", [])
                transcription = [
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                    }
                    for segment in segments
                ]

            # Cache the transcription
            CacheManager.save_cache(transcription_path, transcription)
            return transcription

        except Exception as e:
            print(f"Error during transcription for {Path(video_path).name}: {e}")
            return None

    def extract_subtitle_text(
        self, transcription: List[Dict[str, Union[str, float]]]
    ) -> str:
        """
        Extract and combine all subtitle text from transcription segments.

        Args:
            transcription: List of transcription segments

        Returns:
            str: Combined subtitle text
        """
        if not transcription:
            return ""

        return " ".join(
            segment["text"] for segment in transcription if segment.get("text")
        )

    def is_english(self, text: str) -> bool:
        """
        Detect if text is primarily in English using GPT-4o.

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if text is primarily English, False otherwise
        """
        if not text.strip():
            return True

        # For very short text, use simple heuristic
        if len(text.strip()) < 10:
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            total_chars = len(re.sub(r"[^\w]", "", text))
            return english_chars / max(total_chars, 1) > 0.7

        try:
            print(f"   🔍 Detecting language with GPT-4o...")
            # Use GPT-4o for accurate language detection
            messages = [
                {
                    "role": "system",
                    "content": "You are a language detection expert. Your task is to determine if the given text is primarily in English or not. Respond with only 'YES' if the text is primarily in English, or 'NO' if it's primarily in another language. Consider mixed content as English if more than 70% is English.",
                },
                {
                    "role": "user",
                    "content": f"Is this text primarily in English?\n\nText: {text[:500]}...",  # Limit to first 500 chars for efficiency
                },
            ]

            result = self._call_openai_chat(messages, max_tokens=10, temperature=0.1)
            is_eng = result.upper() == "YES"
            print(
                f"   🔍 Language detection result: {'English' if is_eng else 'Non-English'}"
            )
            return is_eng

        except Exception as e:
            print(f"   ⚠️  Error in language detection: {e}")
            print(f"   🔄 Falling back to character-based detection...")

            # Fallback to character-based detection
            clean_text = re.sub(r"[^\w\s]", "", text)
            english_chars = len(re.findall(r"[a-zA-Z]", clean_text))
            total_chars = len(re.sub(r"\s", "", clean_text))

            if total_chars == 0:
                return True

            english_ratio = english_chars / total_chars
            return english_ratio > 0.7

    def translate_to_english(self, text: str) -> str:
        """
        Translate non-English text to English using OpenAI API.

        Args:
            text (str): Text to translate

        Returns:
            str: Translated English text
        """
        if not text.strip():
            return text

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a professional film translator working on the English subtitles for an international film. Your task is to translate the dialogue naturally and accurately into English, capturing the characters' tone, emotion, and cultural nuance, while also ensuring the translation reads smoothly and fits within standard subtitle constraints. If the text is already in English, return it as is. Only return the translated text, no explanations.",
                },
                {
                    "role": "user",
                    "content": f"Please translate the following subtitles into fluent, idiomatic English. Prioritize meaning over literal word-for-word translation. When there are idioms, slang, or cultural references, adapt them for an English-speaking audience while retaining the original intent and emotional weight: {text}",
                },
            ]

            translated_text = self._call_openai_chat(
                messages, max_tokens=2000, temperature=0.3
            )
            return translated_text

        except Exception as e:
            print(f"Error during translation: {e}")
            return text  # Return original text if translation fails

    def load_existing_results(self, output_file: str) -> List[Dict[str, str]]:
        """
        Load existing results from JSON file if it exists.

        Args:
            output_file (str): Path to the JSON file

        Returns:
            List[Dict[str, str]]: Existing results or empty list
        """
        if Path(output_file).exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Warning: Could not load existing results from {output_file}: {e}"
                )
                return []
        return []

    def save_single_result(
        self,
        result_entry: Dict[str, str],
        output_file: str,
        all_results: List[Dict[str, str]],
    ) -> None:
        """
        Save a single result to the JSON file immediately.

        Args:
            result_entry: The result entry to add
            output_file: Path to the JSON file
            all_results: Current list of all results
        """
        try:
            # Add the new result
            all_results.append(result_entry)

            # Save immediately to file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"  💾 Saved result to {output_file}")

        except Exception as e:
            print(f"  ✗ Error saving result to {output_file}: {e}")

    def process_videos_directory(
        self, videos_dir: str = "videos", output_file: str = "subtitles.json"
    ) -> List[Dict[str, str]]:
        """
        Process all MP4 videos in the specified directory and extract subtitles.
        Saves each result immediately after processing.

        Args:
            videos_dir (str): Directory containing video files
            output_file (str): Output JSON file path

        Returns:
            List[Dict[str, str]]: List of dictionaries with material_id and transcript
        """
        videos_path = Path(videos_dir)
        if not videos_path.exists():
            raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

        # Find all MP4 files
        video_files = list(videos_path.glob("*.mp4"))
        if not video_files:
            logger.info(f"No MP4 files found in {videos_dir}")
            return []

        # Load existing results
        results = self.load_existing_results(output_file)
        processed_ids = {entry["material_id"] for entry in results}

        # Filter out already processed videos
        remaining_videos = [vf for vf in video_files if vf.stem not in processed_ids]

        total_videos = len(video_files)
        already_processed = len(video_files) - len(remaining_videos)

        logger.info(f"📊 Processing Status:")
        logger.info(f"  Total videos found: {total_videos}")
        logger.info(f"  Already processed: {already_processed}")
        logger.info(f"  Remaining to process: {len(remaining_videos)}")

        if already_processed > 0:
            logger.info(f"  ℹ️  Resuming from previous session...")

        if not remaining_videos:
            print(f"✅ All videos already processed!")
            return results

        print(f"\n🚀 Starting processing of {len(remaining_videos)} videos...")

        for i, video_file in enumerate(remaining_videos, 1):
            video_name = video_file.stem  # Filename without .mp4 extension

            print(f"\n📹 [{i}/{len(remaining_videos)}] Processing: {video_name}")
            print(
                f"   Progress: {((already_processed + i - 1) / total_videos * 100):.1f}% overall"
            )

            try:
                # Transcribe the video
                print(f"   🎵 Extracting audio and transcribing...")
                transcription = self.transcribe_video(str(video_file))

                if transcription:
                    # Extract subtitle text
                    subtitle_content = self.extract_subtitle_text(transcription)

                    # Check if translation is needed using GPT-4o
                    if self.is_english(subtitle_content):
                        print(f"   🇺🇸 Text is in English, no translation needed")
                        final_transcript = subtitle_content
                    else:
                        print(
                            f"   🌐 Text is not in English, translating with GPT-4o..."
                        )
                        final_transcript = self.translate_to_english(subtitle_content)
                        print(f"   ✅ Translation completed")

                    # Create result in requested format
                    result_entry = {
                        "material_id": video_name,
                        "transcript": final_transcript,
                    }

                    # Save immediately
                    self.save_single_result(result_entry, output_file, results)

                    print(f"   ✅ Successfully processed {video_name}")
                    print(
                        f"   📝 Transcript length: {len(final_transcript)} characters"
                    )

                else:
                    # Add empty result for failed transcription
                    result_entry = {"material_id": video_name, "transcript": ""}

                    # Save immediately
                    self.save_single_result(result_entry, output_file, results)

                    print(f"   ✗ Failed to extract subtitles for {video_name}")

            except Exception as e:
                print(f"   ✗ Error processing {video_name}: {e}")
                # Add empty result for error cases
                result_entry = {"material_id": video_name, "transcript": ""}

                # Save immediately
                self.save_single_result(result_entry, output_file, results)

        print(f"\n🎉 Processing completed!")
        print(f"📊 Final results saved to {output_file}")

        return results

    def process_single_video_file(self, video_path: str) -> Dict[str, str]:
        """
        Process a single MP4 video and extract subtitles.

        Args:
            video_path (str): Path to the MP4 video file

        Returns:
            Dict[str, str]: Dictionary with material_id and transcript
        """
        video_file = Path(video_path)

        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if video_file.suffix.lower() != ".mp4":
            raise ValueError(f"Unsupported file format: {video_file.suffix}")

        video_name = video_file.stem
        output_file = "subtitles.json"
        logger.info(f"\n📹 Processing single video: {video_name}")
        results = self.load_existing_results(output_file)
        for entry in results:
            if entry["material_id"] == video_name:
                logger.info(f" Already processed {video_name}, skipping...")
                return entry

        try:
            logger.info(f" Extracting audio and transcribing...")
            transcription = self.transcribe_video(str(video_file))

            if transcription:
                subtitle_content = self.extract_subtitle_text(transcription)

                if self.is_english(subtitle_content):
                    logger.info(f"   🇺🇸 Text is in English, no translation needed")
                    final_transcript = subtitle_content
                else:
                    logger.info(
                        f"   🌐 Text is not in English, translating with GPT-4o..."
                    )
                    final_transcript = self.translate_to_english(subtitle_content)
                    logger.info(f"   ✅ Translation completed")

                result_entry = {
                    "material_id": video_name,
                    "transcript": final_transcript,
                }
                # Save result immediately
                self.save_single_result(result_entry, output_file, results)

                logger.info(f"   ✅ Successfully processed {video_name}")
                logger.info(
                    f"   📝 Transcript length: {len(final_transcript)} characters"
                )

            else:
                result_entry = {"material_id": video_name, "transcript": ""}
                logger.info(f"   ✗ Failed to extract subtitles for {video_name}")

        except Exception as e:
            logger.error(f"   ✗ Error processing {video_name}: {e}")
            result_entry = {"material_id": video_name, "transcript": ""}

        return result_entry


def main():
    """Main function to run the subtitle extraction."""
    try:
        print("🎬 Subtitle Extraction & Translation Tool")
        print("=" * 50)
        print(
            f"🔧 OpenAI API Version: {'v1.x (New)' if OPENAI_V1 else 'v0.x (Legacy)'}"
        )

        extractor = SubtitleExtractor()
        results = extractor.process_videos_directory()

        print(f"\n📈 === Final Summary ===")
        print(f"📊 Total videos processed: {len(results)}")
        successful = sum(1 for entry in results if entry["transcript"].strip())
        failed = len(results) - successful

        print(f"✅ Successful extractions: {successful}")
        print(f"❌ Failed extractions: {failed}")

        if successful > 0:
            # Count translations vs English
            english_count = 0
            translated_count = 0

            for entry in results:
                if entry["transcript"].strip():
                    # Simple heuristic: if it contains common English words, likely was English
                    text = entry["transcript"].lower()
                    if any(
                        word in text
                        for word in [
                            "the",
                            "and",
                            "is",
                            "are",
                            "was",
                            "were",
                            "have",
                            "has",
                        ]
                    ):
                        if (
                            len([c for c in text if ord(c) > 127]) / len(text) < 0.1
                        ):  # Less than 10% non-ASCII
                            english_count += 1
                        else:
                            translated_count += 1
                    else:
                        translated_count += 1

            print(f"🇺🇸 Originally in English: {english_count}")
            print(f"🌐 Translated to English: {translated_count}")

            print(f"\n📝 Sample results:")
            sample_count = 0
            for entry in results:
                if entry["transcript"].strip() and sample_count < 3:
                    preview = (
                        entry["transcript"][:100] + "..."
                        if len(entry["transcript"]) > 100
                        else entry["transcript"]
                    )
                    print(f"  📹 {entry['material_id']}: {preview}")
                    sample_count += 1

        print(f"\n🎯 Results saved to: subtitles.json")

    except KeyboardInterrupt:
        print(f"\n⚠️  Process interrupted by user. Partial results may be saved.")
    except Exception as e:
        print(f"❌ Error in main execution: {e}")


if __name__ == "__main__":
    main()
