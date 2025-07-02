import os
import json


with open("subtitles.json") as f:
    data = json.load(f)


from openai import OpenAI, AsyncOpenAI


# client = OpenAI()
client = AsyncOpenAI()  # Use this for asynchronous operations

response_format = {
    "genre": "",
    "num_words": 0,
    "num_sentences": 0,
    "num_emotional_shifts": 0,
    "num_conflict_scenes": 0,
    "num_plot_twists": 0,
    "num_drama_hooks": 0,
}


async def extract_features(transcript):
    # Placeholder for actual feature extraction logic
    # This function should analyze the transcript and return the features
    prompt = f"""
            Please act as a specialist in TV‐trailer dialogue analysis for marketing. Using the dialogue below, identify and extract these elements: 
            Genre,  Number of Words, Number of Sentences, Number of Emotional Sentiment Shifts, Number of High Conflict Scenes, Number of Plot Twists, Number of Drama Hooks\n
            The dialogue is: {transcript}/n

            Please provide the output in the following JSON format:
            {json.dumps(response_format, indent=2)}
        """
    print(f"Extracting features from transcript: {transcript}")
    response = await client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    print(f"Response {response}")
    content = response.choices[0].message.content.strip()
    result = json.loads(content)
    print(f"Extracted features: {result}")
    return result
