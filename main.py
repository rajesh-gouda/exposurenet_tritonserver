from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import status
import shutil
import os
import aiofiles
from extract_subtitles import SubtitleExtractor
from extract_features import extract_features
import json
import httpx
import subprocess
from logging_config import setup_logging
import logging


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

MAX_SIZE = 50 * 1024 * 1024
# Initialize the ExtractSubtitles class
extractor = SubtitleExtractor()


def get_video_length(video_path: str) -> int:
    """Returns video duration in seconds (integer, no milliseconds)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return int(duration)
    except Exception as e:
        logger.error(f"Failed to get video duration: {e}")
        return 0


async def infer_single_input(input_data):
    url = "http://3.80.116.90:8000/v2/models/exposurenet/infer"
    data = {
        "genre": "drama",
        "num_words": 300,
        "num_sentences": 20,
        "num_emotional_shifts": 3,
        "num_conflict_scenes": 2,
        "num_plot_twists": 1,
        "num_drama_hooks": 1,
        "length": 120,
    }
    for key, value in input_data.items():
        if key in data:
            data[key] = value
    logger.info(f"Sending data to {url}: {data}")
    payload = {
        "inputs": [
            {
                "name": "input_str",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(data)],
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info("Response:")
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"❌ HTTP error: {exc.response.status_code} - {exc.response.text}"
            )
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")


async def add_features(features):
    try:
        # add the features to a list of JSON file called features_output.json this file should be created if it does not exist it should be a list of JSON objects
        features_output_path = "features_output.json"
        if not os.path.exists(features_output_path):
            async with aiofiles.open(features_output_path, "w") as f:
                await f.write("[]")  # Initialize with an empty list
        async with aiofiles.open(features_output_path, "r+") as f:
            content = await f.read()
            features_list = json.loads(content)
            features_list.append(features)
            await f.seek(0)
            await f.write(json.dumps(features_list, indent=2))
            await f.truncate()
        return "Features added successfully."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding features: {str(e)}")


# @app.get("/")
# async def root():
#     logger.info("📥 Received request to '/' endpoint")
#     return {"message": "Welcome to the Video Analysis API!"}


@app.get("/", response_class=HTMLResponse)
async def show_upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/analyze_video/")
async def analyze_video(request: Request, video_file: UploadFile = File(...)):
    logger.info("📥 Received request to '/analyze_video/' endpoint")
    try:
        # Save the uploaded video file
        video_path = os.path.join(VIDEO_DIR, video_file.filename)
        async with aiofiles.open(video_path, "wb") as f:
            content = await video_file.read()
            if len(content) > MAX_SIZE:
                logger.warning("File exceeds size limit.")
                return templates.TemplateResponse(
                    "upload.html",
                    {
                        "request": request,
                        "result": {"message": "Error: File exceeds 50MB limit."},
                    },
                )
            await f.write(content)
        logger.info(f"Video saved to {video_path}")

        duration = get_video_length(video_path)
        logger.info(f"⏱️ Video duration: {duration} seconds")
        # Call the ExtractSubtitles class to process the video
        result = extractor.process_single_video_file(video_path)

        if not result:
            raise HTTPException(status_code=400, detail="Failed to process video.")
        material_id = result.get("material_id", video_file.filename)
        # Extract features from the transcript
        transcript = result.get("transcript", "")
        if not transcript:
            raise HTTPException(
                status_code=400, detail="No transcript found in the video."
            )

        # check if features already exist
        features = None
        logger.info(f"Checking if features for {material_id} already exist...")
        features_output_path = "features_output.json"
        if os.path.exists(features_output_path):
            async with aiofiles.open(features_output_path, "r") as f:
                features_list = json.loads(await f.read())
                for feature in features_list:
                    if feature.get("material_id") == material_id:
                        logger.info(
                            f"Features for {material_id} already exist, skipping extraction...."
                        )
                        features = feature
                        break

        if not features:
            logger.info(f"Extracting features for {material_id}...")
            # Extract features using the extract_features function
            features = await extract_features(transcript)
            if not features:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract features from the transcript.",
                )
            features["material_id"] = material_id
            await add_features(features)
        features["length"] = duration
        logger.info(f"Extracted features: {features}")

        # do a post call to http://3.80.116.90:8000/v2/models/exposurenet/infer
        result = await infer_single_input(features)
        logger.info(f"Inference result: {result}")
        result = {
            "message": "Video analysis completed successfully.",
            "video_file": video_path,
            "predicted_class": result["outputs"][0]["data"][0],
        }
        return templates.TemplateResponse(
            "upload.html", {"request": request, "result": result}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {"request": request, "result": {"message": f"Error: {str(e)}"}},
        )
