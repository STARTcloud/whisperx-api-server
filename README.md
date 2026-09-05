## Overview

WhisperX API Server is a FastAPI-based server designed to transcribe audio files using the Whisper ASR (Automatic Speech Recognition) model based on WhisperX (https://github.com/m-bain/WhisperX) Python library. The API offers an OpenAI-like interface that allows users to upload audio files and receive transcription results in various formats. It supports customizable options such as different models, languages, temperature settings, and more.

Features
1. Audio Transcription: Transcribe audio files using the Whisper ASR model.
2. Model Caching: Load and cache models for reusability and faster performance.
3. OpenAI-like API, based on https://platform.openai.com/docs/api-reference/audio/createTranscription and https://platform.openai.com/docs/api-reference/audio/createTranslation

## API Endpoints

### `POST /v1/audio/transcriptions`
https://platform.openai.com/docs/api-reference/audio/createTranscription

**Parameters**:
- `file`: The audio file to transcribe.
- `model (str)`: The Whisper model to use. Default is `config.whisper.model`.
- `language (str)`: The language for transcription. Default is `config.default_language`.
- `prompt (str)`: Optional transcription prompt.
- `response_format (str)`: The format of the transcription output. Defaults to `json`.
- `temperature (float)`: Temperature setting for transcription. Default is `0.0`.
- `timestamp_granularities (list)`: Granularity of timestamps, either `segment` or `word`. Default is `["segment"]`.
- `stream (bool)`: Enable streaming mode for real-time transcription. (Doesn't work.)
- `hotwords (str)`: Optional hotwords for transcription.
- `suppress_numerals (bool)`: Option to suppress numerals in the transcription. Default is `True`.
- `highlight_words (bool)`: Highlight words in the transcription output for formats like VTT and SRT.
- `align (bool)`: Option to do transcription timings alignment. Default is `True`.
- `diarize (bool)`: Option to diarize the transcription. Default is `False`.

**Returns**: Transcription results in the specified format.

### `POST /v1/audio/translations`
https://platform.openai.com/docs/api-reference/audio/createTranslation

**Parameters**:
- `file`: The audio file to translate.
- `model (str)`: The Whisper model to use. Default is `config.whisper.model`.
- `prompt (str)`: Optional translation prompt.
- `response_format (str)`: The format of the translation output. Defaults to `json`.
- `temperature (float)`: Temperature setting for translation. Default is `0.0`.

**Returns**: Translation results in the specified format.

### `POST /v1/audio/transcriptions/jobs`
Queues an asynchronous transcription job, processed one at a time by a background worker. Poll `GET /v1/audio/transcriptions/jobs/{id}` until `status` is `completed` or `failed`; `GET /v1/audio/transcriptions/jobs` lists jobs and `DELETE /v1/audio/transcriptions/jobs/{id}` removes one and its audio.

**Parameters**: `file`, `model`, `language`, `diarize`, `min_speakers`, `max_speakers`, `chunk_size`, `vad_onset`, `vad_offset`.

A diarized job also returns **`speaker_embeddings`**: a voice vector per speaker label. Diarization labels restart from scratch in every request, so a caller transcribing a long recording in parts cannot otherwise tell that part 2's first voice is part 1's second. Comparing these vectors across parts identifies the same person. The field is absent when the installed whisperx does not produce embeddings, which is distinguishable from a job that simply had no speakers.

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions/jobs \
  -F "file=@meeting.wav" -F "diarize=true" -F "model=large-v2"
```

### `GET /healthcheck`
Returns the current health status of the API server.

### `GET /models/list`
Lists all loaded models currently available on the server.

### `POST /models/unload`
Unloads a specific model from memory cache.

### `POST /models/load`
Loads a specified model into memory.

### Running the API

**With Docker**:

For CPU:
```bash
    docker compose build whisperx-api-server-cpu

    docker compose up whisperx-api-server-cpu
```

For CUDA (GPU):
```bash
    docker compose build whisperx-api-server-cuda

    docker compose up whisperx-api-server-cuda

```

## Contributing

Feel free to submit issues, fork the repository, and send pull requests to contribute to the project.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3. See the `LICENSE` file for details.