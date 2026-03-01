# Beat Your Meet

AI meeting facilitator that joins your calls, keeps everyone on-topic, and enforces time limits — so your meetings actually end on time.

  
Checkout out our deployed app: https://beat-your-meet.vercel.app

## How It Works

1. **Describe your meeting** — tell the app what you're meeting about and how long you have.
2. **AI generates an agenda** — Mistral Large creates a structured, time-boxed agenda from your description.
3. **Pick a facilitation style** — choose how aggressive the bot should be (gentle, moderate, or aggressive).
4. **Start the meeting** — everyone joins a LiveKit audio room. The AI bot ("Beat") joins as a voice participant.
5. **Beat keeps you on track** — it transcribes the conversation in real-time, detects tangents, warns when time is running low, and transitions between agenda items.

## Architecture

```
┌─────────────┐       HTTP        ┌─────────────┐
│   Frontend   │ ───────────────▶ │   Server     │
│  (Next.js)   │                  │  (FastAPI)   │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       │  LiveKit audio +                │ LiveKit API
       │  data channels                  │ (create rooms)
       │                                 │
       ▼                                 ▼
┌──────────────────────────────────────────────┐
│              LiveKit Cloud                    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Agent       │
              │ (LiveKit Agents)│
              │                 │
              │ Deepgram (STT)  │
              │ Mistral  (LLM)  │
              │ ElevenLabs(TTS) │
              │ Silero   (VAD)  │
              └─────────────────┘
```

- **Frontend** (`frontend/`) — Next.js 14 + TypeScript + Tailwind. Handles meeting setup (agenda generation, style selection) and the in-meeting room UI with live agenda tracking.
- **Server** (`server/`) — FastAPI backend. Generates LiveKit tokens, calls Mistral for agenda generation, and creates rooms with agenda/style stored in room metadata.
- **Agent** (`agent/`) — LiveKit Agents voice pipeline. Joins rooms, transcribes audio, monitors conversation against the agenda every 15 seconds, and speaks up when needed.

## Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- [Python](https://python.org/) (3.10+)
- [LiveKit Cloud](https://livekit.io/) account (or self-hosted LiveKit server)
- API keys for: [Mistral AI](https://mistral.ai/), [Deepgram](https://deepgram.com/), [ElevenLabs](https://elevenlabs.io/)

## Setup

1. **Clone and configure environment**
  ```bash
   cp .env.example .env
   # Fill in all API keys in .env
  ```
2. **Install and run the frontend**
  ```bash
   cd frontend
   npm install
   npm run dev
  ```
3. **Install and run the server**
  ```bash
   cd server
   pip install -r requirements.txt
   python main.py
  ```
4. **Install and run the agent**
  ```bash
   cd agent
   pip install -r requirements.txt
   python main.py start
  ```

The frontend runs on `http://localhost:3000`, the server on `http://localhost:8000`.

## Environment Variables


| Variable                     | Description                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `LIVEKIT_URL`                | LiveKit server WebSocket URL                                                                      |
| `LIVEKIT_API_KEY`            | LiveKit API key                                                                                   |
| `LIVEKIT_API_SECRET`         | LiveKit API secret                                                                                |
| `MISTRAL_API_KEY`            | Mistral AI API key                                                                                |
| `DEEPGRAM_API_KEY`           | Deepgram STT API key                                                                              |
| `ELEVENLABS_API_KEY`         | ElevenLabs TTS API key                                                                            |
| `DETERMINISTIC_TIME_QUERIES` | Enable deterministic runtime answers for time-related questions (`true`/`false`, default: `true`) |
| `NEXT_PUBLIC_SERVER_URL`     | Backend URL for the frontend (default: `http://localhost:8000`)                                   |
| `NEXT_PUBLIC_LIVEKIT_URL`    | LiveKit URL for the frontend                                                                      |


## Facilitation Styles


| Style          | Tangent tolerance | Personality                                         |
| -------------- | ----------------- | --------------------------------------------------- |
| **Gentle**     | 60 seconds        | Warm, suggestive — nudges without pressure          |
| **Moderate**   | 30 seconds        | Friendly but firm — balances warmth with directness |
| **Aggressive** | 10 seconds        | Direct and action-oriented — prioritizes efficiency |


## Tech Stack

- **LLM**: Mistral Large (agenda generation + facilitator) / Mistral Small (tangent monitoring)
- **STT**: Deepgram Nova 2
- **TTS**: ElevenLabs Turbo v2.5
- **VAD**: Silero
- **Real-time audio**: LiveKit + LiveKit Agents SDK
- **Frontend**: Next.js 14, React 18, Tailwind CSS, LiveKit Components React
- **Backend**: FastAPI, Uvicorn

