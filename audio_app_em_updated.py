import hashlib  # audio fingerprinting
import html  # safe HTML attribute escaping
import io  # in-memory audio buffers (safe for multi-user cloud deployments)
import json  # safe JS string serialization
import os  # operating-system access (used here to read API keys)
import time  # module with timing functions for latency measurements

from dotenv import load_dotenv  # load environment variables from a local .env file
from gtts import gTTS  # TTS service
from openai import OpenAI  # OpenAI-compatible SDK for OpenRouter ASR and LLM calls
import streamlit as st  # application layer
import streamlit.components.v1 as components  # embed HTML/JS widgets (ElevenLabs ConvAI)


# This project compares two conversational voice-agent architectures:
# 1. ElevenLabs Realtime Conversational AI Widget (embedded WebRTC/streaming voice agent)
# 2. Chained Voice-Agent Pipeline: audio input -> Whisper ASR -> LLM reasoning -> gTTS output.

DEFAULT_ELEVENLABS_AGENT_ID = "agent_8601m7maz3kkfvz9009rwvrzjd00"

ELEVENLABS_API_BASES = {
    "us": "https://api.elevenlabs.io",
    "global": "https://api.elevenlabs.io",
    "eu-residency": "https://api.eu.residency.elevenlabs.io",
    "in-residency": "https://api.in.residency.elevenlabs.io",
}


def get_secret(key: str, default: str | None = None) -> str | None:
    """Read configuration from environment variables (.env) or Streamlit Cloud secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# 1. SETUP: Configuration & API Keys
# Configure the Streamlit browser page and introduce the application architecture.
st.set_page_config(
    page_title="Conversational Audio Bot",
    page_icon="🎙️",
    layout="wide",
)

# Reduce excess top padding so the embedded widget canvas fits comfortably above the fold.
st.markdown(
    "<style>.block-container { padding-top: 2rem; padding-bottom: 1.5rem; }</style>",
    unsafe_allow_html=True,
)

# Load local .env if present (on Streamlit Community Cloud, use App Settings -> Secrets).
load_dotenv(override=True)
api_key = get_secret("OPENROUTER_API_KEY")
default_agent_id = get_secret("ELEVENLABS_AGENT_ID", DEFAULT_ELEVENLABS_AGENT_ID)

st.title("🎙️ Conversational Audio Bot")
st.caption(
    "Compare a live **ElevenLabs Realtime Conversational AI** agent with the "
    "**Chained Pipeline** (Whisper + GPT-6 Luna via OpenRouter + gTTS)."
)


# 2. SIDEBAR: SYSTEM CONFIGURATION
# Allow users to inspect/configure the ElevenLabs widget and tune the chained pipeline.
with st.sidebar:
    st.header("⚙️ System Configuration")

    st.subheader("ElevenLabs Voice Widget")
    elevenlabs_agent_id = st.text_input(
        "Agent ID",
        value=default_agent_id or DEFAULT_ELEVENLABS_AGENT_ID,
        help="Your deployed ElevenLabs Conversational AI Agent ID.",
    ).strip()
    server_location = st.selectbox(
        "Server Region",
        options=["us", "eu-residency", "in-residency", "global"],
        index=0,
        help="Match the data residency region where your ElevenLabs agent is deployed.",
    )
    widget_height = st.slider(
        "Widget Canvas Height (px)",
        min_value=260,
        max_value=640,
        value=380,
        step=20,
        help="Adjust if the expanded widget needs more vertical space.",
    )

    st.divider()
    st.subheader("Chained Pipeline Settings")

    reasoning_effort = st.select_slider(
        "Reasoning Effort",
        options=["low", "medium", "high"],
        value="low",
        help="""
Controls how much internal reasoning the model does before it answers.

- **low:** Fastest and cheapest. Good for short, conversational replies.
- **medium:** More deliberate answers with moderate extra latency.
- **high:** Most thorough reasoning, but noticeably slower and more expensive.

Higher effort adds latency to this voice pipeline, and does **not**
guarantee that an answer is grounded or factually correct.
""",
    )


# 3. MAIN INTERFACE: TABS FOR ELEVENLABS WIDGET & CHAINED PIPELINE
tab_elevenlabs, tab_chained = st.tabs(
    [
        "⚡ ElevenLabs Conversational AI",
        "🔗 Chained Pipeline (Whisper + LLM + gTTS)",
    ]
)

with tab_elevenlabs:
    safe_agent_id = html.escape(elevenlabs_agent_id, quote=True)
    server_loc_attr = (
        f' server-location="{html.escape(server_location, quote=True)}"'
        if server_location != "us"
        else ""
    )
    api_base = ELEVENLABS_API_BASES.get(server_location, "https://api.elevenlabs.io")
    js_agent_id = json.dumps(elevenlabs_agent_id)
    js_api_base = json.dumps(api_base)

    elevenlabs_widget_html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          * {{ box-sizing: border-box; }}
          html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background: transparent;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .widget-stage {{
            position: relative;
            width: 100%;
            height: 100%;
            border: 1px solid #d0d7de;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
          }}
          .stage-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
          }}
          .stage-title {{
            font-size: 15px;
            font-weight: 600;
            color: #0f172a;
            margin: 0;
          }}
          .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 999px;
            background: #e2e8f0;
            color: #334155;
          }}
          .status-pill.ok {{
            background: #dcfce7;
            color: #166534;
          }}
          .status-pill.error {{
            background: #fee2e2;
            color: #991b1b;
          }}
          .stage-hint {{
            font-size: 13px;
            color: #475569;
            margin: 8px 0 0 0;
            max-width: 680px;
            line-height: 1.45;
          }}
          .diag-box {{
            margin-top: 12px;
            padding: 10px 12px;
            border-radius: 8px;
            background: #fff1f2;
            border: 1px solid #fecdd3;
            color: #881337;
            font-size: 12.5px;
            line-height: 1.45;
            display: none;
          }}
          .diag-box code {{
            background: rgba(0, 0, 0, 0.06);
            padding: 1px 4px;
            border-radius: 4px;
          }}
        </style>
      </head>
      <body>
        <div class="widget-stage">
          <div>
            <div class="stage-header">
              <p class="stage-title">ElevenLabs Realtime Voice Widget</p>
              <span id="widget-status" class="status-pill">Loading widget script...</span>
            </div>
            <p class="stage-hint">
              The floating ElevenLabs launcher appears in the <strong>bottom-right corner of this box</strong>.
              Click it and allow <strong>microphone access</strong> to start talking.
            </p>
            <div id="widget-diag" class="diag-box"></div>
          </div>
        </div>

        <elevenlabs-convai agent-id="{safe_agent_id}"{server_loc_attr}></elevenlabs-convai>
        <script
          id="elevenlabs-script"
          src="https://unpkg.com/@elevenlabs/convai-widget-embed"
          async
          type="text/javascript"
        ></script>

        <script>
          (function () {{
            const agentId = {js_agent_id};
            const apiBase = {js_api_base};
            const statusEl = document.getElementById("widget-status");
            const diagEl = document.getElementById("widget-diag");
            const primaryScript = document.getElementById("elevenlabs-script");

            function setStatus(text, kind) {{
              statusEl.textContent = text;
              statusEl.className = "status-pill" + (kind ? " " + kind : "");
            }}

            function showDiag(htmlMsg) {{
              diagEl.innerHTML = htmlMsg;
              diagEl.style.display = "block";
            }}

            // Fallback CDN if unpkg is slow or blocked on the user's network.
            primaryScript.addEventListener("error", function () {{
              setStatus("unpkg blocked — trying jsDelivr...", "");
              const fallback = document.createElement("script");
              fallback.src = "https://cdn.jsdelivr.net/npm/@elevenlabs/convai-widget-embed";
              fallback.async = true;
              fallback.type = "text/javascript";
              fallback.onerror = function () {{
                setStatus("Script failed to load", "error");
                showDiag(
                  "Could not load <code>@elevenlabs/convai-widget-embed</code> from unpkg.com or jsDelivr. " +
                  "Check browser console / network or ad-blocker settings."
                );
              }};
              document.body.appendChild(fallback);
            }});

            // Verify agent widget configuration directly from ElevenLabs API.
            async function verifyAgentConfig() {{
              if (!agentId) {{
                setStatus("Missing Agent ID", "error");
                showDiag("Enter a valid ElevenLabs Agent ID in the sidebar.");
                return;
              }}
              try {{
                const url = apiBase + "/v1/convai/agents/" + encodeURIComponent(agentId) + "/widget";
                const resp = await fetch(url);
                if (!resp.ok) {{
                  let detail = "";
                  try {{
                    const body = await resp.json();
                    detail = body.detail ? JSON.stringify(body.detail) : "";
                  }} catch (_) {{}}
                  setStatus("Agent config error (HTTP " + resp.status + ")", "error");
                  showDiag(
                    "ElevenLabs returned <strong>HTTP " + resp.status + "</strong> for agent <code>" +
                    agentId + "</code>" + (detail ? " (" + detail + ")" : "") + ".<br/>" +
                    "Common causes: (1) Agent ID typo or deleted agent, (2) Agent requires authentication / signed URL " +
                    "(disable Auth in ElevenLabs Agent &rarr; Security for public widget embeds), " +
                    "(3) Domain allowlist blocking <code>localhost</code>, or (4) wrong Server Region in the sidebar."
                  );
                  return;
                }}
              }} catch (err) {{
                setStatus("Network / CORS error", "error");
                showDiag(
                  "Could not reach ElevenLabs API (<code>" + apiBase + "</code>): " +
                  String(err)
                );
                return;
              }}

              // Wait for custom element registration & shadow DOM render.
              const timeout = setTimeout(function () {{
                const el = document.querySelector("elevenlabs-convai");
                if (!customElements.get("elevenlabs-convai")) {{
                  setStatus("Widget script timed out", "error");
                  showDiag("The <code>elevenlabs-convai</code> custom element did not register within 8 seconds.");
                }} else if (el && el.shadowRoot && el.shadowRoot.childElementCount > 0) {{
                  setStatus("Ready (bottom-right)", "ok");
                }} else {{
                  setStatus("Registered, waiting for UI...", "");
                }}
              }}, 8000);

              customElements.whenDefined("elevenlabs-convai").then(function () {{
                const checkMount = setInterval(function () {{
                  const el = document.querySelector("elevenlabs-convai");
                  if (el && el.shadowRoot && el.shadowRoot.childElementCount > 0) {{
                    clearInterval(checkMount);
                    clearTimeout(timeout);
                    setStatus("Ready — click bottom-right", "ok");
                  }}
                }}, 250);
              }});
            }}

            verifyAgentConfig();
          }})();
        </script>
      </body>
    </html>
    """

    # Render inside an explicit framed height so the bottom-right launcher is immediately visible.
    components.html(elevenlabs_widget_html, height=widget_height, scrolling=False)


with tab_chained:
    st.subheader("Chained Audio Pipeline")

    if not api_key:
        st.warning(
            "**OPENROUTER_API_KEY** is not configured. To use the Chained Pipeline tab, "
            "set `OPENROUTER_API_KEY` in your local `.env` file or in "
            "**Streamlit Community Cloud → App Settings → Secrets**."
        )
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # AUDIO INPUT: THE ASR STAGE
        # st.audio_input records audio in the browser. The recorded value remains present
        # across Streamlit reruns, so the code fingerprints each recording before processing it.
        audio_value = st.audio_input(
            "Speak to the AI Assistant",
            key="audio_input",
        )

        # Session state persists across Streamlit reruns within the same browser session.
        if "last_processed_audio_hash" not in st.session_state:
            st.session_state.last_processed_audio_hash = None

        if audio_value:
            # Read the recording once and retain the bytes for hashing and transcription.
            audio_bytes = audio_value.getvalue()

            # SHA-256 creates a stable fingerprint for this recording. The same recording
            # produces the same hash, while a new recording produces a different hash.
            audio_hash = hashlib.sha256(audio_bytes).hexdigest()

            # Process only genuinely new audio. This prevents unrelated widget interactions
            # from replaying the previous question.
            if audio_hash != st.session_state.last_processed_audio_hash:
                st.session_state.last_processed_audio_hash = audio_hash
                # start the timer:
                start_time = time.time()

                # A. ASR STAGE: WHISPER TRANSCRIPTION
                # Send the recorded WAV bytes in-memory to Whisper through OpenRouter.
                with st.status("👂 Listening and Transcribing...", expanded=True):
                    transcript_response = client.audio.transcriptions.create(
                        model="openai/whisper-large-v3",
                        file=("input.wav", audio_bytes, "audio/wav"),
                    )

                    user_text = transcript_response.text
                    st.write(f"**User said:** {user_text}")

                # B. LLM STAGE: GPT REASONING
                # GPT-6 Luna receives the behavioral system instruction and the transcript.
                with st.status("🧠 Reasoning...", expanded=True):
                    response = client.chat.completions.create(
                        model="openai/gpt-6-luna",
                        reasoning_effort=reasoning_effort,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a professional assistant. Be concise.",
                            },
                            {
                                "role": "user",
                                "content": user_text,
                            },
                        ],
                    )

                    ai_text = response.choices[0].message.content
                    st.write(f"**AI Response:** {ai_text}")

                # C. TTS STAGE: gTTS VOCALIZATION
                # Convert the model's response text into an in-memory MP3 and play it in the browser.
                with st.status("🗣️ Synthesizing Voice...", expanded=True):
                    # clock latency here (which is as close to first TTS audio as possible)
                    time_to_tts_start = round(time.time() - start_time, 2)

                    tts = gTTS(
                        text=ai_text,
                        lang="en",
                    )
                    mp3_buffer = io.BytesIO()
                    tts.write_to_fp(mp3_buffer)
                    mp3_buffer.seek(0)

                    st.audio(mp3_buffer.getvalue(), format="audio/mp3", autoplay=True)

                    st.divider()
                    st.header("📊 Performance Metrics")
                    st.metric("Time to TTS Start", f"{time_to_tts_start}s")

                # 4. Low latency warning
                if time_to_tts_start > 1.0:
                    st.warning(
                        "⚠️ This batch chained architecture is slower than a streaming "
                        "or native realtime speech-to-speech implementation."
                    )
