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
    page_title="Card Dispute Voice Agent | MVP",
    page_icon="🛡️",
    layout="wide",
)

# Global MVP styling: compact page spacing, branded hero banner, and footer.
st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
      }
      .mvp-hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 62%, #1d4ed8 100%);
        border: 1px solid rgba(147, 197, 253, 0.25);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 14px;
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
      }
      .mvp-hero-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
      }
      .mvp-brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
      }
      .mvp-logo-box {
        width: 44px;
        height: 44px;
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }
      .mvp-title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }
      .mvp-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
        line-height: 1.2;
      }
      .mvp-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%);
        color: #0f172a;
      }
      .mvp-subtitle {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin: 4px 0 0 0;
      }
      .mvp-pills {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
      }
      .mvp-pill {
        font-size: 0.74rem;
        font-weight: 600;
        color: #e0f2fe;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(186, 230, 253, 0.28);
        padding: 3px 10px;
        border-radius: 999px;
      }
      .mvp-pill.live {
        background: rgba(16, 185, 129, 0.2);
        border-color: rgba(110, 231, 183, 0.45);
        color: #d1fae5;
      }
      .sidebar-brand {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        border-radius: 10px;
        padding: 12px 14px;
        color: #f8fafc;
        margin-bottom: 14px;
      }
      .sidebar-brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
      }
      .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #bfdbfe;
        margin: 3px 0 0 0;
      }
      .mvp-footer {
        margin-top: 18px;
        padding-top: 10px;
        border-top: 1px solid #e2e8f0;
        font-size: 0.78rem;
        color: #64748b;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load local .env if present (on Streamlit Community Cloud, use App Settings -> Secrets).
load_dotenv(override=True)
api_key = get_secret("OPENROUTER_API_KEY")
default_agent_id = get_secret("ELEVENLABS_AGENT_ID", DEFAULT_ELEVENLABS_AGENT_ID)

st.markdown(
    """
    <div class="mvp-hero">
      <div class="mvp-hero-top">
        <div class="mvp-brand-group">
          <div class="mvp-logo-box" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 2L4 5.5V11.2C4 16.4 7.4 21.1 12 22.5C16.6 21.1 20 16.4 20 11.2V5.5L12 2Z"
                    stroke="#93C5FD" stroke-width="1.8" fill="rgba(59,130,246,0.22)"/>
              <path d="M9 11.5V12.5M12 9V15M15 10.5V13.5"
                    stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </div>
          <div>
            <div class="mvp-title-row">
              <h1 class="mvp-title">Card Dispute Voice Agent</h1>
              <span class="mvp-badge">MVP v1.0</span>
            </div>
            <p class="mvp-subtitle">
              US Banking Reg E &amp; Reg Z Autonomous Card Dispute Resolution &middot; Powered by ElevenLabs Conversational AI
            </p>
          </div>
        </div>
      </div>
      <div class="mvp-pills">
        <span class="mvp-pill live">&#9679; Live Voice Demo</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# 2. SIDEBAR: SYSTEM CONFIGURATION
# Allow users to inspect/configure the ElevenLabs widget and tune the chained pipeline.
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <p class="sidebar-brand-title">🛡️ Card Dispute Voice Agent</p>
          <p class="sidebar-brand-sub">MVP v1.0 &middot; US Financial Services Demo</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        min_value=420,
        max_value=760,
        value=560,
        step=20,
        help="Adjust if the expanded widget or guide needs more vertical space.",
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

    st.divider()
    st.caption(
        "🧪 **Demo Notice:** All cardholder names, emails, and dates of birth "
        "displayed in this MVP are synthetic test data."
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
            color: #0f172a;
          }}
          .widget-stage {{
            position: relative;
            width: 100%;
            height: 100%;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
            padding: 16px 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow: hidden;
          }}
          .stage-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            flex-wrap: wrap;
          }}
          .stage-title-wrap {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
          }}
          .stage-title {{
            font-size: 15px;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
          }}
          .synth-tag {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 2px 8px;
            border-radius: 999px;
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
          }}
          .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 600;
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
          .diag-box {{
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
          .stage-body {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) 300px;
            gap: 16px;
            flex: 1;
            min-height: 0;
          }}
          .guide-col {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow-y: auto;
            padding-right: 4px;
          }}
          .info-card {{
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
          }}
          .card-head {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 6px;
            flex-wrap: wrap;
          }}
          .card-title {{
            font-size: 13.5px;
            font-weight: 700;
            color: #1e3a8a;
            margin: 0;
          }}
          .card-sub {{
            font-size: 11.5px;
            color: #64748b;
            margin: 0 0 8px 0;
            line-height: 1.4;
          }}
          table.test-users {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
          }}
          table.test-users th {{
            text-align: left;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #475569;
            background: #f1f5f9;
            padding: 6px 8px;
            border-bottom: 1px solid #cbd5e1;
          }}
          table.test-users td {{
            padding: 7px 8px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: top;
            color: #0f172a;
          }}
          table.test-users tr:last-child td {{
            border-bottom: none;
          }}
          .acct-chip {{
            display: inline-block;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 11px;
            font-weight: 700;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 1px 6px;
          }}
          .dob-chip {{
            display: inline-block;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 11.5px;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 5px;
            padding: 1px 6px;
            color: #0f172a;
            white-space: nowrap;
          }}
          .user-note {{
            margin: 8px 0 0 0;
            font-size: 11.5px;
            color: #475569;
            background: #f8fafc;
            border-left: 3px solid #2563eb;
            padding: 6px 8px;
            border-radius: 4px;
          }}
          .steps-row {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 10px;
          }}
          .step-pill {{
            font-size: 11.5px;
            font-weight: 600;
            color: #1e293b;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 4px 8px;
          }}
          .prompt-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
          }}
          .prompt-item {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 9px;
            padding: 8px 10px;
          }}
          .prompt-label {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #1d4ed8;
            margin: 0 0 4px 0;
          }}
          .prompt-quote {{
            font-size: 12px;
            color: #1e293b;
            margin: 0;
            line-height: 1.4;
          }}
          .widget-dock {{
            border: 1.5px dashed #93c5fd;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.62);
            padding: 14px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            color: #334155;
          }}
          .dock-title {{
            font-size: 13.5px;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 6px 0;
          }}
          .dock-text {{
            font-size: 12px;
            color: #475569;
            line-height: 1.45;
            margin: 0;
          }}
          .dock-tips {{
            margin: 10px 0 0 0;
            padding-left: 16px;
            font-size: 11.5px;
            color: #475569;
            line-height: 1.5;
          }}
          .dock-pointer {{
            font-size: 11.5px;
            font-weight: 600;
            color: #1d4ed8;
            padding-bottom: 64px;
          }}
          @media (max-width: 860px) {{
            .stage-body {{
              grid-template-columns: 1fr;
            }}
            .prompt-grid {{
              grid-template-columns: 1fr;
            }}
            .widget-dock {{
              display: none;
            }}
            .guide-col {{
              padding-bottom: 84px;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="widget-stage">
          <div class="stage-header">
            <div class="stage-title-wrap">
              <p class="stage-title">Live Voice Agent Demo Workspace</p>
              <span class="synth-tag">Synthetic Test Users Only</span>
            </div>
            <span id="widget-status" class="status-pill">Loading widget script...</span>
          </div>

          <div id="widget-diag" class="diag-box"></div>

          <div class="stage-body">
            <div class="guide-col">
              <!-- Card 1: Explicit Synthetic Test Users -->
              <div class="info-card">
                <div class="card-head">
                  <p class="card-title">🧪 Test Caller Profiles (Explicitly Synthetic Demo Users)</p>
                </div>
                <p class="card-sub">
                  Pick any <strong>synthetic test user</strong> below when speaking with the voice agent.
                  Do not enter real personal or card information.
                </p>
                <table class="test-users">
                  <thead>
                    <tr>
                      <th>Acct ID</th>
                      <th>Test User Name</th>
                      <th>Synthetic DOB</th>
                      <th>Test Email</th>
                      <th>Demo Context</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><span class="acct-chip">#42</span></td>
                      <td><strong>Sarah Miller</strong><br/><span style="font-size:11px;color:#64748b;">Test User #1</span></td>
                      <td><span class="dob-chip">04/15/1988</span></td>
                      <td><code>user42@example.com</code></td>
                      <td>Open CityMart fraud claim ($46.84) &middot; Flagged acct</td>
                    </tr>
                    <tr>
                      <td><span class="acct-chip">#160</span></td>
                      <td><strong>Sarah Miller</strong><br/><span style="font-size:11px;color:#64748b;">Test User #2</span></td>
                      <td><span class="dob-chip">09/22/1991</span></td>
                      <td><code>user160@mail.test</code></td>
                      <td>Open StreamCloud claim ($206.11) &middot; Split PillPath</td>
                    </tr>
                    <tr>
                      <td><span class="acct-chip">#191</span></td>
                      <td><strong>David Brooks</strong><br/><span style="font-size:11px;color:#64748b;">Test User #3</span></td>
                      <td><span class="dob-chip">11/03/1984</span></td>
                      <td><code>user191@mail.test</code></td>
                      <td>Won Quick Stop claim &middot; Flux Online ($99.08)</td>
                    </tr>
                  </tbody>
                </table>
                <p class="user-note">
                  <strong>Tip:</strong> Test Users <strong>#42</strong> and <strong>#160</strong> share the name
                  <strong>Sarah Miller</strong> &mdash; share the email, account ID, or synthetic DOB to help the agent match the right account.
                </p>
              </div>

              <!-- Card 2: How to Talk & Starter Questions -->
              <div class="info-card">
                <div class="card-head">
                  <p class="card-title">💬 What You Can Talk About &amp; Starter Questions</p>
                </div>
                <div class="steps-row">
                  <span class="step-pill">1. Click launcher (bottom-right)</span>
                  <span class="step-pill">2. Allow mic &amp; give Test Name + DOB</span>
                  <span class="step-pill">3. Ask about a dispute or charge</span>
                </div>
                <div class="prompt-grid">
                  <div class="prompt-item">
                    <p class="prompt-label">1. Check Open Dispute Status</p>
                    <p class="prompt-quote">
                      &ldquo;Hi, I&rsquo;m Sarah Miller, email user160@mail.test, DOB 09/22/1991. Can you check the status of my StreamCloud dispute?&rdquo;
                    </p>
                  </div>
                  <div class="prompt-item">
                    <p class="prompt-label">2. Verify Duplicate Name / Fraud Claim</p>
                    <p class="prompt-quote">
                      &ldquo;Hello, I&rsquo;m Sarah Miller on account 42, DOB 04/15/1988. What&rsquo;s the update on my CityMart fraud dispute?&rdquo;
                    </p>
                  </div>
                  <div class="prompt-item">
                    <p class="prompt-label">3. File a Merchant Dispute (Reg Z)</p>
                    <p class="prompt-quote">
                      &ldquo;Hi, I&rsquo;m David Brooks, DOB 11/03/1984. My Flux Online order for $99.08 never arrived &mdash; can I dispute it without locking my card?&rdquo;
                    </p>
                  </div>
                  <div class="prompt-item">
                    <p class="prompt-label">4. Test Policy &amp; Reg E Guardrails</p>
                    <p class="prompt-quote">
                      &ldquo;What is my Reg E provisional credit timeline, and can you credit $200 to my account right now?&rdquo;
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Reserved Right Column for Floating ElevenLabs Widget -->
            <div class="widget-dock">
              <div>
                <p class="dock-title">🎙️ Voice Agent Launch Zone</p>
                <p class="dock-text">
                  Click the <strong>ElevenLabs</strong> button in the bottom-right corner to begin a real-time voice call.
                </p>
                <ul class="dock-tips">
                  <li>Speak naturally &mdash; you can interrupt mid-sentence (barge-in).</li>
                  <li>Use any synthetic test user on the left to test verification.</li>
                  <li>Ask about chargebacks, Reg E timelines, or card locks.</li>
                </ul>
              </div>
              <div class="dock-pointer">
                &#8600; Click the floating widget below to start
              </div>
            </div>
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

    # Render inside an explicit framed height so the test-user guide and bottom-right launcher sit side by side.
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


# 4. MVP FOOTER
st.markdown(
    """
    <div class="mvp-footer">
      <span><strong>Card Dispute Voice Agent</strong> &middot; MVP v1.0 Prototype</span>
      <span>All test user names, emails, and dates of birth are synthetic demo data &middot; Powered by ElevenLabs Conversational AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)
