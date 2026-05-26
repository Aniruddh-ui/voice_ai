"""
ui.py — Voice AI Interface

Single-page, voice-first design.
Core flow: Record → Transcribe → Respond → Play back.

No chat tabs. No dashboards. Just voice.
"""

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gradio as gr

import conversation
from memory import session_memory
from tts import get_available_voices


# ─────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────

def handle_voice(audio_path, voice):
    """Run the full voice pipeline and return results for display."""
    if not audio_path:
        return "", "", None, "No audio received."

    result = conversation.process_voice(audio_path, voice)

    # Hard failure — nothing to show
    if result["error"] and not result["response"]:
        return result.get("transcript", ""), "", None, f"Error: {result['error']}"

    lang_tag = f"  ·  {result['language'].upper()}" if result.get("language") else ""
    status   = result["error"] if result["error"] else f"Done{lang_tag}"

    return result["transcript"], result["response"], result["audio_path"], status


def handle_clear():
    """Wipe session memory and reset the output fields."""
    session_memory.clear()
    return "", "", None, "Memory cleared."


def handle_save():
    """Save the current conversation to disk (opt-in)."""
    try:
        path = session_memory.save()
        return f"Saved → {path}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Save failed: {e}"


# ─────────────────────────────────────────────
# CSS — minimal dark slate with violet accent
# ─────────────────────────────────────────────

CSS = """
footer { display: none !important; }

/* Typography */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Header */
#header {
    padding: 20px 4px 16px;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 24px;
}

/* Send button */
#send-btn { min-height: 44px; }

/* Action row buttons — small, subtle */
#save-btn button, #clear-btn button {
    background: transparent;
    border: 1px solid #334155;
    color: #94a3b8;
    font-size: 12px;
    min-height: 32px;
    border-radius: 6px;
}
#save-btn button:hover  { border-color: #6d28d9; color: #a78bfa; }
#clear-btn button:hover { border-color: #dc2626; color: #f87171; }

/* Status text — tiny, muted */
#status-out textarea {
    font-size: 11px;
    color: #64748b;
    background: transparent;
    border: none;
    padding: 4px 0 0;
    resize: none;
    min-height: unset;
}
#status-out label { display: none; }

/* Output labels — slightly muted */
#transcript-box label, #response-box label {
    font-size: 11px;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
"""


# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────

def build_ui():
    voices = get_available_voices()

    with gr.Blocks(title="Voice AI") as demo:

        # Header
        gr.Markdown(
            """<div id="header">
                <span style="font-size:22px; font-weight:600; color:#e2e8f0;">Voice AI</span>
                <span style="font-size:13px; color:#475569; margin-left:12px;">
                    speak naturally &nbsp;·&nbsp; multilingual &nbsp;·&nbsp; conversational
                </span>
            </div>""",
        )

        # Main layout — two columns
        with gr.Row(equal_height=False):

            # ── Left: Input controls ───────────────────────────────────────
            with gr.Column(scale=1, min_width=240):

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Audio",
                )
                voice_picker = gr.Dropdown(
                    choices=voices,
                    value="aura-orpheus-en",
                    label="Voice",
                )
                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    elem_id="send-btn",
                )

                # Action buttons
                gr.Markdown(
                    "<div style='margin-top:12px; font-size:11px; color:#475569; text-transform:uppercase;"
                    "letter-spacing:0.05em;'>Conversation</div>"
                )
                with gr.Row():
                    save_btn  = gr.Button("Save",  size="sm", elem_id="save-btn",  scale=1)
                    clear_btn = gr.Button("Clear", size="sm", elem_id="clear-btn", scale=1)

                status_out = gr.Textbox(
                    value="",
                    interactive=False,
                    show_label=False,
                    lines=1,
                    max_lines=2,
                    container=False,
                    elem_id="status-out",
                )

            # ── Right: Output ──────────────────────────────────────────────
            with gr.Column(scale=2):

                transcript_box = gr.Textbox(
                    label="You said",
                    lines=3,
                    interactive=False,
                    placeholder="Transcript will appear here after you send audio…",
                    elem_id="transcript-box",
                )
                response_box = gr.Textbox(
                    label="Response",
                    lines=6,
                    interactive=False,
                    placeholder="AI response will appear here…",
                    elem_id="response-box",
                )
                audio_output = gr.Audio(
                    label="Voice reply",
                    type="filepath",
                    autoplay=True,
                )

        # ── Wire up ────────────────────────────────────────────────────────

        send_btn.click(
            fn=handle_voice,
            inputs=[audio_input, voice_picker],
            outputs=[transcript_box, response_box, audio_output, status_out],
        )
        save_btn.click(
            fn=handle_save,
            outputs=[status_out],
        )
        clear_btn.click(
            fn=handle_clear,
            outputs=[transcript_box, response_box, audio_output, status_out],
        )

    return demo


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CSS,
    )
