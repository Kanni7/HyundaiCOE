"""End-to-end pipeline: audio -> ASR -> normalise -> semantics -> context -> policy -> vehicle."""
from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import numpy as np

from .audio.asr import SpeechRecognizer
from .config import AppConfig
from .nlu.base import LLMError
from .nlu.context_manager import ConversationContext
from .nlu.semantic_parser import SemanticParser
from .policy.command_policy import CommandPolicy
from .schemas import ASRResult, Decision, PipelineResult, PolicyResult
from .vehicle.base import VehicleAdapter, dispatch


def normalize_transcript(text: str) -> str:
    """Unicode NFC + whitespace cleanup only. Never translates, never drops words (e.g. negation)."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")
    return " ".join(text.split())


_REJECT_TEXT = {
    "unsupported_request": "Sorry, that's not something I can do in this car.",
    "cancelled_by_user": "Okay, cancelled.",
    "negated_no_action": "Okay, I won't do that.",
    "no_action": "I'm not sure what you'd like me to do. I can help with climate, fan, volume, media and navigation.",
    "model_error": "Sorry, I could not understand that. Could you say it again?",
}


def build_response(policy: PolicyResult, action_message: str | None, action_ok: bool | None) -> str:
    if policy.decision == Decision.EXECUTE:
        return f"Done: {action_message}." if action_ok else f"I couldn't do that: {action_message}."
    if policy.decision == Decision.ASK_CONFIRMATION:
        return policy.clarifying_question or "Should I go ahead?"
    return policy.clarifying_question or _REJECT_TEXT.get(policy.reason_code, "Sorry, I can't help with that.")


class BhashaDrivePipeline:
    def __init__(self, parser: SemanticParser, vehicle: VehicleAdapter, asr: SpeechRecognizer | None = None,
                 config: AppConfig | None = None, context: ConversationContext | None = None):
        self.config = config or AppConfig()
        self.parser, self.vehicle, self.asr = parser, vehicle, asr
        self.context = context or ConversationContext(self.config.context)
        self.policy = CommandPolicy(self.config.limits)

    # ---- entry points ---------------------------------------------------
    def process_audio_file(self, path: str | Path) -> PipelineResult:
        if self.asr is None:
            raise RuntimeError("No speech recognizer configured")
        t0 = time.perf_counter()
        asr = self.asr.transcribe_file(path)
        return self._after_asr(asr, time.perf_counter() - t0)

    def process_audio_array(self, audio: np.ndarray, sample_rate: int = 16000) -> PipelineResult:
        if self.asr is None:
            raise RuntimeError("No speech recognizer configured")
        if self.config.store_audio:
            self._store(audio, sample_rate)
        t0 = time.perf_counter()
        asr = self.asr.transcribe_array(audio, sample_rate)
        return self._after_asr(asr, time.perf_counter() - t0)

    def process_text(self, text: str, asr: ASRResult | None = None, timings: dict | None = None) -> PipelineResult:
        timings = dict(timings or {})
        t_start = time.perf_counter()
        norm = normalize_transcript(text)
        frame, nlu_error, resolved, verifier = None, None, None, None
        if not norm:
            policy = PolicyResult(decision=Decision.CLARIFY_OR_REJECT, reason_code="empty_transcript",
                                  clarifying_question="I didn't catch that. Could you repeat?")
        else:
            t0 = time.perf_counter()
            try:
                frame = self.parser.parse(norm, self.context.summary())
            except LLMError as exc:  # unavailable / timeout / invalid output -> no action, no keyword fallback
                nlu_error = f"{type(exc).__name__}: {exc}"
            timings["nlu_s"] = round(time.perf_counter() - t0, 3)
            if frame is not None:
                resolved = self.context.resolve(frame)
                if resolved.command is not None and resolved.context_action in ("none", "cleared_pending"):
                    # Standalone request: get an independent second opinion before any action.
                    t1 = time.perf_counter()
                    try:
                        verifier = self.parser.verify(norm)
                    except LLMError as exc:
                        nlu_error = f"verifier {type(exc).__name__}: {exc}"
                    timings["verify_s"] = round(time.perf_counter() - t1, 3)
            context_intents = {c.intent for c in (self.context.pending, self.context.last_executed) if c}
            policy = self.policy.decide(frame, resolved, nlu_error, verifier, context_intents)

        action = None
        if policy.decision == Decision.EXECUTE and policy.command is not None:
            t0 = time.perf_counter()
            try:
                action = dispatch(self.vehicle, policy.command, self.config.limits)
            except NotImplementedError as exc:
                from .schemas import ActionResult

                action = ActionResult(success=False, intent=policy.command.intent, message=str(exc))
            timings["vehicle_s"] = round(time.perf_counter() - t0, 4)

        response = build_response(policy, action.message if action else None, action.success if action else None)
        if norm:
            self.context.record_outcome(policy.command, policy.decision.value, bool(action and action.success))
            self.context.add_turn(norm, response)
        timings["post_asr_total_s"] = round(time.perf_counter() - t_start, 3)
        if "asr_s" in timings:
            timings["end_to_end_s"] = round(timings["asr_s"] + timings["post_asr_total_s"], 3)
        return PipelineResult(transcript=text or "", normalized_transcript=norm, asr=asr, frame=frame,
                              nlu_error=nlu_error, verification=verifier, resolved=resolved, policy=policy, action=action,
                              response_text=response, timings_s=timings)

    # ---- helpers --------------------------------------------------------
    def _after_asr(self, asr: ASRResult, asr_s: float) -> PipelineResult:
        timings = {"asr_s": round(asr_s, 3)}
        if asr.status != "ok":
            reason = {"silence": "silence", "no_audio": "no_audio"}.get(asr.status, "asr_error")
            policy = PolicyResult(decision=Decision.CLARIFY_OR_REJECT, reason_code=reason,
                                  clarifying_question="I didn't catch that. Could you repeat?")
            return PipelineResult(transcript="", normalized_transcript="", asr=asr, policy=policy,
                                  response_text=policy.clarifying_question, timings_s=timings)
        return self.process_text(asr.text, asr=asr, timings=timings)

    def _store(self, audio: np.ndarray, sr: int) -> None:
        import soundfile as sf

        d = self.config.audio_store_dir
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"utt_{int(time.time() * 1000)}.wav", audio, sr)


def build_default_pipeline(config: AppConfig | None = None, with_asr: bool = True) -> BhashaDrivePipeline:
    """Construct the real local pipeline. Raises clearly if a local model is unavailable."""
    from .audio.asr import FasterWhisperASR
    from .nlu.local_llm import OllamaLLM
    from .nlu.semantic_parser import LLMSemanticParser
    from .vehicle.mock_adapter import MockVehicleAdapter

    config = config or AppConfig()
    llm = OllamaLLM(config.llm)
    llm.health_check()
    llm.warmup()
    asr = FasterWhisperASR(config.asr) if with_asr else None
    return BhashaDrivePipeline(LLMSemanticParser(llm), MockVehicleAdapter(config.limits), asr, config)
