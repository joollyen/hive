"""Corpus callosum — arbitration between the two hemispheres.

Every utterance is evaluated by both halves. This module resolves them, and the
resolution rule is asymmetric on purpose:

    The objective hemisphere can always overrule the subjective one.
    The subjective hemisphere can never overrule the objective one.

Concretely that means style never changes what gets executed, permission verdicts
are not negotiable by phrasing, and generated language is checked against ground
truth before it is spoken — but the objective side never decides *how* something
is worded, because that would flatten the personality the subjective side exists
to provide.

The turn pipeline:

    transcript
      -> intent grammar          (deterministic; may answer outright)
      -> objective verdict       (action + facts + permission, or not grounded)
      -> subjective style frame  (tone, length, personalisation)
      -> either execute, confirm, or generate
      -> verify claims           (objective, last line of defence)
      -> speak

The fast path — a matched intent with ALLOW permission — never touches the
language model at all, which is where the sub-500ms responses come from.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

from ..events import PLAN_PROGRESS, STATUS_WORD, EventBus, State, Stopwatch
from .intents import Category, Intent, classify
from .llm import clean_for_speech
from .objective import ActionPlan, ObjectiveHemisphere, Permission, Verdict
from .phrasing import acknowledgement, holding_line
from .planner import Plan, make_plan
from .subjective import SubjectiveHemisphere


# Action kinds whose *result* is the answer, rather than something an
# acknowledgement can stand in for. A launch only needs "opening Chrome"; a
# system reading, a write-up or a search has produced the very thing that was
# asked for, and `run()` returns it as `detail`.
#
# One list, consulted in all three places an action's result is spoken. It was
# three separate literals, and the omission was invisible until it mattered: a
# search returned four papers with titles and abstracts, the kind was not in any
# of the lists, and HIVE said "Done." — the perfect shape of a failure, in that
# everything worked and nothing was communicated.
SPEAKS_ITS_RESULT = ("system", "media", "media_search", "document",
                     "search", "site_search", "query")

# Actions that have to cross the network before they have anything to say.
#
# Every other action in HIVE runs in under 200ms, which is why the action path
# never needed a holding line — the answer arrived before silence could register.
# A search does not: measured on this box, 1.4s to first audio for arXiv and
# 2.1s for a web search, against an 800ms budget. Going quiet for two seconds is
# the one failure mode this whole system is arranged to avoid, and it arrived by
# a door nobody had put a filler behind.
SLOW_ACTIONS = ("search", "site_search", "jobs")


@dataclass
class Outcome:
    handled: bool = True
    sleep: bool = False
    shutdown: bool = False
    spoke: str = ""
    action_label: str = ""
    # Whether the action this turn ran actually worked. Ignored by the voice
    # path — HIVE has already said what went wrong out loud, and a spoken
    # conversation has nowhere to put a status flag. HIVE-Q reads it, because a
    # queue's whole value is the list of things to go back and fix, and a task
    # that reported its own failure politely and was then filed as done is the
    # exact failure that list exists to prevent.
    failed: bool = False
    failure: str = ""
    # Carried out so the Slow Thinker can predict the next turn without
    # re-deriving what this one was.
    category: str = "general"
    escalated: bool = False
    text: str = ""


class Callosum:
    def __init__(self, cfg, bus: EventBus, objective: ObjectiveHemisphere,
                 subjective: SubjectiveHemisphere, llm, speaker, stm, ltm,
                 curiosity=None, calibrator=None, clippings=None):
        self.cfg, self.bus = cfg, bus
        self.objective, self.subjective = objective, subjective
        self.llm, self.speaker = llm, speaker
        self.stm, self.ltm = stm, ltm
        self.curiosity = curiosity
        self.calibrator = calibrator
        self.clippings = clippings
        self.actions_log: list[str] = []
        self._filler_ms = int(cfg.get("latency.filler_after_ms", 620))
        self._budget_ms = int(cfg.get("latency.target_first_audio_ms", 800))
        self._speech_started = threading.Event()
        self._last_escalated = False
        # What this turn may volunteer once the answer is out. Set during
        # evaluation and consumed at the end of the turn, so every exit path
        # spends it or drops it rather than half of them forgetting it exists.
        self._aside = None

    # -- status word --------------------------------------------------------
    def _status(self, category: str, escalated: bool = False) -> None:
        bank = "status_words.escalating" if escalated else f"status_words.{category}"
        word = self.cfg.line(bank) or self.cfg.line("status_words.general")
        self.bus.publish(STATUS_WORD, word=word, category=category)

    # -- filler -------------------------------------------------------------
    def _arm_filler(self, sw: Stopwatch, urgent: bool = False,
                    tailored: str = "") -> threading.Timer:
        """Speak a holding line if the real answer is going to be late.

        This is the single highest-leverage trick in the system for perceived
        responsiveness. The answer is not faster — it just never arrives into
        silence, and silence is what reads as a hang.

        `urgent` fires sooner, and is used when we already know we are escalating
        to the heavy model. If that model is not resident it costs about four
        seconds to load, and waiting the usual 620ms before saying anything makes
        a long silence longer.

        `tailored` is a line built from the request itself — "differences between
        lorazepam and propranolol, coming up". When there is one it fires sooner
        still, because the point of a tailored line is that it proves the request
        was understood, and a proof that arrives two thirds of a second late has
        already lost the argument. Its audio is rendered concurrently with
        generation (see _generate), so by the time the timer fires it is a cache
        hit and costs nothing.

        The filler marks `first_audio`, because from where the operator sits it
        *is* the first audio. The real answer marks `answer_audio` separately, so
        the breakdown still shows what generation actually cost.
        """
        if tailored:
            target = self._filler_ms * 0.42
        elif urgent:
            target = self._filler_ms * 0.55
        else:
            target = float(self._filler_ms)
        delay = max(0.05, (target - sw.elapsed_ms) / 1000.0)

        def fire() -> None:
            if self._speech_started.is_set() or self.speaker.speaking:
                return
            sw.mark("first_audio")
            if tailored:
                self.speaker.say(tailored, cached=True)
            else:
                self.speaker.say_line("filler.thinking" if urgent else "filler.short")

        t = threading.Timer(delay, fire)
        t.daemon = True
        t.start()
        return t

    # -- asides -------------------------------------------------------------
    def _speak_aside(self, spoken: str, lens=None) -> str:
        """Volunteer the turn's one connection or question, if it survived.

        Spoken as a separate utterance rather than appended to the answer's text,
        so the joiner puts a real sentence gap in front of it. An aside that runs
        straight on from the answer sounds like the answer got longer; one that
        arrives after a beat sounds like a thought.

        `lens` picks which bank the question comes from. A HackBot turn should
        wonder whether he has run it in a lab yet; a VOSS turn should wonder what
        the pattern is. Same slot, same restraint, different curiosity.
        """
        aside, self._aside = self._aside, None
        if aside is None or self.curiosity is None:
            return spoken
        line = self.curiosity.resolve(aside, spoken,
                                      bank=getattr(lens, "probes", ""))
        if not line or self.speaker._interrupt.is_set():
            return spoken
        self.speaker.say(line, cached=False)
        self.bus.log(f"aside ({aside.kind}): {line[:60]}", "memory")
        return f"{spoken} {line}".strip()

    # -- main entry ---------------------------------------------------------
    def handle(self, text: str, sw: Stopwatch) -> Outcome:
        self._aside = None
        outcome = self._handle(text, sw)
        # One place to stamp the turn's shape, rather than at every return.
        outcome.text = text
        if not outcome.category or outcome.category == "general":
            outcome.category = classify(text).category.value
        if self.curiosity is not None:
            self.curiosity.note_reply(outcome.spoke)
        return outcome

    def _handle(self, text: str, sw: Stopwatch) -> Outcome:  # noqa: C901
        intent = classify(text)
        sw.mark("intent")

        # A pending confirmation takes precedence over everything except stop.
        if self.stm.pending_confirm and intent.name in ("yes", "no", "stop"):
            return self._resolve_confirmation(intent, sw)

        # An instruction ends whatever thread was running. Calibration is for
        # conversation; "open Spotify" is not a HackBot question just because
        # the last three turns were.
        if self.calibrator is not None and intent.name != "converse":
            self.calibrator.release()

        if intent.name == "stop":
            self.speaker.interrupt()
            self.stm.pending_confirm = None
            return Outcome(spoke="")

        # The name on its own. Wake, show it, say nothing — and do not record a
        # turn, because none happened. The follow-up window is opened by the
        # runtime on the way out of this, so the command that follows needs no
        # name in front of it.
        if intent.name == "wake_only":
            if not bool(self.cfg.get("wake.acknowledge", False)):
                self.bus.set_state(State.LISTENING)
                return Outcome(spoke="")
            intent = Intent("presence", Category.PRESENCE, {}, 1.0,
                            intent.raw, intent.cleaned)

        if intent.name == "sleep":
            self.speaker.say_line("confirm.denied")
            return Outcome(sleep=True)

        if intent.name == "shutdown":
            return Outcome(shutdown=True)

        # Multi-step, before anything else touches it. Only when the split
        # genuinely yields separate actions — see _plan_is_useful.
        plan = make_plan(text)
        if plan is not None and self._plan_is_useful(plan):
            self.stm.add_user(text, "plan")
            self.subjective.observe(intent, text, self.speaker)
            return self._execute_plan(plan, sw)

        self.stm.add_user(text, intent.name)
        self.subjective.observe(intent, text, self.speaker)

        verdict = self.objective.evaluate(intent, self.stm)
        sw.mark("objective")
        style = self.subjective.frame(intent, self.stm)

        # Does this turn touch anything the vault already knows about, and is it
        # worth a question? Decided before generation so the recall can be handed
        # to the model as grounded context rather than tacked on afterwards.
        if self.curiosity is not None:
            self._aside = self.curiosity.consider(intent, text, self.stm)
            if self._aside is not None and self._aside.fact:
                verdict.facts["recalled_from_past_sessions"] = self._aside.fact
            sw.mark("associate")

        # --- understood, but cannot be done --------------------------------
        if verdict.refusal is not None:
            return self._speak_refusal(verdict, sw)

        # --- deterministic answer: objective already knows what to say -----
        if verdict.speak_text or verdict.speak_bank:
            if verdict.action is not None:
                return self._execute_or_confirm(verdict, sw, style)
            return self._speak_deterministic(verdict, sw)

        # --- action with no canned line ------------------------------------
        if verdict.action is not None:
            return self._execute_or_confirm(verdict, sw, style)

        # --- generative path ------------------------------------------------
        return self._generate(intent, verdict, style, sw)

    # -- deterministic ------------------------------------------------------
    def _speak_deterministic(self, verdict: Verdict, sw: Stopwatch) -> Outcome:
        self.bus.set_state(State.SPEAKING)
        text = verdict.speak_text or ""
        self._speech_started.set()
        if text:
            sw.mark("first_audio")
            self.speaker.say(text, cached=True)
        elif verdict.speak_bank:
            sw.mark("first_audio")
            text = self.speaker.say_line(verdict.speak_bank)
        text = self._speak_aside(text)
        self.stm.add_hive(text, sw.elapsed_ms)
        return Outcome(spoke=text)

    # -- multi-step ----------------------------------------------------------
    def _plan_is_useful(self, plan: Plan) -> bool:
        """Is this really several requests, or one sentence with 'and' in it?

        A split only counts if at least two fragments resolve to concrete,
        non-conversational intents. Otherwise the text goes through whole — a
        wrongly-split question becomes two confused ones, which is worse than a
        long one handled properly.
        """
        concrete = 0
        for step in plan.steps:
            name = classify(step.text).name
            if name not in ("converse", "empty", "presence"):
                concrete += 1
        return concrete >= 2

    def _execute_plan(self, plan: Plan, sw: Stopwatch) -> Outcome:
        """Run each step, tracking outcomes, then report on every one of them.

        State lives here rather than in the model's context. τ-Voice measured
        agents completing part of a multi-step request and reporting success
        anyway; the only defence is to hold the list yourself and read it back.
        """
        self.bus.set_state(State.THINKING)
        self._speech_started.set()
        sw.mark("first_audio")
        self.speaker.say(plan.opening_line(), cached=False)

        for step in plan.steps:
            if self.speaker._interrupt.is_set():
                step.status = "skipped"
                continue

            intent = classify(step.text)
            step.label = self._step_label(intent, step.text)
            self.bus.publish(PLAN_PROGRESS, index=step.index, total=len(plan),
                             label=step.label, status="running")

            verdict = self.objective.evaluate(intent, self.stm)

            if verdict.refusal is not None:
                step.status = "refused"
                step.spoken = verdict.refusal.render(self.cfg)
                self.speaker.say(step.spoken, cached=False)
            elif verdict.action is not None:
                if verdict.action.permission is Permission.ALLOW:
                    ok, detail = verdict.action.run()
                    step.status = "done" if ok else "failed"
                    if ok:
                        self.actions_log.append(f"{verdict.action.kind}: {verdict.action.label}")
                    # Narrate as it goes. In night mode this is the only channel
                    # telling Jay what is actually happening.
                    line = (detail if (verdict.action.kind in SPEAKS_ITS_RESULT
                                       and detail)
                            else verdict.speak_text if verdict.speak_text
                            else f"{step.label}, done." if ok
                            else f"{step.label} failed.")
                    step.spoken = line
                    self.speaker.say(line, cached=False)
                else:
                    # A step needing confirmation cannot be silently skipped, and
                    # stopping the whole plan to ask would strand the rest.
                    step.status = "skipped"
                    step.spoken = f"{step.label} needs your say-so; ask me again on its own."
                    self.speaker.say(step.spoken, cached=False)
            elif verdict.speak_text or verdict.speak_bank:
                step.status = "done"
                step.spoken = verdict.speak_text or ""
                if step.spoken:
                    self.speaker.say(step.spoken, cached=True)
                else:
                    step.spoken = self.speaker.say_line(verdict.speak_bank)
            else:
                step.status = "skipped"
                step.spoken = ""

            self.bus.publish(PLAN_PROGRESS, index=step.index, total=len(plan),
                             label=step.label, status=step.status)

        closing = plan.closing_line()
        self.speaker.say(closing, cached=False)
        spoken = " ".join([s.spoken for s in plan.steps if s.spoken] + [closing])
        self.stm.add_hive(spoken, sw.elapsed_ms)
        self.bus.log(f"plan: {len(plan.done)}/{len(plan)} done", "action")
        bad = plan.unfinished
        return Outcome(spoke=spoken, failed=bool(bad),
                       failure=(", ".join(f"{s.label or s.text[:32]} ({s.status})"
                                          for s in bad) if bad else ""))

    @staticmethod
    def _step_label(intent, text: str) -> str:
        target = intent.slots.get("target") or intent.slots.get("what") or ""
        if intent.name in ("launch", "launch_ask") and target:
            return f"opening {target}"
        if intent.name == "note_add":
            return "the note"
        if intent.name.startswith("media_"):
            return "playback"
        if intent.name == "system_query":
            return "the system check"
        if intent.name == "status_update":
            return "the status briefing"
        return text[:34]

    def _speak_refusal(self, verdict: Verdict, sw: Stopwatch) -> Outcome:
        """Say what is in the way, and what can be done instead.

        If the alternative is executable it becomes a pending confirmation, so
        "yes" runs it — the offer is real, not rhetorical.
        """
        refusal = verdict.refusal
        assert refusal is not None
        self.bus.set_state(State.SPEAKING)
        self._speech_started.set()
        text = refusal.render(self.cfg)
        sw.mark("first_audio")
        self.speaker.say(text, cached=False)
        self.stm.add_hive(text, sw.elapsed_ms)
        if refusal.action is not None:
            self.stm.pending_confirm = {"plan": refusal.action, "at": time.time()}
            self.bus.set_state(State.CONFIRMING)
        self.bus.log(f"refused: {refusal.reason[:60]}", "veto")
        return Outcome(spoke=text)

    # -- actions ------------------------------------------------------------
    def _execute_or_confirm(self, verdict: Verdict, sw: Stopwatch, style) -> Outcome:
        plan = verdict.action
        assert plan is not None

        if plan.permission is Permission.DENY:
            self.bus.set_state(State.ALERT)
            line = self.speaker.say_line("veto.destructive")
            self.stm.add_hive(line)
            self.bus.log(f"VETO {plan.kind}: {plan.label}", "veto")
            return Outcome(spoke=line)

        if plan.permission is Permission.CONFIRM:
            self.stm.pending_confirm = {"plan": plan, "at": time.time()}
            self.bus.set_state(State.CONFIRMING)
            self._speech_started.set()
            sw.mark("first_audio")
            # A verdict that composed its own question asks that one. The banks
            # are generic by design — "Confirm?" — and generic is exactly wrong
            # when the thing being confirmed is *which* application was meant.
            if verdict.speak_text:
                self.speaker.say(verdict.speak_text, cached=False)
                line = verdict.speak_text
            else:
                bank = {
                    "system": "confirm.system_query",
                    "shell": "confirm.shell_exec",
                }.get(plan.kind, "confirm.outside_vault")
                line = self.speaker.say_line(bank)
            self.stm.add_hive(line)
            return Outcome(spoke=line, action_label=plan.label)

        return self._run_plan(plan, verdict, sw)

    def _run_plan(self, plan: ActionPlan, verdict: Verdict, sw: Stopwatch) -> Outcome:
        self.bus.set_state(State.SPEAKING)
        filler = None
        if plan.kind in SLOW_ACTIONS:
            # Say what is being looked for while it is being looked for. The
            # line names the subject, so it is also the acknowledgement that the
            # request was understood — which matters more here than anywhere
            # else, because the answer is several seconds away and there is
            # nothing else on screen to confirm HIVE heard correctly.
            self.bus.set_state(State.THINKING)
            self._status(verdict.category)
            self._speech_started.clear()
            subject = verdict.facts.get("search_topic", "")
            tailored = (acknowledgement(self.cfg, "ack.searching",
                                        title_case=False, target=subject)
                        if subject else "")
            if tailored:
                threading.Thread(
                    target=self.speaker.synth, args=(tailored,),
                    kwargs={"use_cache": True}, name="hive-search-synth",
                    daemon=True).start()
            filler = self._arm_filler(sw, tailored=tailored, urgent=True)

        try:
            ok, detail = plan.run()
        finally:
            if filler is not None:
                filler.cancel()
                self.bus.set_state(State.SPEAKING)   # out of THINKING again
        sw.mark("execute")
        self._speech_started.set()

        if not ok:
            self.bus.set_state(State.ALERT)
            # A failure with something to say says it. "That failed" is fine for
            # a launch that did not launch; it is useless for a write-up that got
            # halfway, which is a case where the detail is the whole message.
            if detail and plan.kind in SPEAKS_ITS_RESULT:
                self.speaker.say(detail, cached=False)
                line = detail
            else:
                line = self.speaker.say_line("veto.error")
            self.stm.add_hive(line)
            self.bus.log(f"action failed: {plan.kind} {detail}", "error")
            return Outcome(spoke=line, failed=True,
                           failure=detail or f"the {plan.kind} action failed")

        self.actions_log.append(f"{plan.kind}: {plan.label}")
        sw.mark("first_audio")

        # These kinds return real content to speak; a launch only needs an ack.
        if plan.kind in SPEAKS_ITS_RESULT and detail:
            self.speaker.say(detail, cached=False)
            spoken = detail
        elif verdict.speak_text:
            self.speaker.say(verdict.speak_text, cached=True)
            spoken = verdict.speak_text
        else:
            spoken = self.speaker.say_line(verdict.speak_bank or "ack.done")

        # A job search is the clearest case for this: the answer is deterministic
        # and the useful half is what the vault remembers about the places on it.
        spoken = self._speak_aside(spoken)
        self.stm.add_hive(spoken, sw.elapsed_ms)
        return Outcome(spoke=spoken, action_label=plan.label)

    def _resolve_confirmation(self, intent: Intent, sw: Stopwatch) -> Outcome:
        pending = self.stm.pending_confirm or {}
        self.stm.pending_confirm = None
        plan: ActionPlan | None = pending.get("plan")

        if intent.name != "yes" or plan is None:
            self._speech_started.set()
            line = self.speaker.say_line("confirm.denied")
            self.stm.add_hive(line)
            return Outcome(spoke=line)

        # Consent was for this specific plan and nothing else. Run it directly,
        # bypassing the permission gate that already asked.
        self.bus.set_state(State.SPEAKING)
        ok, detail = plan.run()
        self._speech_started.set()
        sw.mark("first_audio")
        if not ok:
            line = self.speaker.say_line("veto.error")
            self.stm.add_hive(line)
            return Outcome(spoke=line, failed=True,
                           failure=detail or "the confirmed action failed")
        self.actions_log.append(f"{plan.kind}: {plan.label}")
        if plan.kind in SPEAKS_ITS_RESULT and detail:
            self.speaker.say(detail, cached=False)
            spoken = detail
        else:
            spoken = self.speaker.say_line("confirm.granted")
        self.stm.add_hive(spoken, sw.elapsed_ms)
        return Outcome(spoke=spoken, action_label=plan.label)

    # -- generation ---------------------------------------------------------
    def _generate(self, intent: Intent, verdict: Verdict, style, sw: Stopwatch) -> Outcome:
        # The only place a calibration is chosen. Everything upstream of here is
        # either an action or a canned line, and neither has a register to set.
        lens = None
        if self.calibrator is not None:
            lens = self.calibrator.select(intent, intent.cleaned or "")
            self.calibrator.note_turn()
            sw.mark("calibrate")

        escalate = self.llm.should_escalate(intent.cleaned, verdict.category)
        if lens is not None and lens.escalate and not lens.is_base:
            # Both profiles earn the heavy model: a wrong technical answer costs
            # more than five gigabytes of VRAM, and VOSS's register is the one
            # place where the answer is the whole point.
            escalate = True

        # Pasted material — see clip.py. Held across turns, so this is what
        # makes the fourth question about one paste work as well as the first.
        pasted = self.clippings.context_block() if self.clippings is not None else ""
        if pasted and bool(self.cfg.get("clip.escalate", True)):
            # The question is short — "what does this mean?" — and the escalation
            # heuristic reads question length, so a paste would otherwise route
            # four hundred words of somebody else's argument to the 1.7B. The
            # work is in the source, not in the sentence asking about it.
            escalate = True

        self.bus.set_state(State.THINKING)
        self._status(verdict.category, escalated=escalate)

        directive = self._aside.directive if self._aside is not None else ""
        if lens is not None and not lens.is_base:
            verdict.facts.update(lens.facts)
            if lens.budget:
                style.sentence_budget = lens.budget
        system = self.subjective.system_prompt(
            verdict.facts, style, self.stm, directive=directive,
            calibration=lens.spine if lens is not None else "")
        messages = self.stm.messages(system, extra_context=pasted)
        sw.mark("prompt")

        # Compose the holding line from what was actually asked, and render it
        # while the model is still thinking. Two things have to be true for it to
        # cost nothing: the synthesis runs concurrently with generation, and the
        # speaker caches by text — so the timer below finds finished audio.
        tailored = holding_line(self.cfg, intent.cleaned or "")
        if tailored:
            threading.Thread(
                target=self.speaker.synth, args=(tailored,),
                kwargs={"use_cache": True}, name="hive-filler-synth",
                daemon=True).start()

        self._speech_started.clear()
        filler = self._arm_filler(sw, urgent=escalate, tailored=tailored)

        stream = self.llm.stream(messages, heavy=escalate)
        veto_holder: dict[str, str | None] = {"reason": None}
        guarded = self._verified(stream, verdict.facts, veto_holder, sw)

        marked = {"done": False}

        def first_audio() -> None:
            if not marked["done"]:
                marked["done"] = True
                sw.mark("answer_audio")
                if not any(n == "first_audio" for n, _ in sw.marks):
                    sw.mark("first_audio")   # no filler fired; this was the first sound

        try:
            spoken = self.speaker.say_stream(guarded, on_first_audio=first_audio)
        finally:
            filler.cancel()

        if veto_holder["reason"]:
            # Objective overruled generation mid-flight. Say so rather than
            # silently truncating — an answer that stops dead is worse than one
            # that admits it was wrong.
            self.bus.set_state(State.ALERT)
            self.bus.log(f"VETO claim: {veto_holder['reason']}", "veto")
            correction = self.speaker.say_line("veto.ungrounded")
            spoken = f"{spoken} {correction}".strip()

        spoken = clean_for_speech(spoken)
        # The model gets first refusal on the follow-up: it has the whole answer
        # in front of it and can ask something better than a template can. The
        # line built during evaluation is only spoken if it didn't.
        spoken = self._speak_aside(spoken, lens)
        self.stm.add_hive(spoken, sw.elapsed_ms,
                          calibration=lens.key if lens is not None else "hive")
        self._last_escalated = escalate
        if escalate:
            # Give the VRAM back promptly; the visualizer and TTS both want it.
            threading.Timer(2.0, self.llm.release_heavy).start()
        return Outcome(spoke=spoken, category=verdict.category, escalated=escalate)

    def _verified(self, stream: Iterator[str], facts: dict,
                  holder: dict, sw: Stopwatch) -> Iterator[str]:
        """Pass tokens through, checking completed clauses against ground truth.

        Verification has to happen inline because the response is streamed — by
        the time a full answer exists, half of it has already been spoken. So each
        clause is adjudicated as it completes, and generation is cut the moment
        one fails.
        """
        buf = ""
        first = True
        for chunk in stream:
            if first:
                sw.mark("first_token")
                self._speech_started.set()
                first = False
            buf += chunk
            # Check on sentence boundaries only; partial clauses produce false hits.
            if any(p in chunk for p in ".!?"):
                reason = self.objective.verify(buf, facts)
                if reason:
                    holder["reason"] = reason
                    return
            yield chunk
        if buf:
            reason = self.objective.verify(buf, facts)
            if reason:
                holder["reason"] = reason
