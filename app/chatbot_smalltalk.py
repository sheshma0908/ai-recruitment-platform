"""
Conversation layer for the candidate chatbot (small talk).

Handles the short, casual messages people send in a chat window, so they never
fall through to the career model and get "I'm not sure about that":

    greetings        hi, hii, hlo, helo, hey, hai, yo, namaste, good morning, gm ...
    acknowledgements ok, okay, okk, k, kk, fine, alright, cool, got it, sure, hmm ...
    thanks           thanks, thank you, thanku, tq, thx, ty, tnx ...
    goodbye          bye, byee, tata, see you, cya, good night ...
    wellbeing        how are you, how r u, wassup ...
    identity / help  who are you, what can you do, help, menu ...
    yes / no         yes, yeah, no, nope, no thanks ...
    praise           great, awesome, nice, good bot ...
    complaints       wrong, not helpful, useless, you didn't understand ...

Matching is deliberately strict: the WHOLE message must be small talk (after
removing filler like "bot", "there", "sir", punctuation and emoji). Longer
messages that only START with a greeting ("hi, show me python jobs") are split:
the greeting is removed and the rest is answered normally.

Typos and stretched words are tolerated: "hiiii" -> "hi", "okkk" -> "ok",
"thnks" -> "thanks", "helo" -> "hello".
"""

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple

# ------------------------------------------------------------
# Vocabulary (lowercase, already "squashed": no letter repeated 3+ times)
# ------------------------------------------------------------

GREETINGS = {
    "hi", "hii", "hy", "hyy", "hello", "helo", "hllo", "hlo", "hlw", "hey", "heyy", "hai", "hay", "hiya",
    "yo", "hola", "namaste", "namaskar", "vanakkam", "salaam", "greetings", "howdy",
    "good morning", "good afternoon", "good evening", "gm", "morning", "evening",
    "hi there", "hello there", "hey there",
}

ACKS = {
    "ok", "okk", "okay", "okey", "okie", "okies", "oky", "k", "kk", "kay", "fine", "alright",
    "all right", "allright", "cool", "got it", "gotcha", "sure", "noted", "understood", "i see",
    "oh", "oh ok", "oh okay", "ah", "ah ok", "hmm", "hm", "mm", "right", "ohh", "achha", "acha",
    "theek hai", "thik hai", "ok fine", "okay fine", "ok sure", "okay sure", "fair enough",
    "makes sense", "i understand", "understand", "ok got it", "okay got it", "seen",
}

THANKS = {
    "thanks", "thank you", "thankyou", "thanku", "thank u", "thanx", "thnx", "thnks", "thx", "tnx",
    "ty", "tq", "tqsm", "tysm", "thanks a lot", "thank you so much", "thanks so much",
    "many thanks", "much appreciated", "appreciated", "ok thanks", "okay thanks", "ok thank you",
    "okay thank you", "thanks bro", "dhanyavad", "dhanyavaad", "shukriya",
}

BYES = {
    "bye", "goodbye", "good bye", "bye bye", "see you", "see ya", "cya", "see you later",
    "tata", "take care", "good night", "gn", "later", "catch you later", "ok bye", "okay bye",
    "thanks bye", "thank you bye", "bye thanks", "i am done", "im done", "that is all",
    "thats all", "nothing else", "no more questions",
}

WELLBEING = {
    "how are you", "how r u", "how are u", "how r you", "hows it going", "how is it going",
    "how do you do", "wassup", "whats up", "sup", "what s up", "how are things", "how you doing",
}

IDENTITY = {
    "who are you", "who r u", "what are you", "your name", "what is your name", "whats your name",
    "are you a bot", "are you human", "are you real", "are you ai", "introduce yourself",
}

HELP = {
    "help", "help me", "menu", "options", "what can you do", "what do you do",
    "how can you help", "how can you help me", "what can i ask", "what can i ask you",
    "what should i ask", "start", "start over", "restart", "main menu", "show menu", "need help",
    "i need help", "can you help", "can you help me",
}

YES = {"yes", "yeah", "yea", "yep", "yup", "ya", "haan", "han", "sure thing", "of course", "y"}
NO = {"no", "nope", "nah", "no thanks", "no thank you", "not now", "nothing", "na", "n", "no need"}

PRAISE = {
    "great", "awesome", "nice", "good", "very good", "perfect", "excellent", "amazing", "wow",
    "super", "good bot", "nice bot", "you are great", "you are awesome", "good job", "well done",
    "helpful", "very helpful", "that helps", "that helped", "nice one", "good one", "love it",
}

COMPLAINTS = {
    "wrong", "wrong answer", "not helpful", "useless", "bad", "bad bot", "not correct",
    "incorrect", "thats wrong", "that is wrong", "you are wrong", "not what i asked",
    "that is not what i asked", "you didnt understand", "you did not understand",
    "not working", "doesnt help", "does not help", "same answer", "again same answer",
    "you are repeating", "stupid", "what", "huh", "confused", "i dont understand",
    "i did not understand", "didnt get it", "didnt understand",
}

POLITE = {
    "welcome", "wel come", "you are welcome", "youre welcome", "your welcome", "ur welcome",
    "most welcome", "no problem", "np", "no worries", "my pleasure", "pleasure", "cheers",
    "nice to meet you", "pleased to meet you", "glad to meet you", "nice meeting you",
    "nice talking to you", "same to you", "you too", "same here", "good to know", "ok good",
    "okay good", "good to hear", "nice to know",
}

APOLOGY = {"sorry", "so sorry", "my bad", "my mistake", "excuse me", "pardon", "pardon me",
           "oops", "sorry for that", "sorry my mistake", "i am sorry", "im sorry"}

# Extra words allowed after a THANK-YOU only: "thank you for the jobs list", "thanks for the roadmap"
THANKS_TAIL = {"jobs", "job", "list", "listing", "listings", "suggestions", "suggestion", "tips",
               "roadmap", "advice", "explanation", "openings", "career", "information", "update",
               "updates", "quick", "detailed", "clear", "useful", "steps", "guide"}

BYES |= {"have a nice day", "have a good day", "have a great day", "have a good one",
         "good day", "talk to you later", "ttyl", "see you soon", "bye for now", "signing off"}
THANKS |= {"thanks for helping", "thank you for helping", "thanks for your help",
           "thank you for your help", "thanks for the help", "thank you for the help",
           "thanks for the info", "thank you for the information", "thank you very much",
           "thanks a ton", "thanks again", "thank you again", "appreciate it", "i appreciate it",
           "really appreciate it", "that was helpful", "it was helpful", "you helped me",
           "thank you so much for helping", "great help", "big help"}

# Words that may FOLLOW a small-talk phrase without adding a real question:
# "thank you for helping me", "thanks for the info today", "bye take care sir"
TAIL = {
    "for", "the", "your", "ur", "help", "helping", "helped", "info", "information", "details",
    "support", "assistance", "answer", "answers", "reply", "replies", "response", "responses",
    "guidance", "it", "that", "this", "me", "us", "a", "lot", "ton", "so", "much", "very",
    "really", "again", "all", "today", "now", "everything", "time", "and", "too", "also",
    "bot", "chatbot", "assistant", "sir", "madam", "mam", "bro", "buddy", "dear", "friend",
    "you", "u", "of", "with", "my", "doubts", "doubt", "queries", "query", "solving", "clearing",
    "explaining", "replying", "responding", "being", "here", "there", "patience", "great",
    "nice", "good", "awesome", "helpful", "kind", "take", "care", "day", "have",
}

INTENTS = [
    ("greeting", GREETINGS), ("thanks", THANKS), ("bye", BYES), ("wellbeing", WELLBEING),
    ("identity", IDENTITY), ("help", HELP), ("ack", ACKS), ("yes", YES), ("no", NO),
    ("praise", PRAISE), ("polite", POLITE), ("apology", APOLOGY),
    ("complaint", COMPLAINTS),
]

# Words that can surround small talk without changing its meaning.
FILLER = {
    "bot", "chatbot", "assistant", "there", "sir", "madam", "mam", "maam", "bro", "buddy",
    "dear", "friend", "again", "very", "much", "so", "please", "pls", "plz", "just",
    "talentai", "team", "all", "everyone", "guys", "dude", "man", "boss",
}

# Leading greetings / acks that may prefix a real question.
# ("fine", "great", "morning" etc. are left out: "fine arts jobs", "evening shift jobs".)
PREFIX_WORDS = ((GREETINGS - {"morning", "evening", "hay", "greetings"}) |
                {"ok", "okay", "okk", "k", "kk", "alright", "cool", "hmm", "so", "well",
                 "thanks", "thank you", "please", "pls", "plz", "ok so", "okay so", "ok then",
                 "okay then", "ok now", "okay now"})

_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️‍]+")
_THUMBS = re.compile(r"[\U0001F44D\U0001F44C✅\U0001F64F]")


def _squash(word: str) -> str:
    """'hiiii' -> 'hii', 'okkkk' -> 'okk', 'byeee' -> 'byee' (max 2 repeats)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", word)


def clean(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("'", "").replace("’", "")
    text = _EMOJI.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [_squash(w) for w in text.split()]
    return " ".join(words)


def _strip_filler(text: str) -> str:
    words = [w for w in text.split() if w not in FILLER]
    return " ".join(words)


def _variants(phrase: str):
    """A phrase, plus the version with doubled letters reduced to one ('hii' -> 'hi')."""
    yield phrase
    single = re.sub(r"(.)\1+", r"\1", phrase)
    if single != phrase:
        yield single


def _lookup(phrase: str) -> Optional[str]:
    for p in _variants(phrase):
        for intent, vocab in INTENTS:
            if p in vocab:
                return intent
    return None


def _fuzzy(phrase: str) -> Optional[str]:
    """Typo tolerance for single words / short phrases ('thnaks', 'helllo', 'goodbey')."""
    if len(phrase) < 4 or len(phrase.split()) > 3:
        return None
    best, best_intent = 0.0, None
    for intent, vocab in INTENTS:
        if intent in ("yes", "no", "complaint"):
            continue                       # too risky to guess
        for v in vocab:
            if len(v) < 4 or abs(len(v) - len(phrase)) > 2:
                continue
            s = SequenceMatcher(None, phrase, v).ratio()
            if s > best:
                best, best_intent = s, intent
    return best_intent if best >= 0.84 else None


def detect_small_talk(message: str) -> Optional[str]:
    """Return the small-talk intent if the WHOLE message is small talk, else None."""
    raw = str(message or "").strip()
    if not raw:
        return None
    if _THUMBS.search(raw) and not re.search(r"[a-zA-Z]", raw):
        return "ack"
    text = clean(raw)
    if not text:
        return "ack" if _EMOJI.search(raw) else None
    if len(text.split()) > 10:
        return None
    for candidate in (text, _strip_filler(text)):
        if not candidate:
            continue
        intent = _lookup(candidate)
        if intent:
            return intent
    # Two small-talk parts, e.g. "ok thanks", "hi how are you", "great thank you"
    words = _strip_filler(text).split()
    for i in range(1, len(words)):
        a, b = _lookup(" ".join(words[:i])), _lookup(" ".join(words[i:]))
        if a and b:
            for pref in ("bye", "thanks", "wellbeing", "identity", "help", "complaint"):
                if pref in (a, b):
                    return pref
            return b
    # Small-talk phrase followed only by "tail" words: "thank you for helping me",
    # "welcome sir", "bye take care", "hello good morning"
    for i in range(len(words) - 1, 0, -1):
        head = _lookup(" ".join(words[:i]))
        if head and all(w in TAIL or _lookup(w) or (head == "thanks" and w in THANKS_TAIL)
                        for w in words[i:]):
            return head
    # Small-talk phrase PRECEDED only by tail words: "ok so thank you", "and thanks"
    for i in range(1, len(words)):
        head = _lookup(" ".join(words[i:]))
        if head and all(w in TAIL or w in {"ok", "okay", "so", "and", "well"} for w in words[:i]):
            return head
    return _fuzzy(_strip_filler(text))


def split_leading_small_talk(message: str) -> Tuple[Optional[str], str]:
    """
    'hi, show me python jobs'   -> ('greeting', 'show me python jobs')
    'ok what about salary'      -> ('ack', 'what about salary')
    'python jobs'               -> (None, 'python jobs')
    """
    original = str(message or "").strip()
    # "thank you for helping, show me python jobs" -> ('thanks', 'show me python jobs')
    clause = re.match(r"^\s*([^,.!?;]{1,60})[,.!?;]+\s*(.+)$", original)
    if clause:
        head_intent = detect_small_talk(clause.group(1))
        if head_intent and len(clean(clause.group(2))) >= 2:
            return head_intent, clause.group(2).strip()
    m = re.match(r"^\s*([A-Za-z' ]{1,25}?)[\s,.!:;\-]+(.+)$", original)
    if not m:
        return None, original
    words = original.split()
    for n in (3, 2, 1):
        if len(words) <= n:
            continue
        head = clean(" ".join(words[:n]))
        head = _strip_filler(head) or head
        if any(v in PREFIX_WORDS for v in _variants(head)):
            rest = " ".join(words[n:]).lstrip(" ,.!:;-")
            if len(clean(rest)) >= 2:
                return _lookup(head) or "ack", rest
    return None, original


# ------------------------------------------------------------
# Replies
# ------------------------------------------------------------

CAPABILITIES = (
    "Here's what I can help you with:\n"
    "1) Current job openings (search by skill, city, experience or job type)\n"
    "2) How to apply and upload your resume\n"
    "3) Checking your application and interview status\n"
    "4) Career guidance for 50+ job roles: what the role involves, skills, tools, salary, "
    "career growth and a step-by-step roadmap to get started\n\n"
    "Reply with a number, or just ask, e.g. \"python jobs in Hyderabad\", "
    "\"how to become a data analyst\" or \"data analyst vs data scientist\"."
)


def small_talk_reply(intent: str, last_bot: Optional[str] = None, has_jobs: bool = False) -> str:
    last = (last_bot or "").lower()
    listed_jobs = bool(re.search(r"job listings?|i found \d+|matches? for|current openings", last))
    gave_career = "you can also ask" in last or "step-by-step roadmap**" in last

    if intent == "greeting":
        return ("\U0001F44B Hello! I'm your career assistant.\n\n" + CAPABILITIES)

    if intent == "wellbeing":
        return ("I'm doing great, thanks for asking! \U0001F60A How can I help you today? "
                "I can show current job openings, help you apply, or answer career questions "
                "about any job role.")

    if intent == "identity":
        return ("I'm the candidate career assistant for this job portal. " + CAPABILITIES)

    if intent == "help":
        return CAPABILITIES

    if intent == "thanks":
        return "You're welcome! \U0001F60A Is there anything else I can help you with?"

    if intent == "bye":
        return "Goodbye, and all the best with your job search! Come back anytime. \U0001F44B"

    if intent == "polite":
        return ("\U0001F60A Thank you! Is there anything I can help you with? You can ask about "
                "current job openings, how to apply, or career questions about any job role.")

    if intent == "apology":
        return ("No problem at all! \U0001F60A What would you like to know? You can ask about "
                "job openings, applying, or career questions about any job role.")

    if intent == "praise":
        return "Glad I could help! \U0001F60A Anything else you'd like to know?"

    if intent == "complaint":
        return ("Sorry about that! Could you rephrase your question with a little more detail? "
                "For example:\n- \"python jobs in Hyderabad\"\n- \"details of data analyst job\"\n"
                "- \"skills needed for a data analyst\"\n- \"how to become a web developer\"\n"
                "- \"how do I apply for a job?\"")

    if intent == "no":
        return ("Alright! If you need anything later, just type your question or say \"menu\" "
                "to see what I can do.")

    if intent == "yes":
        if listed_jobs:
            return ("Great! Tell me which job you're interested in, e.g. \"details of "
                    "<job title>\", or filter the list like \"jobs in Hyderabad\".")
        return "Great! What would you like to know? You can say \"menu\" to see all options."

    # ack ("ok", "okay", "got it" ...): keep the conversation going based on context
    if listed_jobs:
        return ("\U0001F44D Want more details on any of these? Just say \"details of <job title>\", "
                "or filter them, e.g. \"jobs in Hyderabad\" or \"jobs for freshers\".")
    if gave_career:
        return ("\U0001F44D Feel free to ask about another topic for this role, or about a different "
                "job role. Anything else?")
    if has_jobs:
        return ("\U0001F44D Anything else I can help with? You can ask about current job openings, "
                "how to apply, or career questions about any job role.")
    return "\U0001F44D Anything else I can help you with?"
