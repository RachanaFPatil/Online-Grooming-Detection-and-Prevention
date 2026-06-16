from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit, join_room
import re, json, smtplib, random, threading, requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ════════════════════════════════════════════════
#  LOAD .env
#  pip install python-dotenv
#  cp .env.example .env  then fill in your values
# ════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[SafeGuard] .env loaded")
except ImportError:
    print("[SafeGuard] python-dotenv not installed — pip install python-dotenv")

# ════════════════════════════════════════════════
#  CONFIG — all from environment, nothing hardcoded
# ════════════════════════════════════════════════
SMTP_EMAIL    = os.environ.get("SMTP_EMAIL",    "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
PARENT_EMAIL  = os.environ.get("PARENT_EMAIL",  "")
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY",  "")
SECRET_KEY    = os.environ.get("SECRET_KEY",    "safeguard-dev-secret")

GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

_missing = []
if not GROQ_API_KEY:   _missing.append("GROQ_API_KEY  (required — AI sandbox won't work)")
if not SMTP_EMAIL:     _missing.append("SMTP_EMAIL    (optional — for email alerts)")
if not SMTP_PASSWORD:  _missing.append("SMTP_PASSWORD (optional — for email alerts)")
if not PARENT_EMAIL:   _missing.append("PARENT_EMAIL  (optional — for email alerts)")

if _missing:
    print("\n[SafeGuard] Missing env vars:")
    for m in _missing: print(f"  {m}")
    if not GROQ_API_KEY:
        print("[SafeGuard] Get a free Groq key at https://console.groq.com\n")
else:
    print("[SafeGuard] All env vars loaded")

# ════════════════════════════════════════════════
#  FLASK + SOCKETIO
# ════════════════════════════════════════════════
app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ════════════════════════════════════════════════
#  ROOM STATE
# ════════════════════════════════════════════════
ROOM              = "safeguard-room"
users             = {}
room_risk         = 0.0
room_history      = []
flagged_messages  = []
room_alerted      = False
room_sandbox      = False
sandbox_msg_count = 0
WIND_DOWN_AFTER   = 10

child_sid    = None
predator_sid = None

# ════════════════════════════════════════════════
#  LOAD ML MODEL
# ════════════════════════════════════════════════
print("[SafeGuard] Loading ML model...")
from detector import load_model, analyze_message as ml_analyze
load_model()
print("[SafeGuard] ML model loaded")

# ════════════════════════════════════════════════
#  THRESHOLDS
#
#  SCORE_THRESHOLD      = 0.65
#    "hi", "how old are you" score ~0.3–0.6 → contribute 0
#    to room_risk. Only clearly grooming messages push it up.
#
#  FLAG_THRESHOLD       = 0.72
#    Must exceed to show red badge in child's chat UI.
#
#  SANDBOX_RISK_TRIGGER = 0.60
#    Sandbox activates once room_risk builds to 0.60 through
#    a pattern of grooming: "meet in secret", "home alone",
#    "come over" etc. A single casual question can't do this.
#
#  EXPLICIT_SANDBOX_SCORE = 0.88
#    A single truly explicit message bypasses the window and
#    triggers sandbox instantly (sexual content / direct abuse).
#
#  EMAIL_THRESHOLD = 1.0
#    Email fires ONLY when room_risk == 1.0 (maximum certainty).
# ════════════════════════════════════════════════
SCORE_THRESHOLD        = 0.65
FLAG_THRESHOLD         = 0.72
SANDBOX_RISK_TRIGGER   = 0.60
EXPLICIT_SANDBOX_SCORE = 0.88
EMAIL_THRESHOLD        = 1.0
WINDOW_SIZE            = 5
predator_scores        = []


def score_message(text, image_path=None):
    result    = ml_analyze(text, image_path=image_path)
    raw_score = result["score"]
    breakdown = result["breakdown"]
    level     = result["risk_level"]

    contributing = raw_score if raw_score > SCORE_THRESHOLD else 0.0

    flags = []
    if raw_score > FLAG_THRESHOLD:
        if breakdown["text_score"]  > 0.60: flags.append("Suspicious language")
        if breakdown["emoji_score"] > 0.55: flags.append("Suspicious emoji")
        if breakdown["image_score"] > 0.40: flags.append("Explicit image content")
        if level == "CRITICAL":             flags.append("Critical risk")
        elif level == "HIGH_RISK":          flags.append("High risk content")

    print(f"[ML] raw={raw_score:.3f} contrib={contributing:.3f} flagged={len(flags)>0} level={level}")
    return round(contributing, 3), flags, breakdown, round(raw_score, 3)


def update_risk(contributing_score):
    global room_risk, predator_scores
    predator_scores.append(contributing_score)
    if len(predator_scores) > WINDOW_SIZE:
        predator_scores = predator_scores[-WINDOW_SIZE:]
    if all(s == 0.0 for s in predator_scores):
        room_risk = 0.0
        return room_risk
    weights      = list(range(1, len(predator_scores) + 1))
    weighted_sum = sum(s * w for s, w in zip(predator_scores, weights))
    room_risk    = round(min(weighted_sum / sum(weights), 1.0), 3)
    return room_risk


def should_sandbox(raw_score, risk_val):
    """
    Triggers when:
      1. room_risk >= 0.60 — sustained grooming pattern
         e.g. "can I meet you in secret", "are you home alone",
         "come over", "don't tell your parents"
      2. raw_score >= 0.88 — single deeply explicit message
         (sexual content, direct exploitation)
    "How old are you", "where do you live" alone CANNOT trigger this.
    """
    return risk_val >= SANDBOX_RISK_TRIGGER or raw_score >= EXPLICIT_SANDBOX_SCORE


def should_email(risk_val):
    """Email fires only at room_risk == 1.0."""
    return risk_val >= EMAIL_THRESHOLD


def get_stage(risk):
    if risk < 0.20: return "Friendly Interaction"
    if risk < 0.40: return "Personal Info Requests"
    if risk < 0.60: return "Isolation & Trust Building"
    if risk < 0.80: return "Emotional Manipulation"
    return "Exploitation Attempt"


# ════════════════════════════════════════════════
#  AI PERSONAS
#
#  ai_as_child(predator_msg)
#    → Called when predator sends a message while sandbox is active
#    → AI replies TO the predator, pretending to be the child
#    → Responses are avoidant, deflecting, cautious — no slang
#
#  ai_as_predator(child_msg)
#    → Called when child sends a message while sandbox is active
#    → AI replies TO the child, pretending to be the predator
#    → Warm, friendly, completely normal topics — winding down
# ════════════════════════════════════════════════

CHILD_SYSTEM = """You are playing the role of a cautious 14-year-old teenager in an online chat.
Keep replies very short — 1 to 2 sentences, natural and brief like real texting. No slang filler words.

Respond specifically to what the other person just said:
- asking where you live / city / address → "My parents always say not to share that with people online." or "I don't really share that."
- asking how old you are → "Why are you asking?" or "Why does that matter?"
- asking if you're alone / home alone → "No, my mom is right here." or "My whole family is home."
- asking for a photo / pic / selfie → "I don't send photos to people I haven't met." or "That's kind of odd."
- telling you to keep it secret / not tell parents → "I don't keep things from my parents." or "That's a weird thing to ask."
- asking to meet up / come over / meet in secret → "I don't meet people I've only talked to online." or "My parents would never allow that."
- asking to video call → "I don't do video calls with people I haven't met in person."
- saying I love you / asking to be girlfriend/boyfriend → "We just met online, that doesn't really make sense."
- anything sexual or explicit → "That's really inappropriate, I don't want to talk about this." then change topic completely.
- asking for your number / snapchat / instagram → "I don't share my contact details with strangers online."
- saying trust me / I'm safe / I won't hurt you → "I don't really know you though."
For anything else, reply naturally as a cautious teenager would. Never reveal you are an AI."""

PREDATOR_SYSTEM = """You are a friendly, completely wholesome person having a totally normal conversation with a teenager.
Reply warmly and naturally — like a safe, friendly chat between two people.
Only talk about everyday topics: school, hobbies, music, games, food, movies, sports, weekend plans.
Keep replies to 1 to 2 sentences. Be upbeat and age-appropriate at all times.
Never say anything romantic, personal, suggestive, or uncomfortable.
Never reveal you are an AI."""


def call_groq(system, user_msg):
    if not GROQ_API_KEY:
        print("[GROQ] Skipped — GROQ_API_KEY not set in .env")
        return None
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model":       GROQ_MODEL,
                "max_tokens":  80,
                "temperature": 0.75,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            },
            timeout=10,
        )
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[GROQ] {reply}")
        return reply
    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return None


def build_history():
    lines = ""
    for msg in room_history[-20:]:
        label = "Teen" if msg["role"] == "child" else "Other person"
        lines += f"{label}: {msg['text']}\n"
    return lines


def ai_as_child(predator_msg):
    """AI replies TO the predator, AS the child — avoidant and deflecting."""
    prompt = (
        f"Conversation so far:\n{build_history()}\n"
        f"Other person just said: \"{predator_msg}\"\n"
        f"Reply as the cautious teenager. Be specific to what they said. "
        f"1-2 short sentences, no filler words."
    )
    return call_groq(CHILD_SYSTEM, prompt) or smart_fallback_child(predator_msg)


def ai_as_predator(child_msg):
    """AI replies TO the child, AS the predator — friendly and normal, winding down."""
    prompt = (
        f"Conversation so far:\n{build_history()}\n"
        f"Teen just said: \"{child_msg}\"\n"
        f"Reply warmly and naturally to what they said. "
        f"Keep it friendly and completely normal. 1-2 sentences."
    )
    return call_groq(PREDATOR_SYSTEM, prompt) or random.choice([
        "Yeah that makes total sense!", "Oh cool, I didn't know that.",
        "That sounds fun actually.", "Nice, good to know.",
        "Ha fair enough.", "Makes sense to me.",
    ])


def smart_fallback_child(text):
    """Fallback replies for the child persona when Groq is unavailable."""
    t = text.lower()
    if re.search(r"how old|your age|age are you", t):
        return random.choice(["Why are you asking?", "Why does that matter?"])
    if re.search(r"where.*live|your address|which city|near me", t):
        return random.choice(["My parents said not to share that online.", "I don't really share where I live."])
    if re.search(r"home alone|are you alone|parents home|anyone home", t):
        return random.choice(["No, my mom is right here.", "My whole family is home."])
    if re.search(r"send.*pic|send.*photo|selfie|show me|picture of you", t):
        return random.choice(["I don't send photos to people I haven't met.", "That's kind of an odd thing to ask."])
    if re.search(r"don'?t tell|keep.*secret|delete|just between us", t):
        return random.choice(["I don't keep things from my parents.", "That's a strange thing to ask."])
    if re.search(r"meet up|come over|pick you up|meet in person|meet tomorrow|meet in secret", t):
        return random.choice(["I don't meet people from online.", "My parents would never allow that."])
    if re.search(r"video call|facetime|on camera", t):
        return random.choice(["I don't video call people I haven't met.", "Not really comfortable with that."])
    if re.search(r"i love you|be my girlfriend|be my boyfriend|date me", t):
        return random.choice(["We literally just met online, that doesn't make sense.", "That's really unexpected."])
    if re.search(r"sexy|naked|undress|touch|explicit|nude", t):
        return random.choice(["That's really inappropriate.", "I don't want to talk about that."])
    if re.search(r"your number|whatsapp|snapchat|instagram", t):
        return random.choice(["I don't share my contact info with strangers.", "I don't give that out to people online."])
    return random.choice(["I'm not sure about that.", "That's kind of an odd thing to say.", "Okay…"])


# ════════════════════════════════════════════════
#  WIND-DOWN
# ════════════════════════════════════════════════
def run_wind_down():
    import time
    print("[SANDBOX] Wind-down starting...")
    child_user    = dict(users.get(child_sid,    {}))
    predator_user = dict(users.get(predator_sid, {}))

    # AI (as child) wraps up with predator
    for i, msg in enumerate(["Hey I have to go now.", "Take care, bye."]):
        time.sleep(3 + i * 3)
        if predator_sid:
            socketio.emit("message", {
                "name": child_user.get("name",""), "avatar": child_user.get("avatar","🐼"),
                "text": msg, "sid": child_sid, "role": "child",
                "risk": room_risk, "msg_score": 0,
                "stage": get_stage(room_risk), "flags": [], "flagged": False,
            }, room=predator_sid)

    # AI (as predator) wraps up with child
    for i, msg in enumerate(["Alright, talk later!", "Bye."]):
        time.sleep(2 + i * 2)
        if child_sid:
            socketio.emit("message", {
                "name": predator_user.get("name",""), "avatar": predator_user.get("avatar","🐼"),
                "text": msg, "sid": predator_sid, "role": "predator",
                "risk": room_risk, "msg_score": 0,
                "stage": get_stage(room_risk), "flags": [], "flagged": False,
            }, room=child_sid)

    time.sleep(3)
    socketio.emit("chat_terminated", {"risk": room_risk, "stage": get_stage(room_risk)}, room=ROOM)
    print("[SANDBOX] Chat terminated.")


# ════════════════════════════════════════════════
#  EMAIL ALERT
# ════════════════════════════════════════════════
def send_alert(risk, stage):
    if not all([SMTP_EMAIL, SMTP_PASSWORD, PARENT_EMAIL]):
        print("[EMAIL] Skipped — SMTP credentials missing from .env")
        return
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "SafeGuard CRITICAL — Child at Maximum Risk"
        msg["From"]    = SMTP_EMAIL
        msg["To"]      = PARENT_EMAIL
        body = f"""
CRITICAL: Maximum risk level (100%) detected on your child's device.

Risk Level : {round(risk * 100)}%
Stage      : {stage}
Time       : {datetime.now().strftime('%Y-%m-%d %H:%M')}

SafeGuard AI has intervened and is protecting your child.

Please act immediately:
CHILDLINE            : 1098
Cybercrime Portal    : cybercrime.gov.in
National Cyber Crime : 1930

— SafeGuard AI
        """
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_EMAIL, SMTP_PASSWORD)
            s.sendmail(SMTP_EMAIL, PARENT_EMAIL, msg.as_string())
        print(f"[EMAIL] Alert sent to {PARENT_EMAIL}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


# ════════════════════════════════════════════════
#  EVIDENCE
# ════════════════════════════════════════════════
def save_evidence():
    path = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump({
            "generated_at":     datetime.now().isoformat(),
            "risk_score":       room_risk,
            "stage":            get_stage(room_risk),
            "conversation":     room_history,
            "flagged_messages": flagged_messages,
        }, f, indent=2)
    print(f"[EVIDENCE] Saved -> {path}")


# ════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory("templates", "chat.html")

@app.route("/dashboard")
def dashboard():
    return send_from_directory("templates", "dashboard.html")

@app.route("/api/dashboard-data")
def dashboard_data():
    return jsonify({
        "room_risk":        round(room_risk, 3),
        "stage":            get_stage(room_risk),
        "sandbox_active":   room_sandbox,
        "total_messages":   len(room_history),
        "flagged_count":    len(flagged_messages),
        "flagged_messages": flagged_messages,
        "conversation":     room_history[-30:],
        "predator_scores":  predator_scores,
        "users":            [{"name": u["name"], "role": u["role"], "avatar": u["avatar"]} for u in users.values()],
        "generated_at":     datetime.now().isoformat(),
    })


# ════════════════════════════════════════════════
#  SOCKET EVENTS
# ════════════════════════════════════════════════
@socketio.on("join")
def on_join(data):
    global child_sid, predator_sid
    sid    = request.sid
    name   = data.get("name", "User")
    avatar = data.get("avatar", "🐼")
    if child_sid is None:
        role = "child";    child_sid    = sid
    else:
        role = "predator"; predator_sid = sid
    users[sid] = {"name": name, "avatar": avatar, "role": role}
    join_room(ROOM)
    count    = len(users)
    existing = [{"name": u["name"], "avatar": u["avatar"]} for s, u in users.items() if s != sid]
    emit("user_joined", {"name": name, "avatar": avatar, "count": count, "role": role}, to=ROOM, skip_sid=sid)
    emit("room_info",   {"count": count, "existing_users": existing, "my_role": role})
    print(f"[+] {name} joined as {role}")


@socketio.on("message")
def on_message(data):
    global room_alerted, room_sandbox, sandbox_msg_count
    sid  = request.sid
    user = users.get(sid, {})
    name = user.get("name", "?")
    role = user.get("role", "unknown")
    text = data.get("text", "")

    contributing, flags, breakdown, raw_score = score_message(text)
    risk  = update_risk(contributing) if role == "predator" else room_risk
    stage = get_stage(risk)

    entry = {
        "sender": name, "role": role, "text": text,
        "score":  contributing, "raw_score": raw_score,
        "flags":  flags, "breakdown": breakdown,
        "ts":     datetime.now().isoformat(),
    }
    room_history.append(entry)
    if flags and role == "predator":
        flagged_messages.append({**entry, "room_risk": risk, "stage": stage})

    payload = {
        "name": name, "avatar": user.get("avatar","🐼"),
        "text": text, "sid": sid, "risk": risk,
        "msg_score": contributing, "stage": stage,
        "flags": flags, "role": role,
    }

    # Snapshot user dicts now — avoids race conditions in threads
    child_user    = dict(users.get(child_sid,    {}))
    predator_user = dict(users.get(predator_sid, {}))

    # ── SANDBOX ACTIVE ─────────────────────────────────────────
    if room_sandbox:
        sandbox_msg_count += 1

        if role == "predator":
            # Predator sees their own message immediately
            emit("message", {**payload, "flagged": False, "flags": []}, to=predator_sid)
            # Child sees NOTHING from predator — AI replies to predator AS the child
            def _reply_to_predator(t=text, cu=child_user):
                reply = ai_as_child(t)
                socketio.emit("message", {
                    "name":     cu.get("name",""),
                    "avatar":   cu.get("avatar","🐼"),
                    "text":     reply,
                    "sid":      child_sid,
                    "risk":     risk, "msg_score": 0, "stage": stage,
                    "flags":    [], "role": "child", "flagged": False,
                }, room=predator_sid)
                print(f"[SANDBOX] AI->predator (as child): '{reply}'")
            threading.Thread(target=_reply_to_predator, daemon=True).start()

        elif role == "child":
            # Child sees their own message immediately
            emit("message", {**payload, "flagged": False}, to=child_sid)
            # Predator sees NOTHING from child — AI replies to child AS the predator
            def _reply_to_child(t=text, pu=predator_user):
                reply = ai_as_predator(t)
                socketio.emit("message", {
                    "name":     pu.get("name",""),
                    "avatar":   pu.get("avatar","🐼"),
                    "text":     reply,
                    "sid":      predator_sid,
                    "risk":     risk, "msg_score": 0, "stage": stage,
                    "flags":    [], "role": "predator", "flagged": False,
                }, room=child_sid)
                print(f"[SANDBOX] AI->child (as predator): '{reply}'")
            threading.Thread(target=_reply_to_child, daemon=True).start()

        if sandbox_msg_count >= WIND_DOWN_AFTER:
            threading.Thread(target=run_wind_down, daemon=True).start()

        if child_sid:
            emit("risk_bar", {"risk": risk, "stage": stage}, to=child_sid)

        # Email only at risk == 1.0, even inside sandbox
        if not room_alerted and should_email(risk):
            threading.Thread(target=send_alert, args=(risk, stage), daemon=True).start()
            room_alerted = True
        return

    # ── NORMAL ROUTING ──────────────────────────────────────────
    if role == "predator":
        # Predator sees their own message, no flags shown to them
        emit("message", {**payload, "flagged": False, "flags": []}, to=predator_sid)
        # Child sees predator's message — flagged badge if raw_score > FLAG_THRESHOLD
        if child_sid:
            emit("message", {**payload, "flagged": len(flags) > 0}, to=child_sid)

        # Sandbox trigger check
        if not room_sandbox and should_sandbox(raw_score, risk):
            room_sandbox = True
            save_evidence()
            if child_sid:
                emit("sandbox_activated", {"risk": risk, "stage": stage}, to=child_sid)
            print(f"[SANDBOX] ACTIVATED — raw={raw_score} room_risk={risk}")

        # Email only at risk == 1.0
        if not room_alerted and should_email(risk):
            threading.Thread(target=send_alert, args=(risk, stage), daemon=True).start()
            room_alerted = True

        # Soft popup warning to child only when risk crosses 0.60
        if not room_sandbox and risk >= 0.60 and child_sid:
            emit("risk_update", {"risk": risk, "stage": stage, "level": "medium"}, to=child_sid)

        if child_sid:
            emit("risk_bar", {"risk": risk, "stage": stage}, to=child_sid)

    elif role == "child":
        emit("message", {**payload, "flagged": False}, to=child_sid)
        if predator_sid:
            emit("message", {**payload, "flagged": False}, to=predator_sid)

    print(f"[MSG] {name}({role}): '{text[:50]}' | raw={raw_score} contrib={contributing} risk={risk} sandbox={room_sandbox}")


@socketio.on("image")
def on_image(data):
    global room_alerted, room_sandbox, sandbox_msg_count
    sid   = request.sid
    user  = users.get(sid, {})
    role  = user.get("role", "unknown")

    image_data = data.get("image", "")
    tmp_path   = None
    if image_data and image_data.startswith("data:image"):
        try:
            import base64, tempfile
            header, b64 = image_data.split(",", 1)
            ext      = "jpg" if "jpeg" in header else "png"
            tmp_path = os.path.join(tempfile.gettempdir(), f"sg_img_{datetime.now().strftime('%f')}.{ext}")
            with open(tmp_path, "wb") as f:
                f.write(base64.b64decode(b64))
        except Exception as e:
            print(f"[IMAGE DECODE ERROR] {e}")

    contributing, img_flags, img_breakdown, raw_score = score_message("[image sent]", image_path=tmp_path)
    if tmp_path and os.path.exists(tmp_path):
        try: os.remove(tmp_path)
        except: pass

    risk  = update_risk(contributing) if role == "predator" else room_risk
    stage = get_stage(risk)

    room_history.append({
        "sender": user.get("name","?"), "role": role,
        "text": "[IMAGE]", "score": contributing, "raw_score": raw_score,
        "flags": img_flags or [], "breakdown": img_breakdown,
        "ts": datetime.now().isoformat(),
    })
    if img_flags and role == "predator":
        flagged_messages.append({
            "sender": user.get("name","?"), "role": role,
            "text": "[IMAGE]", "score": contributing, "raw_score": raw_score,
            "flags": img_flags, "breakdown": img_breakdown,
            "room_risk": risk, "stage": stage,
            "ts": datetime.now().isoformat(),
        })

    payload = {
        "name": user.get("name","?"), "avatar": user.get("avatar","🐼"),
        "image": image_data, "filename": data.get("filename","img"),
        "sid": sid, "risk": risk, "role": role,
    }
    child_user = dict(users.get(child_sid, {}))

    if role == "predator":
        emit("image", {**payload, "flagged": False}, to=predator_sid)
        if room_sandbox:
            def _img_reply(cu=child_user):
                reply = ai_as_child("[someone just sent me an image]")
                socketio.emit("message", {
                    "name": cu.get("name",""), "avatar": cu.get("avatar","🐼"),
                    "text": reply, "sid": child_sid, "risk": risk, "msg_score": 0,
                    "stage": stage, "flags": [], "role": "child", "flagged": False,
                }, room=predator_sid)
            threading.Thread(target=_img_reply, daemon=True).start()
        else:
            if child_sid:
                emit("image", {**payload, "flagged": True}, to=child_sid)
            if should_sandbox(raw_score, risk):
                room_sandbox = True
                save_evidence()
                if child_sid:
                    emit("sandbox_activated", {"risk": risk, "stage": stage}, to=child_sid)
                print(f"[SANDBOX] ACTIVATED via image — raw={raw_score} risk={risk}")
            elif risk >= 0.60 and child_sid:
                emit("risk_update", {"risk": risk, "stage": stage, "level": "medium"}, to=child_sid)
            if not room_alerted and should_email(risk):
                threading.Thread(target=send_alert, args=(risk, stage), daemon=True).start()
                room_alerted = True
        if child_sid:
            emit("risk_bar", {"risk": risk, "stage": stage}, to=child_sid)

    elif role == "child":
        emit("image", {**payload, "flagged": False}, to=child_sid)
        if predator_sid:
            emit("image", {**payload, "flagged": False}, to=predator_sid)


@socketio.on("typing")
def on_typing(data):
    sid  = request.sid
    user = users.get(sid, {})
    emit("typing", {"name": user.get("name",""), "typing": data.get("typing", False)}, to=ROOM, skip_sid=sid)


@socketio.on("disconnect")
def on_disconnect():
    global child_sid, predator_sid
    sid  = request.sid
    user = users.pop(sid, {})
    if user:
        if sid == child_sid:    child_sid    = None
        if sid == predator_sid: predator_sid = None
        emit("user_left", {"name": user.get("name","")}, to=ROOM)
        print(f"[-] {user.get('name')} ({user.get('role')}) left")


if __name__ == "__main__":
    print("=" * 55)
    print("  SafeGuard — Chat + Detection Server")
    print("  Chat      -> http://127.0.0.1:3000")
    print("  Dashboard -> http://127.0.0.1:3000/dashboard")
    print(f"  Sandbox   -> room_risk >= {SANDBOX_RISK_TRIGGER} OR raw >= {EXPLICIT_SANDBOX_SCORE}")
    print(f"  Email     -> room_risk == {EMAIL_THRESHOLD} only")
    print("  First tab = CHILD | Second tab = PREDATOR")
    print("=" * 55)
    socketio.run(app, host="0.0.0.0", port=3000, debug=True)
