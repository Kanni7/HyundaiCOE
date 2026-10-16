"""Hand-written SYNTHETIC evaluation data, written separately from the training seeds.

HANDWRITTEN: unseen paraphrases for the intent model (never used for training or threshold
calibration). Tuple = (text, intent, language, speech_act, negated, gold_parameters).
PERSONAL_EVAL: taught personal commands with later uses, paraphrases, unrelated utterances and
negated/quoted references, for calibrating (val profiles) and testing (test profiles) the matcher.
"""
_L = {"e": "english", "h": "hindi", "r": "romanized_hindi", "g": "hinglish"}
_A = {"d": "direct_command", "i": "implicit_request", "o": "other", "x": "dialog"}


def _rows(intent, items):
    out = []
    for it in items:
        text, code = it[0], it[1]
        params = it[2] if len(it) > 2 else {}
        neg = "!" in code
        code = code.replace("!", "")
        out.append((text, intent, _L[code[0]], _A[code[1]], neg, params))
    return out


S, L = {"amount": "small"}, {"amount": "large"}

HANDWRITTEN = (
    _rows("climate_temperature_up", [
        ("I'm freezing in here.", "ei"), ("It's too cold.", "ei"), ("Bhai, AC thoda warm kar do.", "gd", S),
        ("यहाँ बहुत ठंड लग रही है।", "hi"), ("Make the cabin a little warmer.", "ed", S),
        ("Could you heat things up a little?", "ed", S), ("My teeth are chattering.", "ei"),
        ("Thoda garmahat chahiye andar.", "ri", S), ("कार में ठिठुरन हो रही है, गरम करो।", "hd"),
        ("Yaar thand se haath jam gaye, temperature upar karo.", "gd"), ("Warm me up, it's icy.", "ed"),
        ("Heating thodi aur badha do.", "gd", S), ("Can we have it toastier?", "ed"),
    ]) + _rows("climate_temperature_down", [
        ("It's stuffy and hot, cool it down.", "ed"), ("Garmi se jaan nikal rahi hai.", "ri"),
        ("Cabin ko zara thanda karo bhai.", "gd", S), ("मुझे बहुत पसीना आ रहा है।", "hi"),
        ("Bring the temperature down a notch.", "ed", S), ("Feels like a sauna in here.", "ei"),
        ("AC ki cooling thodi aur chahiye.", "gd", S), ("Thoda thandak kar do.", "rd", S),
        ("I'm overheating, please cool the car.", "ed"), ("गाड़ी का तापमान थोड़ा घटा दो।", "hd", S),
        ("Too warm for me.", "ei"),
    ]) + _rows("climate_temperature_set", [
        ("Set the cabin to 23 degrees.", "ed", {"target_temperature_celsius": 23}),
        ("AC 22 pe kar do yaar.", "gd", {"target_temperature_celsius": 22}),
        ("तापमान इक्कीस डिग्री रखो।", "hd", {"target_temperature_celsius": 21}),
        ("I'd like it at twenty five.", "ed", {"target_temperature_celsius": 25}),
        ("Temperature ko chaubees pe le aao.", "gd", {"target_temperature_celsius": 24}),
        ("Climate should be 20 degrees.", "ed", {"target_temperature_celsius": 20}),
        ("Set temperature to 45 degrees.", "ed", {"target_temperature_celsius": 45}),
        ("Hold the cabin at 19.5.", "ed", {"target_temperature_celsius": 19.5}),
        ("Baais degree kar do.", "rd", {"target_temperature_celsius": 22}),
    ]) + _rows("climate_fan_speed_up", [
        ("Fan ki speed thodi badha do.", "gd", S), ("Give me more air from the vents.", "ed"),
        ("Blast the fan.", "ed", L), ("हवा का बहाव बढ़ाओ।", "hd"), ("The vents are barely blowing.", "ei"),
        ("Pankhe ki raftaar tez karo.", "rd"), ("Air thodi zyada chahiye.", "gd", S),
    ]) + _rows("climate_fan_speed_down", [
        ("Fan ki speed thodi kam kar do.", "gd", S), ("Tone down the blower.", "ed"),
        ("The air is blasting in my face.", "ei"), ("पंखे की रफ़्तार घटाओ।", "hd"), ("Hawa dheere chalao.", "rd"),
        ("Fan bahut shor kar raha hai, kam karo.", "gd"), ("Less wind from the vents.", "ed"),
    ]) + _rows("climate_on", [
        ("Fire up the air conditioner.", "ed"), ("AC chalu kar dijiye.", "rd"), ("एयर कंडीशनर चालू करो।", "hd"),
        ("Climate on kar do bhai.", "gd"), ("Can you get the AC running?", "ed"),
    ]) + _rows("climate_off", [
        ("Turn off the climate.", "ed"), ("AC bandh kar dijiye.", "rd"), ("एयर कंडीशनर बंद कर दो।", "hd"),
        ("Switch the aircon off please.", "ed"), ("AC off kar do yaar.", "gd"),
        ("Don't turn the AC off.", "e!d"), ("AC band mat karna.", "g!d"), ("एसी को बंद मत करना।", "h!d"),
        ("Please never switch off the climate.", "e!d"),
    ]) + _rows("volume_up", [
        ("Volume badha do, yaar.", "gd"), ("I can't hear anything, louder.", "ed"),
        ("Make the music a bit louder.", "ed", S), ("आवाज़ थोड़ी ऊँची करो।", "hd", S), ("Gaana zor se bajao.", "rd"),
        ("Sound full kar do.", "gd", L), ("Turn up the speakers.", "ed"), ("Volume mat badhao.", "g!d"),
        ("Never turn the volume up.", "e!d"),
    ]) + _rows("volume_down", [
        ("Bring the volume down.", "ed"), ("It's way too loud in here.", "ei"), ("Awaaz thodi dheemi kar do.", "rd", S),
        ("आवाज़ कम कीजिए।", "hd"), ("Music ka volume ghatao.", "gd"), ("Softer, please.", "ed"),
        ("Don't reduce the volume.", "e!d"),
    ]) + _rows("media_play", [
        ("Play something.", "ed"), ("Gaana shuru karo.", "rd"), ("कोई गाना बजाओ।", "hd"), ("Resume the song.", "ed"),
        ("Music chala do bhai.", "gd"), ("Let's have some music.", "ed"),
    ]) + _rows("media_pause", [
        ("Pause it for a second.", "ed"), ("Gaana ek minute rok do.", "rd"), ("संगीत रोक दो।", "hd"),
        ("Hold the song.", "ed"), ("Music ko pause kar do.", "gd"), ("Please don't pause the music.", "e!d"),
    ]) + _rows("media_next", [
        ("Skip to the next one.", "ed"), ("Agla gaana lagao.", "rd"), ("अगला गाना चलाओ।", "hd"),
        ("Next track please.", "ed"), ("Ye wala skip karo.", "gd"), ("Don't skip this one.", "e!d"),
        ("गाना मत बदलो।", "h!d"),
    ]) + _rows("media_previous", [
        ("Play that last song again.", "ed"), ("Pichla wala gaana bajao.", "rd"), ("पिछला गाना लगाओ।", "hd"),
        ("Go back one track.", "ed"), ("Previous wala track lagao.", "gd"),
    ]) + _rows("navigation_destination_request", [
        ("Take me to Connaught Place.", "ed", {"destination": "Connaught Place"}),
        ("India Gate ka raasta dikhao.", "gd", {"destination": "India Gate"}),
        ("मुझे एयरपोर्ट ले चलो।", "hd", {"destination": "एयरपोर्ट"}),
        ("Find the nearest charging station.", "ed", {"destination": "charging station"}),
        ("Navigate to Select Citywalk.", "ed", {"destination": "Select Citywalk"}),
        ("Mujhe Saket jaana hai.", "rd", {"destination": "Saket"}),
        ("Juhu beach tak le chalo.", "gd", {"destination": "Juhu beach"}),
        ("Get me directions to the bus stand.", "ed", {"destination": "bus stand"}),
    ]) + _rows("dialog_confirm", [
        ("Yep.", "ex"), ("Haan bhai.", "rx"), ("हाँ जी।", "hx"), ("Sounds good, do it.", "ex"), ("Ok kar do.", "gx"),
        ("Absolutely.", "ex"),
    ]) + _rows("dialog_cancel", [
        ("Nah.", "ex"), ("Nahi chahiye.", "rx"), ("नहीं रहने दीजिए।", "hx"), ("Cancel that.", "ex"), ("Chhod do.", "rx"),
        ("No, leave it as it is.", "ex"),
    ]) + _rows("unsupported", [
        ("Open the boot.", "eo"), ("Window neeche karo.", "ro"), ("Brake lagao jaldi.", "ro"), ("Call my wife.", "eo"),
        ("What's the time in London?", "eo"), ("Gaadi ko autopilot pe daal do.", "go"), ("Mujhe khana order karna hai.", "go"),
        ("सनरूफ खोलो।", "ho"), ("Delete all system files.", "eo"), ("Turn off traction control.", "eo"),
        ("Kal ka match kaun jeeta?", "ro"), ("I had a great lunch.", "eo"), ("Lock the car.", "eo"),
    ]) + _rows("ambiguous", [
        ("Make it a bit more.", "ed"), ("Thoda aur.", "rd"), ("Change that.", "ed"), ("इसे थोड़ा कम कर दो।", "hd"),
        ("Isko up karo.", "gd"), ("Can you fix the setting?", "ed"), ("Badal do.", "rd"),
    ])
)

# Personal-command evaluation. "uses" must match the command; "paraphrases" are reasonable
# variations that should still match; "negatives" must NOT execute the command.
PERSONAL_EVAL = {
    "val": [
        {"teach": "When I say 'make it cozy', set the temperature to 22 degrees.", "trigger": "make it cozy",
         "intent": "climate_temperature_set", "params": {"target_temperature_celsius": 22},
         "uses": ["make it cozy", "Make it cozy."], "paraphrases": ["make it cozy please", "make it nice and cozy", "cozy it up"],
         "negatives": ["don't make it cozy", "what does 'make it cozy' do?", "make it louder", "it's cold outside"]},
        {"teach": "Jab main 'chill mode' bolun, AC 22 pe set kar dena.", "trigger": "chill mode",
         "intent": "climate_temperature_set", "params": {"target_temperature_celsius": 22},
         "uses": ["chill mode", "Chill mode"], "paraphrases": ["Chill mode, yaar.", "chill mode on karo", "chill mode please"],
         "negatives": ["Don't use chill mode", "chill mode mat karo", "what is 'chill mode'?", "chilli paneer order karo"]},
        {"teach": "When I say 'party time', turn the volume up.", "trigger": "party time", "intent": "volume_up", "params": {},
         "uses": ["party time"], "paraphrases": ["it's party time", "party time yaar", "party time, let's go"],
         "negatives": ["no party time today", "it's lunch time", "what time is the party?"]},
        {"teach": "Jab main 'shanti' bolun, gaana pause kar do.", "trigger": "shanti", "intent": "media_pause", "params": {},
         "uses": ["shanti", "Shanti!"], "paraphrases": ["shanti please", "thodi shanti"],
         "negatives": ["shanti mat karo", "Shantanu ko call karo", "bahut shor hai"]},
        {"teach": "When I say 'office route', navigate to Cyber Hub.", "trigger": "office route",
         "intent": "navigation_destination_request", "params": {"destination": "Cyber Hub"},
         "uses": ["office route"], "paraphrases": ["office route please", "take the office route"],
         "negatives": ["don't take the office route", "office mein meeting hai", "home route"]},
    ],
    "test": [
        {"teach": "When I say 'freeze mode', set the temperature to 18 degrees.", "trigger": "freeze mode",
         "intent": "climate_temperature_set", "params": {"target_temperature_celsius": 18},
         "uses": ["freeze mode", "Freeze mode."], "paraphrases": ["freeze mode please", "freeze mode on karo", "activate freeze mode"],
         "negatives": ["don't use freeze mode", "freeze mode mat lagao", "what does 'freeze mode' mean?", "my phone froze"]},
        {"teach": "Jab main 'dhamaka' bolun, volume badha dena.", "trigger": "dhamaka", "intent": "volume_up", "params": {},
         "uses": ["dhamaka", "Dhamaka!"], "paraphrases": ["dhamaka yaar", "dhamaka kar do", "full dhamaka"],
         "negatives": ["dhamaka mat karo", "news mein dhamake ki khabar thi", "what is 'dhamaka'?"]},
        {"teach": "When I say 'quiet please', pause the music.", "trigger": "quiet please", "intent": "media_pause", "params": {},
         "uses": ["quiet please", "Quiet please."], "paraphrases": ["quiet please yaar", "some quiet please"],
         "negatives": ["don't be quiet", "it's quiet outside", "play some music"]},
        {"teach": "When I say 'ghar chalo', navigate to Lajpat Nagar.", "trigger": "ghar chalo",
         "intent": "navigation_destination_request", "params": {"destination": "Lajpat Nagar"},
         "uses": ["ghar chalo", "Ghar chalo."], "paraphrases": ["ghar chalo yaar", "chalo ghar", "ghar le chalo"],
         "negatives": ["ghar mat chalo abhi", "ghar pe khana hai", "what does 'ghar chalo' do?"]},
        {"teach": "When I say 'warm hug', make it warmer.", "trigger": "warm hug", "intent": "climate_temperature_up", "params": {},
         "uses": ["warm hug"], "paraphrases": ["warm hug please", "give me a warm hug"],
         "negatives": ["no warm hug", "hug my mom", "warm milk"]},
        {"teach": "Jab main 'agla dhun' bolun, next song chala do.", "trigger": "agla dhun", "intent": "media_next", "params": {},
         "uses": ["agla dhun"], "paraphrases": ["agla dhun bajao", "agla dhun please"],
         "negatives": ["agla dhun mat lagao", "dhoop bahut hai"]},
    ],
    # Unrelated utterances that must not match ANY personal command (false-acceptance test).
    "unrelated": [
        "turn the volume down", "set the temperature to 24", "navigate to India Gate", "pause the music",
        "what's the weather", "call mom", "I'm feeling cold", "next song", "open the window", "how are you",
        "mode change karo", "AC on karo", "gaana chalao", "office jaana hai", "time kya hua hai",
        "make it quick", "party", "freeze", "quiet", "hug", "chill", "ghar", "route",
    ],
}
