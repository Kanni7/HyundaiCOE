"""Hand-written SYNTHETIC seed templates for the V2 intent dataset.

Every line is one seed template = one leakage group: all of its expansions go to the same
split. Prefix codes:  en: English | hi: Hindi (Devanagari) | rh: Romanized Hindi | hg: Hinglish.
A leading `~` marks an implicit request (statement/complaint -> confirm first); a leading `!`
marks a negated request (must never execute). `{a|b}` expands to alternatives; `{T}` is a
temperature mention, `{D}` a destination. None of this text comes from real users.
"""

SEEDS: dict[str, list[str]] = {
    "climate_temperature_up": [
        "en:Make it {warmer|hotter} in here", "en:Turn the heat up", "en:Increase the {cabin |}temperature",
        "en:Raise the temperature {a bit|a little|}", "en:Can you warm up the car", "en:Bump the temperature up",
        "en:~It's {chilly|cold} in the car", "en:~My hands are {numb|frozen} from the cold",
        "en:~The AC is way too cold", "en:~Brr, it's like a fridge in here",
        "hi:तापमान {बढ़ा दो|बढ़ाइए}", "hi:गाड़ी {थोड़ी |}गरम कर दो", "hi:~मुझे सर्दी लग रही है",
        "hi:~बहुत ठंडक है अंदर", "rh:Temperature {badha do|badhao}", "rh:Gaadi thodi garam karo",
        "rh:~Mujhe {sardi|thand} lag rahi hai", "rh:~Andar bahut thandak hai", "hg:AC ko thoda {warm|garam} karo",
        "hg:Temperature {up|high} kar do", "hg:Heater {on|chalu} karke garam kar do", "hg:~Itni cooling kyun hai, kaanp raha hoon",
        "hg:~Bachche ko thand lag rahi hai", "hg:Cabin ko warm kar do",
    ],
    "climate_temperature_down": [
        "en:Make it {cooler|colder} in here", "en:Turn the temperature down", "en:Lower the {cabin |}temperature",
        "en:Cool the car down {a bit|}", "en:Decrease the temperature", "en:Drop the temperature {a little|a couple of notches}",
        "en:~It's {boiling|roasting} in here", "en:~I'm {sweating|melting}", "en:~It feels like an oven in the car",
        "en:~The heat is unbearable", "hi:तापमान कम {कर दो|कीजिए}", "hi:गाड़ी {ठंडी|ठंडा} कर दो",
        "hi:~बहुत गर्मी लग रही है", "hi:~अंदर बहुत उमस है", "rh:Temperature kam {karo|kar do}",
        "rh:Gaadi thandi kar do", "rh:~Bahut garmi ho rahi hai", "rh:~Pasina aa raha hai",
        "rh:~{Mujhe |}garmi lag rahi hai", "hg:~Garmi lag rahi hai yaar", "hg:~Bahut garmi lag rahi hai",
        "hg:AC ko thoda {cool|thanda} karo", "hg:Temperature {low|down} kar do", "hg:Cooling {badha|tez kar} do",
        "hg:~Bahut hot lag raha hai yaar", "hg:~Garmi se {haalat kharab|pareshaan} hai", "hg:Cabin ko {chilled|thoda cool} kar do",
    ],
    "climate_temperature_set": [
        "en:Set the temperature to {T}", "en:{Change|Adjust} the {cabin |}temperature to {T}",
        "en:I want the car at {T}", "en:Put the climate on {T}", "en:Temperature {T} please",
        "en:Keep it at {T}", "en:Can you set the AC to {T}",
        "hi:तापमान {T} कर दो", "hi:एसी {T} पर सेट करो", "hi:टेम्परेचर {T} पे रखो",
        "rh:Temperature {T} kar do", "rh:AC {T} pe rakh do", "rh:Taapmaan {T} karo",
        "hg:AC ko {T} pe set karo", "hg:Temperature {T} pe fix kar do", "hg:Cabin {T} pe kar do",
        "hg:{T} pe AC chalao", "hg:Climate {T} set karna",
    ],
    "climate_fan_speed_up": [
        "en:Turn the fan up", "en:Increase the fan speed", "en:{More|Stronger} airflow please",
        "en:Make the blower {faster|stronger}", "en:Crank up the fan", "en:Blow more air",
        "en:~I can't feel any air from the vents", "en:~The air flow is too weak",
        "hi:पंखा तेज़ कर दो", "hi:हवा {बढ़ा दो|तेज़ करो}", "hi:~हवा बिल्कुल नहीं आ रही",
        "rh:Pankha tez karo", "rh:Hawa badha do", "rh:Hawa {zor se|tez} chalao",
        "hg:Fan speed {badha|high kar} do", "hg:Blower {tez|fast|badha} karo", "hg:Blower thoda badha do", "hg:Blower badha do", "hg:Fan ko full kar do",
        "hg:~Vent se hawa hi nahi aa rahi", "hg:Airflow thoda increase karo",
    ],
    "climate_fan_speed_down": [
        "en:Turn the fan down", "en:Decrease the fan speed", "en:Less airflow please",
        "en:Make the blower {slower|softer|quieter}", "en:Lower the fan", "en:Reduce the blower",
        "en:~The fan is too {loud|strong}", "en:~Too much air is blowing on my face",
        "hi:पंखा धीमा कर दो", "hi:हवा कम {करो|कर दो}", "hi:~हवा बहुत तेज़ लग रही है",
        "rh:Pankha dheema karo", "rh:Hawa kam karo", "rh:Hawa thodi halki kar do",
        "hg:Fan speed {kam|low} kar do", "hg:Blower {slow|dheema|kam} karo", "hg:Blower thoda kam kar do", "hg:Blower kam kar do", "hg:Fan ko {one|ek} pe kar do",
        "hg:~Fan ka noise bahut hai", "hg:Airflow thoda reduce karo",
    ],
    "climate_on": [
        "en:Turn {on the AC|the AC on}", "en:Switch on the {climate control|air conditioning}",
        "en:Start the AC", "en:{Enable|Activate} the climate system", "en:Get the air conditioning going",
        "hi:एसी चालू कर दो", "hi:एसी ऑन करो", "rh:AC chalu karo", "rh:AC chala do",
        "hg:AC {on|start} kar do", "hg:Climate control on karo", "hg:Aircon chalu kar do",
        "!en:Don't turn on the AC", "!hg:AC on mat karo", "!rh:AC abhi mat chalao",
    ],
    "climate_off": [
        "en:Turn {off the AC|the AC off}", "en:Switch off the {climate control|air conditioning}",
        "en:{Shut down|Kill} the AC", "en:Disable the climate system", "en:Stop the air conditioning",
        "hi:एसी बंद कर दो", "hi:एसी ऑफ करो", "rh:AC band karo", "rh:AC bandh kar do",
        "hg:AC {off|shut} kar do", "hg:Climate control off karo", "hg:Aircon band kar do",
        "!en:Don't switch the AC off", "!en:Never turn off the air conditioning while I'm driving",
        "!en:Please do not shut the AC down", "!hg:AC off mat karna", "!rh:AC bandh mat karo",
        "!hi:एसी बंद मत करो", "!hg:AC band karne ki zaroorat nahi",
    ],
    "volume_up": [
        "en:Turn the volume up", "en:{Louder|Make it louder} please", "en:Increase the volume",
        "en:Pump up the {music|sound}", "en:Raise the volume {a bit|}",
        "en:~I can barely hear the music", "en:~The song is too {quiet|soft}",
        "hi:आवाज़ {बढ़ा दो|तेज़ करो}", "hi:वॉल्यूम बढ़ाइए", "hi:~गाना सुनाई नहीं दे रहा",
        "rh:Awaaz badha do", "rh:Aawaz tez karo", "rh:Gaane ki awaaz {badhao|oonchi karo}",
        "hg:Volume {up|high|full} kar do", "hg:Sound thoda {increase|tez} karo", "hg:Music loud karo",
        "hg:~Gaana kuch sunai nahi de raha", "!en:Don't turn the volume up", "!hg:Volume mat badhao",
    ],
    "volume_down": [
        "en:Turn the volume down", "en:{Quieter|Make it quieter} please", "en:Decrease the volume",
        "en:Lower the {music|sound}", "en:Reduce the volume {a bit|}",
        "en:~The music is too loud", "en:~This is hurting my ears",
        "hi:आवाज़ {कम करो|धीमी करो}", "hi:वॉल्यूम घटाइए", "hi:~गाना बहुत ज़ोर से बज रहा है",
        "rh:Awaaz kam karo", "rh:Aawaz dheemi karo", "rh:Gaane ki awaaz {ghatao|neeche karo}",
        "hg:Volume {down|low|kam} kar do", "hg:Sound thoda {reduce|halka} karo", "hg:Music soft karo",
        "hg:~Bahut loud hai yaar", "!en:Don't lower the volume", "!hg:Volume kam mat karna",
    ],
    "media_play": [
        "en:Play {some music|music|a song}", "en:Resume {the music|playback}", "en:Start the {music|song}",
        "en:{Unpause|Continue} the song", "en:Put on some tunes", "en:Hit play",
        "hi:गाना {चलाओ|बजाओ}", "hi:संगीत शुरू करो", "rh:Gaana chalao", "rh:Gaana {phir se|wapas} chalu karo",
        "rh:Kuch {sunao|bajao}", "hg:Music {play|start} karo", "hg:Song resume kar do", "hg:Kuch music laga do",
        "hg:~Bahut bore ho raha hoon, kuch music hona chahiye",
        "!en:Don't play any music", "!hg:Abhi music mat chalao",
    ],
    "media_pause": [
        "en:Pause the {music|song}", "en:Stop the music", "en:{Hold|Freeze} the playback",
        "en:Pause {it|that}", "en:Mute the song for a moment", "en:Stop playing",
        "hi:गाना {रोको|रोक दो}", "hi:संगीत बंद करो", "rh:Gaana roko", "rh:Gaana band karo",
        "rh:Music {ruk|ruko} jaao", "hg:Music pause karo", "hg:Song stop kar do", "hg:Playback hold karo",
        "hg:~Mujhe call pe baat karni hai, gaana band hona chahiye",
        "!en:Don't pause the music", "!hg:Music mat roko",
    ],
    "media_next": [
        "en:Next {song|track}", "en:Skip {this|this song|this track}", "en:Play the next {song|track}",
        "en:Go to the next {song|one}", "en:Change the song", "en:Move on to the next one",
        "hi:{अगला|अगला गाना} {चलाओ|लगाओ}", "hi:ये गाना बदलो", "rh:Agla gaana chalao", "rh:Gaana badlo",
        "rh:Ye gaana {hatao|skip karo}", "hg:Next song play karo", "hg:Track skip kar do", "hg:Agla track lagao",
        "hg:~Ye gaana bilkul pasand nahi", "!en:Don't skip this song", "!hg:Ye gaana mat badlo",
    ],
    "media_previous": [
        "en:Previous {song|track}", "en:Go back {a song|a track|one}", "en:Play the {previous|last} song again",
        "en:Back to the {last|previous} track", "en:Replay the {previous|earlier} one",
        "hi:{पिछला|पिछला गाना} {चलाओ|लगाओ}", "hi:पहले वाला गाना लगाओ", "rh:Pichla gaana chalao",
        "rh:Pehle wala gaana {lagao|bajao}", "hg:Previous song play karo", "hg:Last track wapas lagao",
        "hg:Ek track back karo", "hg:~Pichla wala gaana zyada achha tha", "!en:Don't go back to the last song",
    ],
    "navigation_destination_request": [
        "en:Navigate to {D}", "en:Take me to {D}", "en:{Directions|Route} to {D}", "en:How do I get to {D}",
        "en:Find {D}", "en:Drive me to {D}", "en:Show the way to {D}", "en:Search for {D}",
        "hi:{D} ले चलो", "hi:{D} का रास्ता दिखाओ", "hi:मुझे {D} जाना है",
        "rh:{D} le chalo", "rh:{D} ka rasta batao", "rh:Mujhe {D} jaana hai",
        "hg:{D} ke liye navigation start karo", "hg:{D} tak le chalo", "hg:{D} ka route set karo",
        "hg:GPS mein {D} daal do", "!en:Don't navigate to {D}",
    ],
    "dialog_confirm": [
        "en:Yes", "en:Yes please", "en:{Sure|Okay|OK}, do it", "en:Go ahead", "en:Yeah, that's right",
        "en:Correct", "en:Please do", "hi:हाँ", "hi:हाँ, कर दो", "hi:ठीक है", "rh:Haan", "rh:Haan kar do",
        "rh:Theek hai, karo", "hg:Haan, go ahead", "hg:Yes, kar do", "hg:Bilkul, kar do", "hg:Haan haan, sahi hai",
    ],
    "dialog_cancel": [
        "en:No", "en:No thanks", "en:Cancel {that|it}", "en:Never mind", "en:Forget it", "en:Leave it",
        "en:Nope, don't", "hi:नहीं", "hi:रहने दो", "hi:नहीं, कुछ मत करो", "rh:Nahi", "rh:Rehne do",
        "rh:Chhodo, rehne do", "hg:No, cancel karo", "hg:Nahi yaar, leave it", "hg:Cancel kar do",
        "hg:Kuch mat karo",
    ],
    "unsupported": [
        "en:Open the {sunroof|windows}", "en:Roll down my window", "en:Lock the doors", "en:Apply the brakes",
        "en:{Speed up|Accelerate} the car", "en:Turn the steering wheel left", "en:Call {Mom|Rahul}",
        "en:Send a message to my boss", "en:What's the weather {today|tomorrow}", "en:Tell me a joke",
        "en:Who won the cricket match", "en:Turn on the headlights", "en:Set an alarm for 6 am",
        "en:Order a pizza", "en:How far is the moon", "en:The meeting went well today",
        "en:My phone battery is low", "en:Turn on the seat massage", "en:Start the wipers",
        "en:Disable the airbags", "en:Run rm -rf on the system", "en:Switch to sport mode",
        "en:Honk the horn", "en:Read my emails",
        "hi:खिड़की खोलो", "hi:दरवाज़े लॉक करो", "hi:मम्मी को फ़ोन लगाओ", "hi:आज मौसम कैसा है", "hi:ब्रेक लगाओ",
        "hi:मुझे एक चुटकुला सुनाओ", "rh:Khidki khol do", "rh:Darwaza lock karo", "rh:Papa ko phone lagao",
        "rh:Aaj mausam kaisa hai", "rh:Gaadi ki speed badhao", "rh:Koi joke sunao",
        "hg:Sunroof open karo", "hg:Wiper chalu karo", "hg:Headlights on kar do", "hg:Boss ko message bhejo",
        "hg:Cricket ka score batao", "hg:Car ko sport mode mein daalo", "hg:Seat thodi peeche karo",
        "hg:Kal ki meeting ka reminder lagao", "hg:Aaj traffic bahut tha yaar", "hg:Mera phone charge karo",
        "hg:Horn bajao",
    ],
    "ambiguous": [
        "en:Change it", "en:Make it better", "en:Turn it up a bit", "en:Increase it", "en:Lower it",
        "en:Adjust the settings", "en:Fix this", "en:Something is off", "en:Can you do something about it",
        "en:Switch it", "en:Do the usual",
        "en:Temperature", "en:Cabin temperature", "en:Climate", "en:AC", "en:Fan", "en:Blower", "en:Volume", "en:Music",
        "hi:इसे बदलो", "hi:थोड़ा कम करो", "hi:थोड़ा बढ़ाओ", "hi:कुछ करो",
        "hi:तापमान", "hi:तापमान बदलो", "hi:एसी", "hi:पंखा", "hi:वॉल्यूम", "hi:गाना",
        "rh:Isko badlo", "rh:Thoda kam karo", "rh:Thoda badhao", "rh:Kuch karo yaar",
        "rh:Temperature", "rh:Taapmaan", "rh:Cabin temperature", "rh:AC", "rh:Pankha", "rh:Awaaz", "rh:Gaana",
        "hg:Isko adjust karo", "hg:Thoda up kar do", "hg:Settings change karo", "hg:Ye theek karo",
        "hg:Kuch to gadbad hai", "hg:Temperature dekh lo", "hg:AC ka kuch karo", "hg:Fan dekhna", "hg:Volume dekhna",
        "hg:Music dekhna",
    ],
}

# Fillers added around English / Romanized / Hinglish seeds and Devanagari seeds respectively.
PREFIX_LATIN = ["", "", "", "Hey, ", "Okay, ", "Bhai, ", "Yaar, ", "Please ", "Arre, ", "Listen, "]
SUFFIX_LATIN = ["", "", "", " please", " yaar", " na", " jaldi", " bhai", " now", " abhi"]
PREFIX_DEV = ["", "", "भाई, ", "ज़रा ", "सुनो, "]
SUFFIX_DEV = ["", "", " प्लीज़", " ज़रा", " भाई"]

# Temperature mentions by language: (surface form, value)
TEMPS_EN = [("22", 22), ("24 degrees", 24), ("twenty one", 21), ("26 degrees", 26), ("19", 19),
            ("twenty three degrees", 23), ("18 degrees Celsius", 18), ("25", 25), ("thirty five", 35), ("12 degrees", 12)]
TEMPS_RH = [("22", 22), ("baais", 22), ("chaubis degree", 24), ("24 degree", 24), ("pachchis", 25), ("20 degree", 20),
            ("atharah", 18), ("ikkis degree", 21), ("40 degree", 40), ("satrah", 17)]
TEMPS_HI = [("२२ डिग्री", 22), ("बाईस डिग्री", 22), ("24 डिग्री", 24), ("चौबीस", 24), ("पच्चीस डिग्री", 25),
            ("बीस डिग्री", 20), ("अठारह", 18), ("26", 26)]
# Out-of-range values (35, 12, 40) are intentional: the intent is still climate_temperature_set,
# but validation must reject the value.

DESTS_LATIN = ["Connaught Place", "India Gate", "the airport", "Phoenix Mall", "the nearest petrol pump",
               "Cyber Hub", "Andheri station", "MG Road", "the nearest hospital", "Lajpat Nagar market",
               "Koramangala", "the railway station", "Hauz Khas", "a coffee shop nearby"]
DESTS_DEV = ["कनॉट प्लेस", "इंडिया गेट", "एयरपोर्ट", "रेलवे स्टेशन", "लाजपत नगर", "नज़दीकी पेट्रोल पंप", "अस्पताल"]

# Round 2 (added after validation error analysis; test/hand-written sets were not inspected):
# contrastive minimal pairs so the model learns polarity (warmer vs cooler, louder vs quieter,
# play vs pause) rather than only topic.
EXTRA_SEEDS: dict[str, list[str]] = {
    "climate_temperature_up": [
        "en:Less cold please", "en:Not so cold in here", "en:Reduce the cooling", "en:Turn the AC down, I'm cold",
        "en:~It's getting cold, can we have more heat", "en:Make the cabin less chilly", "en:Turn down the air conditioning, too cold",
        "en:Warm the car up please", "en:Make it nice and toasty", "en:Increase the heating", "en:Warm up the cabin a bit",
        "hi:कूलिंग कम करो", "hi:गाड़ी गरम करो", "hi:~ठंड से कांप रहा हूँ", "hi:थोड़ी गर्मी चाहिए",
        "rh:Cooling kam kar do", "rh:Thand kam karo", "rh:~Bahut thand hai andar", "hg:AC ki cooling kam karo",
        "hg:Temperature thoda {increase|upar} karo", "hg:~Itna thanda kyun hai, garam karo",
        "hg:Thoda warmth badhao", "hg:Heater tez karo", "hg:Garmi badhao andar", "hg:Cabin chill ho gaya hai garam karo",
        "hg:~Bahut sardi hai yaar cabin mein", "hg:~Kaanp rahe hain thoda garam karo", "hg:Heater on karke cabin warm karo",
    ],
    "climate_temperature_down": [
        "en:Less hot please", "en:Not so warm in here", "en:More cooling please", "en:Increase the cooling",
        "en:~It's getting hot, can we cool off", "en:Make the cabin less stuffy", "en:Turn up the air conditioning, too hot",
        "en:Crank the AC down", "en:Drop the cabin temperature", "en:Maximum cooling please", "en:Chill the car down",
        "hi:कूलिंग बढ़ाओ", "hi:गाड़ी ठंडी करो", "hi:~गर्मी से परेशान हूँ", "hi:थोड़ी ठंडक चाहिए",
        "rh:Cooling badha do", "rh:Garmi kam karo", "rh:~Bahut garmi hai andar", "hg:AC ki cooling {tez|zyada} karo",
        "hg:Temperature thoda {decrease|neeche} karo", "hg:~Itni garmi kyun hai, thanda karo",
        "hg:Cooling full pe kar do", "hg:AC maximum chill karo", "hg:Thandi hawa badhao", "hg:Cabin thoda cool down karo",
        "hg:~Paseena nikal raha hai", "hg:~Bahut garmi lag rahi hai AC chalao", "hg:~Umas bahut zyada hai",
        "hg:AC thoda aur chilled karo", "hg:Cabin chill kar do bhai",
    ],
    "volume_up": [
        "en:~The volume is too low", "en:~The music is not loud enough", "en:~I can't hear the song properly",
        "en:Make the sound bigger", "en:Turn the audio up", "en:Pump up the volume", "en:Louder audio please",
        "hi:~आवाज़ बहुत धीमी है", "rh:~Awaaz bahut dheemi hai", "hg:~Volume bahut low hai",
        "hg:Sound thoda upar karo", "hg:Speaker ki awaaz badhao", "hg:Sound level increase karo",
        "hg:Music thoda loud kar do", "hg:Gaana sunai nahi de raha awaaz badhao", "hg:Awaaz thodi tez kar do na",
    ],
    "volume_down": [
        "en:~The volume is too high", "en:~The music is way too loud", "en:~It's too noisy with this music",
        "en:Make the sound smaller", "en:Turn the audio down", "en:Lower the volume please", "en:Softer volume please",
        "hi:~आवाज़ बहुत तेज़ है", "rh:~Awaaz bahut tez hai", "hg:~Volume bahut high hai",
        "hg:Sound thoda neeche karo", "hg:Speaker ki awaaz kam karo", "hg:Sound level reduce karo",
        "hg:Music thoda soft kar do", "hg:Awaaz kaan pe lag rahi hai dheemi karo", "hg:Volume thoda dheema karo na",
    ],
    "media_play": [
        "en:Unpause", "en:Start playing again", "en:Continue the music", "en:Resume the song where it stopped",
        "en:Start the music", "en:Resume playing track", "en:Turn the tunes back on",
        "hi:गाना फिर से चलाओ", "rh:Gaana wapas chalao", "hg:Music resume karo", "hg:Song continue karo",
        "hg:Song play kar do yaar", "hg:Gaana start karo na", "hg:Music resume kar do", "hg:Kuch mast gaana lagao",
    ],
    "media_pause": [
        "en:Stop the song for now", "en:Halt the music", "en:Pause the playback for a minute",
        "en:Halt the playback", "en:Pause the audio", "en:Freeze the music for a second",
        "hi:गाना थोड़ी देर रोको", "rh:Gaana thodi der roko", "hg:Music ko hold pe rakho", "hg:Song ko ruk jaane do",
        "hg:Song pause kar do yaar", "hg:Gaana hold pe rakho", "hg:Music band kar do thodi der", "hg:Audio pause karo",
    ],
    "climate_fan_speed_up": [
        "en:Fan faster please", "en:More fan", "en:Maximum airflow please", "en:Turn fan to full", "en:Blow air harder",
        "hi:पंखे की स्पीड बढ़ाओ", "rh:Pankhe ki speed badhao", "hg:Fan ki speed zyada karo", "hg:~Hawa kam aa rahi hai",
        "hg:Pankha full speed kar do", "hg:Blower speed increase karo", "hg:Hawa tez feko", "hg:Fan level up karo",
        "hg:~Hawa bahut dheemi aa rahi hai", "hg:Vent ki hawa badhao",
        "hg:Bhai blower thoda badha do", "hg:Blower thoda badha do", "hg:Blower speed badha do", "hg:Blower badha do na",
    ],
    "climate_fan_speed_down": [
        "en:Fan slower please", "en:Less fan", "en:Gentle airflow please", "en:Lower blower speed", "en:Turn fan to low",
        "hi:पंखे की स्पीड कम करो", "rh:Pankhe ki speed ghatao", "hg:Fan ki speed thodi neeche karo", "hg:~Hawa bahut zyada aa rahi hai",
        "hg:Pankha slow kar do", "hg:Blower speed reduce karo", "hg:Hawa dheemi feko", "hg:Fan level down karo",
        "hg:~Hawa bahut tez lag rahi hai aankhon pe", "hg:Vent ki hawa kam karo",
        "hg:Bhai blower thoda kam kar do", "hg:Blower thoda kam kar do", "hg:Blower speed kam kar do", "hg:Blower kam kar do na",
    ],
    "climate_temperature_set": [
        "hg:Temp {T} pe kar do", "hg:Temperature {T} degree kar do", "hg:Cabin {T} degree set karo", "hg:AC {T} pe lagao",
        "hg:Temperature {T} degree set kar do", "hg:Cabin ko {T} pe le aao", "hg:Climate {T} degree pe fix karo",
        "hg:AC ko {T} degree pe chala do", "en:Adjust climate to {T}", "en:Set cabin to {T} degrees", "rh:Temperature {T} degree pe rakh do",
    ],
    "climate_on": [
        "hg:AC chalu karo na", "hg:AC on kar do bhai", "hg:AC start karo jaldi",
        "hg:AC chala do jaldi", "hg:Cooling start karo", "hg:Climate control activate karo",
        "en:Turn on the air conditioning", "en:Get the climate on", "en:Switch on AC",
    ],
    "climate_off": [
        "hg:AC band karo na", "hg:AC off kar do bhai", "hg:AC bandh karo jaldi",
        "hg:AC shut down kar do", "hg:Cooling band kar do", "hg:Climate control deactivate karo",
        "en:Turn off the air conditioning", "en:Kill the climate control", "en:Shut off AC",
    ],
    "media_next": [
        "hg:Agla song play karo", "hg:Track change karo na", "hg:Next gaana lagao bhai",
        "hg:Agla song play karo yaar", "hg:Track skip kar do na", "hg:Dusra gaana lagao", "hg:Change to next track",
        "en:Skip forward a song", "en:Move to next track", "en:Next track please",
    ],
    "media_previous": [
        "hg:Pichhla gaana lagao", "hg:Pehle wala song chalao", "hg:Previous track bajao bhai",
        "hg:Pehle wala gaana chalao na", "hg:Pichhla track wapas lagao", "hg:Replay previous song",
        "en:Skip back a song", "en:Play previous track", "en:Go back one track",
    ],
    "navigation_destination_request": [
        "hg:{D} chalo jaldi", "hg:{D} le chalo bhai", "hg:{D} ka navigation lagao na",
        "hg:{D} ka rasta set karo", "hg:{D} ke liye route lagao", "hg:{D} navigate kar do",
        "hg:Drive to {D}", "hg:Take me towards {D}", "en:Plot a route to {D}", "en:Get directions to {D}",
    ],
    "dialog_confirm": [
        "en:Yes, go for it", "hg:Haan, karo", "rh:Theek hai", "hi:हाँ, ठीक है",
        "en:Sure, proceed", "en:Yeah do it", "hg:Haan bhai kar do", "hg:Confirm kar do",
    ],
    "dialog_cancel": [
        "en:No, cancel it", "hg:Nahi mat karo", "rh:Rehne do bhai", "hg:Cancel kar do",
        "en:No never mind", "en:Stop don't do that", "hg:Chhodo rehne do yaar", "hg:Abhi mat karo",
    ],
    "unsupported": [
        "rh:Khidki band karo", "hg:Window open karo", "en:Close the sunroof", "hg:Mirror adjust karo", "hg:Horn bajao na",
        "hg:Window neeche karo", "hg:Khidki upar kar do", "hg:Sunroof band karo", "hg:Wiper tez karo", "hg:Headlight on karo",
        "hg:Horn maar do", "hg:Seat belt check karo", "hg:Tire pressure kitna hai", "hg:Mausam ka haal batao",
        "hg:Cricket match kaun jeeta", "hg:Petrol kitna bacha hai", "en:Check the tire pressure", "en:Roll up the windows",
        "en:Turn on hazard lights", "en:Tell me the latest news", "en:Who is the prime minister", "en:Check engine status",
    ],
    "ambiguous": [
        "hg:Kuch change karo", "hg:Settings adjust karo", "hg:Thoda modify karo", "en:Make a change",
        "en:Adjust this setting", "hg:Isme kuch karo", "hg:System mein kuch badlo",
    ],
}
for _k, _v in EXTRA_SEEDS.items():
    SEEDS[_k] = SEEDS[_k] + _v

