"""UI strings for English, Hindi and Marathi."""
LANG_NAMES = {"English": "English", "Hindi": "हिन्दी", "Marathi": "मराठी"}

STRINGS = {
    "English": {
        "tagline": "Your crop-growing companion",
        "caption": "Crop advice for rice, wheat, maize, cotton, soybean and sugarcane, answered from agriculture PDFs.",
        "new_chat": "➕ New chat", "new_chat_title": "New chat", "your_chats": "**Your chats**",
        "delete_all": "Delete all chats", "delete_chat": "Delete this chat",
        "safety_note": "No pesticide, herbicide or fungicide advice and no fertilizer calculations. Please consult your local KVK for those.",
        "try": "**Try:** *How to grow soybean?* · *When should wheat be sown?* · *Signs of maturity in sugarcane?*",
        "placeholder": "Ask about rice, wheat, maize, cotton, soybean or sugarcane...",
        "spinner": "Searching the crop documents...", "loading": "Loading knowledge base...",
        "sources": "Sources", "page": "page",
        "listen": "🔊 Listen", "stop": "⏹ Stop", "no_voice": "No suitable voice found",
        "suggested": "Suggested questions",
        "kb_error": "Knowledge base not ready: {err}",
        "kb_hint": "Put crop PDFs in the `data/` folder, run `run_ingest.bat`, then refresh this page.",
    },
    "Hindi": {
        "tagline": "फसल की खेती में आपका साथी",
        "caption": "चावल, गेहूँ, मक्का, कपास, सोयाबीन और गन्ने की खेती की सलाह, कृषि PDF दस्तावेज़ों के आधार पर।",
        "new_chat": "➕ नई चैट", "new_chat_title": "नई चैट", "your_chats": "**आपकी चैट**",
        "delete_all": "सभी चैट हटाएँ", "delete_chat": "यह चैट हटाएँ",
        "safety_note": "कीटनाशक, खरपतवारनाशक या फफूंदनाशक की सलाह और उर्वरक की गणना नहीं दी जाती। इसके लिए कृपया अपने स्थानीय KVK से संपर्क करें।",
        "try": "**आज़माएँ:** *सोयाबीन की खेती कैसे करें?* · *गेहूँ की बुवाई कब करें?* · *गन्ने की परिपक्वता के लक्षण क्या हैं?*",
        "placeholder": "चावल, गेहूँ, मक्का, कपास, सोयाबीन या गन्ने के बारे में पूछें...",
        "spinner": "फसल दस्तावेज़ों में खोजा जा रहा है...", "loading": "ज्ञान आधार लोड हो रहा है...",
        "sources": "स्रोत", "page": "पृष्ठ",
        "listen": "🔊 सुनें", "stop": "⏹ रोकें", "no_voice": "उपयुक्त आवाज़ उपलब्ध नहीं है",
        "suggested": "सुझाए गए सवाल",
        "kb_error": "ज्ञान आधार तैयार नहीं है: {err}",
        "kb_hint": "`data/` फ़ोल्डर में फसल PDF रखें, `run_ingest.bat` चलाएँ, फिर यह पेज रीफ़्रेश करें।",
    },
    "Marathi": {
        "tagline": "पीक लागवडीसाठी तुमचा साथीदार",
        "caption": "तांदूळ, गहू, मका, कापूस, सोयाबीन आणि ऊस लागवडीचा सल्ला, कृषी PDF दस्तऐवजांवर आधारित.",
        "new_chat": "➕ नवीन चॅट", "new_chat_title": "नवीन चॅट", "your_chats": "**तुमचे चॅट**",
        "delete_all": "सर्व चॅट हटवा", "delete_chat": "हा चॅट हटवा",
        "safety_note": "कीटकनाशक, तणनाशक किंवा बुरशीनाशकाचा सल्ला आणि खताची गणना दिली जात नाही. यासाठी कृपया स्थानिक KVK शी संपर्क साधा.",
        "try": "**हे विचारून पहा:** *सोयाबीनची लागवड कशी करावी?* · *गव्हाची पेरणी कधी करावी?* · *उसाच्या परिपक्वतेची लक्षणे कोणती?*",
        "placeholder": "तांदूळ, गहू, मका, कापूस, सोयाबीन किंवा ऊस याबद्दल विचारा...",
        "spinner": "पीक दस्तऐवजांमध्ये शोध घेतला जात आहे...", "loading": "ज्ञानसंग्रह लोड होत आहे...",
        "sources": "स्रोत", "page": "पृष्ठ",
        "listen": "🔊 ऐका", "stop": "⏹ थांबा", "no_voice": "योग्य आवाज उपलब्ध नाही",
        "suggested": "सुचवलेले प्रश्न",
        "kb_error": "ज्ञानसंग्रह तयार नाही: {err}",
        "kb_hint": "`data/` फोल्डरमध्ये पिकांचे PDF ठेवा, `run_ingest.bat` चालवा आणि हे पेज रिफ्रेश करा.",
    },
}


def tr(lang: str, key: str, **kw) -> str:
    text = STRINGS.get(lang, STRINGS["English"]).get(key) or STRINGS["English"][key]
    return text.format(**kw) if kw else text
