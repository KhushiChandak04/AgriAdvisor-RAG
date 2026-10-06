"""Detect and handle out-of-scope and price questions locally without LLM."""
import re

# Out-of-scope patterns: non-agriculture questions
OUT_OF_SCOPE = re.compile(
    r"\b(weather|cricket|politics|sports|movie|music|stock|news|bitcoin|crypto|relationship|"
    r"love|marriage|health|medicine|doctor|disease|covid|vaccine|recipe|cooking|"
    r"homework|exam|career|job|salary|tax|police|crime|current affairs)\b|"
    r"(what is .*|who is|when was|where is|why is|how to [^g][^r])",
    re.I
)

PRICE_KEYWORDS = re.compile(
    r"\b(price|cost|rate|cheap|expensive|sell|buy|profit|loss|income|earnings|earn|"
    r"demand|supply|export|import|procurement|minimum support|msp|contract farming|"
    r"value added|commodity price|future|trading|revenue|price today|market rate|"
    r"current price|wholesale|retail|rates)\b|"
    r"market (price|rate|value|trading|news|update)",
    re.I
)

# Hindi/Marathi out-of-scope
OOS_DEVANAGARI = re.compile(
    r"बॉलीवुड|फिल्म|फीचर|गीत|सिनेमा|राजनीति|खेल|क्रिकेट|व्यापार|शेयर|स्वास्थ्य|दवा|"
    r"संबंध|प्रेम|विवाह|नौकरी|वेतन|कानून|पुलिस|अपराध|मौसम|हवामान|आज|कल|"
    r"प्रधान|मंत्री|सरकार|राष्ट्रपति"
)

# Price in Hindi/Marathi
PRICE_DEVANAGARI = re.compile(
    r"दाम|कीमत|भाव|मूल्य|महंगा|सस्ता|बिक्री|खरीद|लाभ|नुकसान|आय|आजीविका|"
    r"बाजार|व्यापार|निर्यात|आयात|खरीफ|रबी|सरकारी|सहायता|न्यूनतम|समर्थन|"
    r"किंमत|दर|भाव|मार्केट|विक्रय|क्रय|नफा|घाटा|उत्पन्न|कमाई|आय|उपार्जन|कमावा|आमदनी"
)


def is_out_of_scope(question: str) -> bool:
    """True if question is unrelated to crop production."""
    return bool(OUT_OF_SCOPE.search(question)) or bool(OOS_DEVANAGARI.search(question))


def is_price_question(question: str) -> bool:
    """True if question is about prices, markets, or economics."""
    return bool(PRICE_KEYWORDS.search(question)) or bool(PRICE_DEVANAGARI.search(question))


def out_of_scope_reply(language: str) -> str:
    """Local response for out-of-scope questions."""
    replies = {
        "English": ("I only answer questions about growing rice, wheat, maize, cotton, soybean "
                    "and sugarcane (climate, soil, sowing, irrigation, harvesting, storage). "
                    "For topics like markets, economics, current events or health, please consult "
                    "other resources."),
        "Hindi": ("मैं केवल चावल, गेहूँ, मक्का, कपास, सोयाबीन और गन्ने की खेती संबंधी सवालों "
                  "का जवाब देता हूँ (जलवायु, मिट्टी, बुवाई, सिंचाई, कटाई, भंडारण)। बाजार, "
                  "आर्थिकी या स्वास्थ्य जैसे विषयों के लिए कृपया अन्य स्रोत देखें।"),
        "Marathi": ("मी केवळ तांदूळ, गहू, मका, कापूस, सोयाबीन आणि ऊस लागवडीबद्दल सवालांची उत्तरे देतो "
                    "(हवामान, मिट्टी, पेरणी, सिंचन, कापणी, साठयन). बाजार, अर्थशास्त्र किंवा स्वास्थ्य "
                    "विषयांसाठी कृपया इतर स्रोत पहा."),
    }
    return replies.get(language, replies["English"])


def price_question_reply(language: str) -> str:
    """Local response for price/market questions."""
    replies = {
        "English": ("I focus on crop production practices. For commodity prices, market rates, "
                    "minimum support price (MSP), contract farming, or economic aspects, please "
                    "check your local agricultural market, state procurement center, or "
                    "commodity exchange platforms."),
        "Hindi": ("मैं फसल उत्पादन तरीकों पर ध्यान केंद्रित करता हूँ। वस्तु मूल्य, बाजार दर, "
                  "न्यूनतम समर्थन मूल्य (MSP), अनुबंध खेती या आर्थिक पहलुओं के लिए कृपया अपने "
                  "स्थानीय कृषि बाजार, राज्य खरीफ केंद्र या कमोडिटी एक्सचेंज से संपर्क करें।"),
        "Marathi": ("मी पीक उत्पादन पद्धतींवर लक्ष केंद्रित करतो. वस्तू किंमत, बाजार दर, "
                    "किमान समर्थन मूल्य (MSP), करार शेती किंवा आर्थिक पहलूंसाठी कृपया आपल्या "
                    "स्थानिक कृषी बाजार, राज्य खरीफ केंद्र किंवा वस्तू एक्सचेंजशी संपर्क साधा."),
    }
    return replies.get(language, replies["English"])
