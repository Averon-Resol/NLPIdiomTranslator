import json

new_idioms = [
    {
        "idiom": "घर की मुर्गी दाल बराबर",
        "literal_meaning": "A home-cooked chicken is equal to lentils.",
        "figurative_meaning": "Familiarity breeds contempt; undervaluing what one has.",
        "example": "Ghar ki murgi daal barabar, uski salah koi nahi sunta.",
        "language": "Hindi"
    },
    {
        "idiom": "चार दिन की चाँदनी फिर अंधेरी रात",
        "literal_meaning": "Four days of moonlight, then dark night.",
        "figurative_meaning": "A nine days' wonder; short-lived happiness.",
        "example": "Ye nayi naukri ki kushi char din ki chandni phir andheri raat hai.",
        "language": "Hindi"
    },
    {
        "idiom": "छाती पर मूंग दलना",
        "literal_meaning": "To grind lentils on someone's chest.",
        "figurative_meaning": "To deliberately annoy someone while staying close to them.",
        "example": "Tum wahan jaa kar kyu rehte ho, sirf uski chhati par moong dalne ke liye?",
        "language": "Hindi"
    },
    {
        "idiom": "छक्के छुड़ाना",
        "literal_meaning": "To make someone lose their sixes.",
        "figurative_meaning": "To defeat completely.",
        "example": "Bhartiya team ne vipakshi team ke chhakke chhuda diye.",
        "language": "Hindi"
    },
    {
        "idiom": "जान हथेली पर रखना",
        "literal_meaning": "To keep life on the palm.",
        "figurative_meaning": "To risk one's life.",
        "example": "Sainik apni jaan hatheli par rakh kar desh ki raksha karte hain.",
        "language": "Hindi"
    },
    {
        "idiom": "टेढ़ी खीर",
        "literal_meaning": "Crooked rice pudding.",
        "figurative_meaning": "A very difficult task.",
        "example": "Bina kisi sifarish ke sarkari naukri pana tedhi kheer hai.",
        "language": "Hindi"
    },
    {
        "idiom": "डूबते को तिनके का सहारा",
        "literal_meaning": "A drowning man gets the support of a straw.",
        "figurative_meaning": "A drowning man catches at a straw.",
        "example": "Aakhir samay mein uski 100 rupaye ki madad bhi doobte ko tinke ka sahara thi.",
        "language": "Hindi"
    },
    {
        "idiom": "तलवे चाटना",
        "literal_meaning": "To lick the soles of feet.",
        "figurative_meaning": "To flatter or bootlick for personal gain.",
        "example": "Promotion paane ke liye woh boss ke talwe chatne laga.",
        "language": "Hindi"
    },
    {
        "idiom": "तिल का ताड़ बनाना",
        "literal_meaning": "To make a palm tree out of a sesame seed.",
        "figurative_meaning": "To make a mountain out of a molehill.",
        "example": "Chhoti si baat thi, tumne toh til ka taad bana diya.",
        "language": "Hindi"
    },
    {
        "idiom": "दाल गलना",
        "literal_meaning": "For lentils to melt/cook.",
        "figurative_meaning": "To succeed in a trick or plan.",
        "example": "Yahan tumhara jhooth nahi chalega, tumhari daal nahi galne wali.",
        "language": "Hindi"
    },
    {
        "idiom": "धोबी का कुत्ता न घर का न घाट का",
        "literal_meaning": "A washerman's dog, belonging neither to the house nor the riverbank.",
        "figurative_meaning": "A person with no fixed place or status.",
        "example": "Dono jagah kaam karne ke chakkar mein woh dhobi ka kutta na ghar ka na ghat ka reh gaya.",
        "language": "Hindi"
    },
    {
        "idiom": "नानी याद आना",
        "literal_meaning": "To remember one's maternal grandmother.",
        "figurative_meaning": "To find oneself in a very difficult or painful situation.",
        "example": "Pahaad chadhte waqt sabko nani yaad aa gayi.",
        "language": "Hindi"
    },
    {
        "idiom": "पीठ थपथपाना",
        "literal_meaning": "To pat the back.",
        "figurative_meaning": "To encourage or praise.",
        "example": "Pitamah ne acche parinam par bete ki peeth thapthapai.",
        "language": "Hindi"
    },
    {
        "idiom": "फूले न समाना",
        "literal_meaning": "Not fitting into oneself while blossoming.",
        "figurative_meaning": "To be extremely happy.",
        "example": "Parixa mein pratham aane par woh phoole na samaya.",
        "language": "Hindi"
    },
    {
        "idiom": "बगुला भगत",
        "literal_meaning": "A crane devotee.",
        "figurative_meaning": "A hypocrite.",
        "example": "Wo dikhta toh sadhu hai, par asal mein bagula bhagat hai.",
        "language": "Hindi"
    },
    {
        "idiom": "बाल-बाल बचना",
        "literal_meaning": "To be saved by a hair.",
        "figurative_meaning": "To have a narrow escape.",
        "example": "Kal sadak durghatna mein woh baal-baal bacha.",
        "language": "Hindi"
    },
    {
        "idiom": "रफूचक्कर होना",
        "literal_meaning": "To become darning and a wheel.",
        "figurative_meaning": "To run away or flee.",
        "example": "Chori karne ke baad woh turant rafuchakkar ho gaya.",
        "language": "Hindi"
    },
    {
        "idiom": "राई का पहाड़ बनाना",
        "literal_meaning": "To make a mountain out of mustard seeds.",
        "figurative_meaning": "To exaggerate a small issue.",
        "example": "Baat itni si thi, lekin tumne toh rai ka pahad bana diya.",
        "language": "Hindi"
    },
    {
        "idiom": "रंगा सियार",
        "literal_meaning": "A painted jackal.",
        "figurative_meaning": "A deceiver or hypocrite.",
        "example": "Uski mithi baaton mein mat aana, woh poora ranga siyar hai.",
        "language": "Hindi"
    },
    {
        "idiom": "हाथ साफ करना",
        "literal_meaning": "To clean hands.",
        "figurative_meaning": "To steal something or show sleight of hand.",
        "example": "Bheed ka fayda utha kar chor ne purse par hath saaf kar diya.",
        "language": "Hindi"
    }
]

with open('data/raw/hindi.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Deduplicate based on idiom text just to be safe
existing_idioms = {item['idiom'] for item in data}
filtered_new_idioms = [item for item in new_idioms if item['idiom'] not in existing_idioms]

data.extend(filtered_new_idioms)

with open('data/raw/hindi.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Added {len(filtered_new_idioms)} new unique idioms to hindi.json! Total is now {len(data)}.")
