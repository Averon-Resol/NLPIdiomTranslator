import json

new_idioms = [
    {
        "idiom": "आग में घी डालना",
        "literal_meaning": "To pour ghee in fire.",
        "figurative_meaning": "To aggravate a situation.",
        "example": "Usne aag mein ghee daalne ka kaam kiya jab usne unki ladai ke beech ek aur jhooth bol diya.",
        "language": "Hindi"
    },
    {
        "idiom": "आसमान से गिरा खजूर में अटका",
        "literal_meaning": "Fell from the sky, got stuck in a date tree.",
        "figurative_meaning": "Out of one trouble and into another.",
        "example": "Pehle naukri gayi, ab gaadi kharab ho gayi, yeh toh wahi baat hui ki aasman se gira, khajur mein atka.",
        "language": "Hindi"
    },
    {
        "idiom": "ईद का चाँद होना",
        "literal_meaning": "To be the moon of Eid.",
        "figurative_meaning": "To be rarely seen.",
        "example": "Bhai, tum toh aajkal Eid ka chaand ho gaye ho, kabhi dikhte hi nahi.",
        "language": "Hindi"
    },
    {
        "idiom": "उल्टा चोर कोतवाल को डांटे",
        "literal_meaning": "The thief scolds the police officer.",
        "figurative_meaning": "The pot calling the kettle black.",
        "example": "Galti tumhari hai aur tum mujhe hi suna rahe ho? Yeh toh ulta chor kotwal ko daante wali baat ho gayi.",
        "language": "Hindi"
    },
    {
        "idiom": "ऊँट के मुँह में जीरा",
        "literal_meaning": "Cumin seed in a camel's mouth.",
        "figurative_meaning": "A drop in the ocean; too little.",
        "example": "Is bade parivaar ke liye yeh thoda sa khana oont ke muh mein jeera hai.",
        "language": "Hindi"
    },
    {
        "idiom": "एक अनार सौ बीमार",
        "literal_meaning": "One pomegranate, a hundred sick people.",
        "figurative_meaning": "Demand is much higher than supply.",
        "example": "Naukri ek hai aur ummeedwar hazaar, yeh toh ek anar sau beemar wali sthiti hai.",
        "language": "Hindi"
    },
    {
        "idiom": "नौ दो ग्यारह होना",
        "literal_meaning": "To become nine two eleven.",
        "figurative_meaning": "To run away or escape.",
        "example": "Police ko aate dekh, chor nau do gyarah ho gaye.",
        "language": "Hindi"
    },
    {
        "idiom": "भैंस के आगे बीन बजाना",
        "literal_meaning": "To play a flute in front of a buffalo.",
        "figurative_meaning": "Casting pearls before swine.",
        "example": "Us murkh ko samjhana bhains ke aage been bajana jaisa hai.",
        "language": "Hindi"
    },
    {
        "idiom": "मुँह में राम बगल में छुरी",
        "literal_meaning": "Rama in the mouth, knife in the side.",
        "figurative_meaning": "A wolf in sheep's clothing.",
        "example": "Wo dikhne mein toh accha hai par andar se munh mein Ram bagal mein chhuri jaisa hai.",
        "language": "Hindi"
    },
    {
        "idiom": "हवा से बातें करना",
        "literal_meaning": "To talk to the wind.",
        "figurative_meaning": "To run very fast.",
        "example": "Uska ghoda maidan mein hawa se baatein kar raha tha.",
        "language": "Hindi"
    },
    {
        "idiom": "आँखों में धूल झोंकना",
        "literal_meaning": "To throw dust in eyes.",
        "figurative_meaning": "To deceive someone.",
        "example": "Chor sabki aankhon mein dhool jhonk kar bhaag gaya.",
        "language": "Hindi"
    },
    {
        "idiom": "अक्ल पर पत्थर पड़ना",
        "literal_meaning": "Stones falling on intellect.",
        "figurative_meaning": "To lose one's senses.",
        "example": "Uski akal par patthar pad gaye the jab usne apna saara paisa juye mein laga diya.",
        "language": "Hindi"
    },
    {
        "idiom": "अपना उल्लू सीधा करना",
        "literal_meaning": "To straighten one's owl.",
        "figurative_meaning": "To serve one's own selfish motives.",
        "example": "Aaj kal log sirf apna ullu seedha karne mein lage rehte hain.",
        "language": "Hindi"
    },
    {
        "idiom": "अपने पैर पर कुल्हाड़ी मारना",
        "literal_meaning": "To hit an axe on one's own foot.",
        "figurative_meaning": "To harm oneself.",
        "example": "Usne naukri chhod kar apne hi pair par kulhadi maar li.",
        "language": "Hindi"
    },
    {
        "idiom": "आस्तीन का साँप",
        "literal_meaning": "Snake in the sleeve.",
        "figurative_meaning": "A hidden enemy or traitor.",
        "example": "Mujhe kya pata tha ki mera dost hi aasteen ka saanp nikalega.",
        "language": "Hindi"
    },
    {
        "idiom": "दाँत खट्टे करना",
        "literal_meaning": "To make teeth sour.",
        "figurative_meaning": "To defeat completely.",
        "example": "Bhartiya sena ne dushman ke daant khatte kar diye.",
        "language": "Hindi"
    },
    {
        "idiom": "दाल में काला होना",
        "literal_meaning": "Something black in the lentils.",
        "figurative_meaning": "Something is fishy.",
        "example": "Wo chupke chupke baat kar rahe hain, zaroor daal mein kuch kaala hai.",
        "language": "Hindi"
    },
    {
        "idiom": "नाच न जाने आँगन टेढ़ा",
        "literal_meaning": "Doesn't know how to dance, blames the crooked courtyard.",
        "figurative_meaning": "A bad workman blames his tools.",
        "example": "Tumhe khelna aata nahi aur bat ko kharab bol rahe ho, naach na jaane aangan tedha.",
        "language": "Hindi"
    },
    {
        "idiom": "लोहे के चने चबाना",
        "literal_meaning": "To chew iron chickpeas.",
        "figurative_meaning": "To do a very difficult task.",
        "example": "UPSC exam pass karna lohe ke chane chabane ke barabar hai.",
        "language": "Hindi"
    },
    {
        "idiom": "श्री गणेश करना",
        "literal_meaning": "To do Shri Ganesh.",
        "figurative_meaning": "To start something new.",
        "example": "Aaj unhone apne naye dukaan ka shri ganesh kiya.",
        "language": "Hindi"
    },
    {
        "idiom": "मुठ्ठी गर्म करना",
        "literal_meaning": "To warm the fist.",
        "figurative_meaning": "To give a bribe.",
        "example": "Bina mutthi garm kiye yahan koi kaam nahi hota.",
        "language": "Hindi"
    },
    {
        "idiom": "रातों की नींद हराम होना",
        "literal_meaning": "Sleep of nights becoming forbidden.",
        "figurative_meaning": "To be extremely worried or anxious.",
        "example": "Exam ke chakkar mein uski raaton ki neend haram ho gayi hai.",
        "language": "Hindi"
    },
    {
        "idiom": "लकीर का फकीर होना",
        "literal_meaning": "To be a beggar of the line.",
        "figurative_meaning": "To blindly follow old traditions.",
        "example": "Zamana badal gaya hai, kab tak lakeer ke fakeer bane rahoge?",
        "language": "Hindi"
    },
    {
        "idiom": "लोहा लेना",
        "literal_meaning": "To take iron.",
        "figurative_meaning": "To face bravely.",
        "example": "Maharana Pratap ne mughalon se kadi takkar li aur loha liya.",
        "language": "Hindi"
    },
    {
        "idiom": "सफेद झूठ",
        "literal_meaning": "White lie.",
        "figurative_meaning": "An obvious or blatant lie.",
        "example": "Wo safed jhooth bol raha hai ki wo wahan nahi tha.",
        "language": "Hindi"
    },
    {
        "idiom": "हाथ मलना",
        "literal_meaning": "To rub hands.",
        "figurative_meaning": "To regret.",
        "example": "Samay nikal jaane ke baad haath malne se kuch nahi hoga.",
        "language": "Hindi"
    },
    {
        "idiom": "हाथ-पाँव मारना",
        "literal_meaning": "To strike hands and feet.",
        "figurative_meaning": "To make desperate efforts.",
        "example": "Naukri paane ke liye usne bahut haath-paon maare.",
        "language": "Hindi"
    },
    {
        "idiom": "आग बबूला होना",
        "literal_meaning": "To become a fire bubble.",
        "figurative_meaning": "To be extremely angry.",
        "example": "Bina wajah uski gaadi thok di gayi toh woh aag baboola ho gaya.",
        "language": "Hindi"
    },
    {
        "idiom": "आपे से बाहर होना",
        "literal_meaning": "To be out of oneself.",
        "figurative_meaning": "To lose control due to anger.",
        "example": "Chhoti si baat par woh aape se bahar ho gaya.",
        "language": "Hindi"
    },
    {
        "idiom": "किताबी कीड़ा होना",
        "literal_meaning": "To be a bookworm.",
        "figurative_meaning": "To be always reading.",
        "example": "Woh sirf padhta rehta hai, poora kitabi keeda hai.",
        "language": "Hindi"
    }
]

with open('data/raw/hindi.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data.extend(new_idioms)

with open('data/raw/hindi.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Added {len(new_idioms)} new idioms to hindi.json!")
