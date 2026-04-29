import json

scraped_idioms = [
    {"idiom": "अंग-अंग ढीला होना", "literal_meaning": "Every limb getting loose.", "figurative_meaning": "To be extremely tired.", "example": "Din bhar kaam karne ke baad mera ang-ang dheela ho gaya.", "language": "Hindi"},
    {"idiom": "अंगारे उगलना", "literal_meaning": "To spew embers.", "figurative_meaning": "To speak very harshly or angrily.", "example": "Jab usne sachai jani, to woh angare ugalne laga.", "language": "Hindi"},
    {"idiom": "अंधे की लाठी", "literal_meaning": "A blind man's stick.", "figurative_meaning": "The only support.", "example": "Shravan Kumar apne andhe maa-baap ki lathi tha.", "language": "Hindi"},
    {"idiom": "अंधे के हाथ बटेर लगना", "literal_meaning": "A quail falling into a blind man's hands.", "figurative_meaning": "Getting something valuable without the necessary qualifications.", "example": "Bina padhai kiye uski sarkari naukri lag gayi, yeh to andhe ke hath bater lagna hai.", "language": "Hindi"},
    {"idiom": "अक्ल का दुश्मन", "literal_meaning": "Enemy of intellect.", "figurative_meaning": "A foolish person.", "example": "Tum toh bilkul akal ke dushman ho jo itna asaan kaam nahi kar sake.", "language": "Hindi"},
    {"idiom": "अपने मुँह मियाँ मिट्ठू बनना", "literal_meaning": "To become a sweet-talker from one's own mouth.", "figurative_meaning": "To blow one's own trumpet; self-praise.", "example": "Apne muh miya mitthu banne ka koi fayda nahi, log tumhara kaam dekhte hain.", "language": "Hindi"},
    {"idiom": "आँख का तारा होना", "literal_meaning": "To be the star of the eye.", "figurative_meaning": "To be the apple of someone's eye; very dear.", "example": "Raju apni maa ki aankh ka tara hai.", "language": "Hindi"},
    {"idiom": "आँखें बिछाना", "literal_meaning": "To lay down one's eyes.", "figurative_meaning": "To welcome someone with great respect and eagerness.", "example": "Humne mehmano ke swagat mein aankhein bicha di.", "language": "Hindi"},
    {"idiom": "आकाश-पाताल एक करना", "literal_meaning": "To make sky and underworld one.", "figurative_meaning": "To move heaven and earth; work very hard.", "example": "Usne pariksha pas karne ke liye aakash-patal ek kar diya.", "language": "Hindi"},
    {"idiom": "ईंट से ईंट बजाना", "literal_meaning": "To strike brick against brick.", "figurative_meaning": "To completely destroy or retaliate fiercely.", "example": "Bhartiya sena ne dushman ki eent se eent baja di.", "language": "Hindi"},
    {"idiom": "उँगली उठाना", "literal_meaning": "To raise a finger.", "figurative_meaning": "To point fingers; blame someone.", "example": "Bina saboot ke kisi par ungali uthana galat hai.", "language": "Hindi"},
    {"idiom": "कमर कसना", "literal_meaning": "To tighten the belt.", "figurative_meaning": "To get ready for a difficult task.", "example": "Pariksha ke liye sabhi chhatron ne kamar kas li hai.", "language": "Hindi"},
    {"idiom": "कलेजे पर साँप लोटना", "literal_meaning": "A snake rolling on the liver.", "figurative_meaning": "To burn with jealousy.", "example": "Meri nayi gaadi dekh kar uske kaleje par saanp lot gaya.", "language": "Hindi"},
    {"idiom": "कान भरना", "literal_meaning": "To fill ears.", "figurative_meaning": "To poison someone's mind against another.", "example": "Rani hamesha saas ke khilaf bahu ke kaan bharti hai.", "language": "Hindi"},
    {"idiom": "खून पसीना एक करना", "literal_meaning": "To make blood and sweat one.", "figurative_meaning": "To work incredibly hard.", "example": "Kisan khet mein khoon pasina ek karke fasal ugate hain.", "language": "Hindi"},
    {"idiom": "गले का हार होना", "literal_meaning": "To be a necklace.", "figurative_meaning": "To be someone's favorite or deeply loved.", "example": "Beti apne pita ke gale ka haar hoti hai.", "language": "Hindi"},
    {"idiom": "गागर में सागर भरना", "literal_meaning": "To fill an ocean in a small pot.", "figurative_meaning": "To express deep or vast meaning in a few words.", "example": "Kabir ke dohe gagar mein sagar bharne ke saman hain.", "language": "Hindi"},
    {"idiom": "घी के दीये जलाना", "literal_meaning": "To light ghee lamps.", "figurative_meaning": "To celebrate a great victory or joy.", "example": "Ram ke Ayodhya lautne par logon ne ghee ke diye jalaye.", "language": "Hindi"},
    {"idiom": "चाँदी काटना", "literal_meaning": "To cut silver.", "figurative_meaning": "To earn a lot of money easily.", "example": "Diwali ke tyohar par dukaandar khub chandi katte hain.", "language": "Hindi"},
    {"idiom": "चुल्लू भर पानी में डूब मरना", "literal_meaning": "To drown in a handful of water.", "figurative_meaning": "To be deeply ashamed.", "example": "Chori pakde jane par uski sthiti chullu bhar pani mein doob marne jaisi thi.", "language": "Hindi"},
    {"idiom": "जान पर खेलना", "literal_meaning": "To play on life.", "figurative_meaning": "To take a massive risk.", "example": "Bachche ko aag se bachane ke liye maa apni jaan par khel gayi.", "language": "Hindi"},
    {"idiom": "जमीन आसमान एक करना", "literal_meaning": "To make earth and sky one.", "figurative_meaning": "To put in an immense effort.", "example": "Chor ko pakadne ke liye police ne zameen aasman ek kar diya.", "language": "Hindi"},
    {"idiom": "डाँवाडोल होना", "literal_meaning": "To be swaying.", "figurative_meaning": "To be unstable or wavering.", "example": "Nuksan ke baad uski aarthik sthiti danwadol ho gayi hai.", "language": "Hindi"},
    {"idiom": "दिन दूनी रात चौगुनी", "literal_meaning": "Double by day, quadruple by night.", "figurative_meaning": "To progress very rapidly.", "example": "Bhagwan kare tumhara vyapar din dooni raat chauguni unnati kare.", "language": "Hindi"},
    {"idiom": "दूध का दूध पानी का पानी", "literal_meaning": "Milk to milk, water to water.", "figurative_meaning": "Absolute and fair justice.", "example": "Nyayalay ne faisla sunakar doodh ka doodh aur pani ka pani kar diya.", "language": "Hindi"},
    {"idiom": "नमक मिर्च लगाना", "literal_meaning": "To apply salt and chili.", "figurative_meaning": "To exaggerate a story.", "example": "Usko har baat namak mirch lagakar batane ki aadat hai.", "language": "Hindi"},
    {"idiom": "पहाड़ टूट पड़ना", "literal_meaning": "A mountain breaking down.", "figurative_meaning": "A sudden major disaster or tragedy.", "example": "Pita ki mrityu se us par jaise pahad toot pada.", "language": "Hindi"},
    {"idiom": "पानी-पानी होना", "literal_meaning": "To become water-water.", "figurative_meaning": "To be extremely embarrassed.", "example": "Sabke samne jhooth pakda jane par woh pani-pani ho gaya.", "language": "Hindi"},
    {"idiom": "बाल की खाल निकालना", "literal_meaning": "To extract skin from hair.", "figurative_meaning": "To nitpick or over-analyze.", "example": "Vakil sahab ko har baat mein baal ki khaal nikalne ki aadat hai.", "language": "Hindi"},
    {"idiom": "मुँह की खाना", "literal_meaning": "To eat of the mouth.", "figurative_meaning": "To suffer a humiliating defeat.", "example": "Dushman sena ko yuddh mein munh ki khani padi.", "language": "Hindi"},
    {"idiom": "रंगे हाथों पकड़ना", "literal_meaning": "To catch with colored hands.", "figurative_meaning": "To catch red-handed.", "example": "Police ne chor ko chori karte hue range hathon pakad liya.", "language": "Hindi"},
    {"idiom": "लाल पीला होना", "literal_meaning": "To become red and yellow.", "figurative_meaning": "To be furious.", "example": "Mera nuksan dekh kar pitaji laal peele ho gaye.", "language": "Hindi"},
    {"idiom": "हवा का रुख देखना", "literal_meaning": "To see the direction of the wind.", "figurative_meaning": "To assess a situation before acting.", "example": "Samajhdar log humesha hawa ka rukh dekh kar kadam uthate hain.", "language": "Hindi"},
    {"idiom": "कलई खुलना", "literal_meaning": "The whitewash coming off.", "figurative_meaning": "A secret being revealed.", "example": "Chori pakde jane par uski saari kalai khul gayi.", "language": "Hindi"},
    {"idiom": "अंधेरे घर का उजाला", "literal_meaning": "Light of a dark house.", "figurative_meaning": "The only son or hope of a family.", "example": "Ravi apne mata-pita ke liye andhere ghar ka ujala hai.", "language": "Hindi"}
]

with open('data/raw/hindi.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

existing_idioms = {item['idiom'] for item in data}
filtered_new_idioms = [item for item in scraped_idioms if item['idiom'] not in existing_idioms]

data.extend(filtered_new_idioms)

with open('data/raw/hindi.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Scraped and added {len(filtered_new_idioms)} new unique idioms to hindi.json! Total is now {len(data)}.")
