import json
import os

new_idioms = [
    {
        "idiom": "മുറ്റത്തെ മുല്ലയ്ക്ക് മണമില്ല (Muttathe mullakku manamilla)",
        "literal_meaning": "The jasmine in the front yard has no fragrance.",
        "figurative_meaning": "Familiarity breeds contempt; undervaluing what is easily available.",
        "example": "Avan swantham nattile kalakaranmare bahumanikkilla, muttathe mullakku manamillallo.",
        "language": "Malayalam"
    },
    {
        "idiom": "പല തുള്ളി പെരുവെള്ളം (Pala thulli peru vellam)",
        "literal_meaning": "Many drops make a great flood.",
        "figurative_meaning": "Little contributions add up to a massive result.",
        "example": "Kooduthal aalkar sahayichal ee project theerkam, pala thulli peru vellam aanallo.",
        "language": "Malayalam"
    },
    {
        "idiom": "മിന്നുന്നതെല്ലാം പൊന്നല്ല (Minnunnathellam ponnalla)",
        "literal_meaning": "All that glitters is not gold.",
        "figurative_meaning": "Appearances can be deceptive.",
        "example": "Aa puthiya investment thട്ടിppaayirunnu, minnunnathellam ponnalla ennu ippol manassilayi.",
        "language": "Malayalam"
    },
    {
        "idiom": "കാക്കയ്ക്കും തൻ കുഞ്ഞ് പൊൻകുഞ്ഞ് (Kakkakkum than kunju pon kunju)",
        "literal_meaning": "Even to a crow, its own child is a golden child.",
        "figurative_meaning": "Everyone loves their own offspring dearly, regardless of flaws.",
        "example": "Avan ethra thettu cheythalum ammakku avan nallavana, kakkakkum than kunju pon kunjanallo.",
        "language": "Malayalam"
    },
    {
        "idiom": "എരിയുന്ന തീയിൽ എണ്ണ ഒഴിക്കുക (Eriyunna theeyil enna ozhikkuka)",
        "literal_meaning": "To pour oil into a burning fire.",
        "figurative_meaning": "To make a bad situation worse; adding fuel to the fire.",
        "example": "Avar thammil ulla prashnathil nee idapedanda, eriyunna theeyil enna ozhikkunnu.",
        "language": "Malayalam"
    },
    {
        "idiom": "കഴുതയ്ക്കറിയാമോ കർപ്പൂരത്തിന്റെ മണം (Kazhuthaykkariyamoo karpoorathinte manam)",
        "literal_meaning": "Does a donkey know the smell of camphor?",
        "figurative_meaning": "An uncultured person cannot appreciate the value of fine things.",
        "example": "Ee nalla sangeetham avan manassilavilla, kazhuthaykkariyamo karpoorathinte manam.",
        "language": "Malayalam"
    },
    {
        "idiom": "മിണ്ടാപ്പൂച്ച കലം ഉ‌ടയ്ക്കും (Minda poocha kalam udaikkum)",
        "literal_meaning": "The silent cat breaks the pot.",
        "figurative_meaning": "Quiet people can sometimes be the most mischievous or surprising.",
        "example": "Avan onnum mindillengilum achanodu ellam paranju, minda poocha kalam udaikkum.",
        "language": "Malayalam"
    },
    {
        "idiom": "പാമ്പിന്റെ കാല് പാമ്പറിയും (Paambinte kaalu paambariyum)",
        "literal_meaning": "Only a snake knows a snake's legs.",
        "figurative_meaning": "It takes one to know one; people of similar nature understand each other.",
        "example": "Kallante budhi kallane ariyu, paambinte kaalu paambariyum.",
        "language": "Malayalam"
    },
    {
        "idiom": "വേലിയിൽ ഇരുന്ന പാമ്പിനെ എടുത്തു തോളിൽ വയ്ക്കുക (Veliyil irunna paambine eduthu tholil vaykuka)",
        "literal_meaning": "Taking a snake sitting on the fence and putting it on the shoulder.",
        "figurative_meaning": "To unnecessarily invite trouble into one's life.",
        "example": "Veruthe avante prashnathil idapettu, veliyil irunna paambine tholil vechathu poleyayi.",
        "language": "Malayalam"
    },
    {
        "idiom": "ചക്ക വീണത് മുയൽ ചത്തത് (Chakka veenathu muyal chathathu)",
        "literal_meaning": "The jackfruit fell, the rabbit died.",
        "figurative_meaning": "A pure coincidence mistaken for cause and effect.",
        "example": "Avan officeil ninnu poyathum current vannu, chakka veenathu muyal chathathu pole.",
        "language": "Malayalam"
    },
    {
        "idiom": "വേലി തന്നെ വിളവു തിന്നുക (Veli thanne vilavu thinnuka)",
        "literal_meaning": "The fence itself eating the crop.",
        "figurative_meaning": "The protector turning into the predator.",
        "example": "Police kaaran thanne moshanam nadathiyal pinne enth cheyyan? Veli thanne vilavu thinnukayanu.",
        "language": "Malayalam"
    },
    {
        "idiom": "കാക്ക കുളിച്ചാൽ കൊക്കാകുമോ (Kaakka kulichal kokkakumo)",
        "literal_meaning": "Will a crow become a crane if it bathes?",
        "figurative_meaning": "You cannot change someone's inherent nature with superficial changes.",
        "example": "Avan ethra panam undakkiyalum avante swabhavam maarilla, kaakka kulichal kokkakumo?",
        "language": "Malayalam"
    },
    {
        "idiom": "വൈദ്യൻ കല്പിച്ചതും രോഗി ഇച്ഛിച്ചതും പാല് (Vaidyan kalpichathum rogi ichichathum paalu)",
        "literal_meaning": "What the doctor prescribed and what the patient desired was milk.",
        "figurative_meaning": "A win-win situation where advice perfectly matches desires.",
        "example": "Enikku pokan madi aayirunnu, appozha event maattivechathu, vaidyan kalpichathum rogi ichichathum paalu.",
        "language": "Malayalam"
    },
    {
        "idiom": "ഉള്ളതുകൊണ്ട് ഓണം പോലെ (Ullathukondu onam pole)",
        "literal_meaning": "Celebrate like Onam with whatever is available.",
        "figurative_meaning": "Make the best out of what you have.",
        "example": "Ee cheriya veettil namukku santhoshikkanam, ullathukondu onam pole.",
        "language": "Malayalam"
    },
    {
        "idiom": "അക്കരെ നിന്നാൽ ഇക്കരെ പച്ച (Akkare ninnal ikkare paccha)",
        "literal_meaning": "From the other shore, this shore looks green.",
        "figurative_meaning": "The grass is always greener on the other side.",
        "example": "Avanu vadeshathe jeevitham valiya ishtamanu, pakshe akkare ninnal ikkare paccha aanu.",
        "language": "Malayalam"
    },
    {
        "idiom": "ആന മെലിഞ്ഞാൽ തൊഴുത്തിൽ കെട്ടാമോ (Aana melinjal thozhuthil kettamo)",
        "literal_meaning": "If an elephant becomes thin, can it be tied in a cowshed?",
        "figurative_meaning": "Even in decline, a great person or thing retains their inherent stature.",
        "example": "Avarude kudumbam ippol daridryathilanengilum avarude anthassu poyitilla, aana melinjal thozhuthil kettamo.",
        "language": "Malayalam"
    },
    {
        "idiom": "പടപേടിച്ച് പന്തളത്ത് ചെന്നപ്പോൾ അവിടെ പന്തം കൊളുത്തി പട (Pada pedichu panthalathu chennappol avide pantham koluthi pada)",
        "literal_meaning": "Feared the war and went to Panthalam, only to find a war with torches there.",
        "figurative_meaning": "Out of the frying pan and into the fire.",
        "example": "Nattile prashnam pedichu townil vannappol ivide athilum valiya prashnam, pada pedichu panthalathu chennappol poleyayi.",
        "language": "Malayalam"
    },
    {
        "idiom": "നായ്ക്കenthu തേങ്ങ (Naaykkenthu thenga)",
        "literal_meaning": "What does a coconut mean to a dog?",
        "figurative_meaning": "Giving something valuable to someone who cannot appreciate or use it.",
        "example": "Avanu ee puthiya laptop koduthathu verutheyayi, naaykkenthu thenga.",
        "language": "Malayalam"
    },
    {
        "idiom": "താൻ പിടിച്ച മുയലിന് കൊമ്പ് മൂന്ന് (Thaan pidicha muyalinu kombu moonnu)",
        "literal_meaning": "The rabbit I caught has three horns.",
        "figurative_meaning": "Stubbornly clinging to a false or absurd claim.",
        "example": "Avan parayunnathu thettanennu thelinjittum sammathikkilla, thaan pidicha muyalinu kombu moonnu enna avastha.",
        "language": "Malayalam"
    },
    {
        "idiom": "ചുട്ടയിലെ ശീലം ചുടല വരെ (Chuttayile sheelam chudala vare)",
        "literal_meaning": "Habits formed in the cradle last till the funeral pyre.",
        "figurative_meaning": "Old habits die hard.",
        "example": "Avan eppozhum kalla parayum, athu maarilla, chuttayile sheelam chudala vare.",
        "language": "Malayalam"
    }
]

file_path = 'data/raw/malayalam.json'

# Load existing if any
data = []
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            data = []

# Deduplicate
existing_idioms = {item['idiom'].split(' (')[0].strip() for item in data if 'idiom' in item}
filtered_new_idioms = []
for item in new_idioms:
    base_idiom = item['idiom'].split(' (')[0].strip()
    if base_idiom not in existing_idioms:
        filtered_new_idioms.append(item)

data.extend(filtered_new_idioms)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Added {len(filtered_new_idioms)} new unique idioms to malayalam.json! Total is now {len(data)}.")
