# ─── Pretty labels for every feature column ───────────────────────
NICE_FEAT = {
    # direct numeric / binary columns
    "cast_power":                 "Cast Star-Power",
    "director_power":             "Director Star-Power",
    "writers_power":              "Writer Star-Power",
    "producers_power":            "Producer Star-Power",
    "production_companies_power": "Studio Power",
    "production_countries_power": "Country Market Power",
    "overview_sentiment":         "Overview Sentiment",
    "release_year":               "Release Year",
    "release_month":              "Release Month",
    "is_holiday_release":         "Holiday Release (Y/N)",
    "big_studio":                 "Major Studio (Y/N)",
    "is_sequel":                  "Sequel Flag (Y/N)",
    "director_hit_ratio":         "Director Track Record",
    "runtime":                    "Runtime",
    "log_budget":                 "Budget",
    "success_prob":               "Success Probability",
}

# ----- add all genre_XXX columns ---------------------------------
GENRE_NAMES = [
    "action","adventure","animation","comedy","crime","documentary","drama",
    "family","fantasy","history","horror","music","mystery","romance",
    "science fiction","thriller","tv movie","war","western"
]
NICE_FEAT.update({f"genre_{g}": f"Genre ({g.title()})" for g in GENRE_NAMES})

# ----- add all lang_XXX columns ----------------------------------
LANG_MAP = {
    "afrikaans":"Afrikaans","akan":"Akan","albanian":"Albanian","amharic":"Amharic",
    "arabic":"Arabic","armenian":"Armenian","azerbaijani":"Azerbaijani","bambara":"Bambara",
    "basque":"Basque","bengali":"Bengali","bosnian":"Bosnian","breton":"Breton",
    "bulgarian":"Bulgarian","burmese":"Burmese","cantonese":"Cantonese",
    "catalan":"Catalan","chechen":"Chechen","chichewa; nyanja":"Chichewa/Nyanja",
    "cornish":"Cornish","corsican":"Corsican","cree":"Cree","croatian":"Croatian",
    "czech":"Czech","danish":"Danish","divehi":"Divehi","dutch":"Dutch",
    "english":"English","esperanto":"Esperanto","estonian":"Estonian",
    "faroese":"Faroese","finnish":"Finnish","french":"French","fulah":"Fulah",
    "gaelic":"Gaelic","galician":"Galician","georgian":"Georgian","german":"German",
    "greek":"Greek","guarani":"Guarani","gujarati":"Gujarati",
    "haitian; haitian creole":"Haitian Creole","hausa":"Hausa","hebrew":"Hebrew",
    "hindi":"Hindi","hungarian":"Hungarian","icelandic":"Icelandic","ido":"Ido",
    "indonesian":"Indonesian","inuktitut":"Inuktitut","irish":"Irish","italian":"Italian",
    "japanese":"Japanese","kannada":"Kannada","kashmiri":"Kashmiri","khmer":"Khmer",
    "korean":"Korean","kurdish":"Kurdish","lao":"Lao","latin":"Latin","latvian":"Latvian",
    "lingala":"Lingala","lithuanian":"Lithuanian","malay":"Malay",
    "malayalam":"Malayalam","maltese":"Maltese","mandarin":"Mandarin","maori":"Maori",
    "marathi":"Marathi","moldavian":"Moldavian","mongolian":"Mongolian","navajo":"Navajo",
    "nepali":"Nepali","no language":"No Dialogue","northern sami":"Northern Sami",
    "norwegian":"Norwegian","persian":"Persian","polish":"Polish","portuguese":"Portuguese",
    "punjabi":"Punjabi","pushto":"Pashto","quechua":"Quechua","romanian":"Romanian",
    "russian":"Russian","samoan":"Samoan","sanskrit":"Sanskrit","serbian":"Serbian",
    "serbo-croatian":"Serbo-Croatian","sindhi":"Sindhi","sinhalese":"Sinhalese",
    "slovak":"Slovak","slovenian":"Slovenian","somali":"Somali","sotho":"Sotho",
    "spanish":"Spanish","swahili":"Swahili","swedish":"Swedish","tagalog":"Tagalog",
    "tahitian":"Tahitian","tamil":"Tamil","tatar":"Tatar","telugu":"Telugu","thai":"Thai",
    "tibetan":"Tibetan","tonga":"Tongan","tswana":"Tswana","turkish":"Turkish",
    "ukrainian":"Ukrainian","urdu":"Urdu","vietnamese":"Vietnamese","welsh":"Welsh",
    "xhosa":"Xhosa","yiddish":"Yiddish","yoruba":"Yoruba","zulu":"Zulu",
}
NICE_FEAT.update({f"lang_{k}": f"Language ({v})" for k, v in LANG_MAP.items()})

# ----- add TF-IDF columns ---------------------------------------
NICE_FEAT.update({f"tfidf_{i}": f"Overview TF-IDF" for i in range(50)})
