#!/usr/bin/env python
WORD_LIST = [
  "abandon",
  "ability",
  "able",
  "about",
  "above",
  "absent",
  "absorb",
  "abstract",
  "absurd",
  "abuse",
  "access",
  "accident",
  "account",
  "accuse",
  "achieve",
  "acid",
  "acoustic",
  "acquire",
  "across",
  "act",
  "action",
  "actor",
  "actress",
  "actual",
  "adapt",
  "add",
  "addict",
  "address",
  "adjust",
  "admit",
  "adult",
  "advance",
  "advice",
  "aerobic",
  "affair",
  "afford",
  "afraid",
  "again",
  "age",
  "agent",
  "agree",
  "ahead",
  "aim",
  "air",
  "airport",
  "aisle",
  "alarm",
  "album",
  "alcohol",
  "alert",
  "alien",
  "all",
  "alley",
  "allow",
  "almost",
  "alone",
  "alpha",
  "already",
  "also",
  "alter",
  "always",
  "amateur",
  "amazing",
  "among",
  "amount",
  "amused",
  "analyst",
  "anchor",
  "ancient",
  "anger",
  "angle",
  "angry",
  "animal",
  "ankle",
  "announce",
  "annual",
  "another",
  "answer",
  "antenna",
  "antique",
  "anxiety",
  "any",
  "apart",
  "apology",
  "appear",
  "apple",
  "approve",
  "april",
  "arch",
  "arctic",
  "area",
  "arena",
  "argue",
  "arm",
  "armed",
  "armor",
  "army",
  "around",
  "arrange",
  "arrest",
  "arrive",
  "arrow",
  "art",
  "artefact",
  "artist",
  "artwork",
  "ask",
  "aspect",
  "assault",
  "asset",
  "assist",
  "assume",
  "asthma",
  "athlete",
  "atom",
  "attack",
  "attend",
  "attitude",
  "attract",
  "auction",
  "audit",
  "august",
  "aunt",
  "author",
  "auto",
  "autumn",
  "average",
  "avocado",
  "avoid",
  "awake",
  "aware",
  "away",
  "awesome",
  "awful",
  "awkward",
  "axis",
  "baby",
  "bachelor",
  "bacon",
  "badge",
  "bag",
  "balance",
  "balcony",
  "ball",
  "bamboo",
  "banana",
  "banner",
  "bar",
  "barely",
  "bargain",
  "barrel",
  "base",
  "basic",
  "basket",
  "battle",
  "beach",
  "bean",
  "beauty",
  "because",
  "become",
  "beef",
  "before",
  "begin",
  "behave",
  "behind",
  "believe",
  "below",
  "belt",
  "bench",
  "benefit",
  "best",
  "betray",
  "better",
  "between",
  "beyond",
  "bicycle",
  "bid",
  "bike",
  "bind",
  "biology",
  "bird",
  "birth",
  "bitter",
  "black",
  "blade",
  "blame",
  "blanket",
  "blast",
  "bleak",
  "bless",
  "blind",
  "blood",
  "blossom",
  "blouse",
  "blue",
  "blur",
  "blush",
  "board",
  "boat",
  "body",
  "boil",
  "bomb",
  "bone",
  "bonus",
  "book",
  "boost",
  "border",
  "boring",
  "borrow",
  "boss",
  "bottom",
  "bounce",
  "box",
  "boy",
  "bracket",
  "brain",
  "brand",
  "brass",
  "brave",
  "bread",
  "breeze",
  "brick",
  "bridge",
  "brief",
  "bright",
  "bring",
  "brisk",
  "broccoli",
  "broken",
  "bronze",
  "broom",
  "brother",
  "brown",
  "brush",
  "bubble",
  "buddy",
  "budget",
  "buffalo",
  "build",
  "bulb",
  "bulk",
  "bullet",
  "bundle",
  "bunker",
  "burden",
  "burger",
  "burst",
  "bus",
  "business",
  "busy",
  "butter",
  "buyer",
  "buzz",
  "cabbage",
  "cabin",
  "cable",
  "cactus",
  "cage",
  "cake",
  "call",
  "calm",
  "camera",
  "camp",
  "can",
  "canal",
  "cancel",
  "candy",
  "cannon",
  "canoe",
  "canvas",
  "canyon",
  "capable",
  "capital",
  "captain",
  "car",
  "carbon",
  "card",
  "cargo",
  "carpet",
  "carry",
  "cart",
  "case",
  "cash",
  "casino",
  "castle",
  "casual",
  "cat",
  "catalog",
  "catch",
  "category",
  "cattle",
  "caught",
  "cause",
  "caution",
  "cave",
  "ceiling",
  "celery",
  "cement",
  "census",
  "century",
  "cereal",
  "certain",
  "chair",
  "chalk",
  "champion",
  "change",
  "chaos",
  "chapter",
  "charge",
  "chase",
  "chat",
  "cheap",
  "check",
  "cheese",
  "chef",
  "cherry",
  "chest",
  "chicken",
  "chief",
  "child",
  "chimney",
  "choice",
  "choose",
  "chronic",
  "chuckle",
  "chunk",
  "churn",
  "cigar",
  "cinnamon",
  "circle",
  "citizen",
  "city",
  "civil",
  "claim",
  "clap",
  "clarify",
  "claw",
  "clay",
  "clean",
  "clerk",
  "clever",
  "click",
  "client",
  "cliff",
  "climb",
  "clinic",
  "clip",
  "clock",
  "clog",
  "close",
  "cloth",
  "cloud",
  "clown",
  "club",
  "clump",
  "cluster",
  "clutch",
  "coach",
  "coast",
  "coconut",
  "code",
  "coffee",
  "coil",
  "coin",
  "collect",
  "color",
  "column",
  "combine",
  "come",
  "comfort",
  "comic",
  "common",
  "company",
  "concert",
  "conduct",
  "confirm",
  "congress",
  "connect",
  "consider",
  "control",
  "convince",
  "cook",
  "cool",
  "copper",
  "copy",
  "coral",
  "core",
  "corn",
  "correct",
  "cost",
  "cotton",
  "couch",
  "country",
  "couple",
  "course",
  "cousin",
  "cover",
  "coyote",
  "crack",
  "cradle",
  "craft",
  "cram",
  "crane",
  "crash",
  "crater",
  "crawl",
  "crazy",
  "cream",
  "credit",
  "creek",
  "crew",
  "cricket",
  "crime",
  "crisp",
  "critic",
  "crop",
  "cross",
  "crouch",
  "crowd",
  "crucial",
  "cruel",
  "cruise",
  "crumble",
  "crunch",
  "crush",
  "cry",
  "crystal",
  "cube",
  "culture",
  "cup",
  "cupboard",
  "curious",
  "current",
  "curtain",
  "curve",
  "cushion",
  "custom",
  "cute",
  "cycle",
  "dad",
  "damage",
  "damp",
  "dance",
  "danger",
  "daring",
  "dash",
  "daughter",
  "dawn",
  "day",
  "deal",
  "debate",
  "debris",
  "decade",
  "december",
  "decide",
  "decline",
  "decorate",
  "decrease",
  "deer",
  "defense",
  "define",
  "defy",
  "degree",
  "delay",
  "deliver",
  "demand",
  "demise",
  "denial",
  "dentist",
  "deny",
  "depart",
  "depend",
  "deposit",
  "depth",
  "deputy",
  "derive",
  "describe",
  "desert",
  "design",
  "desk",
  "despair",
  "destroy",
  "detail",
  "detect",
  "develop",
  "device",
  "devote",
  "diagram",
  "dial",
  "diamond",
  "diary",
  "dice",
  "diesel",
  "diet",
  "differ",
  "digital",
  "dignity",
  "dilemma",
  "dinner",
  "dinosaur",
  "direct",
  "dirt",
  "disagree",
  "discover",
  "disease",
  "dish",
  "dismiss",
  "disorder",
  "display",
  "distance",
  "divert",
  "divide",
  "divorce",
  "dizzy",
  "doctor",
  "document",
  "dog",
  "doll",
  "dolphin",
  "domain",
  "donate",
  "donkey",
  "donor",
  "door",
  "dose",
  "double",
  "dove",
  "draft",
  "dragon",
  "drama",
  "drastic",
  "draw",
  "dream",
  "dress",
  "drift",
  "drill",
  "drink",
  "drip",
  "drive",
  "drop",
  "drum",
  "dry",
  "duck",
  "dumb",
  "dune",
  "during",
  "dust",
  "dutch",
  "duty",
  "dwarf",
  "dynamic",
  "eager",
  "eagle",
  "early",
  "earn",
  "earth",
  "easily",
  "east",
  "easy",
  "echo",
  "ecology",
  "economy",
  "edge",
  "edit",
  "educate",
  "effort",
  "egg",
  "eight",
  "either",
  "elbow",
  "elder",
  "electric",
  "elegant",
  "element",
  "elephant",
  "elevator",
  "elite",
  "else",
  "embark",
  "embody",
  "embrace",
  "emerge",
  "emotion",
  "employ",
  "empower",
  "empty",
  "enable",
  "enact",
  "end",
  "endless",
  "endorse",
  "enemy",
  "energy",
  "enforce",
  "engage",
  "engine",
  "enhance",
  "enjoy",
  "enlist",
  "enough",
  "enrich",
  "enroll",
  "ensure",
  "enter",
  "entire",
  "entry",
  "envelope",
  "episode",
  "equal",
  "equip",
  "era",
  "erase",
  "erode",
  "erosion",
  "error",
  "erupt",
  "escape",
  "essay",
  "essence",
  "estate",
  "eternal",
  "ethics",
  "evidence",
  "evil",
  "evoke",
  "evolve",
  "exact",
  "example",
  "excess",
  "exchange",
  "excite",
  "exclude",
  "excuse",
  "execute",
  "exercise",
  "exhaust",
  "exhibit",
  "exile",
  "exist",
  "exit",
  "exotic",
  "expand",
  "expect",
  "expire",
  "explain",
  "expose",
  "express",
  "extend",
  "extra",
  "eye",
  "eyebrow",
  "fabric",
  "face",
  "faculty",
  "fade",
  "faint",
  "faith",
  "fall",
  "false",
  "fame",
  "family",
  "famous",
  "fan",
  "fancy",
  "fantasy",
  "farm",
  "fashion",
  "fat",
  "fatal",
  "father",
  "fatigue",
  "fault",
  "favorite",
  "feature",
  "february",
  "federal",
  "fee",
  "feed",
  "feel",
  "female",
  "fence",
  "festival",
  "fetch",
  "fever",
  "few",
  "fiber",
  "fiction",
  "field",
  "figure",
  "file",
  "film",
  "filter",
  "final",
  "find",
  "fine",
  "finger",
  "finish",
  "fire",
  "firm",
  "first",
  "fiscal",
  "fish",
  "fit",
  "fitness",
  "fix",
  "flag",
  "flame",
  "flash",
  "flat",
  "flavor",
  "flee",
  "flight",
  "flip",
  "float",
  "flock",
  "floor",
  "flower",
  "fluid",
  "flush",
  "fly",
  "foam",
  "focus",
  "fog",
  "foil",
  "fold",
  "follow",
  "food",
  "foot",
  "force",
  "forest",
  "forget",
  "fork",
  "fortune",
  "forum",
  "forward",
  "fossil",
  "foster",
  "found",
  "fox",
  "fragile",
  "frame",
  "frequent",
  "fresh",
  "friend",
  "fringe",
  "frog",
  "front",
  "frost",
  "frown",
  "frozen",
  "fruit",
  "fuel",
  "fun",
  "funny",
  "furnace",
  "fury",
  "future",
  "gadget",
  "gain",
  "galaxy",
  "gallery",
  "game",
  "gap",
  "garage",
  "garbage",
  "garden",
  "garlic",
  "garment",
  "gas",
  "gasp",
  "gate",
  "gather",
  "gauge",
  "gaze",
  "general",
  "genius",
  "genre",
  "gentle",
  "genuine",
  "gesture",
  "ghost",
  "giant",
  "gift",
  "giggle",
  "ginger",
  "giraffe",
  "girl",
  "give",
  "glad",
  "glance",
  "glare",
  "glass",
  "glide",
  "glimpse",
  "globe",
  "gloom",
  "glory",
  "glove",
  "glow",
  "glue",
  "goat",
  "goddess",
  "gold",
  "good",
  "goose",
  "gorilla",
  "gospel",
  "gossip",
  "govern",
  "gown",
  "grab",
  "grace",
  "grain",
  "grant",
  "grape",
  "grass",
  "gravity",
  "great",
  "green",
  "grid",
  "grief",
  "grit",
  "grocery",
  "group",
  "grow",
  "grunt",
  "guard",
  "guess",
  "guide",
  "guilt",
  "guitar",
  "gun",
  "gym",
  "habit",
  "hair",
  "half",
  "hammer",
  "hamster",
  "hand",
  "happy",
  "harbor",
  "hard",
  "harsh",
  "harvest",
  "hat",
  "have",
  "hawk",
  "hazard",
  "head",
  "health",
  "heart",
  "heavy",
  "hedgehog",
  "height",
  "hello",
  "helmet",
  "help",
  "hen",
  "hero",
  "hidden",
  "high",
  "hill",
  "hint",
  "hip",
  "hire",
  "history",
  "hobby",
  "hockey",
  "hold",
  "hole",
  "holiday",
  "hollow",
  "home",
  "honey",
  "hood",
  "hope",
  "horn",
  "horror",
  "horse",
  "hospital",
  "host",
  "hotel",
  "hour",
  "hover",
  "hub",
  "huge",
  "human",
  "humble",
  "humor",
  "hundred",
  "hungry",
  "hunt",
  "hurdle",
  "hurry",
  "hurt",
  "husband",
  "hybrid",
  "ice",
  "icon",
  "idea",
  "identify",
  "idle",
  "ignore",
  "ill",
  "illegal",
  "illness",
  "image",
  "imitate",
  "immense",
  "immune",
  "impact",
  "impose",
  "improve",
  "impulse",
  "inch",
  "include",
  "income",
  "increase",
  "index",
  "indicate",
  "indoor",
  "industry",
  "infant",
  "inflict",
  "inform",
  "inhale",
  "inherit",
  "initial",
  "inject",
  "injury",
  "inmate",
  "inner",
  "innocent",
  "input",
  "inquiry",
  "insane",
  "insect",
  "inside",
  "inspire",
  "install",
  "intact",
  "interest",
  "into",
  "invest",
  "invite",
  "involve",
  "iron",
  "island",
  "isolate",
  "issue",
  "item",
  "ivory",
  "jacket",
  "jaguar",
  "jar",
  "jazz",
  "jealous",
  "jeans",
  "jelly",
  "jewel",
  "job",
  "join",
  "joke",
  "journey",
  "joy",
  "judge",
  "juice",
  "jump",
  "jungle",
  "junior",
  "junk",
  "just",
  "kangaroo",
  "keen",
  "keep",
  "ketchup",
  "key",
  "kick",
  "kid",
  "kidney",
  "kind",
  "kingdom",
  "kiss",
  "kit",
  "kitchen",
  "kite",
  "kitten",
  "kiwi",
  "knee",
  "knife",
  "knock",
  "know",
  "lab",
  "label",
  "labor",
  "ladder",
  "lady",
  "lake",
  "lamp",
  "language",
  "laptop",
  "large",
  "later",
  "latin",
  "laugh",
  "laundry",
  "lava",
  "law",
  "lawn",
  "lawsuit",
  "layer",
  "lazy",
  "leader",
  "leaf",
  "learn",
  "leave",
  "lecture",
  "left",
  "leg",
  "legal",
  "legend",
  "leisure",
  "lemon",
  "lend",
  "length",
  "lens",
  "leopard",
  "lesson",
  "letter",
  "level",
  "liar",
  "liberty",
  "library",
  "license",
  "life",
  "lift",
  "light",
  "like",
  "limb",
  "limit",
  "link",
  "lion",
  "liquid",
  "list",
  "little",
  "live",
  "lizard",
  "load",
  "loan",
  "lobster",
  "local",
  "lock",
  "logic",
  "lonely",
  "long",
  "loop",
  "lottery",
  "loud",
  "lounge",
  "love",
  "loyal",
  "lucky",
  "luggage",
  "lumber",
  "lunar",
  "lunch",
  "luxury",
  "lyrics",
  "machine",
  "mad",
  "magic",
  "magnet",
  "maid",
  "mail",
  "main",
  "major",
  "make",
  "mammal",
  "man",
  "manage",
  "mandate",
  "mango",
  "mansion",
  "manual",
  "maple",
  "marble",
  "march",
  "margin",
  "marine",
  "market",
  "marriage",
  "mask",
  "mass",
  "master",
  "match",
  "material",
  "math",
  "matrix",
  "matter",
  "maximum",
  "maze",
  "meadow",
  "mean",
  "measure",
  "meat",
  "mechanic",
  "medal",
  "media",
  "melody",
  "melt",
  "member",
  "memory",
  "mention",
  "menu",
  "mercy",
  "merge",
  "merit",
  "merry",
  "mesh",
  "message",
  "metal",
  "method",
  "middle",
  "midnight",
  "milk",
  "million",
  "mimic",
  "mind",
  "minimum",
  "minor",
  "minute",
  "miracle",
  "mirror",
  "misery",
  "miss",
  "mistake",
  "mix",
  "mixed",
  "mixture",
  "mobile",
  "model",
  "modify",
  "mom",
  "moment",
  "monitor",
  "monkey",
  "monster",
  "month",
  "moon",
  "moral",
  "more",
  "morning",
  "mosquito",
  "mother",
  "motion",
  "motor",
  "mountain",
  "mouse",
  "move",
  "movie",
  "much",
  "muffin",
  "mule",
  "multiply",
  "muscle",
  "museum",
  "mushroom",
  "music",
  "must",
  "mutual",
  "myself",
  "mystery",
  "myth",
  "naive",
  "name",
  "napkin",
  "narrow",
  "nasty",
  "nation",
  "nature",
  "near",
  "neck",
  "need",
  "negative",
  "neglect",
  "neither",
  "nephew",
  "nerve",
  "nest",
  "net",
  "network",
  "neutral",
  "never",
  "news",
  "next",
  "nice",
  "night",
  "noble",
  "noise",
  "nominee",
  "noodle",
  "normal",
  "north",
  "nose",
  "notable",
  "note",
  "nothing",
  "notice",
  "novel",
  "now",
  "nuclear",
  "number",
  "nurse",
  "nut",
  "oak",
  "obey",
  "object",
  "oblige",
  "obscure",
  "observe",
  "obtain",
  "obvious",
  "occur",
  "ocean",
  "october",
  "odor",
  "off",
  "offer",
  "office",
  "often",
  "oil",
  "okay",
  "old",
  "olive",
  "olympic",
  "omit",
  "once",
  "one",
  "onion",
  "online",
  "only",
  "open",
  "opera",
  "opinion",
  "oppose",
  "option",
  "orange",
  "orbit",
  "orchard",
  "order",
  "ordinary",
  "organ",
  "orient",
  "original",
  "orphan",
  "ostrich",
  "other",
  "outdoor",
  "outer",
  "output",
  "outside",
  "oval",
  "oven",
  "over",
  "own",
  "owner",
  "oxygen",
  "oyster",
  "ozone",
  "pact",
  "paddle",
  "page",
  "pair",
  "palace",
  "palm",
  "panda",
  "panel",
  "panic",
  "panther",
  "paper",
  "parade",
  "parent",
  "park",
  "parrot",
  "party",
  "pass",
  "patch",
  "path",
  "patient",
  "patrol",
  "pattern",
  "pause",
  "pave",
  "payment",
  "peace",
  "peanut",
  "pear",
  "peasant",
  "pelican",
  "pen",
  "penalty",
  "pencil",
  "people",
  "pepper",
  "perfect",
  "permit",
  "person",
  "pet",
  "phone",
  "photo",
  "phrase",
  "physical",
  "piano",
  "picnic",
  "picture",
  "piece",
  "pig",
  "pigeon",
  "pill",
  "pilot",
  "pink",
  "pioneer",
  "pipe",
  "pistol",
  "pitch",
  "pizza",
  "place",
  "planet",
  "plastic",
  "plate",
  "play",
  "please",
  "pledge",
  "pluck",
  "plug",
  "plunge",
  "poem",
  "poet",
  "point",
  "polar",
  "pole",
  "police",
  "pond",
  "pony",
  "pool",
  "popular",
  "portion",
  "position",
  "possible",
  "post",
  "potato",
  "pottery",
  "poverty",
  "powder",
  "power",
  "practice",
  "praise",
  "predict",
  "prefer",
  "prepare",
  "present",
  "pretty",
  "prevent",
  "price",
  "pride",
  "primary",
  "print",
  "priority",
  "prison",
  "private",
  "prize",
  "problem",
  "process",
  "produce",
  "profit",
  "program",
  "project",
  "promote",
  "proof",
  "property",
  "prosper",
  "protect",
  "proud",
  "provide",
  "public",
  "pudding",
  "pull",
  "pulp",
  "pulse",
  "pumpkin",
  "punch",
  "pupil",
  "puppy",
  "purchase",
  "purity",
  "purpose",
  "purse",
  "push",
  "put",
  "puzzle",
  "pyramid",
  "quality",
  "quantum",
  "quarter",
  "question",
  "quick",
  "quit",
  "quiz",
  "quote",
  "rabbit",
  "raccoon",
  "race",
  "rack",
  "radar",
  "radio",
  "rail",
  "rain",
  "raise",
  "rally",
  "ramp",
  "ranch",
  "random",
  "range",
  "rapid",
  "rare",
  "rate",
  "rather",
  "raven",
  "raw",
  "razor",
  "ready",
  "real",
  "reason",
  "rebel",
  "rebuild",
  "recall",
  "receive",
  "recipe",
  "record",
  "recycle",
  "reduce",
  "reflect",
  "reform",
  "refuse",
  "region",
  "regret",
  "regular",
  "reject",
  "relax",
  "release",
  "relief",
  "rely",
  "remain",
  "remember",
  "remind",
  "remove",
  "render",
  "renew",
  "rent",
  "reopen",
  "repair",
  "repeat",
  "replace",
  "report",
  "require",
  "rescue",
  "resemble",
  "resist",
  "resource",
  "response",
  "result",
  "retire",
  "retreat",
  "return",
  "reunion",
  "reveal",
  "review",
  "reward",
  "rhythm",
  "rib",
  "ribbon",
  "rice",
  "rich",
  "ride",
  "ridge",
  "rifle",
  "right",
  "rigid",
  "ring",
  "riot",
  "ripple",
  "risk",
  "ritual",
  "rival",
  "river",
  "road",
  "roast",
  "robot",
  "robust",
  "rocket",
  "romance",
  "roof",
  "rookie",
  "room",
  "rose",
  "rotate",
  "rough",
  "round",
  "route",
  "royal",
  "rubber",
  "rude",
  "rug",
  "rule",
  "run",
  "runway",
  "rural",
  "sad",
  "saddle",
  "sadness",
  "safe",
  "sail",
  "salad",
  "salmon",
  "salon",
  "salt",
  "salute",
  "same",
  "sample",
  "sand",
  "satisfy",
  "satoshi",
  "sauce",
  "sausage",
  "save",
  "say",
  "scale",
  "scan",
  "scare",
  "scatter",
  "scene",
  "scheme",
  "school",
  "science",
  "scissors",
  "scorpion",
  "scout",
  "scrap",
  "screen",
  "script",
  "scrub",
  "sea",
  "search",
  "season",
  "seat",
  "second",
  "secret",
  "section",
  "security",
  "seed",
  "seek",
  "segment",
  "select",
  "sell",
  "seminar",
  "senior",
  "sense",
  "sentence",
  "series",
  "service",
  "session",
  "settle",
  "setup",
  "seven",
  "shadow",
  "shaft",
  "shallow",
  "share",
  "shed",
  "shell",
  "sheriff",
  "shield",
  "shift",
  "shine",
  "ship",
  "shiver",
  "shock",
  "shoe",
  "shoot",
  "shop",
  "short",
  "shoulder",
  "shove",
  "shrimp",
  "shrug",
  "shuffle",
  "shy",
  "sibling",
  "sick",
  "side",
  "siege",
  "sight",
  "sign",
  "silent",
  "silk",
  "silly",
  "silver",
  "similar",
  "simple",
  "since",
  "sing",
  "siren",
  "sister",
  "situate",
  "six",
  "size",
  "skate",
  "sketch",
  "ski",
  "skill",
  "skin",
  "skirt",
  "skull",
  "slab",
  "slam",
  "sleep",
  "slender",
  "slice",
  "slide",
  "slight",
  "slim",
  "slogan",
  "slot",
  "slow",
  "slush",
  "small",
  "smart",
  "smile",
  "smoke",
  "smooth",
  "snack",
  "snake",
  "snap",
  "sniff",
  "snow",
  "soap",
  "soccer",
  "social",
  "sock",
  "soda",
  "soft",
  "solar",
  "soldier",
  "solid",
  "solution",
  "solve",
  "someone",
  "song",
  "soon",
  "sorry",
  "sort",
  "soul",
  "sound",
  "soup",
  "source",
  "south",
  "space",
  "spare",
  "spatial",
  "spawn",
  "speak",
  "special",
  "speed",
  "spell",
  "spend",
  "sphere",
  "spice",
  "spider",
  "spike",
  "spin",
  "spirit",
  "split",
  "spoil",
  "sponsor",
  "spoon",
  "sport",
  "spot",
  "spray",
  "spread",
  "spring",
  "spy",
  "square",
  "squeeze",
  "squirrel",
  "stable",
  "stadium",
  "staff",
  "stage",
  "stairs",
  "stamp",
  "stand",
  "start",
  "state",
  "stay",
  "steak",
  "steel",
  "stem",
  "step",
  "stereo",
  "stick",
  "still",
  "sting",
  "stock",
  "stomach",
  "stone",
  "stool",
  "story",
  "stove",
  "strategy",
  "street",
  "strike",
  "strong",
  "struggle",
  "student",
  "stuff",
  "stumble",
  "style",
  "subject",
  "submit",
  "subway",
  "success",
  "such",
  "sudden",
  "suffer",
  "sugar",
  "suggest",
  "suit",
  "summer",
  "sun",
  "sunny",
  "sunset",
  "super",
  "supply",
  "supreme",
  "sure",
  "surface",
  "surge",
  "surprise",
  "surround",
  "survey",
  "suspect",
  "sustain",
  "swallow",
  "swamp",
  "swap",
  "swarm",
  "swear",
  "sweet",
  "swift",
  "swim",
  "swing",
  "switch",
  "sword",
  "symbol",
  "symptom",
  "syrup",
  "system",
  "table",
  "tackle",
  "tag",
  "tail",
  "talent",
  "talk",
  "tank",
  "tape",
  "target",
  "task",
  "taste",
  "tattoo",
  "taxi",
  "teach",
  "team",
  "tell",
  "ten",
  "tenant",
  "tennis",
  "tent",
  "term",
  "test",
  "text",
  "thank",
  "that",
  "theme",
  "then",
  "theory",
  "there",
  "they",
  "thing",
  "this",
  "thought",
  "three",
  "thrive",
  "throw",
  "thumb",
  "thunder",
  "ticket",
  "tide",
  "tiger",
  "tilt",
  "timber",
  "time",
  "tiny",
  "tip",
  "tired",
  "tissue",
  "title",
  "toast",
  "tobacco",
  "today",
  "toddler",
  "toe",
  "together",
  "toilet",
  "token",
  "tomato",
  "tomorrow",
  "tone",
  "tongue",
  "tonight",
  "tool",
  "tooth",
  "top",
  "topic",
  "topple",
  "torch",
  "tornado",
  "tortoise",
  "toss",
  "total",
  "tourist",
  "toward",
  "tower",
  "town",
  "toy",
  "track",
  "trade",
  "traffic",
  "tragic",
  "train",
  "transfer",
  "trap",
  "trash",
  "travel",
  "tray",
  "treat",
  "tree",
  "trend",
  "trial",
  "tribe",
  "trick",
  "trigger",
  "trim",
  "trip",
  "trophy",
  "trouble",
  "truck",
  "true",
  "truly",
  "trumpet",
  "trust",
  "truth",
  "try",
  "tube",
  "tuition",
  "tumble",
  "tuna",
  "tunnel",
  "turkey",
  "turn",
  "turtle",
  "twelve",
  "twenty",
  "twice",
  "twin",
  "twist",
  "two",
  "type",
  "typical",
  "ugly",
  "umbrella",
  "unable",
  "unaware",
  "uncle",
  "uncover",
  "under",
  "undo",
  "unfair",
  "unfold",
  "unhappy",
  "uniform",
  "unique",
  "unit",
  "universe",
  "unknown",
  "unlock",
  "until",
  "unusual",
  "unveil",
  "update",
  "upgrade",
  "uphold",
  "upon",
  "upper",
  "upset",
  "urban",
  "urge",
  "usage",
  "use",
  "used",
  "useful",
  "useless",
  "usual",
  "utility",
  "vacant",
  "vacuum",
  "vague",
  "valid",
  "valley",
  "valve",
  "van",
  "vanish",
  "vapor",
  "various",
  "vast",
  "vault",
  "vehicle",
  "velvet",
  "vendor",
  "venture",
  "venue",
  "verb",
  "verify",
  "version",
  "very",
  "vessel",
  "veteran",
  "viable",
  "vibrant",
  "vicious",
  "victory",
  "video",
  "view",
  "village",
  "vintage",
  "violin",
  "virtual",
  "virus",
  "visa",
  "visit",
  "visual",
  "vital",
  "vivid",
  "vocal",
  "voice",
  "void",
  "volcano",
  "volume",
  "vote",
  "voyage",
  "wage",
  "wagon",
  "wait",
  "walk",
  "wall",
  "walnut",
  "want",
  "warfare",
  "warm",
  "warrior",
  "wash",
  "wasp",
  "waste",
  "water",
  "wave",
  "way",
  "wealth",
  "weapon",
  "wear",
  "weasel",
  "weather",
  "web",
  "wedding",
  "weekend",
  "weird",
  "welcome",
  "west",
  "wet",
  "whale",
  "what",
  "wheat",
  "wheel",
  "when",
  "where",
  "whip",
  "whisper",
  "wide",
  "width",
  "wife",
  "wild",
  "will",
  "win",
  "window",
  "wine",
  "wing",
  "wink",
  "winner",
  "winter",
  "wire",
  "wisdom",
  "wise",
  "wish",
  "witness",
  "wolf",
  "woman",
  "wonder",
  "wood",
  "wool",
  "word",
  "work",
  "world",
  "worry",
  "worth",
  "wrap",
  "wreck",
  "wrestle",
  "wrist",
  "write",
  "wrong",
  "yard",
  "year",
  "yellow",
  "you",
  "young",
  "youth",
  "zebra",
  "zero",
  "zone",
  "zoo"
]

import sys
if sys.version_info.major == 2:
  # Base switching
  code_strings = {
      2: '01',
      10: '0123456789',
      16: '0123456789abcdef',
      32: 'abcdefghijklmnopqrstuvwxyz234567',
      58: '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz',
      256: ''.join([chr(x) for x in range(256)])
  }

  def bin_to_b58check(inp, magicbyte=0):
      while magicbyte > 0:
          inp = chr(int(magicbyte % 256)) + inp
          magicbyte //= 256
      leadingzbytes = len(re.match('^\x00*', inp).group(0))
      checksum = bin_dbl_sha256(inp)[:4]
      return '1' * leadingzbytes + changebase(inp+checksum, 256, 58)

  def bytes_to_hex_string(b):
      return b.encode('hex')

  def safe_from_hex(s):
      return s.decode('hex')

  def from_int_to_byte(a):
      return chr(a)

  def from_byte_to_int(a):
      return ord(a)

  def from_string_to_bytes(a):
      return a

  def safe_hexlify(a):
      return binascii.hexlify(a)

  def encode(val, base, minlen=0):
      base, minlen = int(base), int(minlen)
      code_string = get_code_string(base)
      result = ""
      while val > 0:
          result = code_string[val % base] + result
          val //= base
      return code_string[0] * max(minlen - len(result), 0) + result

  def decode(string, base):
      base = int(base)
      code_string = get_code_string(base)
      result = 0
      if base == 16:
          string = string.lower()
      while len(string) > 0:
          result *= base
          result += code_string.find(string[0])
          string = string[1:]
      return result
else:
  # Base switching
  code_strings = {
      2: '01',
      10: '0123456789',
      16: '0123456789abcdef',
      32: 'abcdefghijklmnopqrstuvwxyz234567',
      58: '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz',
      256: ''.join([chr(x) for x in range(256)])
  }

  def bin_to_b58check(inp, magicbyte=0):
      while magicbyte > 0:
          inp = from_int_to_byte(magicbyte % 256) + inp
          magicbyte //= 256

      leadingzbytes = 0
      for x in inp:
          if x != 0:
              break
          leadingzbytes += 1

      checksum = bin_dbl_sha256(inp)[:4]
      return '1' * leadingzbytes + changebase(inp+checksum, 256, 58)

  def bytes_to_hex_string(b):
      if isinstance(b, str):
          return b

      return ''.join('{:02x}'.format(y) for y in b)

  def safe_from_hex(s):
      return bytes.fromhex(s)

  def from_int_to_byte(a):
      return bytes([a])

  def from_byte_to_int(a):
      return a

  def from_string_to_bytes(a):
      return a if isinstance(a, bytes) else bytes(a, 'utf-8')

  def safe_hexlify(a):
      return str(binascii.hexlify(a), 'utf-8')

  def encode(val, base, minlen=0):
      base, minlen = int(base), int(minlen)
      code_string = get_code_string(base)
      result_bytes = bytes()
      while val > 0:
          curcode = code_string[val % base]
          result_bytes = bytes([ord(curcode)]) + result_bytes
          val //= base

      pad_size = minlen - len(result_bytes)

      padding_element = b'\x00' if base == 256 else b'1' \
          if base == 58 else b'0'
      if (pad_size > 0):
          result_bytes = padding_element*pad_size + result_bytes

      result_string = ''.join([chr(y) for y in result_bytes])
      result = result_bytes if base == 256 else result_string

      return result

  def decode(string, base):
      if base == 256 and isinstance(string, str):
          string = bytes(bytearray.fromhex(string))
      base = int(base)
      code_string = get_code_string(base)
      result = 0
      if base == 256:
          def extract(d, cs):
              return d
      else:
          def extract(d, cs):
              return cs.find(d if isinstance(d, str) else chr(d))

      if base == 16:
          string = string.lower()
      while len(string) > 0:
          result *= base
          result += extract(string[0], code_string)
          string = string[1:]

      return result


def bin_dbl_sha256(s):
    bytes_to_hash = from_string_to_bytes(s)
    return hashlib.sha256(hashlib.sha256(bytes_to_hash).digest()).digest()

def lpad(msg, symbol, length):
        if len(msg) >= length:
            return msg
        return symbol * (length - len(msg)) + msg

def get_code_string(base):
    if base in code_strings:
        return code_strings[base]
    else:
        raise ValueError("Invalid base!")

def changebase(string, frm, to, minlen=0):
    if frm == to:
        return lpad(string, get_code_string(frm)[0], minlen)
    return encode(decode(string, frm), to, minlen)


def b58check_to_bin(inp):
    leadingzbytes = len(re.match('^1*', inp).group(0))
    data = b'\x00' * leadingzbytes + changebase(inp, 58, 256)
    assert bin_dbl_sha256(data[:-4])[:4] == data[-4:]
    return data[:-4]

# from http://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python/

def modular_sqrt(a, p):
    """ Find a quadratic residue (mod p) of 'a'. p
    must be an odd prime.
    
    Solve the congruence of the form:
    x^2 = a (mod p)
    And returns x. Note that p - x is also a root.
    
    0 is returned is no square root exists for
    these a and p.
    
    The Tonelli-Shanks algorithm is used (except
    for some simple cases in which the solution
    is known from an identity). This algorithm
    runs in polynomial time (unless the
    generalized Riemann hypothesis is false).
    """
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return p
    elif p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    
    # Partition p-1 to s * 2^e for an odd s (i.e.
    # reduce all the powers of 2 from p-1)
    #
    s = p - 1
    e = 0
    while s % 2 == 0:
        s //= 2
        e += 1
        
    # Find some 'n' with a legendre symbol n|p = -1.
    # Shouldn't take long.
    #
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1
        
    # Here be dragons!
    # Read the paper "Square roots from 1; 24, 51,
    # 10 to Dan Shanks" by Ezra Brown for more
    # information
    #
    
    # x is a guess of the square root that gets better
    # with each iteration.
    # b is the "fudge factor" - by how much we're off
    # with the guess. The invariant x^2 = ab (mod p)
    # is maintained throughout the loop.
    # g is used for successive powers of n to update
    # both a and b
    # r is the exponent - decreases with each update
    #
    x = pow(a, (s + 1) // 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e
    
    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = pow(t, 2, p)
            
        if m == 0:
            return x
        
        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m
        
def legendre_symbol(a, p):
    """ Compute the Legendre symbol a|p using
    Euler's criterion. p is a prime, a is
    relatively prime to p (if p divides
    a, then a|p = 0)
    
    Returns 1 if a has a square root modulo
    p, -1 otherwise.
    """
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls

# much of the code is 'borrowed' from electrum,
# https://gitorious.org/electrum/electrum
# and is under the GPLv3.

import hashlib, base64, ecdsa, re
import hmac
import os, unicodedata
import unittest

def rev_hex(s):
    return bytes_to_hex_string(safe_from_hex(s)[::-1])

def int_to_hex(i, length=1):
    s = hex(i)[2:].rstrip('L')
    s = "0"*(2*length - len(s)) + s
    return rev_hex(s)

def var_int(i):
    # https://en.bitcoin.it/wiki/Protocol_specification#Variable_length_integer
    if i<0xfd:
        return int_to_hex(i)
    elif i<=0xffff:
        return "fd"+int_to_hex(i,2)
    elif i<=0xffffffff:
        return "fe"+int_to_hex(i,4)
    else:
        return "ff"+int_to_hex(i,8)


def sha256(x):
    return hashlib.sha256(x).digest()

def Hash(x):
    if sys.version_info.major == 2:
      if type(x) is unicode: x=x.encode('utf-8')
    else:
      if type(x) is str: x=x.encode('utf-8')
    return sha256(sha256(x))

# pywallet openssl private key implementation

def i2d_ECPrivateKey(pkey, compressed=False):
    if compressed:
        key = '3081d30201010420' + \
              '%064x' % pkey.secret + \
              'a081a53081a2020101302c06072a8648ce3d0101022100' + \
              '%064x' % _p + \
              '3006040100040107042102' + \
              '%064x' % _Gx + \
              '022100' + \
              '%064x' % _r + \
              '020101a124032200'
    else:
        key = '308201130201010420' + \
              '%064x' % pkey.secret + \
              'a081a53081a2020101302c06072a8648ce3d0101022100' + \
              '%064x' % _p + \
              '3006040100040107044104' + \
              '%064x' % _Gx + \
              '%064x' % _Gy + \
              '022100' + \
              '%064x' % _r + \
              '020101a144034200'
        
    return safe_from_hex(key) + i2o_ECPublicKey(pkey.pubkey, compressed)
    
def i2o_ECPublicKey(pubkey, compressed=False):
    # public keys are 65 bytes long (520 bits)
    # 0x04 + 32-byte X-coordinate + 32-byte Y-coordinate
    # 0x00 = point at infinity, 0x02 and 0x03 = compressed, 0x04 = uncompressed
    # compressed keys: <sign> <x> where <sign> is 0x02 if y is even and 0x03 if y is odd
    if compressed:
        if pubkey.point.y() & 1:
            key = '03' + '%064x' % pubkey.point.x()
        else:
            key = '02' + '%064x' % pubkey.point.x()
    else:
        key = '04' + \
              '%064x' % pubkey.point.x() + \
              '%064x' % pubkey.point.y()
            
    return safe_from_hex(key)
            
# end pywallet openssl private key implementation

                                                
            
############ functions from pywallet ##################### 

def hash_160(public_key):
    try:
        md = hashlib.new('ripemd160')
        md.update(sha256(public_key))
        return md.digest()
    except Exception:
        import ripemd
        md = ripemd.new(sha256(public_key))
        return md.digest()


def public_key_to_bc_address(public_key):
    h160 = hash_160(public_key)
    return hash_160_to_bc_address(h160)

def hash_160_to_bc_address(h160, addrtype = 0):
    vh160 = from_string_to_bytes(chr(addrtype)) + from_string_to_bytes(h160)
    return bin_to_b58check(vh160)

def bc_address_to_hash_160(addr):
    bytes = b58check_to_bin(addr)
    return ord(from_int_to_byte(from_byte_to_int(bytes[0]))), bytes[1:21]

def EncodeBase58Check(vchIn):
    return bin_to_b58check(vchIn)

def DecodeBase58Check(psz):
    return b58check_to_bin(psz)

def PrivKeyToSecret(privkey):
    return privkey[9:9+32]

def SecretToASecret(secret, compressed=False, addrtype=0):
    vchIn = from_int_to_byte((addrtype+128)&255) + secret
    if compressed: vchIn += b'\01'
    return EncodeBase58Check(vchIn)

def ASecretToSecret(key, addrtype=0):
    vch = DecodeBase58Check(key)
    if vch and vch[0] == chr((addrtype+128)&255):
        return vch[1:]
    else:
        return False

def regenerate_key(sec):
    b = ASecretToSecret(sec)
    if not b:
        return False
    b = b[0:32]
    return EC_KEY(b)

def GetPubKey(pubkey, compressed=False):
    return i2o_ECPublicKey(pubkey, compressed)

def GetPrivKey(pkey, compressed=False):
    return i2d_ECPrivateKey(pkey, compressed)

def GetSecret(pkey):
    return safe_from_hex(('%064x' % pkey.secret))

def is_compressed(sec):
    b = ASecretToSecret(sec)
    return len(b) == 33


def public_key_from_private_key(sec):
    # rebuild public key from private key, compressed or uncompressed
    pkey = regenerate_key(sec)
    assert pkey
    compressed = is_compressed(sec)
    public_key = GetPubKey(pkey.pubkey, compressed)
    return bytes_to_hex_string(public_key)


def address_from_private_key(sec):
    public_key = public_key_from_private_key(sec)
    address = public_key_to_bc_address(safe_from_hex(public_key))
    return address


def is_valid(addr):
    ADDRESS_RE = re.compile('[1-9A-HJ-NP-Za-km-z]{26,}\\Z')
    if not ADDRESS_RE.match(addr): return False
    try:
        addrtype, h = bc_address_to_hash_160(addr)
    except Exception:
        return False
    return addr == hash_160_to_bc_address(h, addrtype)


########### end pywallet functions #######################

try:
    from ecdsa.ecdsa import curve_secp256k1, generator_secp256k1
except Exception:
    print("cannot import ecdsa.curve_secp256k1. You probably need to upgrade ecdsa.\nTry: sudo pip install --upgrade ecdsa")
    exit()

from ecdsa.curves import SECP256k1
from ecdsa.ellipticcurve import Point
from ecdsa.util import string_to_number, number_to_string

def msg_magic(message):
    varint = var_int(len(message))
    encoded_varint = "".join([chr(int(varint[i:i+2], 16)) for i in range(0, len(varint), 2)])
    return "\x18Bitcoin Signed Message:\n" + encoded_varint + message


def verify_message(address, signature, message):
    try:
        EC_KEY.verify_message(address, signature, message)
        return True
    except Exception as e:
        print("error: Verification error: {0}".format(e))
        return False


def encrypt_message(message, pubkey):
    return EC_KEY.encrypt_message(message, safe_from_hex(pubkey))


def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def ECC_YfromX(x,curved=curve_secp256k1, odd=True):
    _p = curved.p()
    _a = curved.a()
    _b = curved.b()
    for offset in range(128):
        Mx = x + offset
        My2 = pow(Mx, 3, _p) + _a * pow(Mx, 2, _p) + _b % _p
        My = pow(My2, (_p+1)//4, _p )

        if curved.contains_point(Mx,My):
            if odd == bool(My&1):
                return [My,offset]
            return [_p-My,offset]
    raise Exception('ECC_YfromX: No Y found')

def private_header(msg,v):
    assert v<1, "Can't write version %d private header"%v
    r = b''
    if v==0:
        r += safe_from_hex(('%08x'%len(msg)))
        r += sha256(msg.encode('utf-8'))[:2]
    return safe_from_hex(('%02x'%v)) + safe_from_hex(('%04x'%len(r))) + r

def public_header(pubkey,v):
    assert v<1, "Can't write version %d public header"%v
    r = ''
    if v==0:
        r = sha256(pubkey)[:2]
    return b'\x6a\x6a' + safe_from_hex(('%02x'%v)) + safe_from_hex(('%04x'%len(r))) + r


def negative_point(P):
    return Point( P.curve(), P.x(), -P.y(), P.order() )


def point_to_ser(P, comp=True ):
    if comp:
        return safe_from_hex(( ('%02x'%(2+(P.y()&1)))+('%064x'%P.x()) ))
    return safe_from_hex(( '04'+('%064x'%P.x())+('%064x'%P.y()) ))


def ser_to_point(Aser):
    curve = curve_secp256k1
    generator = generator_secp256k1
    _r  = generator.order()
    assert from_int_to_byte(from_byte_to_int(Aser[0])) in [b'\x02',b'\x03',b'\x04']
    if from_int_to_byte(from_byte_to_int(Aser[0])) == b'\x04':
        return Point( curve, str_to_long(Aser[1:33]), str_to_long(Aser[33:]), _r )
    Mx = string_to_number(Aser[1:])
    return Point( curve, Mx, ECC_YfromX(Mx, curve, from_int_to_byte(from_byte_to_int(Aser[0]))==b'\x03')[0], _r )



class EC_KEY(object):
    def __init__( self, k ):
        secret = string_to_number(k)
        self.pubkey = ecdsa.ecdsa.Public_key( generator_secp256k1, generator_secp256k1 * secret )
        self.privkey = ecdsa.ecdsa.Private_key( self.pubkey, secret )
        self.secret = secret

    def get_public_key(self, compressed=True):
        return bytes_to_hex_string(point_to_ser(self.pubkey.point, compressed))

    def sign_message(self, message, compressed, address):
        private_key = ecdsa.SigningKey.from_secret_exponent( self.secret, curve = SECP256k1 )
        public_key = private_key.get_verifying_key()
        signature = private_key.sign_digest_deterministic( Hash( msg_magic(message) ), hashfunc=hashlib.sha256, sigencode = ecdsa.util.sigencode_string )
        assert public_key.verify_digest( signature, Hash( msg_magic(message) ), sigdecode = ecdsa.util.sigdecode_string)
        for i in range(4):
            sig = base64.b64encode( from_int_to_byte(27 + i + (4 if compressed else 0)) + signature )
            try:
                self.verify_message( address, sig, message)
                return sig
            except Exception:
                continue
        else:
            raise Exception("error: cannot sign message")


    @classmethod
    def verify_message(self, address, signature, message):
        """ See http://www.secg.org/download/aid-780/sec1-v2.pdf for the math """
        from ecdsa import numbertheory, util
        curve = curve_secp256k1
        G = generator_secp256k1
        order = G.order()
        # extract r,s from signature
        sig = base64.b64decode(signature)
        if len(sig) != 65: raise Exception("Wrong encoding")
        r,s = util.sigdecode_string(sig[1:], order)
        nV = ord(from_int_to_byte(from_byte_to_int(sig[0])))
        if nV < 27 or nV >= 35:
            raise Exception("Bad encoding")
        if nV >= 31:
            compressed = True
            nV -= 4
        else:
            compressed = False

        recid = nV - 27
        # 1.1
        x = r + (recid//2) * order
        # 1.3
        alpha = ( x * x * x  + curve.a() * x + curve.b() ) % curve.p()
        beta = modular_sqrt(alpha, curve.p())
        y = beta if (beta - recid) % 2 == 0 else curve.p() - beta
        # 1.4 the constructor checks that nR is at infinity
        R = Point(curve, x, y, order)
        # 1.5 compute e from message:
        h = Hash( msg_magic(message) )
        e = string_to_number(h)
        minus_e = -e % order
        # 1.6 compute Q = r^-1 (sR - eG)
        inv_r = numbertheory.inverse_mod(r,order)
        Q = inv_r * ( s * R + minus_e * G )
        public_key = ecdsa.VerifyingKey.from_public_point( Q, curve = SECP256k1 )
        # check that Q is the public key
        public_key.verify_digest( sig[1:], h, sigdecode = ecdsa.util.sigdecode_string)
        # check that we get the original signing address
        addr = public_key_to_bc_address( point_to_ser(public_key.pubkey.point, compressed) )
        if address != addr:
            raise Exception("Bad signature")


    # ecdsa encryption/decryption methods
    # credits: jackjack, https://github.com/jackjack-jj/jeeq

    @classmethod
    def encrypt_message(self, message, pubkey):
        generator = generator_secp256k1
        curved = curve_secp256k1
        r = b''
        msg = private_header(message,0) + from_string_to_bytes(message)
        msg = msg + (b'\x00'*( 32-(len(msg)%32) ))
        msgs = chunks(msg,32)

        _r  = generator.order()
        str_to_long = string_to_number

        P = generator
        if len(pubkey)==33: #compressed
            pk = Point( curve_secp256k1, str_to_long(pubkey[1:33]), ECC_YfromX(str_to_long(pubkey[1:33]), curve_secp256k1, from_int_to_byte(from_byte_to_int(pubkey[0]))==b'\x03')[0], _r )
        else:
            pk = Point( curve_secp256k1, str_to_long(pubkey[1:33]), str_to_long(pubkey[33:65]), _r )

        for i in range(len(msgs)):
            n = ecdsa.util.randrange( pow(2,256) )
            Mx = str_to_long(msgs[i])
            My, xoffset = ECC_YfromX(Mx, curved)
            M = Point( curved, Mx+xoffset, My, _r )
            T = P*n
            U = pk*n + M
            toadd = point_to_ser(T) + point_to_ser(U)
            toadd = from_int_to_byte(from_byte_to_int(toadd[0])-2 + 2*xoffset) + toadd[1:]
            r += toadd

        return base64.b64encode(public_header(pubkey,0) + r)


    def decrypt_message(self, enc):
        G = generator_secp256k1
        curved = curve_secp256k1
        pvk = self.secret
        pubkeys = [point_to_ser(G*pvk,True), point_to_ser(G*pvk,False)]
        enc = base64.b64decode(enc)
        str_to_long = string_to_number

        assert enc[:2]==b'\x6a\x6a'

        phv = str_to_long(from_int_to_byte(from_byte_to_int(enc[2])))
        assert phv==0, "Can't read version %d public header"%phv
        hs = str_to_long(enc[3:5])
        public_header=enc[5:5+hs]
        checksum_pubkey=public_header[:2]
        address=list(filter(lambda x:sha256(x)[:2]==checksum_pubkey, pubkeys))
        assert len(address)>0, 'Bad private key'
        address=address[0]
        enc=enc[5+hs:]
        r = b''
        for Tser,User in map(lambda x:[x[:33],x[33:]], chunks(enc,66)):
            ots = ord(from_int_to_byte(from_byte_to_int(Tser[0])))
            xoffset = ots>>1
            Tser = from_int_to_byte(2+(ots&1))+Tser[1:]
            T = ser_to_point(Tser)
            U = ser_to_point(User)
            V = T*pvk
            Mcalc = U + negative_point(V)
            r += safe_from_hex(('%064x'%(Mcalc.x()-xoffset)))

        pvhv = str_to_long(from_int_to_byte(from_byte_to_int(r[0])))
        assert pvhv==0, "Can't read version %d private header"%pvhv
        phs = str_to_long(r[1:3])
        private_header = r[3:3+phs]
        size = str_to_long(private_header[:4])
        checksum = private_header[4:6]
        r = r[3+phs:]

        msg = r[:size]
        hashmsg = sha256(msg)[:2]
        checksumok = hashmsg==checksum

        return [msg, checksumok, address]





###################################### BIP32 ##############################

random_seed = lambda n: "%032x"%ecdsa.util.randrange( pow(2,n) )
BIP32_PRIME = 0x80000000

def bip32_init(seed):
    import hmac
    seed = safe_from_hex(seed)
    I = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()

    master_secret = I[0:32]
    master_chain = I[32:]

    K, K_compressed = get_pubkeys_from_secret(master_secret)
    return master_secret, master_chain, K, K_compressed


def get_pubkeys_from_secret(secret):
    # public key
    private_key = ecdsa.SigningKey.from_string( secret, curve = SECP256k1 )
    public_key = private_key.get_verifying_key()
    K = public_key.to_string()
    K_compressed = GetPubKey(public_key.pubkey,True)
    return K, K_compressed



# Child private key derivation function (from master private key)
# k = master private key (32 bytes)
# c = master chain code (extra entropy for key derivation) (32 bytes)
# n = the index of the key we want to derive. (only 32 bits will be used)
# If n is negative (i.e. the 32nd bit is set), the resulting private key's
#  corresponding public key can NOT be determined without the master private key.
# However, if n is positive, the resulting private key's corresponding
#  public key can be determined without the master private key.
def CKD(k, c, n):
    import hmac
    from ecdsa.util import string_to_number, number_to_string
    order = generator_secp256k1.order()
    keypair = EC_KEY(k)
    K = GetPubKey(keypair.pubkey,True)

    if n & BIP32_PRIME: # We want to make a "secret" address that can't be determined from K
        data = from_string_to_bytes(chr(0)) + k + safe_from_hex(rev_hex(int_to_hex(n,4)))
        I = hmac.new(c, data, hashlib.sha512).digest()
    else: # We want a "non-secret" address that can be determined from K
        I = hmac.new(c, K + safe_from_hex(rev_hex(int_to_hex(n,4))), hashlib.sha512).digest()
        
    k_n = number_to_string( (string_to_number(I[0:32]) + string_to_number(k)) % order , order )
    c_n = I[32:]
    return k_n, c_n

# Child public key derivation function (from public key only)
# K = master public key 
# c = master chain code
# n = index of key we want to derive
# This function allows us to find the nth public key, as long as n is 
#  non-negative. If n is negative, we need the master private key to find it.
def CKD_prime(K, c, n):
    import hmac
    from ecdsa.util import string_to_number, number_to_string
    order = generator_secp256k1.order()

    if n & BIP32_PRIME: raise

    K_public_key = ecdsa.VerifyingKey.from_string( K, curve = SECP256k1 )
    K_compressed = GetPubKey(K_public_key.pubkey,True)

    I = hmac.new(c, K_compressed + safe_from_hex(rev_hex(int_to_hex(n,4))), hashlib.sha512).digest()

    curve = SECP256k1
    pubkey_point = string_to_number(I[0:32])*curve.generator + K_public_key.pubkey.point
    public_key = ecdsa.VerifyingKey.from_public_point( pubkey_point, curve = SECP256k1 )

    K_n = public_key.to_string()
    K_n_compressed = GetPubKey(public_key.pubkey,True)
    c_n = I[32:]

    return K_n, K_n_compressed, c_n



# def bip32_private_derivation(k, c, branch, sequence):
#     assert sequence.startswith(branch)
#     sequence = sequence[len(branch):]
#     for n in sequence.split('/'):
#         if n == '': continue
#         n = int(n[:-1]) + BIP32_PRIME if n[-1] == "'" else int(n)
#         k, c = CKD(k, c, n)
#     K, K_compressed = get_pubkeys_from_secret(k)
#     return bytes_to_hex_string(k), bytes_to_hex_string(c), bytes_to_hex_string(K), bytes_to_hex_string(K_compressed)


# def bip32_public_derivation(c, K, branch, sequence):
#     assert sequence.startswith(branch)
#     sequence = sequence[len(branch):]
#     for n in sequence.split('/'):
#         n = int(n)
#         K, cK, c = CKD_prime(K, c, n)

#     return bytes_to_hex_string(c), bytes_to_hex_string(K), bytes_to_hex_string(cK)


# def bip32_private_key(sequence, k, chain):
#     for i in sequence:
#         k, chain = CKD(k, chain, i)
#     return SecretToASecret(k, True)

###################################### test_crypto ##############################

def test_crypto():

    G = generator_secp256k1
    _r  = G.order()
    pvk = ecdsa.util.randrange( pow(2,256) ) %_r

    Pub = pvk*G
    pubkey_c = point_to_ser(Pub,True)
    pubkey_u = point_to_ser(Pub,False)
    addr_c = public_key_to_bc_address(pubkey_c)
    addr_u = public_key_to_bc_address(pubkey_u)

    print("Private key            ", '%064x'%pvk)
    print("Compressed public key  ", bytes_to_hex_string(pubkey_c))
    print("Uncompressed public key", bytes_to_hex_string(pubkey_u))

    message = "Chancellor on brink of second bailout for banks"
    enc = EC_KEY.encrypt_message(message,pubkey_c)
    eck = EC_KEY(number_to_string(pvk,_r))
    dec = eck.decrypt_message(enc)
    print("decrypted", dec)

    signature = eck.sign_message(message, True, addr_c)
    print(signature)
    EC_KEY.verify_message(addr_c, signature, message)
    
###################################### KEYTREE ##############################

import binascii
import getpass
"""
You can specify all options at once with the no prompt option. But it is discouraged because on most OS commands are stored in a history file:
./kt.py --noprompt -s "this is a password" --chain "(0-1)'/(6-8)'" -trav levelorder
./kt.py -np -s "this is a password" -c "(0-1)'/(6-8)'" -hs 3 -v
./kt.py -np --extkey xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7 -c "(0-1)'/8"
./kt.py -np --b39 -s "pilot dolphin motion portion survey sock turkey afford destroy knee sock sibling" -c "44'/0'/(0-1)'"
./kt.py -np -sh 936ae011512b96e7ce3ff05d464e3801834d023249baabfebfe13e593dc33610ea68279c271df6bab7cfbea8bbcf470e050fe6589f552f7e1f6c80432c7bcc57 -c "44'/0'/(0-1)'"
"""

noInputEcho = False

cmdName = "./kt.py"
SEED_FORMAT = "seed_format"
SEED_VALUE = "seed_value"
EXTENDEDKEY_VALUE = "extkey_value"
CHAIN_VALUE = "chain_value"
MNEMONIC_VALUE = 'mnemonic_value'

HELP = "-help"
RUN_TESTS = "-runtests"
NO_INPUT_ECHO = "-noecho"
TESTNET = "-testnet"
HASH_SEED = "-hashseed"
NO_PROMPT = "-noprompt"
SEED = "-seed"
SEED_HEX = "-seedhex"
EXTENDEDKEY = "-extkey"
CHAIN = "-chain"
TREE_TRAVERSAL_OPTION = "-traverse"
TREE_TRAVERSAL_TYPE_PREORDER = "preorder"
TREE_TRAVERSAL_TYPE_POSTORDER = "postorder"
TREE_TRAVERSAL_TYPE_LEVELORDER = "levelorder"
OUTPUT_ENTIRE_CHAIN_OPTION = "-all"
VERBOSE_OPTION = "-verbose"
ENFORCE_BIP39_RULE = "-bip39"
GENERATE_MNEMONIC_PASSPHRASE = "-generateBIP39mnemonic"

HELP_SHORT = "h"
RUN_TESTS_SHORT = "rt"
NO_INPUT_ECHO_SHORT = "ne"
TESTNET_SHORT = "tn"
HASH_SEED_SHORT = "hs"
NO_PROMPT_SHORT = "np"
SEED_SHORT = "s"
SEED_HEX_SHORT = "sh"
EXTENDEDKEY_SHORT = "ek"
CHAIN_SHORT = "c"
TREE_TRAVERSAL_OPTION_SHORT = "trav"
TREE_TRAVERSAL_TYPE_PREORDER_SHORT = "pre"
TREE_TRAVERSAL_TYPE_POSTORDER_SHORT = "post"
TREE_TRAVERSAL_TYPE_LEVELORDER_SHORT = "lev"
OUTPUT_ENTIRE_CHAIN_OPTION_SHORT = "a"
VERBOSE_OPTION_SHORT = "v"
ENFORCE_BIP39_RULE_SHORT = "-b39"
GENERATE_MNEMONIC_PASSPHRASE_SHORT = "gb39m"

class StringType:
    HEX = 1
    BASE58 = 2
    ASCII = 3 

class TreeTraversal:
    PREORDER = 1
    POSTORDER = 2
    LEVELORDER = 3 

DEFAULTTREETRAVERSALTYPE = TreeTraversal.PREORDER



class KeyTreeUtil(object):
    NODE_IDX_M_FLAG = sys.maxsize
    MASTER_NODE_LOWERCASE_M = "m"
    LEAD_CHAIN_PATH = "___"

    @staticmethod
    def sha256Rounds(data, rounds):
        for i in range(rounds):
            data = sha256(data)
        return data

    @staticmethod
    def toPrime(i):
        return i + BIP32_PRIME  

    @staticmethod
    def removePrime(i):
        return 0x7fffffff & i

    @staticmethod
    def isPrime(i):
        return BIP32_PRIME & i  

    @staticmethod
    def iToString(i):    
        if KeyTreeUtil.isPrime(i):
            return str(KeyTreeUtil.removePrime(i))+"'"
        else:
            return str(i)  

    @staticmethod
    def parseRange(node, isPrivate):
        #node must be like (123-9345)
        minMaxString = node[1:-1]
        minMaxPair = minMaxString.split('-')
        if len(minMaxPair) != 2:
            raise ValueError('Invalid arguments.')
        min = int(minMaxPair[0])
        max = int(minMaxPair[1])
        if isPrivate:
            return [True, [min, max]]
        else:
            return [False, [min, max]]

    @staticmethod
    def parseChainString(sequence):
        if sequence == None or sequence == "":
            treeChains = []
            treeChains.append([True, [KeyTreeUtil.NODE_IDX_M_FLAG, KeyTreeUtil.NODE_IDX_M_FLAG]])            
            return treeChains

        treeChains = []
        splitChain = sequence.split('/')

        treeChains.append([True, [KeyTreeUtil.NODE_IDX_M_FLAG, KeyTreeUtil.NODE_IDX_M_FLAG]])

        for node in splitChain:
            if node[-1] == "'":
                node = node[:-1]
                if node[0] == '(' and node[-1] == ')':
                    treeChains.append(KeyTreeUtil.parseRange(node, True))
                else:
                    num = int(node)
                    treeChains.append([True, [num, num]])
            else:
                if node[0] == '(' and node[-1] == ')':
                    treeChains.append(KeyTreeUtil.parseRange(node, False))
                else:
                    num = int(node)
                    treeChains.append([False, [num, num]])

        return treeChains

    @staticmethod
    def powMod(x, y, z):
        """Calculate (x ** y) % z efficiently."""
        number = 1
        while y:
            if y & 1:
                number = number * x % z
            y >>= 1
            x = x * x % z
        return number

    @staticmethod
    def compressedPubKeyToUncompressedPubKey(compressedPubKey):
        compressedPubKey = bytes_to_hex_string(compressedPubKey)

        p = 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f
        y_parity = int(compressedPubKey[:2]) - 2

        x = int(compressedPubKey[2:], 16)
        a = (KeyTreeUtil.powMod(x, 3, p) + 7) % p
        y = KeyTreeUtil.powMod(a, (p+1)//4, p)
        if y % 2 != y_parity:
            y = -y % p

        ppp = KeyTreeUtil.powMod(x, 3, p)
        uncompressedPubKey = '{:064x}{:064x}'.format(x, y)
        return safe_from_hex(uncompressedPubKey)



class BIP39(object):
    # code below from https://github.com/trezor/python-mnemonic/blob/master/mnemonic/mnemonic.py
    @staticmethod
    def normalizeString(txt):
        if isinstance(txt, str if sys.version < "3" else bytes):
            utxt = txt.decode("utf8")
        elif isinstance(txt, unicode if sys.version < "3" else str):  # noqa: F821
            utxt = txt
        else:
            raise TypeError("String value expected")

        return unicodedata.normalize("NFKD", utxt)

    @staticmethod
    def getMasterHex(mnemonic, passphrase=""):
        assert(BIP39.phraseIsValid(mnemonic))
        mnemonic = BIP39.normalizeString(mnemonic)
        passphrase = BIP39.normalizeString(passphrase)
        passphrase = "mnemonic" + passphrase
        mnemonic = mnemonic.encode("utf-8")
        passphrase = passphrase.encode("utf-8")
        stretched = hashlib.pbkdf2_hmac("sha512", mnemonic, passphrase, 2048)
        return stretched[:64]


    @staticmethod
    def phraseIsValid(mnemonic):
        mnemonic = BIP39.normalizeString(mnemonic).split(" ")
        # list of valid mnemonic lengths
        if len(mnemonic) not in [12, 15, 18, 21, 24]:
            return False
        try:
            idx = map(lambda x: bin(WORD_LIST.index(x))[2:].zfill(11), mnemonic)
            b = "".join(idx)
        except ValueError:
            return False
        l = len(b)  # noqa: E741
        d = b[: l // 33 * 32]
        h = b[-l // 33 :]
        nd = binascii.unhexlify(hex(int(d, 2))[2:].rstrip("L").zfill(l // 33 * 8))
        nh = bin(int(hashlib.sha256(nd).hexdigest(), 16))[2:].zfill(256)[: l // 33]
        return h == nh


    @staticmethod
    def toMnemonic(data):
        if len(data) not in [16, 20, 24, 28, 32]:
            raise ValueError(
                "Data length should be one of the following: [16, 20, 24, 28, 32], but it is not (%d)."
                % len(data)
            )
        h = hashlib.sha256(data).hexdigest()
        b = (
            bin(int(safe_hexlify(data), 16))[2:].zfill(len(data) * 8)
            + bin(int(h, 16))[2:].zfill(256)[: len(data) * 8 // 32]
        )
        result = []
        for i in range(len(b) // 11):
            idx = int(b[i * 11 : (i + 1) * 11], 2)
            result.append(WORD_LIST[idx])

        result_phrase = " ".join(result)
        return result_phrase


    @staticmethod
    def generateMnemonicPassphrase(strength=128):
        if strength not in [128, 160, 192, 224, 256]:
            raise ValueError(
                "Strength should be one of the following [128, 160, 192, 224, 256], but it is not (%d)."
                % strength
            )
        return BIP39.toMnemonic(os.urandom(strength // 8))



class KeyNode(object):
    priv_version = 0x0488ADE4
    pub_version = 0x0488B21E
    addr_type = 0

    def __init__(self, key = None, chain_code = None, extkey = None, child_num = 0, parent_fp = 0, depth = 0):
        self.version = None
        self.depth = None
        self.parent_fp = None
        self.child_num = None
        self.chain_code = None
        self.key = None
        self.valid = False
        self.pubkey = None
        self.pubkey_compressed = None

        if key and chain_code:
            self.key = key
            self.chain_code = chain_code
            self.child_num = child_num
            self.parent_fp = safe_from_hex(format(parent_fp, '#010x')[2:])
            self.depth = depth
            self.version = KeyNode.priv_version
            if self.key:
                if len(self.key) == 32:
                    self.key = b'\00'+self.key
                elif len(self.key) != 33:
                    raise ValueError('Invalid key.')

                K0, K0_compressed = get_pubkeys_from_secret(self.key[1:])
                self.pubkey = K0
                self.pubkey_compressed = K0_compressed
            
            self.valid = True
        elif extkey:
            self.parseExtKey(extkey)

    def parseExtKey(self, extKey):
        if len(extKey) != 78:
            raise ValueError("Invalid extended key length.")

        self.version = extKey[0:4]
        self.depth = from_int_to_byte(from_byte_to_int(extKey[4]))
        self.parent_fp = extKey[5:9]
        self.child_num = extKey[9:13]
        self.chain_code = extKey[13:45]
        self.key = extKey[45:78]
        self.version = int(bytes_to_hex_string(self.version), 16)
        self.depth = int(bytes_to_hex_string(self.depth), 16)
        self.child_num = int(bytes_to_hex_string(self.child_num), 16)

        if self.isPrivate():
            if self.version != KeyNode.priv_version:
                raise ValueError("Invalid extended key version.")            

            K0, K0_compressed = get_pubkeys_from_secret(self.key[1:])
            self.pubkey = K0
            self.pubkey_compressed = K0_compressed
        else:
            if self.version != KeyNode.pub_version:
                raise ValueError("Invalid extended key version.")            

            self.pubkey_compressed = self.key
            self.pubkey = KeyTreeUtil.compressedPubKeyToUncompressedPubKey(self.key)

        self.valid = True

    def getVersion(self):
        return self.version

    def getDepth(self):
        return self.depth

    def getParentFingerPrint(self):
        return self.parent_fp

    def getChildNum(self):
        return self.child_num

    def getChainCodeBytes(self):
        return self.chain_code

    def getKeyBytes(self):
        return self.key

    def getPubKeyBytes(self, compressed):
        if compressed:
            return self.pubkey_compressed
        else:
            return self.pubkey

    def isPrivate(self):
        return len(self.key) == 33 and from_byte_to_int(self.key[0]) == 0x00

    def getFingerPrint(self):
        return hash_160(self.pubkey_compressed)[:4]

    def getPublic(self):
        if not self.valid:
            raise Exception('Keychain is invalid.')

        pub = KeyNode()
        pub.valid = self.valid
        pub.version = KeyNode.pub_version
        pub.depth = self.depth
        pub.parent_fp = self.parent_fp
        pub.child_num = self.child_num
        pub.chain_code = self.chain_code
        pub.pubkey = self.pubkey
        pub.pubkey_compressed = self.pubkey_compressed
        pub.key = self.pubkey_compressed
        return pub

    def getChild(self, i):
        if not self.valid:
            raise Exception('Keychain is invalid.')

        if not self.isPrivate() and KeyTreeUtil.isPrime(i):
            raise Exception('Cannot do private key derivation on public key.')

        child = KeyNode()
        child.valid = False
        child.version = self.version
        child.depth = self.depth + 1
        child.parent_fp = self.getFingerPrint()
        child.child_num = i

        if self.isPrivate():
            child.key, child.chain_code = CKD(self.key[1:], self.chain_code, i)
            # pad with 0's to make it 33 bytes
            zeroPadding = b'\00'*(33 - len(child.key))
            child.key = zeroPadding + child.key
            child.pubkey, child.pubkey_compressed = get_pubkeys_from_secret(child.key[1:])
        else:
            child.pubkey, child.pubkey_compressed, child.chain_code = CKD_prime(self.pubkey, self.chain_code, i)
            child.key = child.pubkey_compressed

        child.valid = True
        return child

    def getPrivKey(self, compressed):
        return SecretToASecret(self.key[1:], compressed, KeyNode.addr_type)

    def getPubKey(self, compressed):
        if compressed:
            return bytes_to_hex_string(self.pubkey_compressed)
        else:
            return bytes_to_hex_string((b'\04' + self.pubkey))

    def getAddress(self, compressed):
        if compressed:
            return hash_160_to_bc_address(hash_160(self.pubkey_compressed), KeyNode.addr_type)
        else:
            return hash_160_to_bc_address(hash_160(b'\04' + self.pubkey), KeyNode.addr_type)

    def getExtKey(self):
        depthBytes = safe_from_hex(format(self.depth, '#04x')[2:])
        childNumBytes = safe_from_hex(format(self.child_num, '#010x')[2:])
        versionBytes = safe_from_hex(format(self.version, '#010x')[2:])
        extkey = versionBytes+depthBytes+self.parent_fp+childNumBytes+self.chain_code+self.key
        return EncodeBase58Check(extkey)

    @staticmethod
    def setTestNet(enabled):
        if enabled:
            KeyNode.priv_version = 0x04358394
            KeyNode.pub_version = 0x043587CF
            KeyNode.addr_type = 111
        else:
            KeyNode.priv_version = 0x0488ADE4
            KeyNode.pub_version = 0x0488B21E
            KeyNode.addr_type = 0


def testVector1():
    optionsDict = {OUTPUT_ENTIRE_CHAIN_OPTION:True, VERBOSE_OPTION:False}
    #optionsDict[VERBOSE_OPTION] = True
    outputExtKeysFromSeed("000102030405060708090a0b0c0d0e0f", "0'/1/2'/2/1000000000", StringType.HEX, 0, optionsDict)

def testVector2():
    optionsDict = {OUTPUT_ENTIRE_CHAIN_OPTION:True, VERBOSE_OPTION:False}
    #optionsDict[VERBOSE_OPTION] = True
    seed = "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542"
    outputExtKeysFromSeed(seed,"0/2147483647'/1/2147483646'/2", StringType.HEX, 0, optionsDict)

class TestKeyTree(unittest.TestCase):
    def setUp(self):
      KeyNode.setTestNet(False)

    def testVector1(self):
      seedHexStr = "000102030405060708090a0b0c0d0e0f"
      master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
      k = master_secret
      c = master_chain
      keyNode = KeyNode(key = k, chain_code = c)

      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub661MyMwAqRbcFtXgS5sYJABqqG9YLmC4Q1Rdap9gSE8NqtwybGhePY2gZ29ESFjqJoCu1Rupje8YtGqsefD265TMg7usUDFdp6W1EGMcet8')
      self.assertEqual(keyNode.getExtKey(), 'xprv9s21ZrQH143K3QTDL4LXw2F7HEK3wJUD2nW2nRk4stbPy6cq3jPPqjiChkVvvNKmPGJxWUtg6LnF5kejMRNNU3TGtRBeJgk33yuGBxrMPHi')
      self.assertEqual(keyNode.getPrivKey(True), 'L52XzL2cMkHxqxBXRyEpnPQZGUs3uKiL3R11XbAdHigRzDozKZeW')
      self.assertEqual(keyNode.getAddress(True), '15mKKb2eos1hWa6tisdPwwDC1a5J1y9nma')

      keyNode = keyNode.getChild(KeyTreeUtil.toPrime(0))
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub68Gmy5EdvgibQVfPdqkBBCHxA5htiqg55crXYuXoQRKfDBFA1WEjWgP6LHhwBZeNK1VTsfTFUHCdrfp1bgwQ9xv5ski8PX9rL2dZXvgGDnw')
      self.assertEqual(keyNode.getExtKey(), 'xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7')
      self.assertEqual(keyNode.getPrivKey(True), 'L5BmPijJjrKbiUfG4zbiFKNqkvuJ8usooJmzuD7Z8dkRoTThYnAT')
      self.assertEqual(keyNode.getAddress(True), '19Q2WoS5hSS6T8GjhK8KZLMgmWaq4neXrh')

      keyNode = keyNode.getChild(1)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6ASuArnXKPbfEwhqN6e3mwBcDTgzisQN1wXN9BJcM47sSikHjJf3UFHKkNAWbWMiGj7Wf5uMash7SyYq527Hqck2AxYysAA7xmALppuCkwQ')
      self.assertEqual(keyNode.getExtKey(), 'xprv9wTYmMFdV23N2TdNG573QoEsfRrWKQgWeibmLntzniatZvR9BmLnvSxqu53Kw1UmYPxLgboyZQaXwTCg8MSY3H2EU4pWcQDnRnrVA1xe8fs')
      self.assertEqual(keyNode.getPrivKey(True), 'KyFAjQ5rgrKvhXvNMtFB5PCSKUYD1yyPEe3xr3T34TZSUHycXtMM')
      self.assertEqual(keyNode.getAddress(True), '1JQheacLPdM5ySCkrZkV66G2ApAXe1mqLj')

      keyNode = keyNode.getChild(KeyTreeUtil.toPrime(2))
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6D4BDPcP2GT577Vvch3R8wDkScZWzQzMMUm3PWbmWvVJrZwQY4VUNgqFJPMM3No2dFDFGTsxxpG5uJh7n7epu4trkrX7x7DogT5Uv6fcLW5')
      self.assertEqual(keyNode.getExtKey(), 'xprv9z4pot5VBttmtdRTWfWQmoH1taj2axGVzFqSb8C9xaxKymcFzXBDptWmT7FwuEzG3ryjH4ktypQSAewRiNMjANTtpgP4mLTj34bhnZX7UiM')
      self.assertEqual(keyNode.getPrivKey(True), 'L43t3od1Gh7Lj55Bzjj1xDAgJDcL7YFo2nEcNaMGiyRZS1CidBVU')
      self.assertEqual(keyNode.getAddress(True), '1NjxqbA9aZWnh17q1UW3rB4EPu79wDXj7x')

      keyNode = keyNode.getChild(2)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6FHa3pjLCk84BayeJxFW2SP4XRrFd1JYnxeLeU8EqN3vDfZmbqBqaGJAyiLjTAwm6ZLRQUMv1ZACTj37sR62cfN7fe5JnJ7dh8zL4fiyLHV')
      self.assertEqual(keyNode.getExtKey(), 'xprvA2JDeKCSNNZky6uBCviVfJSKyQ1mDYahRjijr5idH2WwLsEd4Hsb2Tyh8RfQMuPh7f7RtyzTtdrbdqqsunu5Mm3wDvUAKRHSC34sJ7in334')
      self.assertEqual(keyNode.getPrivKey(True), 'KwjQsVuMjbCP2Zmr3VaFaStav7NvevwjvvkqrWd5Qmh1XVnCteBR')
      self.assertEqual(keyNode.getAddress(True), '1LjmJcdPnDHhNTUgrWyhLGnRDKxQjoxAgt')

      keyNode = keyNode.getChild(1000000000)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6H1LXWLaKsWFhvm6RVpEL9P4KfRZSW7abD2ttkWP3SSQvnyA8FSVqNTEcYFgJS2UaFcxupHiYkro49S8yGasTvXEYBVPamhGW6cFJodrTHy')
      self.assertEqual(keyNode.getExtKey(), 'xprvA41z7zogVVwxVSgdKUHDy1SKmdb533PjDz7J6N6mV6uS3ze1ai8FHa8kmHScGpWmj4WggLyQjgPie1rFSruoUihUZREPSL39UNdE3BBDu76')
      self.assertEqual(keyNode.getPrivKey(True), 'Kybw8izYevo5xMh1TK7aUr7jHFCxXS1zv8p3oqFz3o2zFbhRXHYs')
      self.assertEqual(keyNode.getAddress(True), '1LZiqrop2HGR4qrH1ULZPyBpU6AUP49Uam')

    def testVector2(self):
      seedHexStr = "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542"
      master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
      k = master_secret
      c = master_chain
      keyNode = KeyNode(key = k, chain_code = c)

      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub661MyMwAqRbcFW31YEwpkMuc5THy2PSt5bDMsktWQcFF8syAmRUapSCGu8ED9W6oDMSgv6Zz8idoc4a6mr8BDzTJY47LJhkJ8UB7WEGuduB')
      self.assertEqual(keyNode.getExtKey(), 'xprv9s21ZrQH143K31xYSDQpPDxsXRTUcvj2iNHm5NUtrGiGG5e2DtALGdso3pGz6ssrdK4PFmM8NSpSBHNqPqm55Qn3LqFtT2emdEXVYsCzC2U')
      self.assertEqual(keyNode.getPrivKey(True), 'KyjXhyHF9wTphBkfpxjL8hkDXDUSbE3tKANT94kXSyh6vn6nKaoy')
      self.assertEqual(keyNode.getAddress(True), '1JEoxevbLLG8cVqeoGKQiAwoWbNYSUyYjg')

      keyNode = keyNode.getChild(0)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub69H7F5d8KSRgmmdJg2KhpAK8SR3DjMwAdkxj3ZuxV27CprR9LgpeyGmXUbC6wb7ERfvrnKZjXoUmmDznezpbZb7ap6r1D3tgFxHmwMkQTPH')
      self.assertEqual(keyNode.getExtKey(), 'xprv9vHkqa6EV4sPZHYqZznhT2NPtPCjKuDKGY38FBWLvgaDx45zo9WQRUT3dKYnjwih2yJD9mkrocEZXo1ex8G81dwSM1fwqWpWkeS3v86pgKt')
      self.assertEqual(keyNode.getPrivKey(True), 'L2ysLrR6KMSAtx7uPqmYpoTeiRzydXBattRXjXz5GDFPrdfPzKbj')
      self.assertEqual(keyNode.getAddress(True), '19EuDJdgfRkwCmRzbzVBHZWQG9QNWhftbZ')

      keyNode = keyNode.getChild(KeyTreeUtil.toPrime(2147483647))
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6ASAVgeehLbnwdqV6UKMHVzgqAG8Gr6riv3Fxxpj8ksbH9ebxaEyBLZ85ySDhKiLDBrQSARLq1uNRts8RuJiHjaDMBU4Zn9h8LZNnBC5y4a')
      self.assertEqual(keyNode.getExtKey(), 'xprv9wSp6B7kry3Vj9m1zSnLvN3xH8RdsPP1Mh7fAaR7aRLcQMKTR2vidYEeEg2mUCTAwCd6vnxVrcjfy2kRgVsFawNzmjuHc2YmYRmagcEPdU9')
      self.assertEqual(keyNode.getPrivKey(True), 'L1m5VpbXmMp57P3knskwhoMTLdhAAaXiHvnGLMribbfwzVRpz2Sr')
      self.assertEqual(keyNode.getAddress(True), '1Lke9bXGhn5VPrBuXgN12uGUphrttUErmk')

      keyNode = keyNode.getChild(1)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6DF8uhdarytz3FWdA8TvFSvvAh8dP3283MY7p2V4SeE2wyWmG5mg5EwVvmdMVCQcoNJxGoWaU9DCWh89LojfZ537wTfunKau47EL2dhHKon')
      self.assertEqual(keyNode.getExtKey(), 'xprv9zFnWC6h2cLgpmSA46vutJzBcfJ8yaJGg8cX1e5StJh45BBciYTRXSd25UEPVuesF9yog62tGAQtHjXajPPdbRCHuWS6T8XA2ECKADdw4Ef')
      self.assertEqual(keyNode.getPrivKey(True), 'KzyzXnznxSv249b4KuNkBwowaN3akiNeEHy5FWoPCJpStZbEKXN2')
      self.assertEqual(keyNode.getAddress(True), '1BxrAr2pHpeBheusmd6fHDP2tSLAUa3qsW')

      keyNode = keyNode.getChild(KeyTreeUtil.toPrime(2147483646))
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6ERApfZwUNrhLCkDtcHTcxd75RbzS1ed54G1LkBUHQVHQKqhMkhgbmJbZRkrgZw4koxb5JaHWkY4ALHY2grBGRjaDMzQLcgJvLJuZZvRcEL')
      self.assertEqual(keyNode.getExtKey(), 'xprvA1RpRA33e1JQ7ifknakTFpgNXPmW2YvmhqLQYMmrj4xJXXWYpDPS3xz7iAxn8L39njGVyuoseXzU6rcxFLJ8HFsTjSyQbLYnMpCqE2VbFWc')
      self.assertEqual(keyNode.getPrivKey(True), 'L5KhaMvPYRW1ZoFmRjUtxxPypQ94m6BcDrPhqArhggdaTbbAFJEF')
      self.assertEqual(keyNode.getAddress(True), '15XVotxCAV7sRx1PSCkQNsGw3W9jT9A94R')

      keyNode = keyNode.getChild(2)
      keyNodePub = keyNode.getPublic()
      self.assertEqual(keyNodePub.getExtKey(), 'xpub6FnCn6nSzZAw5Tw7cgR9bi15UV96gLZhjDstkXXxvCLsUXBGXPdSnLFbdpq8p9HmGsApME5hQTZ3emM2rnY5agb9rXpVGyy3bdW6EEgAtqt')
      self.assertEqual(keyNode.getExtKey(), 'xprvA2nrNbFZABcdryreWet9Ea4LvTJcGsqrMzxHx98MMrotbir7yrKCEXw7nadnHM8Dq38EGfSh6dqA9QWTyefMLEcBYJUuekgW4BYPJcr9E7j')
      self.assertEqual(keyNode.getPrivKey(True), 'L3WAYNAZPxx1fr7KCz7GN9nD5qMBnNiqEJNJMU1z9MMaannAt4aK')
      self.assertEqual(keyNode.getAddress(True), '14UKfRV9ZPUp6ZC9PLhqbRtxdihW9em3xt')

    def testBIP39AndBIP44_1(self):
      mnemonic = 'pilot dolphin motion portion survey sock turkey afford destroy knee sock sibling'
      seedHexStr = '936ae011512b96e7ce3ff05d464e3801834d023249baabfebfe13e593dc33610ea68279c271df6bab7cfbea8bbcf470e050fe6589f552f7e1f6c80432c7bcc57'
      self.assertEqual(BIP39.phraseIsValid(mnemonic), True)
      self.assertEqual(safe_hexlify(BIP39.getMasterHex(mnemonic)), seedHexStr)

      master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
      k = master_secret
      c = master_chain
      keyNode = KeyNode(key = k, chain_code = c)

      # "44'/0'/(0-1)'/(0-1)/(0-1)

      # "44'/0'/0'
      keyNodeBIP44Account0 = keyNode.getChild(KeyTreeUtil.toPrime(44)).getChild(KeyTreeUtil.toPrime(0)).getChild(KeyTreeUtil.toPrime(0))
      keyNodePubkeyNodeBIP44Account0 = keyNodeBIP44Account0.getPublic()
      self.assertEqual(keyNodePubkeyNodeBIP44Account0.getExtKey(), 'xpub6C8GhTSMvJj3sxXBn2MExN2gNNNheLcp6n82KW2cPbvrMPAB9ph7REGW3NCb1SYVkV8B2Jkkg3YH9k1n9wvV8BdBq87hTHDP9rAo1ajg2zi')
      self.assertEqual(keyNodeBIP44Account0.getExtKey(), 'xprv9y8vHwuU5wAkfUSifzpEbE5wpLYDEstxjZCRX7czqGPsUaq2cHNrsRx2C4zdbcLguGqsAeJQ82kxnEXcwsYr859mLmd8Z619aAjmoaarJYr')

      k = keyNodeBIP44Account0.getChild(0).getChild(0)
      # "44'/0'/0'/0/0
      self.assertEqual(k.getPrivKey(True), 'KxCM9pZVYWQ1KVhsgqs3ityzC6ix934uR4XXurzo2rm2qPhZNCb7')
      self.assertEqual(k.getAddress(True), '141Cx3X4fy22kBCprmQaFSbEz8R7bPtY6r')
      self.assertEqual(k.getPubKey(True), '039dfdf237c90bc58d4a00810fea1881c146ada2b0b5a24a9058587d5a5f7ee56b')
      self.assertEqual(k.getPrivKey(False), '5J37qDb6ZWmdmyM8rosJqTjN7bNjrQ8EEcEiSa5V6rnmpiozgj7')
      self.assertEqual(k.getAddress(False), '193CmiHk68uvwBm3EWzUT9xBwEV5axRLRc')
      self.assertEqual(k.getPubKey(False), '049dfdf237c90bc58d4a00810fea1881c146ada2b0b5a24a9058587d5a5f7ee56b6f7f197628446e66f048920b1e2ecba96f209120a7cdc9965fc920d2103b8fb1')
      k = keyNodeBIP44Account0.getChild(0).getChild(1)
      # "44'/0'/0'/0/1
      self.assertEqual(k.getPrivKey(True), 'Kz5JBbALpaf8GmopZSVXqoCvJ42NWBf16jCZaJG7rzuqCARpnjYy')
      self.assertEqual(k.getAddress(True), '1ARLhiAbSZNC9vv8R1AB5cbc6aWQEpNXJ6')
      self.assertEqual(k.getPubKey(True), '022742251a2cabaf82c647641481b00e8c6c8c74134046c881ab76af5191baaffd')
      self.assertEqual(k.getPrivKey(False), '5JToVEBGpV1wDBc5S2FAhxnJ4Qo5RHTjcje4JWjr8hm9CcxZrEr')
      self.assertEqual(k.getAddress(False), '17n9XZTpSuyLBrLboh1v9Hx4c9tyVreFJp')
      self.assertEqual(k.getPubKey(False), '042742251a2cabaf82c647641481b00e8c6c8c74134046c881ab76af5191baaffd6b369b7c88db765b920039b447817b82c53e7ae6ad75465e249a7a0f61f35f56')

      k = keyNodeBIP44Account0.getChild(1).getChild(0)
      # "44'/0'/0'/1/0
      self.assertEqual(k.getPrivKey(True), 'L43BWnRwRrq6cSMECCqALAT9M1rGZxQNr64WcRGyGQUaysC9vTYW')
      self.assertEqual(k.getAddress(True), '12e3sfXMoZ2afB8sxTSKyiQgu9cZtzvxHa')
      self.assertEqual(k.getPubKey(True), '03638f7c478e63deae7200a474d10b5519d0352898c37b3a66aa365578930ce20c')
      self.assertEqual(k.getPrivKey(False), '5KMtKXgdoQyfRo58XBPEtvLyGQknfjTbQMdbMG1uAqNCn4Uu7s4')
      self.assertEqual(k.getAddress(False), '1GtKGcMKckcQCJpN1DqKhYN4eKNgm2AwD2')
      self.assertEqual(k.getPubKey(False), '04638f7c478e63deae7200a474d10b5519d0352898c37b3a66aa365578930ce20c8048261edf4b86af1a7053fd84826cbcbe95cd5dc33c04c9ef82656cac8ccc19')
      k = keyNodeBIP44Account0.getChild(1).getChild(1)
      # "44'/0'/0'/1/1
      self.assertEqual(k.getPrivKey(True), 'Kz59nEqpedMQxM4oi5zYSQcF3a7rksCVnh4agonXXRRdUxRBeXzr')
      self.assertEqual(k.getAddress(True), '1JB6H4KxgGbnYiNNxuHHqhddyPqCbBvtLX')
      self.assertEqual(k.getPubKey(True), '03fcbe014cab36a7c2a4d0babfd8185de0e7cabe5b4e5158f2ba6d1066a5210f73')
      self.assertEqual(k.getPrivKey(False), '5JTmap6AtaAA42Nhwd2Fpu2BtLHWQ5aRCFy6NFgKF9FeGTJWKuN')
      self.assertEqual(k.getAddress(False), '17A9hhFSsrC6JEYgjiCni7H4UDwewDNZRK')
      self.assertEqual(k.getPubKey(False), '04fcbe014cab36a7c2a4d0babfd8185de0e7cabe5b4e5158f2ba6d1066a5210f734e9985a9d1ab4b5ae240b23b82b48e0acf18a8babcbcf4ec89872dacec0b6b49')

      # "44'/0'/1'
      keyNodeBIP44Account1 = keyNode.getChild(KeyTreeUtil.toPrime(44)).getChild(KeyTreeUtil.toPrime(0)).getChild(KeyTreeUtil.toPrime(1))
      keyNodePubkeyNodeBIP44Account1 = keyNodeBIP44Account1.getPublic()
      self.assertEqual(keyNodePubkeyNodeBIP44Account1.getExtKey(), 'xpub6C8GhTSMvJj3xRbmbq6uneY49PHP9qBm8pRRv2G4FZHLwWdhTE6Gv1BqpYrL4jCWNB3K7yYCEYxn1TTC7dT749nftPCD8BFnPPNQxi5T2oo')
      self.assertEqual(keyNodeBIP44Account1.getExtKey(), 'xprv9y8vHwuU5wAkjwXJVoZuRWbKbMStkNTumbVq7drShDkN4iJYugn2NCsMyGqfibc9wyQk9bpZdS2D4wtFHVi2jGB6TcJTA34oJUfJYKZ37XU')

      k = keyNodeBIP44Account1.getChild(0).getChild(0)
      # "44'/0'/1'/0/0
      self.assertEqual(k.getPrivKey(True), 'L3w3wFQZZTvgYPXn9ep1MJeYfSenTZ86zqivE1V2ijsJ1vSncrdm')
      self.assertEqual(k.getAddress(True), '1HaPoaEHYSDZjHR18yxm9DRLrkfX8qpXEX')
      self.assertEqual(k.getPubKey(True), '0220aa64a35e8f46d94d8acf8f82581e539aa8db34723bf7c00376ccaea9bc06eb')
      self.assertEqual(k.getPrivKey(False), '5KLVm1YcvaHhE17ySybfbWNeXuwKuguMquGqmBXpWr8mdxWEYrq')
      self.assertEqual(k.getAddress(False), '1HETVjsvDcjiyiiuHUPVw1TbgNU7KR7gxT')
      self.assertEqual(k.getPubKey(False), '0420aa64a35e8f46d94d8acf8f82581e539aa8db34723bf7c00376ccaea9bc06eb43867ceaebcdd3360b5187d1e0920d9f8ef28063a466b84b0d732e4f1c45ade4')
      k = keyNodeBIP44Account1.getChild(0).getChild(1)
      # "44'/0'/1'/0/1
      self.assertEqual(k.getPrivKey(True), 'Kz2EfWsP8toKdbBpdGyrQL3rR83bpLbrqmhpGjotMXnbrvGebsum')
      self.assertEqual(k.getAddress(True), '18bGTq3Q8AnKwSMFjsuHvG8pVcYJFa6v7A')
      self.assertEqual(k.getPubKey(True), '02e4ec0fc3919b38acea7a0331fe2720ad89f678ae65922285a7359df3ad25365f')
      self.assertEqual(k.getPrivKey(False), '5JT7GXL7q1Qdr57z1WuJY7aj1UstPWN3EKHVqReoym8cYJYwJQc')
      self.assertEqual(k.getAddress(False), '18Pwyap9XdnZrqqqmCskPbCv4ZN3vyLSxU')
      self.assertEqual(k.getPubKey(False), '04e4ec0fc3919b38acea7a0331fe2720ad89f678ae65922285a7359df3ad25365f4dc76e1f622d554fea1dfa45974ad3330b4770731944b1b59a97309a7a9f6808')

      k = keyNodeBIP44Account1.getChild(1).getChild(0)
      # "44'/0'/1'/1/0
      self.assertEqual(k.getPrivKey(True), 'L1w4DP1b98vaghjsqK7g9vWE4ANkBjfgXfBaKYrS7HofQ4NKombU')
      self.assertEqual(k.getAddress(True), '1DjZygViqqRRfwwZxK7CssnnpQooAHheuh')
      self.assertEqual(k.getPubKey(True), '0368b105d83122aabfcf3c64f20d4b6a4ef280b1aaea82a843ba6c6d7961885427')
      self.assertEqual(k.getPrivKey(False), '5JtDWYGYJFUzvrZNVAYRH4y1h91rCwyo7Vb7eHHBg2zhRKZXQYo')
      self.assertEqual(k.getAddress(False), '1BVn1pqN6njq9zvqus9PPeDtbdJKc2bi9m')
      self.assertEqual(k.getPubKey(False), '0468b105d83122aabfcf3c64f20d4b6a4ef280b1aaea82a843ba6c6d7961885427c200026b6bedda94ca37c36c5f485fd814c723b30c8061457bbeb279c884c603')

      k = keyNodeBIP44Account1.getChild(1).getChild(1)
      # "44'/0'/1'/1/1
      self.assertEqual(k.getPrivKey(True), 'L5jS5VLsGGYg2XnkUBS4x9qfpeeASx6ZpAFy8Uh39ggc2SEaneop')
      self.assertEqual(k.getAddress(True), '155P6c59ixhpQyfn7a9mZbH9qsNaVE4Shv')
      self.assertEqual(k.getPubKey(True), '02a59b9a6e1d0552c16efc958dadeb30cd783c4fa5d79a1364e3db6c7b44533fce')
      self.assertEqual(k.getPrivKey(False), '5Kk9QDmNoxddUzGoYzC7naGZMmKqp7bLfn9NnCFxmZZ5mHG4HY9')
      self.assertEqual(k.getAddress(False), '1Dx2QUVVMbjx55AtamXweRXSEadxRipXuJ')
      self.assertEqual(k.getPubKey(False), '04a59b9a6e1d0552c16efc958dadeb30cd783c4fa5d79a1364e3db6c7b44533fce0eaaf3f2414b931409758c6b1f5a62e7f2ce48aa1b5340ac9467e5de8261130c')

    def testBIP39AndBIP44_2(self):
      mnemonic = 'payment minute try rifle weekend spin sentence slush iron fury artist slogan'
      seedHexStr = '3a73c9afd0c9c585d5d1197252ea64030d099c931ebc388b85ba61d95c42503c51e42d3161e0e7fe9933b4dc8866ac390f70e296661462b2deb27f59cb87389c'
      self.assertEqual(BIP39.phraseIsValid(mnemonic), True)
      self.assertEqual(safe_hexlify(BIP39.getMasterHex(mnemonic)), seedHexStr)

      master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
      k = master_secret
      c = master_chain
      keyNode = KeyNode(key = k, chain_code = c)

      # "44'/0'/(0-1)'/(0-1)/(0-1)

      # "44'/0'/0'
      keyNodeBIP44Account0 = keyNode.getChild(KeyTreeUtil.toPrime(44)).getChild(KeyTreeUtil.toPrime(0)).getChild(KeyTreeUtil.toPrime(0))
      keyNodePubkeyNodeBIP44Account0 = keyNodeBIP44Account0.getPublic()
      self.assertEqual(keyNodePubkeyNodeBIP44Account0.getExtKey(), 'xpub6CKJAMrz4m23L1tHakEGy4X9dr9Re39YES4XunwZkZn78HCpcXj9kYWfYaeAtA7H1rrmebffgg6fAPR1GJbHJqVnUS7pmBkwoMB7uW3Ftju')
      self.assertEqual(keyNodeBIP44Account0.getExtKey(), 'xprv9yKwkrL6EPTk7XopUihGbvaR5pJwEaRgsD8w7QXxCEF8FUsg4zQuCkCBhKxA8zrXMQ2rqSEcmDEc5DcPKYz4Jbbk1ryX4eSWfpgYRZqWW3b')

      k = keyNodeBIP44Account0.getChild(0).getChild(0)
      # "44'/0'/0'/0/0
      self.assertEqual(k.getPrivKey(True), 'L2485SWvCq24gLnkaWhJB7hgKhCuWKNrFUPoZYivrhtPHDLVucwc')
      self.assertEqual(k.getAddress(True), '1FaRiujjaQSC6585ixvRepBionjyzM6drN')
      self.assertEqual(k.getPubKey(True), '02c309b6898ed9b9156ff9847dfd3ea3753a1bce18ea3d8f916473ad1f5c9aa637')
      self.assertEqual(k.getPrivKey(False), '5JupNPzyuyeQ37kSjV2RjcxSDqz5CB48zCeLcU8EZZNcQfGvhb1')
      self.assertEqual(k.getAddress(False), '1GfPr7Wvpay8ihWZ52WS5ibNnkivnJJQSy')
      self.assertEqual(k.getPubKey(False), '04c309b6898ed9b9156ff9847dfd3ea3753a1bce18ea3d8f916473ad1f5c9aa637754d0f09e44155e21a0d0310aa16f3b3f611af4ad3fb3436693c68f74a976396')
      k = keyNodeBIP44Account0.getChild(0).getChild(1)
      # "44'/0'/0'/0/1
      self.assertEqual(k.getPrivKey(True), 'L2wXUmKTwKsQM5DiproJpwgzhnN7p5FCzd6XoteXAb1ky1bi7LeG')
      self.assertEqual(k.getAddress(True), '1NR4SxQioSepwBS6N6pdJ18uBT4H9Yxct')
      self.assertEqual(k.getPubKey(True), '033feacb414cabac1e4a69951d30deeea1201ff72151b26b362dda63c2ee9fb57a')
      self.assertEqual(k.getPrivKey(False), '5K7TqtGxwuJmFyEkuz3nj8QZZMFJQMSskR3AFFRMt7M7LWA2Lx5')
      self.assertEqual(k.getAddress(False), '16ahGxDzFYnK7RG4PmvjMWCEBj8DHHz2i')
      self.assertEqual(k.getPubKey(False), '043feacb414cabac1e4a69951d30deeea1201ff72151b26b362dda63c2ee9fb57a70174cdfc3c04914eae40a8a3c4b439faa81571bd570fa25e4eebcab9930edd7')

      k = keyNodeBIP44Account0.getChild(1).getChild(0)
      # "44'/0'/0'/1/0
      self.assertEqual(k.getPrivKey(True), 'L3grZRytSJxVberN8Z5yNocvq4vEgX1ng2oZm1SG6MSJD25F72ZA')
      self.assertEqual(k.getAddress(True), '1HhCwuYhxtG8Ums95ZTuYciayJbDNy9ofQ')
      self.assertEqual(k.getPubKey(True), '0237723b2125bf63afded5184ddeddabb728912f02cc2e2ff2a4e0c128f6aee212')
      self.assertEqual(k.getPrivKey(False), '5KHHDLFheuZ4JpYaLPDueZmPcp7Au11BMAvbcM36gjEtUgDoRGF')
      self.assertEqual(k.getAddress(False), '126sEodKKqTxBGRz7H7ZDtoKYwTJZ2cGJH')
      self.assertEqual(k.getPubKey(False), '0437723b2125bf63afded5184ddeddabb728912f02cc2e2ff2a4e0c128f6aee21255b11b1045d7bff24bcdc0d648ca9109d06bc39e676f3ed761877c333fa958be')
      k = keyNodeBIP44Account0.getChild(1).getChild(1)
      # "44'/0'/0'/1/1
      self.assertEqual(k.getPrivKey(True), 'KwraHz3vtY22aaGP1jfpoY7xk1fHRHsURLLUcitzDNs7dz321e1p')
      self.assertEqual(k.getAddress(True), '1AXvaXfeUyrod7xfxjQzDfKTipfeNdktg8')
      self.assertEqual(k.getPubKey(True), '031aeabce44a716a55d46a79a5d3a1d2353389f1230dc5e5a24595dcc98b884e58')
      self.assertEqual(k.getPrivKey(False), '5HxdzmeNGkjNey2Dh3RCjiVK4rYL8KKgk86mmoCKrUcNShEu7Yn')
      self.assertEqual(k.getAddress(False), '1LWvFbks5Z7d4neXJe5z4MNffAgn6vWK5f')
      self.assertEqual(k.getPubKey(False), '041aeabce44a716a55d46a79a5d3a1d2353389f1230dc5e5a24595dcc98b884e58652d187da697210f43512e94afd15f93cc34d8cd796c6eeeef9c7435300c0a89')

      # "44'/0'/1'
      keyNodeBIP44Account1 = keyNode.getChild(KeyTreeUtil.toPrime(44)).getChild(KeyTreeUtil.toPrime(0)).getChild(KeyTreeUtil.toPrime(1))
      keyNodePubkeyNodeBIP44Account1 = keyNodeBIP44Account1.getPublic()
      self.assertEqual(keyNodePubkeyNodeBIP44Account1.getExtKey(), 'xpub6CKJAMrz4m23Q36BMYQmR5eWaG19Gc41knevzToxToqpc5j5owUXiEBg1DG8pdiQ7vvq5prrB7HPLTzS7CtV7rzexSZ33sWmXnRD6yMB9G1')
      self.assertEqual(keyNodeBIP44Account1.getExtKey(), 'xprv9yKwkrL6EPTkBZ1iFWsm3whn2EAes9LAPZjLC5QLuUJqjHPwGQAHARsC9xFXi9H96w5x7fSNn1GWD7xEo1Mam5PY87yzPNP5KMPQMhuR3ip')

      k = keyNodeBIP44Account1.getChild(0).getChild(0)
      # "44'/0'/1'/0/0
      self.assertEqual(k.getPrivKey(True), 'L4RidpdkZSbVRnjtCRtXZdnpQe6GwG2GDB6315tMbAmfkdo4DZsD')
      self.assertEqual(k.getAddress(True), '1LEtY7Vqfg6QEvNnQrrbYLpLQjFmqt84XD')
      self.assertEqual(k.getPubKey(True), '020ed67f822cf5b08a1b46737ef4a572874b36f87aa0f29c851be3c846eb69b1fe')
      self.assertEqual(k.getPrivKey(False), '5KSzTvPNei9GWBN3Y8uTQhxhCFJ4A82eRxkjbTUwxmopkVKM74q')
      self.assertEqual(k.getAddress(False), '1FFV5TjKKixJN3bu4MiGnxBWr9voF3TZ3X')
      self.assertEqual(k.getPubKey(False), '040ed67f822cf5b08a1b46737ef4a572874b36f87aa0f29c851be3c846eb69b1fec705b286fd4b88a933b682061f0e01179b31e140a0f57bdb5bc128f7100ff046')
      k = keyNodeBIP44Account1.getChild(0).getChild(1)
      # "44'/0'/1'/0/1
      self.assertEqual(k.getPrivKey(True), 'KyTWJGcG6BgsUsR3LE5SEEqCo3ZD7sh5sgLSFe5vyXkbraJcwmZy')
      self.assertEqual(k.getAddress(True), '14rNR9AWgiHCDtjUYUWn1JwEWSxtTeiaui')
      self.assertEqual(k.getPubKey(True), '024b6cb1ea39a0283dcabe7206b6d56a414014ad18d12411466a4fb630244d7483')
      self.assertEqual(k.getPrivKey(False), '5JKhAoapNZnHRSUCQPSvkKzbPfeCosTJRApT7VLhKhVuksTd22L')
      self.assertEqual(k.getAddress(False), '19ZPEMom1UE1t2cYQzsDzgVK6KLpg6zdQP')
      self.assertEqual(k.getPubKey(False), '044b6cb1ea39a0283dcabe7206b6d56a414014ad18d12411466a4fb630244d7483b072ce80ef43780badc5fe5cdbb13644afe6327eaf45971bf0c5e55ab34b720e')

      k = keyNodeBIP44Account1.getChild(1).getChild(0)
      # "44'/0'/1'/1/0
      self.assertEqual(k.getPrivKey(True), 'L2dfzRoFiQK2VgaGyYWnuFz82CqTdzJXEmMRsBMxh7eZsMV4iC4J')
      self.assertEqual(k.getAddress(True), '1D28BJPdrdhRzYzPHsdktDPKSP5ZKX54df')
      self.assertEqual(k.getPubKey(True), '029e878cf81b3a0a24c3f2af8b0c50b01b86da656305e93ff1f52d62b69716b458')
      self.assertEqual(k.getPrivKey(False), '5K3REvCPNAMrH1oxjGJAwtRivHjFC6jV9ZFbdk8U7pj3TpfVZkj')
      self.assertEqual(k.getAddress(False), '18YHWphSmjj1FTPuz3sbqZGNTfzPCrotye')
      self.assertEqual(k.getPubKey(False), '049e878cf81b3a0a24c3f2af8b0c50b01b86da656305e93ff1f52d62b69716b4588a3cb05ce968b26d6af16fefda76e2fb84b75a5d9475f1ebe9ecea8ff320e3fa')
      k = keyNodeBIP44Account1.getChild(1).getChild(1)
      # "44'/0'/1'/1/1
      self.assertEqual(k.getPrivKey(True), 'L54L4MYkj4sBMzW5Zs44EMEtiRk51ng7PB2p69B8ZBynonu6EsxX')
      self.assertEqual(k.getAddress(True), '1Hko6Wt7D2NL6djdjGz7vxgXwhTEMWaQKQ')
      self.assertEqual(k.getPubKey(True), '033529ad53c5ece0b87a237299ad4541bf9dc8e31171495b5f9f70c28e4dc88412')
      self.assertEqual(k.getPrivKey(False), '5KbHZ2V3CoBv6fQMX6uMDYKZ7DQMPqQpDGsFkJiGJiTTh51Kx6R')
      self.assertEqual(k.getAddress(False), '133XTyugjyV8NLzvkfRRLri6uLFz4acBCs')
      self.assertEqual(k.getPubKey(False), '043529ad53c5ece0b87a237299ad4541bf9dc8e31171495b5f9f70c28e4dc88412f5d124ed7b3b0ad23a2d1d2d8251f6504dc833a3f55fbb3ac7628d0cf658dbc1')

    def testGenerateMnemonic(self):
      mnemonic = BIP39.generateMnemonicPassphrase()
      self.assertEqual(BIP39.phraseIsValid(mnemonic), True)
      self.assertEqual(len(mnemonic.split()), 12)

    def testExtendedKeyPrimeDerivation(self):
      extKey = 'xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7'
      extKeyBytes = DecodeBase58Check(extKey)
      keyNode = KeyNode(extkey = extKeyBytes)
      self.assertEqual(keyNode.getPublic().getExtKey(), 'xpub68Gmy5EdvgibQVfPdqkBBCHxA5htiqg55crXYuXoQRKfDBFA1WEjWgP6LHhwBZeNK1VTsfTFUHCdrfp1bgwQ9xv5ski8PX9rL2dZXvgGDnw')

      keyNode = keyNode.getChild(KeyTreeUtil.toPrime(0)).getChild(8)
      keyNodePublic = keyNode.getPublic()
      self.assertEqual(keyNodePublic.getExtKey(), 'xpub6DQgP643Y39n9c1tbmwb6opi5ndTBYFK8K7UbFt8XfYcrM4aBZFL8KXBEfFg5ngnUrUd9T5ZouvoWuVgU8KguhTjzm9VnUx8x4tySSCSUXf')
      self.assertEqual(keyNode.getExtKey(), 'xprv9zRKyaX9hfbUw7wRVkQajfsyXknxn5XTm6BsnsUWyL1dyYjRe1w5aXChPP2JXdbUU7DMgy2KC8w9MBjz78GjxGxTa7MCnPHAgFQaYLv2oay')
      self.assertEqual(keyNode.getPrivKey(True), 'L5UUipv5Fp94yK3xLqqfNKhXA1BZPt64JBw4jhnvqax36AsqHA3W')
      self.assertEqual(keyNode.getAddress(True), '14hDwAhNkhCprB9Xbk1drgZiamitcHRkQD')
      self.assertEqual(keyNode.getPubKey(True), '0394580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1')
      self.assertEqual(keyNode.getPrivKey(False), '5Kgktcn6x1iFVcotWRCw1JNsLFzg4NYWJ32Nh2zXmzY3jumJ6CD')
      self.assertEqual(keyNode.getAddress(False), '167eGArJCtpd1vxBTwzucW33oSNGyUYq5S')
      self.assertEqual(keyNode.getPubKey(False), '0494580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1e989476e1fd426c56e1a0284a1c8430a597233669ae614a4335becd0ff9b0c49')

      self.assertEqual(keyNodePublic.getAddress(True), '14hDwAhNkhCprB9Xbk1drgZiamitcHRkQD')
      self.assertEqual(keyNodePublic.getPubKey(True), '0394580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1')
      self.assertEqual(keyNodePublic.getAddress(False), '167eGArJCtpd1vxBTwzucW33oSNGyUYq5S')
      self.assertEqual(keyNodePublic.getPubKey(False), '0494580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1e989476e1fd426c56e1a0284a1c8430a597233669ae614a4335becd0ff9b0c49')

      self.assertEqual(is_valid(keyNode.getAddress(True)), True)
      self.assertEqual(is_valid(keyNode.getAddress(False)), True)

      extKey = 'xpub6DQgP643Y39n9c1tbmwb6opi5ndTBYFK8K7UbFt8XfYcrM4aBZFL8KXBEfFg5ngnUrUd9T5ZouvoWuVgU8KguhTjzm9VnUx8x4tySSCSUXf'
      extKeyBytes = DecodeBase58Check(extKey)
      keyNodePublic = KeyNode(extkey = extKeyBytes)

      self.assertEqual(keyNodePublic.getAddress(True), '14hDwAhNkhCprB9Xbk1drgZiamitcHRkQD')
      self.assertEqual(keyNodePublic.getPubKey(True), '0394580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1')
      self.assertEqual(keyNodePublic.getAddress(False), '167eGArJCtpd1vxBTwzucW33oSNGyUYq5S')
      self.assertEqual(keyNodePublic.getPubKey(False), '0494580846dd10ad927b46eeecdb691073d560fbfeca28b3df87d1af41435e8ac1e989476e1fd426c56e1a0284a1c8430a597233669ae614a4335becd0ff9b0c49')

      self.assertEqual(is_valid(keyNodePublic.getAddress(True)), True)
      self.assertEqual(is_valid(keyNodePublic.getAddress(False)), True)

    def testExtendedKeyNonPrimeDerivation(self):
      extKey = 'xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7'
      extKeyBytes = DecodeBase58Check(extKey)
      keyNode = KeyNode(extkey = extKeyBytes)

      keyNode = keyNode.getChild(0).getChild(8)
      keyNodePublic = keyNode.getPublic()
      self.assertEqual(keyNodePublic.getExtKey(), 'xpub6DEF9h3BTTA8Ka3SSg9aek42rcAKSsXbpczRPzKiFTjCfJg3oeqy6VcUoYLxsEqggRFpXF97XN1pqVYu5FwKJKu2Xsw2PXnDrk8h7cr8rEi')
      self.assertEqual(keyNode.getExtKey(), 'xprv9zEtkBWHd5bq75xyLecaHc7JJaKq3QokTQ4pbbv6h8CDnWLuG7XiYhHzxFdTUsvYaZBEtAXNeB4Rdm1Unk3nbtbzxF8h6sy6ZJnazdeX7ep')
      self.assertEqual(keyNode.getPrivKey(True), 'L1Zku8j3mCiiHxZdo6NDLHv6jcA1JyNufUSHBMiznML38vNr9Agh')
      self.assertEqual(keyNode.getAddress(True), '17JbSP83rPWmbdcdtiiTNqBE8MgGN8kmUk')
      self.assertEqual(keyNode.getPubKey(True), '035f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1')
      self.assertEqual(keyNode.getPrivKey(False), '5JoPdjXpkYezqhgZohBhQQKNh1iTofbsSwFLXSYGMhQLyDJ3beh')
      self.assertEqual(keyNode.getAddress(False), '1LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVrC')
      self.assertEqual(keyNode.getPubKey(False), '045f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1ae9d3053c9a526cd97dd5e4c2483992963a061b6271f89482b5e30da450234e1')

      self.assertEqual(is_valid(keyNode.getAddress(True)), True)
      self.assertEqual(is_valid(keyNode.getAddress(False)), True)

      self.assertEqual(keyNodePublic.getAddress(True), '17JbSP83rPWmbdcdtiiTNqBE8MgGN8kmUk')
      self.assertEqual(keyNodePublic.getPubKey(True), '035f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1')
      self.assertEqual(keyNodePublic.getAddress(False), '1LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVrC')
      self.assertEqual(keyNodePublic.getPubKey(False), '045f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1ae9d3053c9a526cd97dd5e4c2483992963a061b6271f89482b5e30da450234e1')


      extKey = 'xpub68Gmy5EdvgibQVfPdqkBBCHxA5htiqg55crXYuXoQRKfDBFA1WEjWgP6LHhwBZeNK1VTsfTFUHCdrfp1bgwQ9xv5ski8PX9rL2dZXvgGDnw'
      extKeyBytes = DecodeBase58Check(extKey)
      keyNode = KeyNode(extkey = extKeyBytes)

      keyNodePublic = keyNode.getChild(0).getChild(8)
      self.assertEqual(keyNodePublic.getExtKey(), 'xpub6DEF9h3BTTA8Ka3SSg9aek42rcAKSsXbpczRPzKiFTjCfJg3oeqy6VcUoYLxsEqggRFpXF97XN1pqVYu5FwKJKu2Xsw2PXnDrk8h7cr8rEi')

      self.assertEqual(keyNodePublic.getAddress(True), '17JbSP83rPWmbdcdtiiTNqBE8MgGN8kmUk')
      self.assertEqual(keyNodePublic.getPubKey(True), '035f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1')
      self.assertEqual(keyNodePublic.getAddress(False), '1LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVrC')
      self.assertEqual(keyNodePublic.getPubKey(False), '045f980e89a2c5d5805e44ac5c55441be195deba5aa46910f66be3d6e0b6c7c3c1ae9d3053c9a526cd97dd5e4c2483992963a061b6271f89482b5e30da450234e1')


      self.assertEqual(is_valid(keyNodePublic.getAddress(True)), True)
      self.assertEqual(is_valid(keyNodePublic.getAddress(False)), True)

    def testIsValidAddress(self):
      self.assertEqual(is_valid('14hDwAhNkhCprB9Xbk1drgZiamitcHRkQD'), True)
      self.assertEqual(is_valid('167eGArJCtpd1vxBTwzucW33oSNGyUYq5S'), True)
      self.assertEqual(is_valid('17JbSP83rPWmbdcdtiiTNqBE8MgGN8kmUk'), True)
      self.assertEqual(is_valid('1LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVrC'), True)
      self.assertEqual(is_valid('123'), False)
      self.assertEqual(is_valid('1LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVr'), False)
      self.assertEqual(is_valid('LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVrC'), False)
      self.assertEqual(is_valid('LdwPM9Qpt5ucZck9SGjwkZh1VzTfrYVr'), False)

def suite():
  suite = unittest.TestSuite()
  suite.addTest(TestKeyTree('testVector1'))
  suite.addTest(TestKeyTree('testVector2'))
  suite.addTest(TestKeyTree('testBIP39AndBIP44_1'))
  suite.addTest(TestKeyTree('testBIP39AndBIP44_2'))
  suite.addTest(TestKeyTree('testGenerateMnemonic'))
  suite.addTest(TestKeyTree('testExtendedKeyPrimeDerivation'))
  suite.addTest(TestKeyTree('testExtendedKeyNonPrimeDerivation'))
  suite.addTest(TestKeyTree('testExtendedKeyNonPrimeDerivation'))
  suite.addTest(TestKeyTree('testIsValidAddress'))
  return suite

def parse_arguments(argv):
    argsDict = {}
    it = 0
    while it < len(argv):
        arg = argv[it]
        if arg[0] != '-':
            raise ValueError("Invalid arguments.")

        arg = arg[1:]
        if arg == HELP or arg == HELP_SHORT:
            argsDict[HELP] = HELP
            break
        elif arg == RUN_TESTS or arg == RUN_TESTS_SHORT:
            argsDict[RUN_TESTS] = "Y"
            break
        elif arg == SEED or arg == SEED_SHORT:
            argsDict[SEED_FORMAT] = "" #assumes ascii
            argsDict[SEED] = "Y"
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[SEED_VALUE] = argv[it]
        elif arg == SEED_HEX or arg == SEED_HEX_SHORT:
            argsDict[SEED_FORMAT] = "hex"
            argsDict[SEED] = "Y"
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[SEED_VALUE] = argv[it]
        elif arg == EXTENDEDKEY or arg == EXTENDEDKEY_SHORT:
            argsDict[EXTENDEDKEY] = "Y"
            
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[EXTENDEDKEY_VALUE] = argv[it]
        elif arg == CHAIN or arg == CHAIN_SHORT:
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[CHAIN_VALUE] = argv[it]
        elif arg == TREE_TRAVERSAL_OPTION or arg == TREE_TRAVERSAL_OPTION_SHORT:
            it += 1
            argsDict[TREE_TRAVERSAL_OPTION] = argv[it]
            argsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = "Y"
        elif arg == OUTPUT_ENTIRE_CHAIN_OPTION or arg == OUTPUT_ENTIRE_CHAIN_OPTION_SHORT:
            argsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = "Y"
        elif arg == VERBOSE_OPTION or arg == VERBOSE_OPTION_SHORT:
            argsDict[VERBOSE_OPTION] = "Y"
        elif arg == NO_INPUT_ECHO or arg == NO_INPUT_ECHO_SHORT:
            global noInputEcho
            noInputEcho = True
        elif arg == TESTNET or arg == TESTNET_SHORT:
            argsDict[TESTNET] = "Y"
        elif arg == HASH_SEED or arg == HASH_SEED_SHORT:
            argsDict[HASH_SEED] = "Y"
            
            if getOptionValue(argsDict.get(NO_PROMPT)):
                it += 1
                argsDict[HASH_SEED] = argv[it]
        elif arg == NO_PROMPT or arg == NO_PROMPT_SHORT:
            argsDict[NO_PROMPT] = "Y"
        elif arg == ENFORCE_BIP39_RULE or arg == ENFORCE_BIP39_RULE_SHORT:
            argsDict[ENFORCE_BIP39_RULE] = "Y"
        elif arg == GENERATE_MNEMONIC_PASSPHRASE or arg == GENERATE_MNEMONIC_PASSPHRASE_SHORT:
            argsDict[GENERATE_MNEMONIC_PASSPHRASE] = "Y"
        else:
            raise ValueError("Invalid arguments.")

        it += 1

    # default to seed if no option provided
    if argsDict.get(EXTENDEDKEY) == None and argsDict.get(SEED) == None:
        argsDict[SEED] = "Y"
    
    return argsDict

def outputExamples():
    outputString("Extended Keys can be in hex or base58. Seed can be in ASCII or hex. Examples below.")
    outputString("")
    
    outputString("To use KeyTree simply do the following:")
    outputString(cmdName)
    outputString("Enter Seed:")
    outputString("correct horse battery staple")
    outputString("Enter Chain:")
    outputString("0'/0")
    outputString("")
    
    outputString("Use the BIP39 option to check if the seed/mnemonic conforms to bip39 rules:")
    outputString(cmdName+" --bip39")
    outputString("Enter Seed:")
    outputString("correct horse battery staple")
    outputString("Enter Chain:")
    outputString("0'/0")
    outputString("")

    outputString("Use the hex option to enter the seed in hex:")
    outputString(cmdName+" --seedhex")
    outputString("Enter Seed in Hex:")
    outputString("000102030405060708090a0b0c0d0e0f")
    outputString("Enter Chain:")
    outputString("0'/1/2")
    outputString("")

    outputString("Use the hash seed option to do a number of specific rounds of sha256 on your seed. If the bip39 option is used hash seed option will be ignored:")
    outputString(cmdName+" --seedhex --hashseed")
    outputString("Enter Seed in Hex:")
    outputString("000102030405060708090a0b0c0d0e0f")
    outputString("Enter Chain:")
    outputString("0'/1/2")
    outputString("Enter number of rounds of Sha256 hash:")
    outputString("2")
    outputString("")

    outputString("Use the generate BIP39 mnemonic option to generate a mnemonic using os.urandom:")
    outputString(cmdName+" --generateBIP39mnemonic")
    outputString(cmdName+" -gb39m")
    outputString("")

    outputString("Use the extended key option to enter the extended key in lieu of the seed:")
    outputString(cmdName+" --extkey")
    outputString(cmdName+" -ek")
    outputString("")
    
    outputString("It is also possible to print multiple chain paths together:")
    outputString(cmdName)
    outputString("Enter Extended Key:")
    outputString("xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7")
    outputString("Enter Chain:")
    outputString("0'/(3-6)'/(1-2)/8")
    outputString("")
    
    outputString("To output all the node data on the chain, use the all option:")
    outputString(cmdName+" --all")
    outputString(cmdName+" -a")
    outputString("")
    
    outputString("It is also possible to output the nodes in a different order:")
    outputString(cmdName+" --traverse levelorder")
    outputString(cmdName+" -trav postorder")
    outputString(cmdName+" -trav preorder")
    outputString("")

    outputString("For more information on the node use the verbose option:")
    outputString(cmdName+" --verbose")
    outputString(cmdName+" -v")
    outputString("")
    
    outputString("There is also the testnet option:")
    outputString(cmdName+" --testnet")
    outputString(cmdName+" -tn")
    outputString("")

    outputString("Use the no echo option to not echo your inputs:")
    outputString(cmdName+" --noecho")
    outputString(cmdName+" -ne")
    outputString("")

    outputString("To run unit tests do the following:")
    outputString(cmdName+" --runtests")
    outputString(cmdName+" -rt")
    outputString("")

    outputString("For information on how to use KeyTree do:")
    outputString(cmdName+" --help")
    outputString(cmdName+" -h")
    outputString("")

    outputString("You can specify all options at once with the no prompt option. But it is discouraged because on most OSs, commands are stored in a history file:")
    outputString(cmdName+" --noprompt -s \"this is a password\" --chain \"(0-1)'/(6-8)'\" -trav levelorder")
    outputString(cmdName+" -np -s \"this is a password\" -c \"(0-1)'/(6-8)'\" -hs 3 -v")
    outputString(cmdName+" -np --extkey xprv9uHRZZhk6KAJC1avXpDAp4MDc3sQKNxDiPvvkX8Br5ngLNv1TxvUxt4cV1rGL5hj6KCesnDYUhd7oWgT11eZG7XnxHrnYeSvkzY7d2bhkJ7 -c \"(0-1)'/8\"")
    outputString(cmdName+" -np --b39 -s \"pilot dolphin motion portion survey sock turkey afford destroy knee sock sibling\" -c \"44'/0'/(0-1)'/(0-1)/0\"")
    outputString(cmdName+" -np -sh 936ae011512b96e7ce3ff05d464e3801834d023249baabfebfe13e593dc33610ea68279c271df6bab7cfbea8bbcf470e050fe6589f552f7e1f6c80432c7bcc57 -c \"44'/0'/(0-1)'/(0-1)/0\"")
    outputString("")

def getTreeTraversalOption(treeTraversalOption):
    if treeTraversalOption == TREE_TRAVERSAL_TYPE_LEVELORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_LEVELORDER_SHORT:
        return TreeTraversal.LEVELORDER
    elif treeTraversalOption == TREE_TRAVERSAL_TYPE_POSTORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_POSTORDER_SHORT:
        return TreeTraversal.POSTORDER
    elif treeTraversalOption == TREE_TRAVERSAL_TYPE_PREORDER or treeTraversalOption == TREE_TRAVERSAL_TYPE_PREORDER:
        return TreeTraversal.PREORDER
    else:
        return DEFAULTTREETRAVERSALTYPE

def getOptionValue(option):
    if option == "Y": return True
    else: return False

def get_input(pretext):
    if noInputEcho:
        return getpass.getpass(pretext+'\n')
    else:   
        if sys.version_info.major == 2:
          return raw_input(pretext+'\n')
        else:
          return input(pretext+'\n')

def enter_prompt(argsDict):
    if argsDict.get(HELP) == HELP:
        outputExamples()
    elif argsDict.get(RUN_TESTS):
        runner = unittest.TextTestRunner()
        runner.run(suite())
    else:
        optionsDict = {}
        optionsDict[TESTNET] = getOptionValue(argsDict.get(TESTNET))
        optionsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = getOptionValue(argsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION))
        optionsDict[VERBOSE_OPTION] = getOptionValue(argsDict.get(VERBOSE_OPTION))
        traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))

        if getOptionValue(argsDict.get(SEED)):
            seed = None
            seed_format = None
            if (argsDict.get(SEED_FORMAT) == "hex"):
                seed_format = StringType.HEX
                seed = get_input("Enter Seed in Hex:")
                try: int(seed, 16)
                except ValueError: raise ValueError("Invalid hex string \"" + seed + "\"")
            else:
                seed_format = StringType.ASCII
                seed = get_input("Enter Seed:")
            
            chain = get_input("Enter Chain:")
            
            optionsDict[ENFORCE_BIP39_RULE] = getOptionValue(argsDict.get(ENFORCE_BIP39_RULE))

            roundsToHash = 0
            if getOptionValue(argsDict.get(HASH_SEED)) and optionsDict[ENFORCE_BIP39_RULE] == False:
                roundsToHashStr = get_input("Enter number of rounds of Sha256 hash:")
                if roundsToHashStr:
                    roundsToHash = int(roundsToHashStr)
            
            outputExtKeysFromSeed(seed, chain, seed_format, roundsToHash, optionsDict, traverseType)
            
        elif getOptionValue(argsDict.get(EXTENDEDKEY)):
            extkey = get_input("Enter Extended Key:")
            chain = get_input("Enter Chain:")
            
            if chain != "":
                outputExtKeysFromExtKey(extkey, chain, optionsDict, traverseType)
            else:
                outputKeyAddressofExtKey(extkey, optionsDict)
    
    return 0

def handle_arguments(argsDict):
    # outputString("Arguments:")
    # for key in argsDict:
    #     outputString("\tkey: " + key + " value: " + argsDict[key])
    # outputString("")

    if argsDict.get(HELP) == HELP:
        outputExamples()
    elif argsDict.get(RUN_TESTS):
        runner = unittest.TextTestRunner()
        runner.run(suite())
    else:
        optionsDict = {}
        optionsDict[TESTNET] = getOptionValue(argsDict.get(TESTNET))
        optionsDict[OUTPUT_ENTIRE_CHAIN_OPTION] = getOptionValue(argsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION))
        optionsDict[VERBOSE_OPTION] = getOptionValue(argsDict.get(VERBOSE_OPTION))

        if argsDict.get(SEED_VALUE) != None:
            seed = argsDict.get(SEED_VALUE)
            seed_format = None
            if argsDict.get(SEED_FORMAT) == "hex":
                seed_format = StringType.HEX
            else:
                seed_format = StringType.ASCII
            
            chain = None
            if argsDict.get(CHAIN_VALUE) != None:
                chain = argsDict.get(CHAIN_VALUE)
            
            roundsToHashStr = argsDict.get(HASH_SEED)
            roundsToHash = 0
            if roundsToHashStr:
                roundsToHash = int(roundsToHashStr)
            
            traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))
            optionsDict[HASH_SEED] = getOptionValue(argsDict.get(HASH_SEED))
            optionsDict[ENFORCE_BIP39_RULE] = getOptionValue(argsDict.get(ENFORCE_BIP39_RULE))

            outputExtKeysFromSeed(seed, chain, seed_format, roundsToHash, optionsDict, traverseType)
        elif argsDict.get(EXTENDEDKEY_VALUE) != None and argsDict.get(CHAIN_VALUE) != None:
            extkey = argsDict.get(EXTENDEDKEY_VALUE)
            chain = argsDict.get(CHAIN_VALUE)
            
            traverseType = getTreeTraversalOption(argsDict.get(TREE_TRAVERSAL_OPTION))
            outputExtKeysFromExtKey(extkey, chain, optionsDict, traverseType)
        elif argsDict.get(EXTENDEDKEY) != None:
            extkey = argsDict.get(EXTENDEDKEY_VALUE)
            outputKeyAddressofExtKey(extkey, optionsDict)
        elif argsDict.get(GENERATE_MNEMONIC_PASSPHRASE) != None:
            outputGeneratedMnemonicPassphrase(optionsDict)
        else:
            raise ValueError("Invalid arguments.")

    return 0

def outputString(string):
    print(string)

def visit(keyNode, chainName, isLeafNode, optionsDict):
    if not isLeafNode and not optionsDict.get(OUTPUT_ENTIRE_CHAIN_OPTION):
        return

    outputString("* [Chain " + chainName + "]")
    if keyNode.isPrivate():
        keyNodePub = keyNode.getPublic()
        outputString("  * ext pub:  " + keyNodePub.getExtKey())
        outputString("  * ext prv:  " + keyNode.getExtKey())
        if optionsDict.get(VERBOSE_OPTION) == False:
            outputString("  * priv key: " + keyNode.getPrivKey(True))
            outputString("  * address:  " + keyNode.getAddress(True))
        else:
            outputString("  * uncompressed priv key: " + keyNode.getPrivKey(False))
            outputString("  * uncompressed pub key:  " + keyNode.getPubKey(False))
            outputString("  * uncompressed address:  " + keyNode.getAddress(False))
            outputString("  * compressed priv key: " + keyNode.getPrivKey(True))
            outputString("  * compressed pub key:  " + keyNode.getPubKey(True))
            outputString("  * compressed address:  " + keyNode.getAddress(True))
    else:
        outputString("  * ext pub:  " + keyNode.getExtKey())
        if optionsDict[VERBOSE_OPTION] == False:
            outputString("  * address:  " + keyNode.getAddress(True))
        else:
            outputString("  * uncompressed pub key:  " + keyNode.getPubKey(False))
            outputString("  * uncompressed address:  " + keyNode.getAddress(False))
            outputString("  * compressed pub key:  " + keyNode.getPubKey(True))
            outputString("  * compressed address:  " + keyNode.getAddress(True))

def traversePreorder(keyNode, treeChains, chainName, optionsDict):
    if treeChains:
        isPrivateNPathRange = treeChains.pop(0)
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        isLeafNode = False
        if not treeChains: isLeafNode = True
        if min == KeyTreeUtil.NODE_IDX_M_FLAG and max == KeyTreeUtil.NODE_IDX_M_FLAG:
            visit(keyNode, chainName, isLeafNode, optionsDict)
            traversePreorder(keyNode, treeChains[:], chainName, optionsDict)
        else:
            for i in range(min, max+1):
                childChainName = chainName + "/" + str(i) + "'" if isPrivate else chainName + "/" + str(i)
                if isPrivate:
                    childNode = keyNode.getChild(KeyTreeUtil.toPrime(i))
                else:
                    childNode = keyNode.getChild(i)

                visit(childNode, childChainName, isLeafNode, optionsDict)
                traversePreorder(childNode, treeChains[:], childChainName, optionsDict)

def traversePostorder(keyNode, treeChains, chainName, optionsDict):
    if treeChains:
        isPrivateNPathRange = treeChains.pop(0)
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        isLeafNode = False
        if not treeChains: isLeafNode = True
        if min == KeyTreeUtil.NODE_IDX_M_FLAG and max == KeyTreeUtil.NODE_IDX_M_FLAG:
            traversePostorder(keyNode, treeChains[:], chainName, optionsDict)
            visit(keyNode, chainName, isLeafNode, optionsDict)
        else:
            for i in range(min, max+1):
                if isPrivate: i = KeyTreeUtil.toPrime(i)
                childChainName = chainName + "/" + KeyTreeUtil.iToString(i)
                childNode = keyNode.getChild(i)

                traversePostorder(childNode, treeChains[:], childChainName, optionsDict)
                visit(childNode, childChainName, isLeafNode, optionsDict)

def traverseLevelorder(keyNode, treeChains, chainName, level, keyNodeDeq, levelNChainDeq, optionsDict):
    isLeafNode = False
    if level < len(treeChains):
        isPrivateNPathRange = treeChains[level]
        isPrivate = isPrivateNPathRange[0]
        min = isPrivateNPathRange[1][0]
        max = isPrivateNPathRange[1][1]
        level += 1
        for i in range(min, max+1):
            if isPrivate: i = KeyTreeUtil.toPrime(i)
            childChainName = chainName + "/" + KeyTreeUtil.iToString(i)
            childNode = keyNode.getChild(i)
            keyNodeDeq.append(childNode)            
            levelNChainDeq.append([level, childChainName])            
    else:
        isLeafNode = True

    visit(keyNode, chainName, isLeafNode, optionsDict)

    if keyNodeDeq:
        pair = levelNChainDeq.pop(0)
        level = pair[0] 
        cc = pair[1] 
        node = keyNodeDeq.pop(0)
        traverseLevelorder(node, treeChains, cc, level, keyNodeDeq, levelNChainDeq, optionsDict)

def outputExtraKeyNodeData(keyNode):
    outputString("  * depth:              " + str(keyNode.getDepth()))
    outputString("  * child number:       " + KeyTreeUtil.iToString(keyNode.getChildNum()))
    outputString("  * parent fingerprint: " + bytes_to_hex_string(keyNode.getParentFingerPrint()))
    outputString("  * fingerprint:        " + bytes_to_hex_string(keyNode.getFingerPrint()))

def outputExtKeysFromSeed(seed, chainStr, seedStringFormat, roundsToHash, optionsDict, traverseType = DEFAULTTREETRAVERSALTYPE):
    seedHexStr = None

    if optionsDict.get(ENFORCE_BIP39_RULE):
        # if doing bip39 then rounds to hash option will not be used because it is not part of the bip39 standard
        if seedStringFormat == StringType.ASCII:
            masterSeed = BIP39.getMasterHex(seed)
            seedHexStr = safe_hexlify(masterSeed)
        elif seedStringFormat == StringType.HEX:
            try: int(seed, 16)
            except ValueError: raise ValueError("Invalid hex string \"" + seed + "\"")
            seedHexStr = seed
        else:
            raise ValueError("Invalid seed string format.")
    else:
        if seedStringFormat == StringType.ASCII:
            seedHexStr = safe_hexlify(from_string_to_bytes(seed))
        elif seedStringFormat == StringType.HEX:
            try: int(seed, 16)
            except ValueError: raise ValueError("Invalid hex string \"" + seed + "\"")
            seedHexStr = seed
        else:
            raise ValueError("Invalid seed string format.")

        if roundsToHash > 0:
            seedHexStr = bytes_to_hex_string(KeyTreeUtil.sha256Rounds(safe_from_hex(seedHexStr) , roundsToHash))


    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    master_secret, master_chain, master_public_key, master_public_key_compressed = bip32_init(seedHexStr)
    k = master_secret
    c = master_chain

    keyNodeSeed = KeyNode(key = k, chain_code = c)

    treeChains = KeyTreeUtil.parseChainString(chainStr)

    if optionsDict.get(VERBOSE_OPTION):
        outputString("Master (hex): " + seedHexStr)

    if traverseType == TreeTraversal.POSTORDER:
        traversePostorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, optionsDict)
    elif traverseType == TreeTraversal.LEVELORDER:
        treeChains.pop(0)
        traverseLevelorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, 0, [], [], optionsDict)
    else:        
        traversePreorder(keyNodeSeed, treeChains, KeyTreeUtil.MASTER_NODE_LOWERCASE_M, optionsDict)

def outputExtKeysFromExtKey(extKey, chainStr, optionsDict, traverseType = DEFAULTTREETRAVERSALTYPE):
    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    keyNode = None
    try:
        int(extKey, 16)
        keyNode = KeyNode(extkey = safe_from_hex(extKey))
    except ValueError:
        extKeyBytes = DecodeBase58Check(extKey)
        if not extKeyBytes:
            raise ValueError('Invalid extended key.')
        keyNode = KeyNode(extkey = extKeyBytes)

    treeChains = KeyTreeUtil.parseChainString(chainStr)

    if optionsDict.get(VERBOSE_OPTION): outputExtraKeyNodeData(keyNode)

    if traverseType == TreeTraversal.POSTORDER:
        traversePostorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, optionsDict)
    elif traverseType == TreeTraversal.LEVELORDER:
        treeChains.pop(0)
        traverseLevelorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, 0, [], [], optionsDict)
    else:        
        traversePreorder(keyNode, treeChains, KeyTreeUtil.LEAD_CHAIN_PATH, optionsDict)

def outputKeyAddressofExtKey(extKey, optionsDict):
    if optionsDict.get(TESTNET) == None or optionsDict.get(TESTNET) == False:
        KeyNode.setTestNet(False)
    else:
        KeyNode.setTestNet(True)

    extKeyBytes = DecodeBase58Check(extKey)
    if not extKeyBytes:
        raise ValueError('Invalid extended key.')

    keyNode = KeyNode(extkey = DecodeBase58Check(extKey))
    if optionsDict.get(VERBOSE_OPTION): outputExtraKeyNodeData(keyNode)
    visit(keyNode, KeyTreeUtil.LEAD_CHAIN_PATH, True, optionsDict)
    outputString("")

def outputGeneratedMnemonicPassphrase(optionsDict):
    mnemonic = BIP39.generateMnemonicPassphrase()
    assert(BIP39.phraseIsValid(mnemonic))
    outputString("mnemonic:")
    outputString(mnemonic)
    if optionsDict.get(VERBOSE_OPTION):
        outputString("Master (hex):")
        outputString(safe_hexlify(BIP39.getMasterHex(mnemonic)))

def main():
    argv = sys.argv[1:]
    argsDict = parse_arguments(argv)

    if getOptionValue(argsDict.get(NO_PROMPT)) or getOptionValue(argsDict.get(GENERATE_MNEMONIC_PASSPHRASE)):
        return handle_arguments(argsDict)
    else:
        return enter_prompt(argsDict)


if __name__ == '__main__':
    #test_crypto()
    #testVector1()
    #testVector2()
    main()