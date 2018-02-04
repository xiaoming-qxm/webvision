# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create artifical word level feature for different classes. """

import json
from os.path import join as pjoin


def save_feature(data_root, feat, class_name):

    with open(pjoin(data_root, "{}.txt".format(class_name)), 'w') as f:
        json.dump(feat, f, indent=4)

data_root = "../data/word_feat"

tench = {"pos": ["fishery", "fishes", "fish", "fishing",
                 "angling", "angler", "anglers", "ichthyology",
                 "golden", "tail", "fin", "cyprinidae", "river",
                 "rivers", "water", "lake", "lakes", "canals",
                 "carp", "habitats", "habitat", "species",
                 "freshwater", "baits", "tails", "fins", "waters"],
         "neg": ["submarines", "people", "prison", "person",
                 "man", "woman", "musician", "site", "submarine",
                 "men", "women", "officer", "actor", "ablum",
                 "film", "song", "singer", "commune"]}
save_feature(data_root, tench, 'tench')

bulbul = {"pos": ["passerine", "bird", "birds", "songbird", "songbirds",
                  "greenbul", "greenbuls", "brownbul", "brownbuls",
                  "leafloves", "bristlebills", "species", "forest",
                  "habitat", "habitats", "rainforest", "tail", "tails",
                  "wing", "wings", "neck", "necked"],
          "neg": ["town", "syria", "singer", "dancer", "player", "film",
                  "lyricist", "composer", "director", "cricketer",
                  "actor", "activist"]}
save_feature(data_root, bulbul, 'bulbul')

terrapin = {"pos": ["turtle", "turtles", "water", "brackish", "sea",
                    "cheloniology", "testudinology", "chelonian", "chelonians",
                    "shell", "leatherback", "freshwater", "carapace", "river",
                    "marsh", "emydidae", "geoemydidae", "neck", "necked"],
            "neg": ["allies", "vehicle", "song", "ship", "beer", "company",
                    "athletic", "portable", "classroom", "album", "studio"]}
save_feature(data_root, terrapin, 'terrapin')

scorpion = {"pos": ["arthropod", "arachnid", "scorpiones", "scorpione",
                    "legs", "pedipalp", "pedipalps", "tail", "species",
                    "sting", "stings", "medical", "venom"],
            "neg": ["people", "pharaoh", "wrestling", "comic", "comics",
                    "marvel", "superhero", "film", "game", "video", "wrestler",
                    "tank", "novel", "music", "band", "album", "song",
                    "drama", "series", "episode", "computer", "cpu", "insect",
                    "paramilitary", "racehorse", "team", "sport", "sports",
                    "vessels", "vessel", "vehicle", "vehicles", "gun", "weaponry",
                    "weapon"]}
save_feature(data_root, scorpion, 'scorpion')

coucal = {"pos": ["bird", "birds", "cuckoo", "centropodinae", "centropus", "claw",
                  "toe", "hallux", "feather", "feathers", "nests", "nest", "fly",
                  "toed", "neck", "necked", "extinct", "feet", "nestling", "nestlings",
                  "pheasant"],
          "neg": []}
save_feature(data_root, coucal, 'coucal')

sea_anemone = {"pos": ["marine", "predatory", "animal", "animals", "actiniaria",
                       "cnidaria", "cnidarians", "anthozoa", "hexacorallia",
                       "float", "polyp", "water", "muscles", "nerves", "swimming",
                       "mouth", "gullet", "predator", "fish"],
               "neg": ["plant", "plants", "flowering", "flower", "botanist",
                       "pulsatilla", "hepatica", "leaves", "leaf", "flowers",
                       "fruits", "fruit", "shrimp", "song", "album", "radar",
                       "actress", "actor", "series", "ship", "navy"]}

save_feature(data_root, sea_anemone, 'sea_anemone')

black_stork = {"pos": ["bird", "birds", "ciconiidae", "plumage", "legs", "leg",
                       "beak", "species", "water", "waters", "migrate", "eggs",
                       "egg", "nest", ],
               "neg": ["film", "motion", "star", "starring", "pakistani",
                       "army", "forces", "navy", "service"]}
save_feature(data_root, black_stork, 'black_stork')

sea_lion = {"pos": ["aquatic", "mammal", "mammals", "foreflipper", "ear", "eared",
                    "fur", "seals", "otariidae", "foreflippers", "belly",
                    "water", "waters", "ocean", "animal", "animals", "marine",
                    "zoo", "species", "specie", "feeding", "feed"],
            "neg": ["ablum", "studio", "invasion", "operation", "submarine",
                    "submarines", "base", "helicopter", "antarctica", "supermarine",
                    "locomotive", "park"]}
save_feature(data_root, sea_lion, 'sea_lion')

beagle = {"pos": ["dog", "dogs", "breed", "hound", "hunt", "hunting",
                  "detection", "pet", "pets", "hunted", "hares", "scent",
                  "beagling", "breeds", "hunter", "hunters", "puppy",
                  "puppies", "animal", "animals"],
          "neg": ["channel", "island", "islands", "conflict", "gulf",
                  "territory", "beer", "brand", "crater", "software",
                  "ship", "darwin", "planet", "computer", "aircraft",
                  "nato", "hockey", "writer", "bay"]}
save_feature(data_root, beagle, 'beagle')

irish_setter = {"pos": ["dog", "dogs", "breed", "breeds", "gundog", "bred",
                        "hunting", "hunt", "hunting", "hunter", "hunters",
                        "gamebirds", "gamebird", "gundogs", "puppy",
                        "puppies", "animal", "animals"],
                "neg": []}
save_feature(data_root, irish_setter, 'irish_setter')

malinois = {"pos": ["dog", "dogs", "breed", "breeds", "detection", "explosives",
                    "accelerants", "narcotics", "belgian", "police",
                    "search", "rescue", "sheepdog", "working", "puppy",
                    "hunting", "hunt", "hunting", "hunter", "hunters",
                    "puppies", "animal", "animals"],
            "neg": []}
save_feature(data_root, malinois, 'malinois')

collie = {"pos": ["dog", "dogs", "breed", "breeds", "herding", "sports", "fur",
                  "pet", "pets", "sport", "showing", "shepherd", "blanco",
                  "reveille", "shep", "border", "bearded", "farm", "puppy",
                  "puppies", "animal", "animals"],
          "neg": ["village", "river", "mountain", "nickname", "surname",
                  "artist", "naval", "squirrel", "cannabis"]}
save_feature(data_root, collie, 'collie')

siberian_husky = {"pos": ["dog", "dogs", "breed", "breeds", "working",
                          "spitz", "hunt", "hunting", "hunter", "hunters",
                          "puppy", "puppies", "animal", "animals"],
                  "neg": []}
save_feature(data_root, siberian_husky, 'siberian_husky')

langur = {"pos": ["monkey", "colobus", "colobine", "primates", "primate",
                  "tail", "tails", "leaves", "fruits", "monkeys",
                  "slender", "tailed", "colobinae", "ape"],
          "neg": ["bala"]}
save_feature(data_root, langur, 'langur')

buckle = {"pos": ["loose", "fasten", "fastening", "belt", "device",
                  "chape", "frame", "bar", "prong", "clasp" "attach",
                  "attached", "fastened", " ornament", "ornaments"
                  ],
          "neg": ["casting", "defect", "surname", "comic", "dessert",
                  "food", "episode", "sport", "sports", "veteran",
                  "automobile", "blimp", "inc", "accessory", "footwear"]}
save_feature(data_root, buckle, 'buckle')

candle = {"pos": ["wick", "ignitable", "flammable", "tallow", "fragrance",
                  "chandlery", "tabletop", "heat", "illumination",
                  "lighting", "flame", "light", "burn", "lights", "luminous"],
          "neg": ["place", "places", "lake", "novel", "music", "band",
                  "rock", "record", "albums", "song", "singer", "singers",
                  "corporation", "facility", "company", "fish", "hap",
                  "ice", "tree", "trees", "conical", "cone", "geometry",
                  "rod", "machinery", "forum", "ratio", "haircut",
                  "surname", "historian", "developer", "financier"]}
save_feature(data_root, candle, 'candle')

canoe = {"pos": ["vessel", "vessels", "paddlers", "paddler", "paddle",
                 "transport", "racing", "canoeing", "touring", "camping",
                 "wood", "fiberglass", "sporting", "sail", "sails",
                 "outboard", "outtriggers", "outtrigger", "hull", "ship",
                 "hulls", "canvas", "sprint", "slalom", "wildwater",
                 "freestyle", "blade", "paddles", "boat"],
         "neg": ["city", "district", "volcano", "reserve", "park",
                 "portal", "war", "leader", "restaurant"]}
save_feature(data_root, canoe, 'canoe')

carton = {"pos": ["box", "container", "paperboard", "fiberboard", "packaging",
                  "package", "folding", "egg", "eggs", "trays", "tray",
                  "aseptic", "liquids", "liquid", "gable", "containers",
                  "cardboard", "paper", "packed", "storage"],
          "neg": ["surname", "personality", "singer", "songwriter",
                  "guitarist", "politician", "hurler", "physician",
                  "actress", "philosopher", "hero", "heroes"]}
save_feature(data_root, carton, 'carton')

cowboy_boot = {"pos": ["riding", "worn", "cuban", "toe", "shaft",
                       "leatuer", "arched", "heel", "stitching",
                       "tooling", "chunky", "slanted", "footwear"],
               "neg": []}
save_feature(data_root, cowboy_boot, 'cowboy_boot')

dam = {"pos": ["barrier", "water", "impoundment", "streams", "stream",
               "flower", "floods", "flood", "hydropower", "floodgates",
               "levees", "levee", "floodgate", "gravity", "barrage",
               "arch", "embankment", "saddle", "weir", "tailings",
               "spillways", "spillway", "hydroelectric", "reclamation",
               "rock"],
       "neg": ["square", "band", "reservoir", "coin", "chemotherapy",
               "methylase", "dance", "animal", "people",
               "minster", "biochemist", "physiologist", "footballer",
               "boxer", "politician", "artist"]}
save_feature(data_root, dam, 'dam')

dutch_oven = {"pos": ["cooking", "pot", "walled", "lid", "iron",
                      "casserole", "dishes", "cookware", "braadpan",
                      "roasting", "sudderpan", "simmerpan", "simmering",
                      "heating", "camping", "chuckwagon", "bedourie",
                      "cookpot", "potjie", "chugunok"],
              "neg": ["racehorse", "thoroughbred", "pillar", "newspaper",
                      "satirical", "pitcher", "derek"]}
save_feature(data_root, dutch_oven, 'dutch_oven')

fountain = {"pos": ["fons", "spring", "water", "basin", "jets", "drinking",
                    "decorative", "dramatic", "springs", "aqueducts",
                    "decoration", "decorated", "basins", "musical",
                    "splash", "source"],
            "neg": ["heraldry", "soda", "space", "elevator", "pen",
                    "writing", "clarinetist", "writer", "minister", "lawyer"]}
save_feature(data_root, fountain, 'fountain')

hard_disk = {"pos": ["data", "storage", "device", "magnetic", "retrieve",
                     "digital", "hdd", "stored", "retrieved", "computer",
                     "capacity", "hdds", "usb", "sata", "cylinder", "recording",
                     "information", "spindle", "logical", "blocks", "latency",
                     "noise", "desktop", "laptop", "recorders"],
             "neg": ["album", "fictional", "character", "radio", "film", "novel"]
             }
save_feature(data_root, hard_disk, 'hard_disk')

harvester = {"pos": ["mechanical", "machine", "automated", "mechanised", "wheel",
                     "machinery", "engine", "engines", "vehicle", "arm"],
             "neg": ["grim", "death", "painting", "paint", "sculpture", "miniatures",
                     "alien", "game", "ufo", "comic", "comics", "protagonist",
                     "marvel", "fiction", "fictional", "ends", "antagonists",
                     "shannara", "strain", "blade", "vampire", "infamous", "faction",
                     "subnautica", "overwatch", "film", "sarandon", "catherine",
                     "novel", "discworld", "pratchett", "magazine", "theristai",
                     "band", "album", "albums", "song", "songs", "series", "episode",
                     "miniseries", "episodes", "atomics", "workstation", "software",
                     "program", "operating", "zombie", "sailing", "vessel",
                     "aircraft", "football", "team", "butterflies", "butterfly",
                     "arachnids", "arachnid", "spider", "spiders", "opiliones",
                     "miletinae", "tarquinius", "bioinformatic", "download", "web",
                     "website", "websites", "community", "restaurant", "fictional",
                     "navy", "ship", "ships", "horse", "satyr"]}
save_feature(data_root, harvester, 'harvester')

paperknife = {"pos": ["tool", "cutting", "pages", "book", "books", "cut",
                      "envelopes", "edge", "cutting", "instrument", "blade",
                      "sharp", "handle"],
              "neg": []}
save_feature(data_root, paperknife, 'paperknife')

lifeboat = {"pos": ["boat", "boats", "emergency", "evacuation", "disaster",
                    "shop", "inflatable", "inflation", "ships", "inflating",
                    "life", "rafts", "raft", "liferafts", "deck", "sink",
                    "sinking"],
            "neg": ["film", "television", "episode", "sketch", "album",
                    "albums", "song", "songs", "software", "foundation",
                    "distributor", "distribution", "music"]}
save_feature(data_root, lifeboat, 'lifeboat')

lotion = {"pos": ["shampoo", "skin", "treatment", "cream", "creams",
                  "viscosity", "hand", "body", "moisturize", "care",
                  "emulsion", "aqueous", "oily", "gel", "fragrances",
                  "glycerol", "lubricants", "absorption", "artistry",
                  "gold", "bond", "olay", "aveeno", "jergens", "sebamed",
                  "bath", "johnson", "skincare", "cetaphil", "lux", "suave",
                  "dial", "neutrogena", "vaseline", "eucerin", "nivea",
                  "moisturizer", "liquid", "soothing", "cleansing", "antiseptic",
                  "washing", "bathing", "hands"],
          "neg": ["band", "song", "album"]}
save_feature(data_root, lotion, 'lotion')

matchstick = {"pos": ["fire", "fires", "matches", "wooden", "sticks", "heat",
                      "frictional", "ignite", "igniting", "burn", "ignition",
                      "friction", "candle", "burning", "light", "lights",
                      "lighting", "cigar", "cigars", "wood", "lighter", "cardboard"],
              "neg": ["bandsia", "girl", "fictional", "graph", "geometric", "movie",
                      "film", "model", "models", "rock", "music", "tv", "brand"]}
save_feature(data_root, matchstick, 'matchstick')

minivan = {"pos": ["carrier", "mpv", "muv", "vehicle", "layout", "roof",
                   "hatchback", "engine", "mechanicals", "car", "wheel",
                   "sedan", "wagon", "seating", "seat", "seats", "doors",
                   "doors", "cars", "motors", "mpvs", "muvs", "carrying",
                   "cargo", "passengers", "passenger"],
           "neg": []}
save_feature(data_root, minivan, 'minivan')

padlock = {"pos": ["lock", "locks", "shackle", "chain", "hasp", "rusty",
                   "attached", "attach", "key", "keys", "gate", "door",
                   "doors", "safety", "locksmiths", "machined", "security",
                   "locking", "secured", "shackles", "unlock", "secure",
                   ],
           "neg": ["comic", "opera", "issac", "bickerstaffe",
                   "gwen", "guthrie", "album", "via",
                   "processor", "cpu", "cpus"]}
save_feature(data_root, padlock, 'padlock')

rocking_chair = {"pos": ["wooden", "curved", "wood", "sit", "sat", "sits",
                         "furniture", "legs", "springs", "sitter"],
                 "neg": ["song", "music", "single", "album", "band", "pitcher", "musician",
                         "songwriter", "neighborhood", "painter", "illustrator",
                         "writer", "historian", "coach", "football", "drummer", "film",
                         "literature", "band", "brand", "branded", "rock", "looptroop",
                         "dolly", "bombay", "revenge", "hop", "hip", "soundtrack",
                         "singer", "drumbeat", "beer", "subculture", "motorcycle",
                         "laboratory", "mixing", "engine", "vehicle", "panel", "team",
                         "basketball", "skis", "skating", "wrestling"]}
save_feature(data_root, rocking_chair, 'rocking_chair')

rubber_eraser = {"pos": ["stationery", "writing", "paper", "pencils", "mistakes",
                         "gum", "marks", "pen", "chalk", "brushes"],
                 "neg": ["film", "album", "song", "latex", "natural", "tree",
                         "novel", "band", "bridge", "baseball", "cricket",
                         "game"]}
save_feature(data_root, rubber_eraser, 'rubber_eraser')

screw = {"pos": ["fastener", "bolt", "metal", "helical", "ridge", "thread",
                 "nail", "threads", "nut", "groove", "screwdrivers",
                 "screwdriver", "wrenches", "wrench", "bolts", "nuts",
                 "tighten", "tightened", "threaded", "insertion", "assemble",
                 "assembled", "torque", "torquing", "torqued", "released",
                 "release", "assembly", "wood", "fasteners"],
         "neg": ["band", "game", "magazine", "ablum", "motion", "propeller",
                 "music", "song"]}
save_feature(data_root, screw, 'screw')

strainer = {"pos": ["plumbing", "cooking", "cook", "utensil", "colander",
                    "sieve", "metal", "filter", "sink", "water", "basket",
                    "perforated", "perforate", "corrugated", "corrugate",
                    "flow", "bowl", "holes", "draining", "drain",
                    "drains", "drained", " aluminium", "stainless",
                    "plastic", "mesh", "dry", "sifters", "sifter"],
            "neg": ["composer", "organist", "professor", "athlete",
                    "rower", "music", "coefficient", "band"]}
save_feature(data_root, strainer, 'strainer')

studio_couch = {"pos": ["sleep", "sleeping", "sofa", "sofas", "bed",
                        "cot", "lie", "lying", "sleeper", "lies",
                        "lay", "sleeps", "slept", "upholstered",
                        "furniture"],
                "neg": []}
save_feature(data_root, studio_couch, 'studio_couch')

trifle = {"pos": ["recipe", "recipes", "taste", "eat", "ate", "food",
                  "dessert", "cake", "cream", "brandy", "rum", "preserve",
                  "preserves", "fruit", "fruits", "custard", "custards",
                  "jam", "sherry", "wine", "sponge", "jelly"],
          "neg": []}
save_feature(data_root, trifle, 'trifle')

bell_pepper = {"pos": ["sweet", "cultivars", "cultivar", "food",
                       "plant", "capsicum", "capsaicin", "taste",
                       "vegetables", "vegetable", "recipe", "recipes",
                       "vitamin", "annuum", "pungent"],
               "neg": []}
save_feature(data_root, bell_pepper, 'bell_pepper')

granny_smith = {"pos": ["apple", "apples", "cultivar", "tree",
                        "fruit", "fruits", "juicy", "fresh",
                        "flesh", "crisp", "acidity", "green",
                        "acidity", "mellows", "mellow", "ripening",
                        "flavour", "vitamin", "vitamins", "antioxidant",
                        "diets", "dishes", "tart", "eat", "ate", "eaten"],
                "neg": ["gold", "tv", "television", "series",
                        "festival", "character"]}
save_feature(data_root, granny_smith, 'granny_smith')

dough = {"pos": ["cook", "cooking", "ingredient", "malleable", "grains",
                 "flour", "water", "shaping", "foodstuffs", "bread",
                 "biscuits", "cakes", "cookies", "noodles", "paste",
                 "flatbreads", "dumplings", "pizza", "flours", "piecrusts",
                 "wheat", "maize", "rice", "rye", "legumes", "almonds",
                 "yeast", "yeasts", "knead", "kneading", "kneaded",
                 "baking", "mixture", "pastry"],
         "neg": ["episode", "series", "television", "sitcom", "broadcast"]}
save_feature(data_root, dough, 'dough')

cup = {"pos": ["drink", "coffe", "milk", "water", "container",
               "drinking", "drinks", "glass", "plastic", "wood",
               "thirst", "teacups", "teacup", "drinkable", "tea",
               "coffee", "mazagran", "mug", "thermos", "paper",
               "beer", "wine", "cocktail", "liquor", "glasses",
               "tankard", "bottom", "handle"],
       "neg": ["bra", "breast", "breasts", "single", "song", "game",
               "equipment", "series", "film", "book", "payment"]}
save_feature(data_root, cup, 'cup')
