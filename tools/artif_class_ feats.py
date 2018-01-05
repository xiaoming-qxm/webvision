# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

""" Create artifical word level feature for different classes. """

import json
from os.path import join as pjoin


def save_feature(data_root, feat, class_name):

    with open(pjoin(data_root, "{}.txt".format(class_name)), 'w') as f:
        json.dump(feat, f, indent=4)


tench = {"pos": ["fishery", "fishes", "fish", "fishing",
                 "angling", "angler", "anglers", "ichthyology",
                 "golden", "tail", "fin", "12lbs", "1oz",
                 "cyprinidae", "river", "rivers", "water",
                 "lake", "lakes", "canals", "freshwater", "baits",
                 "carp", "habitats", "habitat", "species",
                 "tails", "fins", "waters"],
         "neg": ["submarines", "people", "prison", "person",
                 "man", "woman", "musician", "site", "submarine",
                 "men", "women", "officer", "actor"]}

bulbul = {"pos": ["passerine", "bird", "birds", "songbird", "songbirds",
                  "greenbul", "greenbuls", "brownbul", "brownbuls",
                  "leafloves", "bristlebills", "species", "forest",
                  "habitat", "habitats", "rainforest", "tail", "tails",
                  "wing", "wings", "neck", "necked"],
          "neg": ["town", "syria", "singer", "dancer", "player", "film",
                  "lyricist", "composer", "director", "cricketer",
                  "actor", "activist"]}

terrapin = {"pos": ["turtle", "turtles", "water", "brackish", "sea",
                    "cheloniology", "testudinology", "chelonian", "chelonians",
                    "shell", "leatherback", "freshwater", "carapace", "river",
                    "marsh", "emydidae", "geoemydidae", "neck", "necked"],
            "neg": ["allies", "vehicle", "song", "ship", "beer", "company",
                    "athletic", "portable", "classroom", "album", "studio"]}

scorpion = {"pos": ["arthropod", "arachnid", "scorpiones", "scorpione",
                    "legs", "pedipalp", "pedipalps", "tail", "species",
                    "sting", "stings", "medical", "venom"],
            "neg": ["people", "pharaoh", "wrestling", "comic", "comics",
                    "marvel", "superhero", "film", "game", "video", "wrestler".
                    "tank", "novel", "music", "band", "album", "song",
                    "drama", "series", "episode", "computer", "cpu", "insect",
                    "paramilitary", "racehorse", "team", "sport", "sports",
                    "vessels", "vessel", "vehicle", "vehicles", "gun", "weaponry",
                    "weapon"]}

coucal = {"pos": ["bird", "birds", "cuckoo", "centropodinae", "centropus", "claw",
                  "toe", "hallux", "feather", "feathers", "nests", "nest", "fly",
                  "toed", "neck", "necked", "extinct", "feet", "nestling", "nestlings",
                  "pheasant"],
          "neg": []}


sea_anemone = {"pos": ["marine", "predatory", "animal", "animals", "actiniaria",
                       "cnidaria", "cnidarians", "anthozoa", "hexacorallia",
                       "float", "polyp", "water", "muscles", "nerves", "swimming",
                       "mouth", "gullet", "predator", "fish"],
               "neg": ["plant", "plants", "flowering", "flower", "botanist",
                       "pulsatilla", "hepatica", "leaves", "leaf", "flowers",
                       "fruits", "fruit", "shrimp", "song", "album", "radar",
                       "actress", "actor", "series", "ship", "navy"]}

black_stork = {"pos": ["bird", "birds", "ciconiidae", "plumage", "legs", "leg",
                       "beak", "species", "water", "waters", "migrate", "eggs",
                       "egg", "nest", ],
               "neg": ["film", "motion", "star", "starring", "pakistani",
                       "army", "forces", "navy", "service"]}

sea_lion = {"pos": ["aquatic", "mammal", "mammals", "foreflipper", "ear", "eared",
                    "fur", "seals", "otariidae", "foreflippers", "belly",
                    "water", "waters", "ocean", "animal", "animals", "marine",
                    "zoo", "species", "specie", "feeding", "feed"],
            "neg": ["ablum", "studio", "invasion", "operation", "submarine",
                    "submarines", "base", "helicopter", "antarctica", "supermarine",
                    "locomotive", "park"]}

beagle = {"pos": ["dog", "dogs", "breed", "hound", "hunt", "hunting",
                  "detection", "pet", "pets", "hunted", "hares", "scent",
                  "beagling", "breeds", "hunter", "hunters"],
          "neg": ["channel", "island", "islands", "conflict", "gulf",
                  "territory", "beer", "brand", "crater", "software",
                  "ship", "darwin", "planet", "computer", "aircraft",
                  "nato", "hockey", "writer", "bay"]}

irish_setter = {"pos": ["dog", "dogs", "breed", "breeds", "gundog", "bred",
                        "hunting", "hunt", "hunting", "hunter", "hunters",
                        "gamebirds", "gamebird", "gundogs"],
                "neg": []}

malinois = {"pos": ["dog", "dogs", "breed", "breeds", "detection", "explosives",
                    "accelerants", "narcotics", "belgian", "police",
                    "search", "rescue", "sheepdog", "working"],
            "neg": []}

collie = {"pos": ["dog", "dogs", "breed", "breeds", "herding", "sports", "fur",
                  "pet", "pets", "sport", "showing", "shepherd", "blanco",
                  "reveille", "shep", "border", "bearded", "farm"],
          "neg": ["village", "river", "mountain", "nickname", "surname",
                  "artist", "naval", "squirrel", "cannabis"]}

siberian_husky = {"pos": ["dog", "dogs", "breed", "breeds", "working",
                          "spitz", "hunt", "hunting", "hunter", "hunter"],
                  "neg": []}

langur = {"pos": ["monkey", "colobus", "colobine", "primates", "primate",
                  "tail", "tails", "leaves", "fruits"],
          "neg": ["bala"]}

buckle = {"pos": ["loose", "fasten", "fastening", "belt", "device",
                  "chape", "frame", "bar", "prong"],
          "neg": ["casting", "defect", "surname", "comic", "dessert",
                  "food", "episode", "sport", "sports", "veteran",
                  "automobile", "blimp", "inc", "accessory", "footwear"]}

candle = {"pos": ["wick", "ignitable", "flammable", "tallow", "fragrance",
                  "chandlery", "tabletop", "heat", "illumination",
                  "lighting", "flame", "light"],
          "neg": ["place", "places", "lake", "novel", "music", "band",
                  "rock", "record", "albums", "song". "singer", "singers",
                  "corporation", "facility", "company", "fish", "hap",
                  "ice", "tree", "trees", "conical", "cone", "geometry",
                  "rod", "machinery", "forum", "ratio", "haircut",
                  "surname", "historian", "developer", "financier"]}

canoe = {"pos": ["vessel", "vessels", "paddlers", "paddler", "paddle",
                 "transport", "racing", "canoeing", "touring", "camping",
                 "wood", "fiberglass", "sporting", "sail", "sails",
                 "outboard", "outtriggers", "outtrigger", "hull", "ship",
                 "hulls", "canvas", "sprint", "slalom", "wildwater",
                 "freestyle", "blade", "paddles", "boat"],
         "neg": ["city", "district", "volcano", "reserve", "park",
                 "portal", "war", "leader", "restaurant"]}

carton = {"pos": ["box", "container", "paperboard", "fiberboard", "packaging",
                  "package", "folding", "egg", "eggs", "trays", "tray",
                  "aseptic", "liquids", "liquid", "gable"],
          "neg": ["surname", "personality", "singer", "songwriter",
                  "guitarist", "politician", "hurler", "physician",
                  "actress", "philosopher", "hero", "heroes"]}

cowboy_boot = {"pos": ["riding", "worn", "cuban", "toe", "shaft",
                       "leatuer"],
               "neg": []}

dam = {"pos": ["barrier", "water", "impoundment", "streams", "stream",
               "flower", "floods", "flood", "hydropower", "floodgates",
               "levees", "levee", "floodgate", "gravity", "barrage".
               "arch", "embankment", "saddle", "weir", "tailings",
               "spillways", "spillway", "hydroelectric", "reclamation",
               "rock"],
       "neg": ["square", "band", "reservoir", "coin", "chemotherapy",
               "methylase", "dance", "animal", "people",
               "minster", "biochemist", "physiologist", "footballer",
               "boxer", "politician", "artist"]}

dutch_oven = {"pos": ["cooking", "pot", "walled", "lid", "iron",
                      "casserole", "dishes", "cookware", "braadpan",
                      "roasting", "sudderpan", "simmerpan", "simmering",
                      "heating", "camping", "chuckwagon", "bedourie",
                      "cookpot", "potjie", "chugunok"],
              "neg": ["racehorse", "thoroughbred", "pillar", "newspaper",
                      "satirical", "pitcher", "derek"]}

fountain = {"pos": ["fons", "spring", "water", "basin", "jets", "drinking",
                    "decorative", "dramatic", "springs", "aqueducts",
                    "decoration", "decorated", "basins", "musical",
                    "splash", "source"],
            "neg": ["heraldry", "soda", "space", "elevator", "pen",
                    "writing", "clarinetist", "writer", "minister", "lawyer"]
            }

hard_disk = {"pos": ["data", "storage", "device", "magnetic", "retrieve",
                     "digital", "hdd", "stored", "retrieved", "computer",
                     "capacity", "hdds", "usb", "sata", "cylinder", "recording",
                     "information", "spindle", "logical", "blocks", "latency",
                     "noise", "desktop", "laptop", "recorders"],
             "neg": ["album", "fictional", "character", "radio", "film", "novel"]
             }

data_root = "../data/hc_word_feat"
save_feature(data_root, tench, 'tench')
